import modal
import torch
import io
import ast
import base64
import sqlite3
import uuid
import re
import numpy as np
import nltk
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.io import wavfile
from pydub import AudioSegment
from accelerate import Accelerator
from fasthtml.common import fast_app, H1, P, Div, Form, Input, Button, Group, Title, Main, Audio
from bark import SAMPLE_RATE, preload_models, generate_audio

def download_nltk_data():
    nltk.download("punkt", download_dir="/tmp/nltk_data")

app = modal.App("bark-llama-app")
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg")
    .pip_install(
        "git+https://github.com/suno-ai/bark.git",
        "nltk",
        "pydub",
        "python-fasthtml==0.12.0",
        "scipy",
        "tqdm",
        "transformers==4.46.1",
        "accelerate>=0.26.0"
    )
    .run_function(download_nltk_data)
)

BARK_DIR = "/bark"
LLAMA_DIR = "/llama_mini"
DATA_DIR = "/data"

try:
    data_volume = modal.Volume.lookup("my_data_volume", create_if_missing=True)
except modal.exception.NotFoundError:
    data_volume = modal.Volume.persisted("my_data_volume")

try:
    llm_volume = modal.Volume.lookup("llama_mini", create_if_missing=False)
except modal.exception.NotFoundError:
    raise Exception("Download your Llama model files first with the appropriate script.")

NLTK_DATA_DIR = "/tmp/nltk_data"
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DATA_DIR)

def sentence_splitter(text):
    try:
        sents = nltk.sent_tokenize(text)
    except LookupError:
        nltk.download("punkt", download_dir=NLTK_DATA_DIR)
        sents = nltk.sent_tokenize(text)
    return [s.strip() for s in sents if s.strip()]

def preprocess_text(text):
    text = text.strip()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

def numpy_to_audio_segment(audio_arr, sr):
    audio_int16 = (audio_arr * 32767).astype(np.int16)
    bio = io.BytesIO()
    wavfile.write(bio, sr, audio_int16)
    bio.seek(0)
    return AudioSegment.from_wav(bio)

def generate_speaker_audio_longform(full_text, speaker):
    speaker_voice_mapping = {
        "Speaker 1": "v2/en_speaker_0",
        "Speaker 2": "v2/en_speaker_6"
    }
    voice_preset = speaker_voice_mapping.get(speaker, "v2/en_speaker_0")
    full_text = preprocess_text(full_text)
    sentences = sentence_splitter(full_text)
    all_audio = []
    prev_generation_dict = None
    for sent in sentences:
        generation_dict, audio_array = generate_audio(
            text=sent,
            history_prompt=prev_generation_dict if prev_generation_dict else voice_preset,
            output_full=True,
            text_temp=0.7,
            waveform_temp=0.7,
        )
        prev_generation_dict = generation_dict
        all_audio.append(audio_array)
        chunk_silence = np.zeros(int(0.15 * SAMPLE_RATE), dtype=np.float32)
        all_audio.append(chunk_silence)
    if not all_audio:
        return np.zeros(24000, dtype=np.float32), SAMPLE_RATE
    return np.concatenate(all_audio, axis=0), SAMPLE_RATE

def concatenate_audio_segments(segments, rates):
    final_audio = None
    for seg, sr in zip(segments, rates):
        audio_seg = numpy_to_audio_segment(seg, sr)
        if final_audio is None:
            final_audio = audio_seg
        else:
            final_audio = final_audio.append(audio_seg, crossfade=150)
    return final_audio

def audio_player(file_path):
    if not os.path.exists(file_path):
        return P("No audio file found.")
    with open(file_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return Audio(src=f"data:audio/wav;base64,{b64}", controls=True)

@app.function(
    image=image,
    gpu=modal.gpu.A100(count=1, size="80GB"),
    container_idle_timeout=10*60,
    timeout=24*60*60,
    allow_concurrent_inputs=100,
    volumes={BARK_DIR: modal.Volume.lookup("bark"), LLAMA_DIR: llm_volume, DATA_DIR: data_volume}
)
@modal.asgi_app()
def serve():
    UPLOAD_FOLDER = "/data/uploads"
    DB_PATH = "/data/uploads.db"
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")
    conn.commit()

    fasthtml_app, rt = fast_app()

    preload_models()

    accelerator = Accelerator()
    model = AutoModelForCausalLM.from_pretrained(
        LLAMA_DIR,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_DIR)
    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        tokenizer.add_special_tokens({'pad_token': '<pad>'})
        tokenizer.pad_token = '<pad>'
    model, tokenizer = accelerator.prepare(model, tokenizer)

    @rt("/")
    def homepage():
        upload_input = Input(type="file", name="document", accept=".txt", required=True)
        form = Form(
            Group(upload_input, Button("Upload")),
            hx_post="/upload",
            hx_swap="afterbegin",
            enctype="multipart/form-data",
            method="post"
        )
        return Title("Bark + Llama"), Main(
            H1("Upload a .txt file and generate podcast audio."),
            form,
            Div(id="upload-info")
        )

    @rt("/upload", methods=["POST"])
    async def upload_doc(request):
        file_uuid = uuid.uuid4().hex
        form = await request.form()
        docfile = form.get("document")
        if not docfile:
            return Div(P("No file uploaded.", cls="text-red-500"), id="upload-info")

        contents = await docfile.read()
        save_path = os.path.join(UPLOAD_FOLDER, docfile.filename)
        with open(save_path, "wb") as f:
            f.write(contents)
        cursor.execute("INSERT INTO uploads (filename) VALUES (?)", (docfile.filename,))
        conn.commit()

        text = open(save_path, "r", encoding="utf-8").read()
        sentences = sentence_splitter(text)

        dialogue = []
        for i, sentence in enumerate(sentences):
            speaker = "Speaker 1" if i % 2 == 0 else "Speaker 2"
            dialogue.append((speaker, sentence))

        segments, rates = [], []
        for speaker, text in tqdm(dialogue, desc="Generating audio", unit="segment"):
            audio_arr, sr = generate_speaker_audio_longform(text, speaker)
            segments.append(audio_arr)
            rates.append(sr)

        final_audio = concatenate_audio_segments(segments, rates)
        final_audio_path = f"/data/final_podcast_audio_{file_uuid}.wav"
        final_audio.export(final_audio_path, format="wav")

        return Div(
            P(f"✅ File '{docfile.filename}' processed!", cls="text-green-500"),
            Div(audio_player(final_audio_path), id="audio-player"),
            id="upload-results"
        )

    return fasthtml_app

if __name__ == "__main__":
    serve()



