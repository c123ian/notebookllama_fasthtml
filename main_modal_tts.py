import modal
import torch
import io
import ast
import base64
import sqlite3
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.io import wavfile
from pydub import AudioSegment
from accelerate import Accelerator
from fasthtml.common import (
    fast_app, H1, P, Div, Form, Input, Button, Group,
    Title, Main, Progress, Audio
)
from parler_tts import ParlerTTSForConditionalGeneration

app = modal.App("simple-fasthtml-example")
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "python-fasthtml==0.12.0", 
        "transformers==4.46.1", 
        "accelerate>=0.26.0", 
        "scipy", 
        "pydub", 
        "tqdm", 
        "parler_tts"
    )
)
MODELS_DIR = "/llama_mini"
TTS_DIR = "/tts"  
device = "cuda" if torch.cuda.is_available() else "cpu"

SYS_PROMPT = """
........ 

ALWAYS START YOUR RESPONSE DIRECTLY WITH SPEAKER 1: 
DO NOT GIVE EPISODE TITLES SEPERATELY, LET SPEAKER 1 TITLE IT IN HER SPEECH
DO NOT GIVE CHAPTER TITLES
IT SHOULD STRICTLY BE THE DIALOGUES
"""
SYSTEMP_PROMPT ="""
You are an international oscar winnning screenwriter

You have been working with multiple award winning podcasters.

Example of response:
[
    ("Speaker 1", "Welcome to our podcast, where we explore the latest advancements in AI and technology. I'm your host, and today we're joined by a renowned expert in the field of AI. We're going to dive into the exciting world of Llama 3.2, the latest release from Meta AI."),
    ("Speaker 2", "Hi, I'm excited to be here! So, what is Llama 3.2?"),
    ("Speaker 1", "Ah, great question! Llama 3.2 is an open-source AI model that allows developers to fine-tune, distill, and deploy AI models anywhere. It's a significant update from the previous version, with improved performance, efficiency, and customization options."),
    ("Speaker 2", "That sounds amazing! What are some of the key features of Llama 3.2?")
]
"""

try:
    data_volume = modal.Volume.lookup("my_data_volume", create_if_missing=True)
except modal.exception.NotFoundError:
    data_volume = modal.Volume.persisted("my_data_volume")
try:
    llm_volume = modal.Volume.lookup("llama_mini", create_if_missing=False)
except modal.exception.NotFoundError:
    raise Exception("Download your Llama model files first with the appropriate script.")
try:
    tts_volume = modal.Volume.lookup("tts", create_if_missing=False)
except modal.exception.NotFoundError:
    raise Exception("Download your TTS model files first with the appropriate script.")

def numpy_to_audio_segment(audio_arr, sampling_rate):
    audio_int16 = (audio_arr * 32767).astype(np.int16)
    byte_io = io.BytesIO()
    wavfile.write(byte_io, sampling_rate, audio_int16)
    byte_io.seek(0)
    return AudioSegment.from_wav(byte_io)

@app.function(
    image=image,
    gpu=modal.gpu.A100(count=1, size="40GB"),
    container_idle_timeout=10 * 60,
    timeout=24 * 60 * 60,
    allow_concurrent_inputs=100,
    volumes={MODELS_DIR: llm_volume, "/data": data_volume, TTS_DIR: tts_volume}
)
@modal.asgi_app()
def serve():
    import os
    UPLOAD_FOLDER = "/data/uploads"
    AUDIO_FILE_PATH = "/data/placeholder_audio.mp3"
    DB_PATH = "/data/uploads.db"
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    fasthtml_app, rt = fast_app()
    print("🚀 Loading language model...")
    accelerator = Accelerator()
    model = AutoModelForCausalLM.from_pretrained(
        MODELS_DIR,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(MODELS_DIR)
    model, tokenizer = accelerator.prepare(model, tokenizer)
    tts_model = ParlerTTSForConditionalGeneration.from_pretrained(TTS_DIR,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    tts_tokenizer = AutoTokenizer.from_pretrained(TTS_DIR)
    speaker1_description = """
Laura's voice is expressive and dramatic in delivery, speaking at a moderately fast pace with a very close recording that almost has no background noise.
"""
    speaker2_description = """ 
Gary's voice is expressive and dramatic in delivery, speaking at a slow pace with a very close recording that almost has no background noise.
"""
    def generate_speaker_audio(text, speaker="Speaker 1"):
        if not isinstance(text, str):
            text = str(text)
        desc = speaker1_description if speaker == "Speaker 1" else speaker2_description
        print(f"🎤 Generating TTS audio for {speaker}...")
        input_ids = tts_tokenizer(desc, return_tensors="pt").input_ids.to(device)
        prompt_input_ids = tts_tokenizer(text, return_tensors="pt").input_ids.to(device)
        generation = tts_model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
        # Convert the generated tensor to float32 on CPU
        generation = generation.cpu().float()
        audio_arr = generation.numpy().squeeze()
        return audio_arr, tts_model.config.sampling_rate
    def create_word_bounded_chunks(text, target_chunk_size):
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        for word in words:
            word_length = len(word) + 1
            if current_length + word_length > target_chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        return chunks
    def process_chunk(text_chunk, chunk_num, model, tokenizer):
        conversation = [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": text_chunk},
        ]
        prompt = tokenizer.apply_chat_template(conversation, tokenize=False)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                temperature=0.7,
                top_p=0.9,
                max_new_tokens=512
            )
        processed_text = tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):].strip()
        return processed_text
    async def process_uploaded_file(filename):
        print(f"📥 File '{filename}' uploaded!")
        output_file = os.path.join("/data", f"clean_{filename}")
        input_file = os.path.join(UPLOAD_FOLDER, filename)
        with open(input_file, "r", encoding="utf-8") as file:
            text = file.read()
        chunks = create_word_bounded_chunks(text, 1000)
        with open(output_file, "w", encoding="utf-8") as out_file:
            for chunk_num, chunk in enumerate(chunks):
                cleaned_chunk = process_chunk(chunk, chunk_num, model, tokenizer)
                out_file.write(cleaned_chunk + "\n")
                out_file.flush()
        print(f"🧹 File '{filename}' cleaned and saved to '{output_file}'!")
        return output_file
    def load_audio_base64(audio_path: str):
        with open(audio_path, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")
    def audio_player():
        if not os.path.exists(AUDIO_FILE_PATH):
            return P("No placeholder audio file found.")
        audio_data = load_audio_base64(AUDIO_FILE_PATH)
        return Audio(src=f"data:audio/mp4;base64,{audio_data}", controls=True)
    def progress_bar(percent):
        return Progress(
            id="progress_bar",
            value=str(percent),
            max="1",
            hx_get=f"/update_progress?percent={percent}",
            hx_trigger="every 500ms",
            hx_swap="outerHTML",
        )
    def read_file_to_string(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    @rt("/")
    def homepage():
        upload_input = Input(type="file", name="document", accept=".txt", required=True)
        form = Form(
            Group(upload_input, Button("Upload")),
            hx_post="/upload",
            hx_swap="afterbegin",
            enctype="multipart/form-data",
            method="post",
        )
        return Title("Simple Upload + Progress & Audio"), Main(
            H1("Simple Uploader with Progress & Audio"),
            form,
            Div(id="upload-info"),
            cls="container mx-auto p-4"
        )
    @rt("/upload", methods=["POST"])
    async def upload_doc(request):
        form = await request.form()
        docfile = form.get("document")
        if not docfile:
            return Div(
                P("⚠️ No file uploaded. Please try again.", cls="text-red-500"),
                id="upload-info"
            )
        contents = await docfile.read()
        save_path = os.path.join(UPLOAD_FOLDER, docfile.filename)
        with open(save_path, "wb") as f:
            f.write(contents)
        cursor.execute("INSERT INTO uploads (filename) VALUES (?)", (docfile.filename,))
        conn.commit()
        cursor.execute("SELECT filename FROM uploads ORDER BY uploaded_at DESC LIMIT 1")
        recent_file = cursor.fetchone()[0]
        output_file_path = await process_uploaded_file(recent_file)
        INPUT_PROMPT = read_file_to_string(output_file_path)
        print("📝 Generating first script...")
        messages = [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": INPUT_PROMPT},
        ]
        first_pipeline = __import__("transformers").pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto",
        )
        first_outputs = first_pipeline(
            messages,
            max_new_tokens=8126,
            temperature=1,
        )
        first_generated_text = first_outputs[0]["generated_text"]
        print("✍️  First script generated.")
        print("🔄 Rewriting script with disfluencies...")
        rewriting_messages = [
            {"role": "system", "content": SYSTEMP_PROMPT},
            {"role": "user", "content": first_generated_text},
        ]
        second_outputs = first_pipeline(
            rewriting_messages,
            max_new_tokens=8126,
            temperature=1,
        )
        final_rewritten_text = second_outputs[0]["generated_text"]
        print("✅ Script rewritten:")
        print(final_rewritten_text)
        try:
            dialogue = ast.literal_eval(final_rewritten_text)
        except Exception as e:
            print("❌ Error parsing final_rewritten_text to a Python literal:", e)
            dialogue = [("Speaker 1", final_rewritten_text)]
        final_audio = None
        print("🎧 Generating podcast segments (TTS audio)...")
        for speaker, text in tqdm(dialogue, desc="Generating podcast segments", unit="segment"):
            audio_arr, rate = generate_speaker_audio(text, speaker=speaker)
            audio_segment = numpy_to_audio_segment(audio_arr, rate)
            if final_audio is None:
                final_audio = audio_segment
            else:
                final_audio += audio_segment
        final_audio_path = "/data/final_podcast_audio.wav"
        final_audio.export(final_audio_path, format="wav")
        print("🎶 Final podcast audio generated and saved to", final_audio_path)
        return Div(
            P(f"✅ File '{docfile.filename}' uploaded and processed successfully!", cls="text-green-500"),
            progress_bar(0),
            id="processing-results"
        )
    return fasthtml_app

def audio_player(file_path="/data/final_podcast_audio.wav"):
    import os
    if not os.path.exists(file_path):
        return P("No audio file found.")
    with open(file_path, "rb") as f:
        audio_data = base64.b64encode(f.read()).decode("ascii")
    return Audio(src=f"data:audio/wav;base64,{audio_data}", controls=True)

if __name__ == "__main__":
    serve()











































































