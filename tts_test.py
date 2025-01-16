import modal
import torch
import io
import ast
import base64
import pickle
import re
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

app = modal.App("test-audiogen-example")
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

pdf_volume = modal.Volume.lookup("pdf_uploads")  # Must exist
MODELS_DIR = "/llama_mini"
TTS_DIR = "/tts"
device = "cuda" if torch.cuda.is_available() else "cpu"

@app.function(
    image=image,
    gpu=modal.gpu.A100(count=1, size="80GB"),
    container_idle_timeout=10 * 60,
    timeout=24 * 60 * 60,
    allow_concurrent_inputs=100,
    volumes={MODELS_DIR: modal.Volume.lookup("llama_mini"),
             TTS_DIR: modal.Volume.lookup("tts"),
             "/pdf_uploads": pdf_volume}
)
@modal.asgi_app()
def serve():
    import os
    fasthtml_app, rt = fast_app()
    print("🚀 Loading TTS model...")
    accelerator = Accelerator()
    tts_model = ParlerTTSForConditionalGeneration.from_pretrained(
        TTS_DIR, torch_dtype=torch.bfloat16, device_map=device
    )
    tts_tokenizer = AutoTokenizer.from_pretrained(TTS_DIR)
    if tts_tokenizer.pad_token is None:
        tts_tokenizer.add_special_tokens({'pad_token': '<pad>'})
        tts_tokenizer.pad_token = '<pad>'
    tts_model, tts_tokenizer = accelerator.prepare(tts_model, tts_tokenizer)
    def preprocess_text(text):
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        return text
    def numpy_to_audio_segment(audio_arr, sr):
        audio_int16 = (audio_arr * 32767).astype(np.int16)
        bio = io.BytesIO()
        wavfile.write(bio, sr, audio_int16)
        bio.seek(0)
        return AudioSegment.from_wav(bio)
    speaker1_description = "Laura's voice is expressive and dramatic in delivery, speaking at a moderately fast pace with a very close recording that almost has no background noise."
    speaker2_description = "Gary's voice is monotone, speaking at a slow pace with a very close recording that almost has no background noise."
    def generate_speaker_audio(text, speaker):
        text = preprocess_text(str(text))
        desc = speaker1_description if speaker == "Speaker 1" else speaker2_description
        input_data = tts_tokenizer(desc, return_tensors="pt", max_length=512, padding="max_length").to(device)
        prompt_data = tts_tokenizer(text, return_tensors="pt", max_length=512, padding="max_length").to(device)
        try:
            gen = tts_model.generate(input_data["input_ids"], attention_mask=input_data["attention_mask"],
                                     prompt_input_ids=prompt_data["input_ids"], temperature=0.9)
            audio_arr = gen.cpu().float().numpy().squeeze()
            if np.isnan(audio_arr).any() or np.isinf(audio_arr).any():
                audio_arr = np.zeros(int(tts_model.config.sampling_rate))
        except:
            audio_arr = np.zeros(int(tts_model.config.sampling_rate))
        return audio_arr, tts_model.config.sampling_rate
    def concatenate_audio_segments(segments, rates):
        final_audio = None
        for seg, sr in zip(segments, rates):
            audio_seg = numpy_to_audio_segment(seg, sr)
            final_audio = audio_seg if final_audio is None else final_audio.append(audio_seg, crossfade=100)
        return final_audio
    def audio_player(file_path):
        if not os.path.exists(file_path):
            return P("No audio file found.")
        with open(file_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return Audio(src=f"data:audio/wav;base64,{b64}", controls=True)
    @rt("/")
    def homepage():
        form = Form(
            Group(Button("Test")),
            hx_post="/test",
            hx_swap="afterbegin",
            method="post"
        )
        return Title("Podcast TTS"), Main(H1("Click 'Test' to Generate Audio"), form, Div(id="result"))
    @rt("/test", methods=["POST"])
    async def test_gen(request):
        file_uuid = "pkl_tts_" + os.urandom(4).hex()
        pickle_path = "/pdf_uploads/podcast_ready_data.pkl"
        if not os.path.exists(pickle_path):
            return Div(P("⚠️ No 'podcast_ready_data.pkl' found."), id="result")
        with open(pickle_path, "rb") as f:
            final_rewritten_text = pickle.load(f)
        # Expect final_rewritten_text to be a list of tuples
        if isinstance(final_rewritten_text, str):
            try:
                final_rewritten_text = ast.literal_eval(final_rewritten_text)
            except:
                final_rewritten_text = [("Speaker 1", final_rewritten_text)]
        segments, rates = [], []
        for speaker, text in tqdm(final_rewritten_text, desc="Generating segments"):
            arr, sr = generate_speaker_audio(text, speaker)
            segments.append(arr); rates.append(sr)
        final_audio = concatenate_audio_segments(segments, rates)
        final_audio_path = f"/pdf_uploads/final_podcast_audio_{file_uuid}.wav"
        final_audio.export(final_audio_path, format="wav")
        return Div(
            P("✅ TTS Generation Completed!", cls="text-green-500"),
            Div(audio_player(final_audio_path), id="audio-player"),
            id="result"
        )
    return fasthtml_app

if __name__ == "__main__":
    serve()


