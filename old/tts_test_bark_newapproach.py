import os, io, base64, ast, pickle, re
import shutil
import nltk
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile
from pydub import AudioSegment
from fasthtml.common import fast_app, H1, P, Div, Form, Button, Group, Title, Main, Audio


# Download NLTK punkt if needed
NLTK_DATA_DIR = "/tmp/nltk_data"
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DATA_DIR)
nltk.download("punkt", download_dir=NLTK_DATA_DIR)

import modal
app = modal.App("bark-tts-example")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "TTS",
        "nltk",
        "pydub",
        "python-fasthtml==0.12.0",
        "scipy",
        "tqdm",
    )
)

pdf_volume = modal.Volume.lookup("pdf_uploads")


def sentence_splitter(text):
    sents = nltk.sent_tokenize(text)
    return [s.strip() for s in sents if s.strip()]

def preprocess_text(text):
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s.,!?-]", "", text)
    return text

def numpy_to_audio_segment(audio_arr, sr):
    audio_int16 = (audio_arr * 32767).astype(np.int16)
    bio = io.BytesIO()
    wavfile.write(bio, sr, audio_int16)
    bio.seek(0)
    return AudioSegment.from_wav(bio)

# Example speaker mapping.
# For custom voices, ensure you have a folder "bark_voices/<speaker>/" containing a speaker.wav file.
speaker_voice_mapping = {"Speaker 1": "ljspeech", "Speaker 2": "random"}
default_speaker_id = "random"

def generate_speaker_audio_longform(full_text, speaker):
    speaker_id = speaker_voice_mapping.get(speaker, default_speaker_id)
    processed_text = preprocess_text(full_text)
    sentences = sentence_splitter(processed_text)
    if not sentences:
        # Return one second of silence if no valid text.
        return np.zeros(24000, dtype=np.float32), 24000
    # The TTS API supports batch synthesis. Here we use it for our list of sentences.
    # If speaker_id is 'random', a new random voice is used each time.
    # For a specific speaker, ensure that the speaker embedding is available in "bark_voices/".
    output_wavs = []
    for sent in sentences:
        wav = tts.tts(sent, speaker=speaker_id)
        output_wavs.append(wav)
    final_arr = np.concatenate(output_wavs, axis=0)
    return final_arr, 24000

def concatenate_audio_segments(segments, rates):
    final_audio = None
    for seg, sr in zip(segments, rates):
        audio_seg = numpy_to_audio_segment(seg, sr)
        if final_audio is None:
            final_audio = audio_seg
        else:
            final_audio = final_audio.append(audio_seg, crossfade=100)
    return final_audio

def audio_player(file_path):
    if not os.path.exists(file_path):
        return P("No audio file found.")
    with open(file_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return Audio(src=f"data:audio/wav;base64,{b64}", controls=True)

@modal.function(
    image=image,
    gpu=modal.gpu.A100(count=1, size="80GB"),
    container_idle_timeout=10*60,
    timeout=24*60*60,
    allow_concurrent_inputs=100,
    volumes={"/pdf_uploads": pdf_volume}
)
@modal.asgi_app()
def serve():
    fasthtml_app, rt = fast_app()
    
    # Use Coqui TTS API to load the official Bark model (auto-downloads proper weights)
    # Note: Bark on CPU is quite slow so we recommend GPU.
    from TTS.api import TTS
    tts = TTS("tts_models/multilingual/multi-dataset/bark", gpu=True)

    @rt("/")
    def homepage():
        form = Form(
            Group(Button("Generate Podcast")),
            hx_post="/test",
            hx_swap="afterbegin",
            method="post"
        )
        return (
            Title("Bark Podcast TTS"),
            Main(
                H1("Click 'Generate Podcast' to Create Audio"),
                form,
                Div(id="result")
            )
        )

    @rt("/test", methods=["POST"])
    async def test_gen(request):
        file_uuid = "pkl_tts_" + os.urandom(4).hex()
        pickle_path = "/pdf_uploads/podcast_ready_data.pkl"
        if not os.path.exists(pickle_path):
            return Div(P("⚠️ No 'podcast_ready_data.pkl' found."), id="result")
        with open(pickle_path, "rb") as f:
            try:
                transcript = pickle.load(f)
            except Exception as e:
                return Div(P(f"Error loading pickle: {e}"), id="result")

        if isinstance(transcript, str):
            transcript = transcript.strip()
            if transcript.startswith("PODCAST_TEXT ="):
                transcript = transcript.split("=", 1)[1].strip()
            try:
                transcript = ast.literal_eval(transcript)
            except Exception as e:
                return Div(P(f"Error parsing transcript data: {e}"), id="result")

        segments, rates = [], []
        for speaker, text in tqdm(transcript, desc="Generating podcast segments", unit="segment"):
            audio_arr, sr = generate_speaker_audio_longform(text, speaker)
            segments.append(audio_arr)
            rates.append(sr)

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

























