import modal, torch, os, pickle, ast
import numpy as np
import nltk
from tqdm import tqdm
from bark import SAMPLE_RATE, preload_models, generate_audio
from pydub import AudioSegment
from fasthtml.common import fast_app, H1, P, Div, Form, Button, Group, Title, Main, Input, Audio
from scipy.io import wavfile
import io, base64

app = modal.App("audio_gen_new")
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
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
)

device = "cuda" if torch.cuda.is_available() else "cpu"
NLTK_DATA_DIR = "/tmp/nltk_data"
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DATA_DIR)
nltk.download("punkt", download_dir=NLTK_DATA_DIR)
nltk.download("punkt_tab", download_dir=NLTK_DATA_DIR)

@app.function(
    image=image,
    gpu=modal.gpu.A100(count=1, size="80GB"),
    container_idle_timeout=10*60,
    timeout=24*60*60,
    allow_concurrent_inputs=100,
)
@modal.asgi_app()
def serve():
    fasthtml_app, rt = fast_app()
    preload_models()

    def sentence_splitter(text):
        return nltk.sent_tokenize(text)

    speaker_map = {"Speaker_1": "v2/en_speaker_9", "Speaker_2": "v2/en_speaker_6"}

    @rt("/")
    def index():
        form = Form(
            Group(Input(type="file", name="pklfile", accept=".pkl", required=True), Button("Upload")),
            hx_post="/upload", method="post", enctype="multipart/form-data"
        )
        return Title("Bark Podcast TTS"), Main(
            H1("Upload PKL with Script"),
            form,
            Div(id="result")
        )

    @rt("/upload", methods=["POST"])
    async def upload_file(request):
        formdata = await request.form()
        file_field = formdata.get("pklfile")
        if not file_field:
            return Div(P("No .pkl file uploaded."))

        try:
            script_tuples = pickle.load(file_field.file)
        except Exception as e:
            return Div(P(f"Error reading .pkl: {e}"))

        # script_tuples should be: [("Speaker_1", "..."), ("Speaker_2", "..."), ...]
        audio_segments = []
        silence = np.zeros(int(0.25 * SAMPLE_RATE), dtype=np.float32)

        for (speaker, text) in tqdm(script_tuples, desc="Generating audio"):
            # Which speaker voice to use:
            bark_preset = speaker_map.get(speaker, "v2/en_speaker_9")
            # For each line, optionally split into sentences:
            lines = sentence_splitter(text)
            for sent in lines:
                _, audio_array = generate_audio(
                    text=sent,
                    history_prompt=bark_preset,
                    output_full=True
                )
                audio_segments.append(audio_array)
                audio_segments.append(silence)

        # Concatenate
        final_audio = np.concatenate(audio_segments, axis=0)

        # Convert to WAV
        wav_bytes = io.BytesIO()
        wavfile.write(wav_bytes, SAMPLE_RATE, (final_audio * 32767).astype(np.int16))
        wav_bytes.seek(0)
        b64 = base64.b64encode(wav_bytes.read()).decode("ascii")

        return Div(
            P("✅ TTS Completed"),
            Audio(src=f"data:audio/wav;base64,{b64}", controls=True)
        )

    return fasthtml_app

if __name__ == "__main__":
    serve()
