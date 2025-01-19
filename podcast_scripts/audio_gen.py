import modal, torch, io, ast, base64, pickle, re, numpy as np, nltk, os
from tqdm import tqdm
from scipy.io import wavfile
from pydub import AudioSegment
from fasthtml.common import fast_app, H1, P, Div, Form, Button, Group, Title, Main, Audio

# Import from the Bark repo (make sure you installed Bark from GitHub).
from bark import SAMPLE_RATE, preload_models, generate_audio

app = modal.App("audio_gen")
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

pdf_volume = modal.Volume.lookup("pdf_uploads")
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
    volumes={
        "/pdf_uploads": pdf_volume
    }
)
@modal.asgi_app()
def serve():
    fasthtml_app, rt = fast_app()

    # Preload all Bark models (only needed once).
    preload_models()

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

    # speaker mapping 
    # for testing reasons choose 9 'feminine' voice / speaker 6 'masculine' voice
    speaker_voice_mapping = {"Speaker 1": "v2/en_speaker_9", "Speaker 2": "v2/en_speaker_6"}
    default_preset = "v2/en_speaker_9"

    # ------------------------------------------------------------------------
    # NEW LONG-FORM APPROACH using generate_audio(..., output_full=True)
    # ------------------------------------------------------------------------
    def generate_speaker_audio_longform(full_text, speaker):
        """
        Splits `full_text` into sentences and uses `generate_audio` in a loop.
        Each time, we pass the 'full generation' dict as `history_prompt` to the next chunk
        to keep the same voice across chunks.
        """
        voice_preset = speaker_voice_mapping.get(speaker, default_preset)
        full_text = preprocess_text(full_text)
        sentences = sentence_splitter(full_text)

        # We will accumulate multiple audio chunks and short silences in a list
        all_audio = []
        chunk_silence = np.zeros(int(0.25 * SAMPLE_RATE), dtype=np.float32)
        prev_generation_dict = None

        for sent in sentences:
            # Generate chunk with the high-level Bark call.
            # By passing output_full=True, we get (generation_dict, audio_array).
            generation_dict, audio_array = generate_audio(
                text=sent,
                history_prompt=prev_generation_dict if prev_generation_dict else voice_preset,
                output_full=True,
                text_temp=0.7,      # optional
                waveform_temp=0.7,  # optional
            )
            # Chain for next time
            prev_generation_dict = generation_dict

            # Add to final list
            all_audio.append(audio_array)
            all_audio.append(chunk_silence)

        if not all_audio:
            # Return 1 second of silence if there's no text
            return np.zeros(24000, dtype=np.float32), SAMPLE_RATE

        # Concatenate all the chunk arrays
        final_arr = np.concatenate(all_audio, axis=0)
        return final_arr, SAMPLE_RATE

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

    @rt("/")
    def homepage():
        form = Form(Group(Button("Generate Podcast Audio")), hx_post="/test", hx_swap="afterbegin", method="post")
        return Title("Bark Podcast TTS"), Main(H1("Click 'Generate Podcast' to Create Audio"), form, Div(id="result"))

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

        # If transcript is a string, parse it as Python data
        if isinstance(transcript, str):
            transcript = transcript.strip()
            if transcript.startswith("PODCAST_TEXT ="):
                transcript = transcript.split("=", 1)[1].strip()
            try:
                transcript = ast.literal_eval(transcript)
            except Exception as e:
                return Div(P(f"Error parsing transcript data: {e}"), id="result")

        # 'transcript' is presumably a list of (speaker, text) pairs
        segments, rates = [], []

        for speaker, text in tqdm(transcript, desc="Generating podcast segments", unit="segment"):
            speaker = speaker.strip()
            audio_arr, sr = generate_speaker_audio_longform(text, speaker)
            segments.append(audio_arr)
            rates.append(sr)

        # Concatenate all speaker blocks into final WAV
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






















