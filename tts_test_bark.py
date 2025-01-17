import modal, torch, io, ast, base64, pickle, re, numpy as np, nltk, os
from tqdm import tqdm
from transformers import AutoProcessor, BarkModel
from scipy.io import wavfile
from pydub import AudioSegment
from accelerate import Accelerator
from fasthtml.common import fast_app, H1, P, Div, Form, Button, Group, Title, Main, Audio


app = modal.App("bark-tts-example")
image = (modal.Image.debian_slim(python_version="3.10")
    .pip_install("python-fasthtml==0.12.0", "transformers==4.46.1", "accelerate>=0.26.0", "scipy", "pydub", "tqdm","nltk"))
pdf_volume = modal.Volume.lookup("pdf_uploads")
BARK_DIR = "/bark"
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
    volumes={BARK_DIR: modal.Volume.lookup("bark"), "/pdf_uploads": pdf_volume}
)
@modal.asgi_app()
def serve():
    fasthtml_app, rt = fast_app()
    accelerator = Accelerator()
    bark_model = BarkModel.from_pretrained(BARK_DIR, torch_dtype=torch.float16).to(device)
    bark_processor = AutoProcessor.from_pretrained(BARK_DIR)
    bark_model, bark_processor = accelerator.prepare(bark_model, bark_processor)

    def sentence_splitter(text):
        sentences = nltk.sent_tokenize(text)
        return [s.strip() for s in sentences if s.strip()]

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

    speaker_voice_mapping = {"Speaker 1": "v2/en_speaker_0", "Speaker 2": "v2/en_speaker_6"}
    default_preset = "v2/en_speaker_0"

    def generate_speaker_audio(text, speaker):
        text = preprocess_text(str(text))
        voice_preset = speaker_voice_mapping.get(speaker, default_preset)
        inputs = bark_processor(text, voice_preset=voice_preset, return_tensors="pt").to(device)
        try:
            gen = bark_model.generate(
                **inputs,
                temperature=0.9,
                semantic_temperature=0.8,
                min_eos_p=0.05
            )
            audio_arr = gen.cpu().float().numpy().squeeze()
            if np.isnan(audio_arr).any() or np.isinf(audio_arr).any():
                audio_arr = np.zeros(24000)
        except Exception as e:
            audio_arr = np.zeros(24000)
        return audio_arr, 24000

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
        form = Form(Group(Button("Generate Podcast")), hx_post="/test", hx_swap="afterbegin", method="post")
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
        if isinstance(transcript, str):
            transcript = transcript.strip()
            if transcript.startswith("PODCAST_TEXT ="):
                transcript = transcript.split("=", 1)[1].strip()
            try:
                transcript = ast.literal_eval(transcript)
            except Exception as e:
                return Div(P(f"Error parsing transcript data: {e}"), id="result")
        segments = []
        rates = []
        for speaker, text in tqdm(transcript, desc="Generating podcast segments", unit="segment"):
            speaker = speaker.strip()
            all_sentences = sentence_splitter(text)
            for sentence in all_sentences:
                arr, sr = generate_speaker_audio(sentence, speaker)
                segments.append(arr)
                rates.append(sr)
        final_audio = concatenate_audio_segments(segments, rates)
        final_audio_path = f"/pdf_uploads/final_podcast_audio_{file_uuid}.wav"
        final_audio.export(final_audio_path, format="wav")
        return Div(P("✅ TTS Generation Completed!", cls="text-green-500"), Div(audio_player(final_audio_path), id="audio-player"), id="result")
    return fasthtml_app

if __name__ == "__main__":
    serve()






















