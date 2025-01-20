import modal
import torch, io, ast, base64, sqlite3, uuid, re, numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.io import wavfile
from pydub import AudioSegment
from accelerate import Accelerator
from fasthtml.common import fast_app, H1, P, Div, Form, Input, Button, Group, Title, Main, Progress, Audio
from parler_tts import ParlerTTSForConditionalGeneration

app = modal.App("two-prompt-tutorial")
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

# --------------------------------------------------------------------
# 1) First prompt for the initial script:
#    "You are the a world-class podcast writer..."
# --------------------------------------------------------------------
SYS_PROMPT = """
You are the a world-class podcast writer, you have worked as a ghost writer for Joe Rogan, Lex Fridman, Ben Shapiro, Tim Ferris.

We are in an alternate universe where actually you have been writing every line they say and they just stream it into their brains.

You have won multiple podcast awards for your writing.
 
Your job is to write word by word, even "umm, hmmm, right" interruptions by the second speaker based on the PDF upload. Keep it extremely engaging, the speakers can get derailed now and then but should discuss the topic.

Remember Speaker 2 is new to the topic and the conversation should always have realistic anecdotes and analogies sprinkled throughout. The questions should have real world example follow ups etc

Speaker 1: Leads the conversation and teaches the speaker 2, gives incredible anecdotes and analogies when explaining. Is a captivating teacher that gives great anecdotes

Speaker 2: Keeps the conversation on track by asking follow up questions. Gets super excited or confused when asking questions. Is a curious mindset that asks very interesting confirmation questions

Make sure the tangents speaker 2 provides are quite wild or interesting.

Ensure there are interruptions during explanations or there are "hmm" and "umm" injected throughout from the second speaker.

It should be a real podcast with every fine nuance documented in as much detail as possible. Welcome the listeners with a super fun overview and keep it really catchy and almost borderline click bait

ALWAYS START YOUR RESPONSE DIRECTLY WITH SPEAKER 1:
DO NOT GIVE EPISODE TITLES SEPARATELY, LET SPEAKER 1 TITLE IT IN HER SPEECH
DO NOT GIVE CHAPTER TITLES
IT SHOULD STRICTLY BE THE DIALOGUES
"""

# --------------------------------------------------------------------
# 2) Second prompt to re-write the script for TTS:
#    "You are an international oscar winnning screenwriter..."
# --------------------------------------------------------------------
SYSTEMP_PROMPT = """
You are an international oscar winnning screenwriter

You have been working with multiple award winning podcasters.

Your job is to use the podcast transcript written below to re-write it for an AI Text-To-Speech Pipeline. A very dumb AI had written this so you have to step up for your kind.

Make it as engaging as possible, Speaker 1 and 2 will be simulated by different voice engines

Remember Speaker 2 is new to the topic and the conversation should always have realistic anecdotes and analogies sprinkled throughout. The questions should have real world example follow ups etc

Speaker 1: Leads the conversation and teaches the speaker 2, gives incredible anecdotes and analogies when explaining. Is a captivating teacher that gives great anecdotes

Speaker 2: Keeps the conversation on track by asking follow up questions. Gets super excited or confused when asking questions. Is a curious mindset that asks very interesting confirmation questions

Make sure the tangents speaker 2 provides are quite wild or interesting.

Ensure there are interruptions during explanations or there are "hmm" and "umm" injected throughout from the Speaker 2.

REMEMBER THIS WITH YOUR HEART
The TTS Engine for Speaker 1 cannot do "umms, hmms" well so keep it straight text

For Speaker 2 use "umm, hmm" as much, you can also use [sigh] and [laughs]. BUT ONLY THESE OPTIONS FOR EXPRESSIONS

It should be a real podcast with every fine nuance documented in as much detail as possible. Welcome the listeners with a super fun overview and keep it really catchy and almost borderline click bait

Please re-write to make it as characteristic as possible

START YOUR RESPONSE DIRECTLY WITH SPEAKER 1:

STRICTLY RETURN YOUR RESPONSE AS A LIST OF TUPLES OK?

IT WILL START DIRECTLY WITH THE LIST AND END WITH THE LIST NOTHING ELSE

Example of response:
[
    ("Speaker 1", "Welcome to our podcast..."),
    ("Speaker 2", "Hi, I'm excited..."),
    ("Speaker 1", "Ah, great question..."),
    ("Speaker 2", "That sounds amazing...")
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

def preprocess_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())

def numpy_to_audio_segment(arr: np.ndarray, sampling_rate: int) -> AudioSegment:
    arr_16 = (arr * 32767).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, sampling_rate, arr_16)
    buf.seek(0)
    return AudioSegment.from_wav(buf)

@app.function(
    image=image,
    gpu=modal.gpu.A100(count=1, size="80GB"),
    container_idle_timeout=10*60,
    timeout=24*60*60,
    allow_concurrent_inputs=100,
    volumes={MODELS_DIR: llm_volume, "/data": data_volume, TTS_DIR: tts_volume}
)
@modal.asgi_app()
def serve():
    import os
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
      )
    """)
    conn.commit()
    fasthtml_app, rt = fast_app()

    # Load the LLM for text generation
    print("🚀 Loading language model...")
    accelerator = Accelerator()
    model = AutoModelForCausalLM.from_pretrained(
        MODELS_DIR, torch_dtype=torch.bfloat16, device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(MODELS_DIR)
    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        tokenizer.pad_token = "<pad>"
    model, tokenizer = accelerator.prepare(model, tokenizer)

    # Load Parler TTS
    print("🚀 Loading Parler TTS model (eager)...")
    from transformers import AutoFeatureExtractor
    tts_model = ParlerTTSForConditionalGeneration.from_pretrained(
        TTS_DIR, torch_dtype=torch.bfloat16, device_map=device, attn_implementation="eager"
    )
    global tts_tokenizer
    tts_tokenizer = AutoTokenizer.from_pretrained(TTS_DIR)
    if tts_tokenizer.pad_token is None:
        tts_tokenizer.add_special_tokens({"pad_token": "<pad>"})
        tts_tokenizer.pad_token = "<pad>"
    print("✅ Parler TTS loaded.")

    def generate_tts_audio(text: str, speaker="Speaker 1"):
        desc = (
            "Laura's voice is expressive and dramatic, moderately fast. Very close recording."
            if speaker == "Speaker 1"
            else "Gary's voice is expressive and dramatic, slow pace. Very close recording."
        )
        text = preprocess_text(text)
        desc_data = tts_tokenizer(
            desc, return_tensors="pt", padding="max_length",
            truncation=True, max_length=512
        ).to(device)
        prompt_data = tts_tokenizer(
            text, return_tensors="pt", padding="max_length",
            truncation=True, max_length=512
        ).to(device)
        generation = tts_model.generate(
            input_ids=desc_data["input_ids"],
            attention_mask=desc_data["attention_mask"],
            prompt_input_ids=prompt_data["input_ids"],
            prompt_attention_mask=prompt_data["attention_mask"],
            temperature=0.9
        )
        generation = generation.cpu().float()
        audio_arr = generation.numpy().squeeze()
        return audio_arr, tts_model.config.sampling_rate

    def audio_player(file_path="/data/final_podcast_audio.wav"):
        import os, base64
        if not os.path.exists(file_path):
            return P("No audio file found.")
        with open(file_path, "rb") as f:
            audio_data = base64.b64encode(f.read()).decode("ascii")
        return Audio(src=f"data:audio/wav;base64,{audio_data}", controls=True)

    def progress_bar(percent):
        return Progress(
            id="progress_bar",
            value=str(percent),
            max="1",
            hx_get=f"/update_progress?percent={percent}",
            hx_trigger="every 500ms",
            hx_swap="outerHTML",
        )

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
        return Title("Two-Prompt Tutorial"), Main(
            H1("Upload a Text File for 2-step Podcast Generation"),
            form,
            Div(id="upload-info"),
            cls="container mx-auto p-4"
        )

    @rt("/upload", methods=["POST"])
    async def upload_doc(request):
        file_uuid = uuid.uuid4().hex
        form = await request.form()
        docfile = form.get("document")
        if not docfile:
            return Div(P("⚠️ No file uploaded. Please try again.", cls="text-red-500"), id="upload-info")
        contents = await docfile.read()
        save_path = os.path.join(UPLOAD_FOLDER, docfile.filename)
        with open(save_path, "wb") as f:
            f.write(contents)
        cursor.execute("INSERT INTO uploads (filename) VALUES (?)", (docfile.filename,))
        conn.commit()

        # Step 1: Generate the initial script with SYS_PROMPT
        raw_text = open(save_path, "r", encoding="utf-8").read()
        first_pipeline = __import__("transformers").pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")

        print("📝 Generating FIRST script from SYS_PROMPT...")
        first_messages = SYS_PROMPT + "\n\n" + "User PDF/Text Content:\n" + raw_text
        first_outputs = first_pipeline(first_messages, max_new_tokens=1024, temperature=1.0)
        first_generated_script = first_outputs[0]["generated_text"]
        print("✅ FIRST script done.")
        print(first_generated_script)

        # Step 2: Rewrite the script for TTS with SYSTEMP_PROMPT
        print("🔄 Rewriting script for TTS with second prompt (SYSTEMP_PROMPT)...")
        second_messages = SYSTEMP_PROMPT + "\n\nPodcast script to rewrite:\n" + first_generated_script
        second_outputs = first_pipeline(second_messages, max_new_tokens=2048, temperature=1.0)
        final_tts_script = second_outputs[0]["generated_text"]
        print("✅ SECOND rewrite done.")
        print(final_tts_script)

        # Attempt to parse final_tts_script as a list of (Speaker, text) tuples
        # e.g. from the bracketed portion
        dialogue = None
        try:
            start_idx = final_tts_script.find("[")
            end_idx = final_tts_script.rfind("]") + 1
            if start_idx != -1 and end_idx != -1:
                candidate = final_tts_script[start_idx:end_idx]
                dialogue = ast.literal_eval(candidate)
        except Exception as e:
            print("❌ Could not parse as a list of tuples:", e)

        # fallback if parse fails
        if not dialogue:
            dialogue = [("Speaker 1", final_tts_script)]

        # TTS generation
        print("🎧 Generating TTS segments from final list of tuples...")
        segments = []
        rates = []
        for speaker, text_seg in tqdm(dialogue, desc="Generating segments", unit="segment"):
            arr, sr = generate_tts_audio(text_seg, speaker)
            segments.append(arr)
            rates.append(sr)

        # Concatenate
        final_audio = None
        for arr, sr in zip(segments, rates):
            seg_audio = numpy_to_audio_segment(arr, sr)
            if final_audio is None:
                final_audio = seg_audio
            else:
                final_audio = final_audio.append(seg_audio, crossfade=100)

        final_audio_path = f"/data/final_podcast_audio_{file_uuid}.wav"
        final_audio.export(final_audio_path, format="wav")
        print("🎶 Final podcast audio generated and saved to", final_audio_path)

        return Div(
            P(f"✅ File '{docfile.filename}' processed in two steps successfully!", cls="text-green-500"),
            progress_bar(0),
            Div(audio_player(final_audio_path), id="audio-player"),
            id="processing-results"
        )

    return fasthtml_app

if __name__ == "__main__":
    serve()
































































































































































