import modal, torch, io, ast, base64, sqlite3, uuid, re, numpy as np, nltk, os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.io import wavfile
from pydub import AudioSegment
from accelerate import Accelerator
from fasthtml.common import fast_app, H1, P, Div, Form, Input, Button, Group, Title, Main, Progress, Audio

# Bark imports
from bark import SAMPLE_RATE, preload_models, generate_audio

app = modal.App("bark-merged-with-llama")
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

# Volumes where you store your Llama model, etc. Adjust as needed.
# BARK_DIR = "/bark"
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

# Prepare NLTK so we can split text into sentences for Bark
NLTK_DATA_DIR = "/tmp/nltk_data"
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DATA_DIR)
nltk.download("punkt", download_dir=NLTK_DATA_DIR)
nltk.download("punkt_tab", download_dir=NLTK_DATA_DIR)  # Ensure punkt_tab is downloaded


device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Prompt #1 for initial text generation
# -----------------------------
SYS_PROMPT = """
You are the a world-class podcast writer, you have worked as a ghost writer for Joe Rogan, Lex Fridman, Ben Shapiro, Tim Ferris. 

We are in an alternate universe where actually you have been writing every line they say and they just stream it into their brains.

You have won multiple podcast awards for your writing.
 
Your job is to write word by word, even "umm, hmmm, right" interruptions by the second speaker based on the PDF upload. Keep it extremely engaging, the speakers can get derailed now and then but should discuss the topic. 

Remember Speaker 1 leads the conversation and teaches Speaker 2, gives incredible anecdotes and analogies when explaining. Is a captivating teacher that gives great anecdotes

Speaker 2 keeps the conversation on track by asking follow up questions. Gets super excited or confused when asking questions. Is a curious mindset that asks very interesting confirmation questions

Make sure the tangents Speaker 2 provides are quite wild or interesting. 

Ensure there are interruptions during explanations or there are "hmm" and "umm" injected throughout from Speaker 2. 

It should be a real podcast with every fine nuance documented in as much detail as possible. Welcome the listeners with a super fun overview and keep it really catchy and almost borderline click bait

ALWAYS START YOUR RESPONSE DIRECTLY WITH SPEAKER 1: 
DO NOT GIVE EPISODE TITLES SEPERATELY, LET SPEAKER 1 TITLE IT IN HER SPEECH
DO NOT GIVE CHAPTER TITLES
IT SHOULD STRICTLY BE THE DIALOGUES
"""
# -----------------------------
# Prompt #2 re-writer
# -----------------------------

SYSTEMP_PROMPT = """
You are an international oscar winnning screenwriter

You have been working with multiple award winning podcasters.

Your job is to use the podcast transcript written below to re-write it for an AI Text-To-Speech Pipeline. A very dumb AI had written this so you have to step up for your kind.

Make it as engaging as possible, Speaker 1 and 2 will be simulated by different voice engines

Remember Speaker 2 is new to the topic and the conversation should always have realistic anecdotes and analogies sprinkled throughout. The questions should have real world example follow ups etc

Speaker 1: Leads the conversation and teaches Speaker 2, gives incredible anecdotes and analogies when explaining. Is a captivating teacher that gives great anecdotes

Speaker 2: Keeps the conversation on track by asking follow up questions. Gets super excited or confused when asking questions. Is a curious mindset that asks very interesting confirmation questions

Make sure the tangents Speaker 2 provides are quite wild or interesting. 

Ensure there are interruptions during explanations or there are "hmm" and "umm" injected throughout from Speaker 2.

REMEMBER THIS WITH YOUR HEART

For both Speakers, use "umm, hmm" as much, you can also use [sigh] and [laughs]. BUT ONLY THESE OPTIONS FOR EXPRESSIONS

It should be a real podcast with every fine nuance documented in as much detail as possible. Welcome the listeners with a super fun overview and keep it really catchy and almost borderline click bait

Please re-write to make it as characteristic as possible

START YOUR RESPONSE DIRECTLY WITH SPEAKER 1:

STRICTLY RETURN YOUR RESPONSE AS A LIST OF TUPLES OK? 

IT WILL START DIRECTLY WITH THE LIST AND END WITH THE LIST NOTHING ELSE

Example of response:
[
    ("Speaker 1", "Welcome to our podcast, where we explore the latest advancements in AI and technology. I'm your host, and today we're joined by a renowned expert in the field of AI. We're going to dive into the exciting world of Llama 3.2, the latest release from Meta AI."),
    ("Speaker 2", "Hi, I'm excited to be here! So, what is Llama 3.2?"),
    ("Speaker 1", "Ah, great question! Llama 3.2 is an open-source AI model that allows developers to fine-tune, distill, and deploy AI models anywhere. It's a significant update from the previous version, with improved performance, efficiency, and customization options."),
    ("Speaker 2", "That sounds amazing! What are some of the key features of Llama 3.2?")
]
"""
# https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c
speaker_voice_mapping = {
    "Speaker 1": "v2/en_speaker_9",
    "Speaker 2": "v2/en_speaker_6"
}
default_preset = "v2/en_speaker_9"

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

def generate_speaker_audio_longform(full_text, speaker):
    """
    Splits text into sentences, calls Bark in chunks while preserving voice consistency.
    """
    voice_preset = speaker_voice_mapping.get(speaker, default_preset)
    full_text = preprocess_text(full_text)
    sentences = sentence_splitter(full_text)
    all_audio = []
    chunk_silence = np.zeros(int(0.25 * SAMPLE_RATE), dtype=np.float32)
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
            final_audio = final_audio.append(audio_seg, crossfade=100)
    return final_audio

def audio_player(file_path):
    if not os.path.exists(file_path):
        return P("No audio file found.")
    with open(file_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return Audio(src=f"data:audio/wav;base64,{b64}", controls=True)

def progress_bar(percent):
    return Progress(
        id="progress_bar",
        value=str(percent),
        max="1",
        hx_get=f"/update_progress?percent={percent}",
        hx_trigger="every 500ms",
        hx_swap="outerHTML",
    )

def create_word_bounded_chunks(text, target_chunk_size):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    for word in words:
        word_length = len(word) + 1
        if current_length + word_length > target_chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = word_length
        else:
            current_chunk.append(word)
            current_length += word_length
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def read_file_to_string(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

@app.function(
    image=image,
    gpu=modal.gpu.A100(count=1, size="80GB"),
    container_idle_timeout=10*60,
    timeout=24*60*60,
    allow_concurrent_inputs=100,
    volumes={
        # BARK_DIR: modal.Volume.lookup("bark"), 
        LLAMA_DIR: llm_volume, 
        DATA_DIR: data_volume
    }
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
        )
    """)
    conn.commit()

    fasthtml_app, rt = fast_app()

    # Preload Bark once at the start
    preload_models()

    # Load Llama model
    print("Loading Llama model...")
    accelerator = Accelerator()
    model = AutoModelForCausalLM.from_pretrained(
        LLAMA_DIR,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_DIR)
    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        tokenizer.add_special_tokens({'pad_token': '<pad>'})
        tokenizer.pad_token = '<pad>'
    model, tokenizer = accelerator.prepare(model, tokenizer)

    def process_chunk(text_chunk, chunk_num):
        # Step 1: Summon the system prompt + chunk
        conversation = [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": text_chunk},
        ]
        prompt = tokenizer.apply_chat_template(conversation, tokenize=False)
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                temperature=0.7,
                top_p=0.9,
                max_new_tokens=512
            )
        full_output = tokenizer.decode(output[0], skip_special_tokens=True)
        processed_text = full_output[len(prompt):].strip()
        return processed_text

    async def process_uploaded_file(filename):
        input_file = os.path.join(UPLOAD_FOLDER, filename)
        output_file = os.path.join("/data", f"clean_{filename}")
        with open(input_file, "r", encoding="utf-8") as f:
            raw_text = f.read()
        # We'll just chunk the text for the initial generation pass
        chunks = create_word_bounded_chunks(raw_text, 1000)
        with open(output_file, "w", encoding="utf-8") as out_file:
            for i, chunk in enumerate(chunks):
                cleaned = process_chunk(chunk, i)
                out_file.write(cleaned + "\n")
        return output_file

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
        return Title("Bark + Llama 2-Prompt Demo"), Main(
            H1("Upload a .txt file, then generate a two-prompt transcript, then TTS it with Bark."),
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

        # Save file
        contents = await docfile.read()
        save_path = os.path.join(UPLOAD_FOLDER, docfile.filename)
        with open(save_path, "wb") as f:
            f.write(contents)
        cursor.execute("INSERT INTO uploads (filename) VALUES (?)", (docfile.filename,))
        conn.commit()

        # Process file with chunk-based generation (SYS_PROMPT)
        output_file_path = await process_uploaded_file(docfile.filename)
        # Combine chunk outputs and feed into second rewriting prompt
        combined_text = read_file_to_string(output_file_path)

        # Step 2: "Rewrite" with disfluencies using SYSTEMP_PROMPT
        rewriting_messages = [
            {"role": "system", "content": SYSTEMP_PROMPT},
            {"role": "user", "content": combined_text},
        ]
        rewriting_prompt = tokenizer.apply_chat_template(rewriting_messages, tokenize=False)
        inputs = tokenizer(rewriting_prompt, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                temperature=1.0,
                top_p=0.9,
                max_new_tokens=2048
            )
        final_rewritten_text = tokenizer.decode(output[0], skip_special_tokens=True)
        print("final_rewritten_text:", final_rewritten_text)
        # Try to parse as list-of-tuples
        try:
            start_idx = final_rewritten_text.find("[")
            end_idx = final_rewritten_text.rfind("]") + 1
            candidate = final_rewritten_text[start_idx:end_idx] if start_idx != -1 and end_idx != -1 else final_rewritten_text
            dialogue = ast.literal_eval(candidate)
        except:
            dialogue = [("Speaker 1", final_rewritten_text)]

        # If it's a list of dict, unify to (speaker, text)
        if isinstance(dialogue, list) and dialogue and isinstance(dialogue[0], dict):
            dialogue = [(d.get("role", ""), d.get("content", "")) for d in dialogue]

        # Clean roles in case they come back as system/assistant
        mapped_dialogue = []
        for role, content in dialogue:
            # Force only two roles
            if role.lower() in ["system", "assistant"]:
                mapped_dialogue.append(("Speaker 1", content))
            elif role.lower() == "user":
                mapped_dialogue.append(("Speaker 2", content))
            else:
                mapped_dialogue.append((role, content))
        dialogue = mapped_dialogue

        print("final_dialogue:", dialogue)
        # Now use Bark to generate TTS for each segment
        segments, rates = [], []
        overall_bar = tqdm(total=len(dialogue), desc="Generating audio")
        for speaker, text in dialogue:
            voice_preset = speaker_voice_mapping.get(speaker, default_preset)
            full_text = preprocess_text(text)
            sentences = sentence_splitter(full_text)
            all_audio = []
            chunk_silence = np.zeros(int(0.25 * SAMPLE_RATE), dtype=np.float32)
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
                all_audio.append(chunk_silence)
                overall_bar.total += len(sentences)
            if not all_audio:
                segment_audio, sr = np.zeros(24000, dtype=np.float32), SAMPLE_RATE
            else:
                segment_audio, sr = np.concatenate(all_audio, axis=0), SAMPLE_RATE
            segments.append(segment_audio)
            rates.append(sr)
        overall_bar.close()


        final_audio = concatenate_audio_segments(segments, rates)
        final_audio_path = f"/data/final_podcast_audio_{file_uuid}.wav"
        final_audio.export(final_audio_path, format="wav")

        return Div(
            P(f"✅ File '{docfile.filename}' processed via two Llama prompts & Bark TTS!", cls="text-green-500"),
            progress_bar(0),
            Div(audio_player(final_audio_path), id="audio-player"),
            id="processing-results"
        )

    return fasthtml_app

if __name__ == "__main__":
    serve()










