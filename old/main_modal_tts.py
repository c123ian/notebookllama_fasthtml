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

# Initial podcast writer
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
DO NOT GIVE EPISODE TITLES SEPERATELY, LET SPEAKER 1 TITLE IT IN HER SPEECH
DO NOT GIVE CHAPTER TITLES
IT SHOULD STRICTLY BE THE DIALOGUES
"""
# Second podcast re-writer
SYSTEMP_PROMPT ="""
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
        # Tokenize without adding extra special tokens (we assume the prompt template already includes what’s needed)
        tokens = tokenizer(prompt, add_special_tokens=False)['input_ids']
        processed_outputs = []
        # Split the token list into segments of max 512
        for i in range(0, len(tokens), 512):
            token_segment = tokens[i:i+512]
            input_tensor = torch.tensor([token_segment]).to(device)
            with torch.no_grad():
                output = model.generate(
                    input_tensor,
                    temperature=0.7,
                    top_p=0.9,
                    max_new_tokens=512
                )
            processed_text = tokenizer.decode(output[0], skip_special_tokens=True)
            processed_outputs.append(processed_text)
        return " ".join(processed_outputs)



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
    def audio_player(file_path="/data/final_podcast_audio.wav"):
        import os, base64
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
        
        # Generate the podcast audio
        print("🎧 Generating podcast segments (TTS audio)...")
        final_audio_path = "/data/final_podcast_audio.wav"
        # (audio generation code omitted for brevity)
        
        # Return the upload success message + audio player
        return Div(
            P(f"✅ File '{docfile.filename}' uploaded and processed successfully!", cls="text-green-500"),
            progress_bar(0),
            audio_player(final_audio_path),
            id="processing-results"
        )
    return fasthtml_app


if __name__ == "__main__":
    serve()











































































