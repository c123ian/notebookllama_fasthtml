import modal
import torch
import io
import ast
import base64
import sqlite3
import uuid
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

app = modal.App("script_gen")
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

LLAMA_DIR = "/llama_mini"
DATA_DIR = "/data"

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

Make it as engaging as possible, Speaker 1 and 2 will be using different voices

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

try:
    data_volume = modal.Volume.lookup("my_data_volume", create_if_missing=True)
except modal.exception.NotFoundError:
    data_volume = modal.Volume.persisted("my_data_volume")
try:
    llm_volume = modal.Volume.lookup("llama_mini", create_if_missing=False)
except modal.exception.NotFoundError:
    raise Exception("Download your Llama model files first with the appropriate script.")

def numpy_to_audio_segment(audio_arr, sampling_rate):
    audio_int16 = (audio_arr * 32767).astype(np.int16)
    byte_io = io.BytesIO()
    wavfile.write(byte_io, sampling_rate, audio_int16)
    byte_io.seek(0)
    return AudioSegment.from_wav(byte_io)

def preprocess_text(text):
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    return text

@app.function(
    image=image,
    gpu=modal.gpu.A100(count=1, size="80GB"),
    container_idle_timeout=10 * 60,
    timeout=24 * 60 * 60,
    allow_concurrent_inputs=100,
    volumes={LLAMA_DIR: llm_volume, "/data": data_volume}
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
        print(f"🧩 Split text into {len(chunks)} chunk(s).")
        return chunks

    def process_chunk(text_chunk, chunk_num, model, tokenizer):
        conversation = [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": text_chunk},
        ]
        prompt = tokenizer.apply_chat_template(conversation, tokenize=False)
        inputs_pre = tokenizer(prompt, add_special_tokens=True)
        token_count = len(inputs_pre['input_ids'])
        print(f"Chunk {chunk_num}: Prompt has {token_count} tokens and {len(prompt)} characters.")
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            output = model.generate(
                **inputs,
                temperature=0.7,
                top_p=0.9,
                max_new_tokens=512
            )
        full_output = tokenizer.decode(output[0], skip_special_tokens=True)
        processed_text = full_output[len(prompt):].strip()
        new_token_count = len(tokenizer.tokenize(processed_text))
        print(f"Chunk {chunk_num}: Generated {new_token_count} new tokens.")
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
        return Title("Simple Upload + Test Script Gen"), Main(
            H1("Simple Upload + Test Script Gen"),
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
        if isinstance(final_rewritten_text, str):
            try:
                start_idx = final_rewritten_text.find('[')
                end_idx = final_rewritten_text.rfind(']') + 1
                candidate = final_rewritten_text[start_idx:end_idx] if start_idx != -1 and end_idx != -1 else final_rewritten_text
                dialogue = ast.literal_eval(candidate)
            except Exception as e:
                print("❌ Error parsing final_rewritten_text to a Python literal:", e)
                dialogue = [("Speaker 1", final_rewritten_text)]
        else:
            dialogue = final_rewritten_text
        if isinstance(dialogue, list) and dialogue and isinstance(dialogue[0], dict):
            dialogue = [(item.get("role", ""), item.get("content", "")) for item in dialogue]
        try:
            # Force Role Mapping: remap roles to "Speaker 1" and "Speaker 2"
            if isinstance(dialogue, list):
                mapped_dialogue = []
                for role, content in dialogue:
                    if role.lower() in ["system", "assistant"]:
                        mapped_dialogue.append(("Speaker 1", content))
                    elif role.lower() in ["user"]:
                        mapped_dialogue.append(("Speaker 2", content))
                    else:
                        mapped_dialogue.append((role, content))
                dialogue = mapped_dialogue
        except Exception as e:
            print("❌ Error during dialogue mapping:", e)

        print("✅ dialogue written:")
        print(dialogue)
        return Div(
            P(f"✅ File '{docfile.filename}' uploaded and processed successfully!", cls="text-green-500"),
            id="processing-results"
        )
    return fasthtml_app

if __name__ == "__main__":
    serve()























































































































































