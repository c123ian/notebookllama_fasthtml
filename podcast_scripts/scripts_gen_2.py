import modal
import torch
import io
import ast
import base64
import sqlite3
import uuid
import re
import pickle
import numpy as np
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.io import wavfile
from pydub import AudioSegment
from accelerate import Accelerator
from fasthtml.common import (
    fast_app, H1, P, Div, Form, Input, Button, Group,
    Title, Main, Progress, Audio
)

try:
    podcast_volume = modal.Volume.lookup("podcast_volume", create_if_missing=True)
except modal.exception.NotFoundError:
    podcast_volume = modal.Volume.persisted("podcast_volume")

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
DO NOT GIVE EPISODE TITLES SEPARATELY, LET SPEAKER 1 TITLE IT IN HER SPEECH
DO NOT GIVE CHAPTER TITLES
IT SHOULD STRICTLY BE THE DIALOGUES
"""

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
    llm_volume = modal.Volume.lookup("llama_mini", create_if_missing=False)
except modal.exception.NotFoundError:
    raise Exception("Download your Llama model files first.")

@app.function(
    image=image,
    gpu=modal.gpu.A100(count=1, size="80GB"),
    container_idle_timeout=10 * 60,
    timeout=24 * 60 * 60,
    allow_concurrent_inputs=100,
    volumes={LLAMA_DIR: llm_volume, "/data": podcast_volume}
)
@modal.asgi_app()
def serve():
    import os

    UPLOAD_FOLDER = "/data/uploads"
    SCRIPTS_FOLDER = "/data/podcast_scripts_table"
    DB_PATH = "/data/uploads.db"

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(SCRIPTS_FOLDER, exist_ok=True)

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
            wlen = len(word) + 1
            if current_length + wlen > target_chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = wlen
            else:
                current_chunk.append(word)
                current_length += wlen
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        return chunks

    def process_chunk(text_chunk, chunk_num):
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

    def process_uploaded_file(original_name):
        file_uuid = uuid.uuid4().hex
        input_path = os.path.join(UPLOAD_FOLDER, original_name)
        cleaned_path = os.path.join(UPLOAD_FOLDER, f"upload_{file_uuid}.txt")
        with open(input_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        chunks = create_word_bounded_chunks(raw_text, 1000)
        with open(cleaned_path, "w", encoding="utf-8") as out_file:
            for i, c in enumerate(chunks):
                out_file.write(process_chunk(c, i) + "\n")
        print(f"🧹 File '{original_name}' cleaned and saved to '{cleaned_path}'!")
        return cleaned_path

    def read_file_to_string(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def extract_assistant_dialogue(obj):
        """
        Recursively collect only the 'assistant' outputs as ("Speaker 1", content).
        """
        result = []
        if isinstance(obj, list):
            for item in obj:
                result.extend(extract_assistant_dialogue(item))
        elif isinstance(obj, dict):
            if obj.get("role", "").lower() in ["assistant"]:
                content = obj.get("content", "")
                result.append(("Speaker 1", content))
            elif "content" in obj:
                result.extend(extract_assistant_dialogue(obj["content"]))
        elif isinstance(obj, tuple):
            if len(obj) >= 2:
                role, content = obj[0], obj[1]
                if isinstance(role, str) and role.lower() == "assistant":
                    result.append(("Speaker 1", content))
                elif isinstance(content, (list, dict, tuple)):
                    result.extend(extract_assistant_dialogue(content))
        return result

    def prepare_for_audio(dialogue):
        """
        Build a valid Python literal string from a list of (speaker, content) tuples.
        - We escape any quotes and backslashes
        - We replace literal newlines with \n
        - We wrap with [ ... ] so ast.literal_eval(...) -> list-of-tuples
        """
        lines = ["["]
        for speaker, text in dialogue:
            # Replace backslash first, then quotes, then newlines
            safe_text = text.replace("\\", "\\\\")\
                            .replace('"', '\\"')\
                            .replace("\n", "\\n")
            lines.append(f'    ("{speaker}", "{safe_text}"),')
        lines.append("]")
        return "\n".join(lines)

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
        return Title("Simple Upload + Script Gen"), Main(
            H1("Simple Upload + Script Gen"),
            form,
            Div(id="upload-info")
        )

    @rt("/upload", methods=["POST"])
    async def upload_doc(request):
        form = await request.form()
        docfile = form.get("document")
        if not docfile:
            return Div(P("No file uploaded."), id="upload-info")

        contents = await docfile.read()
        save_path = os.path.join(UPLOAD_FOLDER, docfile.filename)
        with open(save_path, "wb") as f:
            f.write(contents)
        cursor.execute("INSERT INTO uploads (filename) VALUES (?)", (docfile.filename,))
        conn.commit()

        # Clean the file
        cleaned_file_path = process_uploaded_file(docfile.filename)
        input_text = read_file_to_string(cleaned_file_path)

        print("📝 Generating first script...")
        first_pipeline = __import__("transformers").pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto",
        )
        first_outputs = first_pipeline(
            [
                {"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": input_text}
            ],
            max_new_tokens=1500,
            temperature=1.0,
        )
        first_generated_text = first_outputs[0]["generated_text"]
        print("✍️  First script generated.")

        print("🔄 Rewriting script with disfluencies...")
        second_outputs = first_pipeline(
            [
                {"role": "system", "content": SYSTEMP_PROMPT},
                {"role": "user", "content": first_generated_text}
            ],
            max_new_tokens=1500,
            temperature=1.0,
        )
        final_rewritten_text = second_outputs[0]["generated_text"]

        # Attempt to parse out a list of tuples from final_rewritten_text
        try:
            start_idx = final_rewritten_text.find("[")
            end_idx = final_rewritten_text.rfind("]") + 1
            candidate = final_rewritten_text[start_idx:end_idx] if start_idx != -1 and end_idx != -1 else final_rewritten_text
            parsed = ast.literal_eval(candidate)
        except Exception:
            parsed = [("Speaker 1", final_rewritten_text)]

        assistant_only = extract_assistant_dialogue(parsed)
        if not assistant_only:
            # fallback if none found
            flattened = []
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, tuple) and len(item) >= 2:
                        flattened.append(item)
            # filter by role
            assistant_only = [(r, t) for r, t in flattened if r.lower() == "assistant"]
        if not assistant_only:
            assistant_only = parsed

        # Force roles: 'assistant' -> Speaker 1, 'user' -> Speaker 2
        final_dialogue = []
        for role, content in assistant_only:
            lr = role.lower() if isinstance(role, str) else ""
            if lr in ["assistant", "system"]:
                final_dialogue.append(("Speaker 1", content))
            elif lr == "user":
                final_dialogue.append(("Speaker 2", content))
            else:
                final_dialogue.append((role, content))

        # Reverse so second output is first
        final_dialogue = final_dialogue[::-1]

        # Build a valid literal string
        final_formatted = prepare_for_audio(final_dialogue)

        # Save as pickled string
        file_uuid = uuid.uuid4().hex
        final_pickle_path = os.path.join(SCRIPTS_FOLDER, f"final_rewritten_text_{file_uuid}.pkl")
        dialogue_pickle_path = os.path.join(SCRIPTS_FOLDER, f"dialogue_{file_uuid}.pkl")

        with open(final_pickle_path, "wb") as f:
            pickle.dump(final_formatted, f)

        with open(dialogue_pickle_path, "wb") as f:
            pickle.dump(final_formatted, f)

        print(f"✅ final_rewritten_text saved to {final_pickle_path}")
        print(f"✅ dialogue saved to {dialogue_pickle_path}")

        return Div(
            P(f"✅ File '{docfile.filename}' uploaded and processed successfully!", cls="text-green-500"),
            id="processing-results"
        )

    return fasthtml_app

if __name__ == "__main__":
    serve()











