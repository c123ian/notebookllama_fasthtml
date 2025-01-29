import modal
import torch
import io
import os
import ast
import base64
import sqlite3
import uuid
import re
import pickle
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from accelerate import Accelerator
from fasthtml.common import (
    fast_app, H1, P, Div, Form, Input, Button, Group,
    Title, Main
)

###############################################################################
# Volume and model lookups
###############################################################################
try:
    podcast_volume = modal.Volume.lookup("podcast_volume", create_if_missing=True)
except modal.exception.NotFoundError:
    raise Exception("No 'podcast_volume' volume found or could not be created.")

try:
    llm_volume = modal.Volume.lookup("llamas_8b", create_if_missing=False)
except modal.exception.NotFoundError:
    raise Exception("Download your Llama model files first (named 'llamas_8b').")

LLAMA_DIR = "/llamas_8b"
DATA_DIR = "/data"
device = "cuda" if torch.cuda.is_available() else "cpu"

###############################################################################
# Modal image with all dependencies
###############################################################################
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

###############################################################################
# Example system prompts
###############################################################################
SYS_PROMPT = """
You are a world-class podcast writer, having ghostwritten for top shows.
Your job is to write a lively, engaging script with two speakers:
Speaker 1 leads the conversation, teaching Speaker 2, giving anecdotes.
Speaker 2 asks follow-up questions, interrupts with "umm, hmm" occasionally.

ALWAYS START YOUR RESPONSE WITH 'SPEAKER 1:' (but we'll refine later).
Keep the conversation extremely engaging, welcome the audience with a fun overview, etc.
"""

SYSTEMP_PROMPT = """
You are an Oscar-winning screenwriter rewriting a transcript for an AI TTS pipeline.
Re-inject disfluencies like "umm, hmm, [laughs]" and ensure there's a real back-and-forth.
Return your final answer as a Python LIST of (Speaker, text) TUPLES, e.g.:

[
    ("Speaker 1", "Hello, and welcome..."),
    ("Speaker 2", "Hmm, that is fascinating!")
]
"""

app = modal.App("script_gen")

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
    ###############################################################################
    # Setup local folders and DB
    ###############################################################################
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

    ###############################################################################
    # Load Llama model & tokenizer
    ###############################################################################
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

    # A text-generation pipeline for convenience
    generation_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        max_new_tokens=1500,
        temperature=1.0,
    )

    ###############################################################################
    # Routes
    ###############################################################################
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
        return Title("Script Generation"), Main(
            H1("Upload a .txt to Generate Podcast Script"),
            form,
            Div(id="upload-info")
        )

    @rt("/upload", methods=["POST"])
    async def upload_doc(request):
        form = await request.form()
        docfile = form.get("document")
        if not docfile:
            return Div(P("No file uploaded."), id="upload-info")

        # Save .txt
        contents = await docfile.read()
        original_path = os.path.join(UPLOAD_FOLDER, docfile.filename)
        with open(original_path, "wb") as f:
            f.write(contents)

        # Log in DB
        cursor.execute("INSERT INTO uploads (filename) VALUES (?)", (docfile.filename,))
        conn.commit()

        # Read raw text
        with open(original_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        ###########################################################################
        # 1) First pass generation
        ###########################################################################
        print("📝 Generating first script...")
        prompt_1 = SYS_PROMPT + "\n\n" + raw_text
        first_result = generation_pipe(prompt_1)[0]["generated_text"]
        print("✍️  First script generated.")

        ###########################################################################
        # 2) Rewrite w/ disfluencies
        ###########################################################################
        print("🔄 Rewriting script with disfluencies...")
        prompt_2 = SYSTEMP_PROMPT + "\n\n" + first_result
        second_result = generation_pipe(prompt_2)[0]["generated_text"]

        ###########################################################################
        # 3) Parse final text as [(Speaker, text), ...]
        ###########################################################################
        try:
            start_idx = second_result.find("[")
            end_idx = second_result.rfind("]") + 1
            candidate = second_result[start_idx:end_idx] if start_idx != -1 and end_idx != -1 else second_result
            final_script = ast.literal_eval(candidate)
            if not isinstance(final_script, list):
                final_script = [("Speaker 1", second_result)]
        except Exception as e:
            print("Parsing error:", e)
            final_script = [("Speaker 1", second_result)]

        ###########################################################################
        # 4) Store final list-of-tuples as a .pkl
        ###########################################################################
        file_uuid = uuid.uuid4().hex
        final_pickle_path = os.path.join(SCRIPTS_FOLDER, f"final_rewritten_text_{file_uuid}.pkl")
        with open(final_pickle_path, "wb") as f:
            pickle.dump(final_script, f)

        return Div(
            P(f"✅ File '{docfile.filename}' uploaded and processed successfully!"),
            P(f"Pickle saved to: {final_pickle_path}"),
            id="upload-info"
        )

    return fasthtml_app

if __name__ == "__main__":
    serve()
