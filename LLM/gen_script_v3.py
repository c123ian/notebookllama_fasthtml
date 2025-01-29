import modal
import torch
import os
import ast
import pickle
import sqlite3
import uuid
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
# Modal image with dependencies
###############################################################################
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "python-fasthtml==0.12.0",
        "transformers==4.46.1",
        "accelerate>=0.26.0"
    )
)

###############################################################################
# System prompts 
###############################################################################
SYSTEM_PROMPT = """
You are a world-class podcast writer, having ghostwritten for top shows.
Your job is to write a lively, engaging script with two speakers:
Speaker 1 leads the conversation, teaching Speaker 2, giving anecdotes.
Speaker 2 asks follow-up questions, interrupts with "umm, hmm" occasionally.

ALWAYS START YOUR RESPONSE WITH 'SPEAKER 1:' (but we'll refine later).
Keep the conversation extremely engaging, welcome the audience with a fun overview, etc.
"""

REWRITE_PROMPT = """
You are an Oscar-winning screenwriter rewriting a transcript for an AI TTS pipeline.
Re-inject disfluencies like "umm, hmm, [laughs]" and ensure there's a real back-and-forth.
Return your final answer as a Python LIST of (Speaker, text) TUPLES, e.g.:

[
    ("Speaker 1", "Hello, and welcome..."),
    ("Speaker 2", "Hmm, that is fascinating!")
]

IMPORTANT: Your response must be a valid Python list of tuples.
"""

def parse_podcast_script(text: str):
    """Parse the generated text into a list of speaker-text tuples"""
    try:
        # Extract everything between the first [ and last ]
        start_idx = text.find("[")
        end_idx = text.rfind("]") + 1
        if start_idx == -1 or end_idx == 0:
            # If no brackets found, try to parse line by line
            lines = text.split('\n')
            script = []
            current_speaker = None
            current_text = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                if ':' in line:
                    if current_speaker and current_text:
                        script.append((current_speaker, ' '.join(current_text)))
                        current_text = []
                    
                    parts = line.split(':', 1)
                    current_speaker = parts[0].strip()
                    if len(parts) > 1:
                        current_text.append(parts[1].strip())
                else:
                    if current_speaker:
                        current_text.append(line)
            
            if current_speaker and current_text:
                script.append((current_speaker, ' '.join(current_text)))
            
            return script if script else [("Speaker 1", text)]
            
        # Try to parse the content between brackets
        candidate = text[start_idx:end_idx]
        final_script = ast.literal_eval(candidate)
        
        if not isinstance(final_script, list):
            return [("Speaker 1", text)]
            
        return final_script
        
    except Exception as e:
        print(f"Parsing error: {str(e)}")
        # Fallback to basic parsing if ast.literal_eval fails
        try:
            lines = text.split('\n')
            script = []
            for line in lines:
                if ':' in line:
                    parts = line.split(':', 1)
                    speaker = parts[0].strip()
                    text = parts[1].strip() if len(parts) > 1 else ""
                    script.append((speaker, text))
            return script if script else [("Speaker 1", text)]
        except Exception as e2:
            print(f"Fallback parsing error: {str(e2)}")
            return [("Speaker 1", text)]

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
    # Setup folders and DB
    ###############################################################################
    UPLOAD_FOLDER = "/data/uploads"
    SCRIPTS_FOLDER = "/data/podcast_scripts_table"  # Updated to match your path
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

    ###############################################################################
    # Load model and create app
    ###############################################################################
    print("Loading Llama model...")
    accelerator = Accelerator()
    model = AutoModelForCausalLM.from_pretrained(
        LLAMA_DIR,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_DIR)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<pad>'})
        
    model, tokenizer = accelerator.prepare(model, tokenizer)
    
    generation_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        max_new_tokens=1500,
        temperature=1.0,
    )

    fasthtml_app, rt = fast_app()

    @rt("/")
    def homepage():
        """Render the homepage with upload form"""
        upload_input = Input(type="file", name="document", accept=".txt", required=True)
        form = Form(
            Group(upload_input, Button("Generate Script")),
            hx_post="/upload",
            hx_swap="afterbegin",
            enctype="multipart/form-data",
            method="post",
        )
        return Title("Podcast Script Generator"), Main(
            H1("Upload Text to Generate Podcast Script"),
            form,
            Div(id="upload-info")
        )

    @rt("/upload", methods=["POST"])
    async def upload_and_generate(request):
        """Handle file upload and script generation"""
        try:
            form = await request.form()
            docfile = form.get("document")
            if not docfile:
                return Div(P("No file uploaded."), id="upload-info")

            # Save uploaded file
            contents = await docfile.read()
            upload_path = os.path.join(UPLOAD_FOLDER, docfile.filename)
            with open(upload_path, "wb") as f:
                f.write(contents)

            # Read text content
            with open(upload_path, "r", encoding="utf-8") as f:
                source_text = f.read()

            # Generate initial script
            print("Generating initial script...")
            prompt_1 = SYSTEM_PROMPT + "\n\n" + source_text
            first_draft = generation_pipe(prompt_1)[0]["generated_text"]

            # Rewrite with disfluencies
            print("Adding natural speech patterns...")
            prompt_2 = REWRITE_PROMPT + "\n\n" + first_draft
            final_text = generation_pipe(prompt_2)[0]["generated_text"]

            # Parse into structured format
            final_script = parse_podcast_script(final_text)

            # Save script
            file_uuid = uuid.uuid4().hex
            final_pickle_path = os.path.join(SCRIPTS_FOLDER, f"final_rewritten_text_{file_uuid}.pkl")
            with open(final_pickle_path, "wb") as f:
                pickle.dump(final_script, f)

            # Log in database
            try:
                cursor.execute(
                    "INSERT INTO uploads (filename) VALUES (?)",
                    (docfile.filename,)
                )
                conn.commit()
            except sqlite3.Error as e:
                print(f"Database error: {str(e)}")

            # Display results
            speakers = set(speaker for speaker, _ in final_script)
            return Div(
                P(f"✅ Generated script from '{docfile.filename}'"),
                P(f"Number of lines: {len(final_script)}"),
                P(f"Speakers: {', '.join(speakers)}"),
                P(f"Script saved as: {os.path.basename(final_pickle_path)}"),
                id="upload-info"
            )

        except Exception as e:
            print(f"Error during processing: {str(e)}")
            return Div(
                P("❌ An error occurred while processing the file."),
                P(f"Error details: {str(e)}"),
                id="upload-info"
            )

    return fasthtml_app

if __name__ == "__main__":
    serve()
