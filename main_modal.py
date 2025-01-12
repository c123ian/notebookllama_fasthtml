import modal
import torch
from transformers import AutoTokenizer

# Initialize Modal app
app = modal.App("simple-fasthtml-example")

# Build an image with FastHTML and necessary packages installed
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("python-fasthtml==0.12.0", "transformers", "accelerate")
)

MODELS_DIR = "/llama_mini"
MODEL_NAME = "Llama-3.2-3B-Instruct"

device = "cuda" if torch.cuda.is_available() else "cpu"

SYS_PROMPT = """
You are a world-class podcast writer who has ghostwritten for Joe Rogan, Lex Fridman, Ben Shapiro, and Tim Ferriss...
"""

# Persisted volume to store uploads and model files
try:
    data_volume = modal.Volume.lookup("my_data_volume", create_if_missing=True)
except modal.exception.NotFoundError:
    data_volume = modal.Volume.persisted("my_data_volume")

try:
    volume = modal.Volume.lookup("llama_mini", create_if_missing=False)
except modal.exception.NotFoundError:
    raise Exception("Download models first with the appropriate script")

@app.function(
    image=image,
    gpu=modal.gpu.A100(count=1, size="40GB"),
    container_idle_timeout=10 * 60,
    timeout=24 * 60 * 60,
    allow_concurrent_inputs=100,
    volumes={MODELS_DIR: volume, "/data": data_volume}
)
@modal.asgi_app()
def serve():
    import os
    import base64
    import sqlite3
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from accelerate import Accelerator
    from fasthtml.common import (
        fast_app, H1, P, Div, Form, Input, Button, Group,
        Title, Main, Progress, Audio
    )

    # Directories and DB setup
    UPLOAD_FOLDER = "/data/uploads"
    AUDIO_FILE_PATH = "/data/placeholder_audio.mp3"
    DB_PATH = "/data/uploads.db"

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    # Initialize SQLite database
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

    # First, create the FastHTML app
    fasthtml_app, rt = fast_app()

    # Load model at server startup
    print("Loading model...")
    accelerator = Accelerator()
    model = AutoModelForCausalLM.from_pretrained(MODELS_DIR, torch_dtype=torch.bfloat16, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(MODELS_DIR)
    model, tokenizer = accelerator.prepare(model, tokenizer)

    # Function to create word-bounded chunks
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

    # Function to process text chunks
    def process_chunk(text_chunk, chunk_num, model, tokenizer):
        conversation = [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": text_chunk},
        ]
        prompt = tokenizer.apply_chat_template(conversation, tokenize=False)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                temperature=0.7,
                top_p=0.9,
                max_new_tokens=512
            )
        processed_text = tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):].strip()
        return processed_text

    # Function to process the uploaded file
    async def process_uploaded_file(filename):
        print(f"📂 Processing file: {filename}")
        output_file = os.path.join("/data", f"clean_{filename}")
        
        # Read the uploaded file
        input_file = os.path.join(UPLOAD_FOLDER, filename)
        with open(input_file, "r", encoding="utf-8") as file:
            text = file.read()

        # Split text into chunks
        chunks = create_word_bounded_chunks(text, 1000)

        # Process chunks and save output
        with open(output_file, "w", encoding="utf-8") as out_file:
            for chunk_num, chunk in enumerate(chunks):
                processed_chunk = process_chunk(chunk, chunk_num, model, tokenizer)
                out_file.write(processed_chunk + "\n")
                out_file.flush()

        print(f"✅ Processing complete. Output saved to {output_file}")
        return output_file

    def load_audio_base64(audio_path: str):
        with open(audio_path, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")

    def audio_player():
        if not os.path.exists(AUDIO_FILE_PATH):
            return P("No placeholder audio file found.")
        audio_data = load_audio_base64(AUDIO_FILE_PATH)
        return Audio(src=f"data:audio/mp4;base64,{audio_data}", controls=True)

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
        return Title("Simple Upload + Progress"), Main(
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
        
        # Read the processed file
        INPUT_PROMPT = read_file_to_string(output_file_path)
        
        # Now use the INPUT_PROMPT with your model
        messages = [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": INPUT_PROMPT},
        ]

        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto",
        )

        outputs = pipeline(
            messages,
            max_new_tokens=8126,
            temperature=1,
        )

        print(outputs[0]["generated_text"][-1]['content'])

        return Div(
            P(f"✅ File '{docfile.filename}' uploaded and processed successfully!", cls="text-green-500"),
            progress_bar(0),
            id="processing-results"
        )

    @rt("/update_progress", methods=["GET"])
    async def update_progress(request):
        percent_str = request.query_params.get("percent", "0")
        try:
            percent_val = float(percent_str)
        except ValueError:
            percent_val = 0.0

        if percent_val >= 1.0:
            return Div(
                P("🎉 Upload complete!"),
                audio_player(),
                id="progress_bar"
            )
        else:
            percent_val += 0.1
            if percent_val > 1.0:
                percent_val = 1.0
            return progress_bar(percent_val)

    return fasthtml_app

if __name__ == "__main__":
    serve()













