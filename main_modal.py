import modal
from transformers import AutoTokenizer


app = modal.App("simple-fasthtml-example")

# Build an image with FastHTML installed
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("python-fasthtml==0.12.0","transformers","accelerate")
)

MODELS_DIR = "/llama_mini"
MODEL_NAME = "Llama-3.2-3B-Instruct"  

# Optional: Persisted volume to store uploads & audio
try:
    data_volume = modal.Volume.lookup("my_data_volume", create_if_missing=True)
except modal.exception.NotFoundError:
    data_volume = modal.Volume.persisted("my_data_volume")

# Download the model weights
try:
    volume = modal.Volume.lookup("llama_mini", create_if_missing=False)
except modal.exception.NotFoundError:
    raise Exception("Download models first with the appropriate script")



@app.function(image=image, 
              gpu=modal.gpu.A100(count=1, size="40GB"),
              container_idle_timeout=10 * 60,
              timeout=24 * 60 * 60,
              allow_concurrent_inputs=100,
              volumes={MODELS_DIR: volume, "/data": data_volume})

@modal.asgi_app()
def serve():
    import os
    import base64
    import sqlite3
    import torch
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


    # download model
    # Function to find the model path by searching for 'config.json'
    def find_model_path(base_dir):
        for root, dirs, files in os.walk(base_dir):
            if "config.json" in files:
                return root
        return None

    # Function to find the tokenizer path by searching for 'tokenizer_config.json'
    def find_tokenizer_path(base_dir):
        for root, dirs, files in os.walk(base_dir):
            if "tokenizer_config.json" in files:
                return root
        return None

    # Check if model files exist
    model_path = find_model_path(MODELS_DIR)
    if not model_path:
        raise Exception(f"Could not find model files in {MODELS_DIR}")

    # Check if tokenizer files exist
    tokenizer_path = find_tokenizer_path(MODELS_DIR)
    if not tokenizer_path:
        raise Exception(f"Could not find tokenizer files in {MODELS_DIR}")

    print(f"Initializing model path: {model_path} and tokenizer path: {tokenizer_path}")

    # Let's load in the model and start processing the text chunks

    accelerator = Accelerator()
    model = AutoModelForCausalLM.from_pretrained(
        MODELS_DIR,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        #device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODELS_DIR, use_safetensors=True)
    model, tokenizer = accelerator.prepare(model, tokenizer)

    fasthtml_app, rt = fast_app()


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

        # Insert into SQLite DB
        cursor.execute("INSERT INTO uploads (filename) VALUES (?)", (docfile.filename,))
        conn.commit()

        # Return progress bar and success message
        return Div(
            P(f"✅ File '{docfile.filename}' uploaded successfully!", cls="text-green-500"),
            progress_bar(0),
            id="upload-info",
            hx_swap_oob="true",
        )

    @rt("/update_progress", methods=["GET"])
    def update_progress(request):
        percent_str = request.query_params.get("percent", "0")
        try:
            percent_val = float(percent_str)
        except ValueError:
            percent_val = 0.0

        if percent_val >= 1.0:
            return Div(
                P("Upload complete!"),
                audio_player(),
                id="progress_bar"
            )
        else:
            percent_val += 0.1
            if percent_val > 1.0:
                percent_val = 1.0
            return progress_bar(percent_val)

    return fasthtml_app

# If run locally for debugging
if __name__ == "__main__":
    serve()






