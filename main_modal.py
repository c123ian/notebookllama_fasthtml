import modal

app = modal.App("simple-fasthtml-example")

# Build an image with FastHTML installed
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("python-fasthtml==0.12.0")
)

# Optional: Persisted volume to store uploads & audio
try:
    data_volume = modal.Volume.lookup("my_data_volume", create_if_missing=True)
except modal.exception.NotFoundError:
    data_volume = modal.Volume.persisted("my_data_volume")

@app.function(image=image, volumes={"/data": data_volume})
@modal.asgi_app()
def serve():
    import os
    import base64
    import sqlite3
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






