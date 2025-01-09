from fasthtml.common import *
import os, random, base64, requests

# Create app
app, rt = fast_app()

# Define folders
UPLOAD_FOLDER = "uploads"
PODCAST_FOLDER = "podcasts"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PODCAST_FOLDER, exist_ok=True)

# 🔄 Placeholder audio file path
AUDIO_FILE_PATH = f"{PODCAST_FOLDER}/podcast_test.mp3"

# Route: Main dashboard
@rt("/")
def get():
    inp = Input(type="file", name="document", accept=".txt,.pdf", required=True)
    add = Form(
        Group(inp, Button("Upload")),
        hx_post="/upload",
        hx_target="#document-list,#progress_bar",
        hx_swap="afterbegin",
        enctype="multipart/form-data",
        method="post",
    )
    document_list = Div(id="document-list")
    return Title("Podcast Generator"), Main(
        H1("Podcast Generator"), add, document_list, cls="container"
    )

# Route: Upload document handler
@rt("/upload", methods=["POST"])
async def upload_document(request):
    form = await request.form()
    document = form.get("document")

    if document is None:
        return P("No file uploaded. Please try again.")

    # Save the file
    file_path = os.path.join(UPLOAD_FOLDER, document.filename)
    contents = await document.read()
    with open(file_path, "wb") as f:
        f.write(contents)

    # Trigger progress bar
    return Div(
        P(f"File '{document.filename}' uploaded successfully!"),
        progress_bar(0),
    )

# ✅ Route: Simulate progress updates
@rt("/update_progress", methods=["GET"])
def update_progress(request):
    try:
        percent_complete = float(request.query_params.get("percent_complete", 0))
        if percent_complete >= 1:
            return Div(
                H3("Upload Complete!", id="progress_bar"),
                audio_player(),
            )

        percent_complete += 0.1
        return progress_bar(min(percent_complete, 1.0))
    except (ValueError, TypeError):
        return progress_bar(0)

# ✅ Progress bar component
def progress_bar(percent_complete: float):
    return Progress(
        id="progress_bar",
        value=str(percent_complete),
        max="1",
        hx_get=f"/update_progress?percent_complete={percent_complete}",
        hx_trigger="every 500ms",
        hx_swap="outerHTML",
        cls="progress-bar",
    )

# ✅ Audio player component
def audio_player():
    if not os.path.exists(AUDIO_FILE_PATH):
        return P("Audio file not found.")
    
    audio_base64 = load_audio_base64(AUDIO_FILE_PATH)
    return Audio(src=f"data:audio/mp4;base64,{audio_base64}", controls=True)

# ✅ Helper to load audio file as base64
def load_audio_base64(audio_path: str):
    with open(audio_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode("ascii")

# Route: Serve uploaded files
@rt("/{path:path}")
def serve_file(path: str):
    return FileResponse(path)

# Start the server
serve()






