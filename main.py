from fasthtml.common import *
import os, random

# Create app
app, rt = fast_app()

# Define upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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
    return Title("Document Upload Demo"), Main(
        H1("Document Upload"), add, document_list, cls="container"
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
        # Get current progress, defaulting to 0 if not present or invalid
        percent_complete = float(request.query_params.get("percent_complete", 0))
        if percent_complete >= 1:
            return H3("Upload Complete!", id="progress_bar")

        # Increment by a fixed amount (0.1 = 10% each time)
        percent_complete += 0.1
        
        # Return updated progress bar with the new value
        return progress_bar(min(percent_complete, 1.0))
    except (ValueError, TypeError):
        # If there's any error parsing the value, start from 0
        return progress_bar(0)

def progress_bar(percent_complete: float):
    return Progress(
        id="progress_bar",
        value=str(percent_complete),  # Make sure to convert to string
        max="1",
        hx_get=f"/update_progress?percent_complete={percent_complete}",  # Add the current progress to the URL
        hx_trigger="every 500ms",
        hx_swap="outerHTML",
        cls="progress-bar",
    )

# Route: Serve uploaded files
@rt("/{path:path}")
def serve_file(path: str):
    return FileResponse(path)

# Start the server
serve()






