from fasthtml.common import *
import os

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
        hx_target="#document-list",
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

    file_ext = os.path.splitext(document.filename)[1].lower()
    file_path = os.path.join(UPLOAD_FOLDER, os.path.basename(document.filename))

    # Save the file
    contents = await document.read()
    with open(file_path, "wb") as f:
        f.write(contents)

    return P(f"File '{os.path.basename(document.filename)}' uploaded successfully!")

# Route: Serve uploaded files
@rt("/{path:path}")
def serve_file(path: str):
    return FileResponse(path)

# Start the server
serve()


