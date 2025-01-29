import modal, torch, os, pickle, ast, uuid
from transformers import AutoTokenizer, AutoModelForCausalLM
from fasthtml.common import fast_app, H1, P, Div, Form, Input, Button, Group, Title, Main

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

Ensure there are interruptions during explanations or there are "hmm" and "umm" injected throughout from both Speakers.

REMEMBER THIS WITH YOUR HEART

For both Speakers, use "umm, hmm" as much, you can also use [sigh], [laughs], [gasps] or [clears throat]. BUT ONLY THESE OPTIONS FOR EXPRESSIONS

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

LLAMA_DIR = "/llamas_8b"
DATA_DIR = "/data"
device = "cuda" if torch.cuda.is_available() else "cpu"


try:
    podcast_volume = modal.Volume.lookup("podcast_volume", create_if_missing=True)
except modal.exception.NotFoundError:
    raise Exception("No podcast_volume for storing history")

try:
    llm_volume = modal.Volume.lookup("llamas_8b", create_if_missing=False)
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
    fasthtml_app, rt = fast_app()
    # load model & tokenizer, etc. as you did

    @rt("/")
    def index():
        upload_input = Input(type="file", name="document", accept=".txt", required=True)
        form = Form(
            Group(upload_input, Button("Upload")),
            hx_post="/upload",
            method="post",
            enctype="multipart/form-data"
        )
        return Title("Simple Upload + Script Gen"), Main(
            H1("Script Gen"),
            form,
            Div(id="upload-info")
        )

    @rt("/upload", methods=["POST"])
    async def upload_doc(request):
        form = await request.form()
        docfile = form.get("document")
        if not docfile:
            return Div(P("No file uploaded."))

        contents = await docfile.read()
        # Save the uploaded .txt
        original_path = f"/data/uploads/{docfile.filename}"
        with open(original_path, "wb") as f:
            f.write(contents)

        # 1) Possibly chunk & pass to LLM (simple version here just loads the file text):
        with open(original_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        # 2) First generation (rough script)
        first_prompt = [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": raw_text},
        ]
        # run it through your model’s `generate` or pipeline ...
        first_output = "YOUR_LLM_CALL(first_prompt)"  # pseudo-code
        # get the raw text from the model

        # 3) Re-inject disfluencies
        second_prompt = [
            {"role": "system", "content": SYSTEMP_PROMPT},
            {"role": "user", "content": first_output},
        ]
        final_output = "YOUR_LLM_CALL(second_prompt)"

        # 4) Parse to a list of tuples
        try:
            start_idx = final_output.find("[")
            end_idx = final_output.rfind("]") + 1
            candidate = final_output[start_idx:end_idx]
            parsed_script = ast.literal_eval(candidate)
            # Make sure it's a list of (speaker, text)
            if not isinstance(parsed_script, list):
                parsed_script = [("Speaker_1", final_output)]
        except Exception:
            parsed_script = [("Speaker_1", final_output)]

        # 5) Pickle the final list
        file_uuid = uuid.uuid4().hex
        final_pickle_path = f"/data/podcast_scripts_table/final_{file_uuid}.pkl"
        with open(final_pickle_path, "wb") as f:
            pickle.dump(parsed_script, f)

        return Div(
            P("File uploaded, script generated!"),
            P(f"Pickled script saved as {final_pickle_path}")
        )

    return fasthtml_app

if __name__ == "__main__":
    serve()
