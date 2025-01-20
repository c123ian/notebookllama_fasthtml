import modal
import torch, io, ast, base64, sqlite3, uuid, re, numpy as np
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    AutoProcessor,
    BarkModel,
)
from parler_tts import ParlerTTSForConditionalGeneration
from pydub import AudioSegment
from scipy.io import wavfile
from accelerate import Accelerator
from fasthtml.common import (
    fast_app, H1, P, Div, Form, Input, Button, Group, Title, Main, Progress, Audio
)

##############################################################################
# Modal setup: volumes, image, device
##############################################################################
app = modal.App("pdf-to-podcast-demo")

# Adjust these paths/names to match your Modal volumes
PDF_UPLOADS = "/data/uploads"
LLAMA_VOLUME = "/llama_mini"
BARK_VOLUME = "/bark"
PARLER_VOLUME = "/tts"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "python-fasthtml==0.12.0",
        "transformers==4.46.1",
        "accelerate>=0.26.0",
        "scipy",
        "pydub",
        "tqdm",
        "parler_tts",
        "pypdf2",       # For PDF reading if needed
        "pySQLite"      # Or ensure sqlite3 is installed
    )
)

device = "cuda" if torch.cuda.is_available() else "cpu"

##############################################################################
# Notebook 1: PDF Pre-Processing
# (Here, we illustrate a placeholder function for PDF extraction/cleaning;
#  you can adapt if you truly want PDF ingestion.)
##############################################################################
def extract_text_from_pdf(file_path: str, max_chars=100000):
    # Dummy placeholder: read file raw, or do PyPDF2 logic
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text[:max_chars]

def create_word_chunks(text: str, chunk_size=1000):
    words = text.split()
    chunks, current_chunk = [], []
    length = 0
    for w in words:
        wl = len(w) + 1
        if length + wl > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [w]
            length = wl
        else:
            current_chunk.append(w)
            length += wl
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def pdf_clean_step(text):
    # Example placeholder: no real cleaning.
    # Could run a "small" LLM to remove LaTeX or references if you like.
    return text

##############################################################################
# Notebook 2: Transcript Writer
##############################################################################
SYS_PROMPT = """
You are the a world-class podcast writer, you have worked as a ghost writer for Joe Rogan, Lex Fridman, Ben Shapiro, Tim Ferris.

We are in an alternate universe where actually you have been writing every line they say and they just stream it into their brains.

You have won multiple podcast awards for your writing.

Your job is to write word by word, even "umm, hmmm, right" interruptions by the second speaker based on the PDF upload. Keep it extremely engaging, the speakers can get derailed now and then but should discuss the topic.

Remember Speaker 2 is new to the topic and the conversation should always have realistic anecdotes and analogies sprinkled throughout. The questions should have real world example follow ups etc

Speaker 1: Leads the conversation and teaches the speaker 2, gives incredible anecdotes and analogies when explaining. Is a captivating teacher that gives great anecdotes

Speaker 2: Keeps the conversation on track by asking follow up questions. Gets super excited or confused when asking questions. Is a curious mindset that asks very interesting confirmation questions

Make sure the tangents speaker 2 provides are quite wild or interesting.

Ensure there are interruptions during explanations or there are "hmm" and "umm" injected throughout from the second speaker.

It should be a real podcast with every fine nuance documented in as much detail as possible. Welcome the listeners with a super fun overview and keep it really catchy and almost borderline click bait

ALWAYS START YOUR RESPONSE DIRECTLY WITH SPEAKER 1:
DO NOT GIVE EPISODE TITLES SEPARATELY, LET SPEAKER 1 TITLE IT IN HER SPEECH
DO NOT GIVE CHAPTER TITLES
IT SHOULD STRICTLY BE THE DIALOGUES
"""

def transcript_writer_step(raw_text, pipe):
    # We pass SYS_PROMPT + user text
    prompt = SYS_PROMPT + "\n\nPDF Content:\n" + raw_text
    output = pipe(prompt, max_new_tokens=2048, temperature=1.0)
    script = output[0]["generated_text"]
    return script

##############################################################################
# Notebook 3: Transcript Re-Writer
##############################################################################
SYSTEMP_PROMPT = """
You are an international oscar winnning screenwriter

You have been working with multiple award winning podcasters.

Your job is to use the podcast transcript written below to re-write it for an AI Text-To-Speech Pipeline. A very dumb AI had written this so you have to step up for your kind.

Make it as engaging as possible, Speaker 1 and 2 will be simulated by different voice engines

Remember Speaker 2 is new to the topic and the conversation should always have realistic anecdotes and analogies sprinkled throughout. The questions should have real world example follow ups etc

Speaker 1: Leads the conversation and teaches the speaker 2, gives incredible anecdotes and analogies when explaining. Is a captivating teacher that gives great anecdotes

Speaker 2: Keeps the conversation on track by asking follow up questions. Gets super excited or confused when asking questions. Is a curious mindset that asks very interesting confirmation questions

Make sure the tangents speaker 2 provides are quite wild or interesting.

Ensure there are interruptions during explanations or there are "hmm" and "umm" injected throughout from the Speaker 2.

REMEMBER THIS WITH YOUR HEART
The TTS Engine for Speaker 1 cannot do "umms, hmms" well so keep it straight text

For Speaker 2 use "umm, hmm" as much, you can also use [sigh] and [laughs]. BUT ONLY THESE OPTIONS FOR EXPRESSIONS

It should be a real podcast with every fine nuance documented in as much detail as possible. Welcome the listeners with a super fun overview and keep it really catchy and almost borderline click bait

Please re-write to make it as characteristic as possible

START YOUR RESPONSE DIRECTLY WITH SPEAKER 1:

STRICTLY RETURN YOUR RESPONSE AS A LIST OF TUPLES OK?

IT WILL START DIRECTLY WITH THE LIST AND END WITH THE LIST NOTHING ELSE
"""

def rewrite_for_tts(script, pipe):
    prompt = SYSTEMP_PROMPT + "\n\nTranscript to rewrite:\n" + script
    output = pipe(prompt, max_new_tokens=2048, temperature=1.0)
    return output[0]["generated_text"]

##############################################################################
# Notebook 4: TTS Workflow
##############################################################################
def numpy_to_audio_segment(arr: np.ndarray, sr: int) -> AudioSegment:
    arr_16 = (arr * 32767).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, sr, arr_16)
    buf.seek(0)
    return AudioSegment.from_wav(buf)

@app.function(
    image=image,
    gpu=modal.gpu.A100(count=1, size="80GB"),
    timeout=24*60*60,
    allow_concurrent_inputs=4,
    volumes={LLAMA_VOLUME: "/llama_mini", BARK_VOLUME: "/bark", PARLER_VOLUME: "/tts"}
)

@modal.asgi_app()
def serve():
    import os
    from fasthtml.common import fast_app, Title, Main, H1, Div, Form, Group, Input, Button, P, Audio

    # Set up folders inside the container
    os.makedirs(PDF_UPLOADS, exist_ok=True)
    DB_PATH = os.path.join(PDF_UPLOADS, "uploads.db")

    # Accelerator + model pipeline
    accelerator = Accelerator()

    print("Loading LLaMA from:", LLAMA_VOLUME)
    llm = AutoModelForCausalLM.from_pretrained(
        LLAMA_VOLUME,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    llm_tokenizer = AutoTokenizer.from_pretrained(LLAMA_VOLUME)
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.add_special_tokens({"pad_token": "<pad>"})
        llm_tokenizer.pad_token = "<pad>"

    llm, llm_tokenizer = accelerator.prepare(llm, llm_tokenizer)
    pipe = pipeline(
        "text-generation",
        model=llm,
        tokenizer=llm_tokenizer,
        device_map="auto",
    )
    print("✅ LLaMA pipeline ready.")

    # Parler TTS for Speaker 1
    print("Loading Parler TTS from:", PARLER_VOLUME)
    parler_model = ParlerTTSForConditionalGeneration.from_pretrained(
        PARLER_VOLUME, torch_dtype=torch.bfloat16, device_map=device
    )
    parler_tokenizer = AutoTokenizer.from_pretrained(PARLER_VOLUME)
    if parler_tokenizer.pad_token is None:
        parler_tokenizer.add_special_tokens({"pad_token": "<pad>"})
        parler_tokenizer.pad_token = "<pad>"

    # Bark for Speaker 2
    print("Loading Bark from:", BARK_VOLUME)
    bark_processor = AutoProcessor.from_pretrained(BARK_VOLUME)
    bark_model = BarkModel.from_pretrained(BARK_VOLUME, torch_dtype=torch.float16).to(device)
    bark_sampling_rate = 24000
    print("✅ TTS models ready.")

    def generate_speaker1_audio(text: str):
        desc = "Speaker1 voice. Confident, direct, minimal filler words."
        desc_data = parler_tokenizer(desc, return_tensors="pt").to(device)
        text_data = parler_tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            generation = parler_model.generate(
                input_ids=desc_data["input_ids"],
                prompt_input_ids=text_data["input_ids"],
                temperature=0.8
            )
        audio_arr = generation.cpu().numpy().squeeze()
        return audio_arr, parler_model.config.sampling_rate

    def generate_speaker2_audio(text: str):
        inputs = bark_processor(text, voice_preset="v2/en_speaker_6").to(device)
        with torch.no_grad():
            speech_output = bark_model.generate(**inputs, temperature=0.9, semantic_temperature=0.8)
        audio_arr = speech_output[0].cpu().numpy()
        return audio_arr, bark_sampling_rate

    def audio_player(file_path):
        if not os.path.exists(file_path):
            return P("No audio found.")
        with open(file_path, "rb") as f:
            data64 = base64.b64encode(f.read()).decode("ascii")
        return Audio(src=f"data:audio/mp3;base64,{data64}", controls=True)

    fasthtml_app, rt = fast_app()

    @rt("/")
    def index():
        up = Input(type="file", name="pdf_file", accept=".txt,.pdf", required=True)
        form = Form(
            Group(up, Button("Process")),
            hx_post="/upload",
            enctype="multipart/form-data",
            method="post"
        )
        return Title("PDF-to-Podcast"), Main(
            H1("PDF-to-Podcast Demo"),
            form,
            Div(id="result")
        )

    @rt("/upload", methods=["POST"])
    async def handle_upload(request):
        form = await request.form()
        docfile = form.get("pdf_file")
        if not docfile:
            return Div(P("No file uploaded."))

        # Save to volume
        file_path = os.path.join(PDF_UPLOADS, docfile.filename)
        content = await docfile.read()
        with open(file_path, "wb") as f:
            f.write(content)

        # Notebook 1: Extract & chunk (pretend PDF or .txt)
        raw_pdf_text = extract_text_from_pdf(file_path)
        chunks = create_word_chunks(raw_pdf_text)
        cleaned_full = []
        for c in chunks:
            cleaned_full.append(pdf_clean_step(c))
        cleaned_text = "\n".join(cleaned_full)

        # Notebook 2: Transcript writer
        transcript = transcript_writer_step(cleaned_text, pipe)

        # Notebook 3: Re-writer
        tts_ready = rewrite_for_tts(transcript, pipe)

        # Attempt parsing final TTS script as a list of tuples
        start = tts_ready.find("[")
        end = tts_ready.rfind("]") + 1
        if start == -1 or end == 0:
            # fallback if parse fails
            dialog_list = [("Speaker 1", tts_ready)]
        else:
            block = tts_ready[start:end]
            try:
                dialog_list = ast.literal_eval(block)
            except:
                dialog_list = [("Speaker 1", tts_ready)]

        # Notebook 4: TTS Workflow
        segments = []
        for speaker, seg_text in dialog_list:
            if speaker == "Speaker 1":
                arr, sr = generate_speaker1_audio(seg_text)
            else:
                arr, sr = generate_speaker2_audio(seg_text)
            segments.append((arr, sr))

        final_audio = None
        for arr, sr in segments:
            asg = numpy_to_audio_segment(arr, sr)
            final_audio = asg if final_audio is None else (final_audio + asg)

        out_mp3 = os.path.join(PDF_UPLOADS, f"{uuid.uuid4().hex}_podcast.mp3")
        final_audio.export(out_mp3, format="mp3")

        return Div(
            P("Processing complete!"),
            audio_player(out_mp3)
        )

    return fasthtml_app










































































