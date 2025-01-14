import modal

app = modal.App("download_model")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        [
            "huggingface_hub",
            "hf-transfer",
            "transformers",
            "torch",
        ]
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

TTS_DIR = "/tts"
DEFAULT_NAME_TTS = "parler-tts/parler-tts-mini-v1"
volume = modal.Volume.from_name("tts", create_if_missing=True)

MINUTES = 60
HOURS = 60 * MINUTES

app = modal.App(image=image, secrets=[modal.Secret.from_name("huggingface")])

@app.function(volumes={TTS_DIR: volume}, timeout=4 * HOURS)
def download_model(model_name=DEFAULT_NAME_TTS, force_download=False):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    volume.reload()

    snapshot_download(
        repo_id=model_name,
        local_dir=TTS_DIR,
        local_dir_use_symlinks=False,
        resume_download=True,
        ignore_patterns=None,
        force_download=force_download,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(TTS_DIR)

    volume.commit()

@app.local_entrypoint()
def main(
    model_name: str = DEFAULT_NAME_TTS,
    force_download: bool = False,
):
    download_model.remote(model_name, force_download)
