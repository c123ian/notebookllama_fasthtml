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

BARK_VOLUME = "/bark"
DEFAULT_NAME_TTS = "suno/bark"
volume = modal.Volume.from_name("bark", create_if_missing=True)

MINUTES = 60
HOURS = 60 * MINUTES

app = modal.App(image=image, secrets=[modal.Secret.from_name("huggingface")])

@app.function(volumes={BARK_VOLUME: volume}, timeout=4 * HOURS)
def download_model(model_name=DEFAULT_NAME_TTS, force_download=False):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    volume.reload()

    snapshot_download(
        repo_id=model_name,
        local_dir=BARK_VOLUME,
        local_dir_use_symlinks=False,
        resume_download=True,
        ignore_patterns=None,
        force_download=force_download,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(BARK_VOLUME)

    volume.commit()

@app.local_entrypoint()
def main(
    model_name: str = DEFAULT_NAME_TTS,
    force_download: bool = False,
):
    download_model.remote(model_name, force_download)
