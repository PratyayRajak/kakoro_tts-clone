"""Download Kokoro voice tensors from HuggingFace."""

from huggingface_hub import hf_hub_download, list_repo_files
from pathlib import Path
import shutil

VOICES_DIR = Path("./voices")
REPO_ID = "hexgrad/Kokoro-82M"

def download_voices():
    VOICES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Fetching file list from {REPO_ID}...")
    files = list_repo_files(REPO_ID)

    # Find voice tensor files
    voice_files = [f for f in files if f.endswith('.pt') and 'voice' in f.lower()]

    # If no files with 'voice' in name, look for any .pt files in voices folder
    if not voice_files:
        voice_files = [f for f in files if f.endswith('.pt')]

    print(f"Found {len(voice_files)} voice files")

    for vf in voice_files:
        print(f"Downloading: {vf}")
        try:
            local_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=vf,
                local_dir="./kokoro_cache"
            )
            # Copy to voices folder
            dest = VOICES_DIR / Path(vf).name
            shutil.copy(local_path, dest)
            print(f"  Saved to: {dest}")
        except Exception as e:
            print(f"  Error: {e}")

    print(f"\nDone! Voice files saved to {VOICES_DIR}")
    print(f"Total voices: {len(list(VOICES_DIR.glob('*.pt')))}")

if __name__ == "__main__":
    download_voices()
