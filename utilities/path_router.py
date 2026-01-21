from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

# Input directory for target audio files
IN_DIR = ROOT_DIR / "in"

# Output directory for generated voices and audio
OUT_DIR = ROOT_DIR / "out"
CONVERTED_DIR = OUT_DIR / "converted_audio"

# Interpolated voices directory
INTERPOLATED_DIR = ROOT_DIR / "interpolated"

# Texts directory for transcriptions
TEXTS_DIR = ROOT_DIR / "texts"

# Voices directory for voice tensors
VOICES_DIR = ROOT_DIR / "voices"

# Examples directory
EXAMPLES_DIR = ROOT_DIR / "examples"
