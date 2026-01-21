"""Audio processing utilities for format conversion and transcription."""

import datetime
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import soundfile as sf

from utilities.path_router import CONVERTED_DIR, TEXTS_DIR


def convert_to_wav_mono_24k(input_path: str, output_path: Optional[str] = None) -> str:
    """Convert audio file to 24kHz mono WAV format.

    Args:
        input_path: Path to input audio file
        output_path: Optional output path (default: converted_audio directory)

    Returns:
        Path to the converted file
    """
    CONVERTED_DIR.mkdir(parents=True, exist_ok=True)

    if output_path is None:
        input_name = Path(input_path).stem
        output_path = str(CONVERTED_DIR / f"{input_name}_24k.wav")

    # Read audio file
    audio, sr = sf.read(input_path, dtype="float32")

    # Convert to mono if stereo
    if len(audio.shape) > 1 and audio.shape[1] > 1:
        audio = np.mean(audio, axis=1)

    # Resample to 24kHz if needed
    if sr != 24000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=24000)

    # Write output
    sf.write(output_path, audio, 24000)
    print(f"Converted: {input_path} -> {output_path}")

    return output_path


class Transcriber:
    """Transcribes audio files to text using Whisper."""

    def __init__(self, model_size: str = "large-v3", device: str = "cpu"):
        """Initialize the transcriber.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large-v3)
            device: Device to run on (cpu or cuda)
        """
        try:
            from faster_whisper import WhisperModel
            self.model = WhisperModel(
                model_size,
                device=device,
                compute_type="int8" if device == "cpu" else "float16"
            )
            self.available = True
        except ImportError:
            print("Warning: faster-whisper not installed. Transcription disabled.")
            self.available = False

    def transcribe(self, audio_path: str, save_text: bool = True) -> str:
        """Transcribe an audio file to text.

        Args:
            audio_path: Path to the audio file
            save_text: Whether to save transcription to a text file

        Returns:
            Transcribed text
        """
        if not self.available:
            raise RuntimeError("Transcription not available. Install faster-whisper.")

        print(f"Transcribing: {audio_path}")
        start_time = datetime.datetime.now()

        segments, info = self.model.transcribe(audio_path, beam_size=5)
        print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")

        text_parts = []
        for segment in segments:
            text_parts.append(segment.text)

        full_text = " ".join(text_parts).strip()

        duration = datetime.datetime.now() - start_time
        print(f"Transcription completed in {duration}")

        if save_text:
            TEXTS_DIR.mkdir(parents=True, exist_ok=True)
            text_path = TEXTS_DIR / f"{Path(audio_path).stem}.txt"
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(full_text)
            print(f"Saved transcription to: {text_path}")

        return full_text

    def transcribe_many(self, audio_paths: list) -> dict:
        """Transcribe multiple audio files.

        Args:
            audio_paths: List of paths to audio files

        Returns:
            Dictionary mapping audio paths to transcriptions
        """
        results = {}
        for path in audio_paths:
            try:
                results[path] = self.transcribe(path)
            except Exception as e:
                print(f"Error transcribing {path}: {e}")
                results[path] = None
        return results
