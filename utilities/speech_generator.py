import warnings
from typing import Optional

import numpy as np
import torch
from kokoro import KPipeline


class SpeechGenerator:
    """Generates speech audio using Kokoro TTS pipeline."""

    def __init__(self, lang_code: str = "a", repo_id: str = "hexgrad/Kokoro-82M"):
        """Initialize the Kokoro TTS pipeline.

        Args:
            lang_code: Language code for the pipeline (default: "a" for American English)
            repo_id: HuggingFace repository ID for the model
        """
        self._suppress_warnings()
        self.pipeline = KPipeline(lang_code=lang_code, repo_id=repo_id)

    def generate_audio(
        self,
        text: str,
        voice: torch.Tensor,
        speed: float = 1.0
    ) -> np.ndarray:
        """Generate audio from text using a voice tensor.

        Args:
            text: The text to synthesize
            voice: Voice tensor defining the voice characteristics
            speed: Speech speed multiplier (default: 1.0)

        Returns:
            Audio data as a float32 numpy array at 24kHz sample rate
        """
        generator = self.pipeline(text, voice, speed)
        audio_chunks = []
        for gs, ps, chunk in generator:
            audio_chunks.append(chunk)
        return np.concatenate(audio_chunks).astype(np.float32)

    def _suppress_warnings(self):
        """Suppress common library warnings that clutter the console."""
        warnings.filterwarnings(
            "ignore",
            message=".*RNN module weights are not part of single contiguous chunk of memory.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=".*is deprecated in favor of*",
            category=FutureWarning
        )
        warnings.filterwarnings(
            "ignore",
            message=".*dropout option adds dropout after all but last recurrent layer*",
            category=UserWarning,
        )
