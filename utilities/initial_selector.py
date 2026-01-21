"""Initial voice selection and interpolation for finding good starting points."""

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from utilities.fitness_scorer import FitnessScorer
from utilities.path_router import VOICES_DIR, INTERPOLATED_DIR
from utilities.speech_generator import SpeechGenerator


def interpolate(voice1: torch.Tensor, voice2: torch.Tensor, alpha: float) -> torch.Tensor:
    """Interpolate between two voice tensors.

    Args:
        voice1: First voice tensor
        voice2: Second voice tensor
        alpha: Interpolation factor (-1.5 to 1.5 for extrapolation)

    Returns:
        Interpolated/extrapolated voice tensor
    """
    midpoint = (voice1 + voice2) / 2
    diff = voice2 - voice1
    return midpoint + alpha * diff


def safe_load_voice(path: str) -> torch.Tensor:
    """Safely load a voice tensor file.

    Args:
        path: Path to the .pt file

    Returns:
        Loaded voice tensor
    """
    return torch.load(path, weights_only=True)


class InitialSelector:
    """Selects initial voices and finds optimal starting points for random walk.

    This class evaluates available voice tensors against a target audio file
    and can perform interpolation search to find better starting positions.
    """

    def __init__(
        self,
        target_audio: str,
        target_text: str,
        other_text: str,
        voice_folder: str = str(VOICES_DIR)
    ):
        """Initialize the selector.

        Args:
            target_audio: Path to the target audio file
            target_text: Text that matches the target audio content
            other_text: Different text for self-similarity testing
            voice_folder: Folder containing voice tensor files (.pt)
        """
        self.target_audio = target_audio
        self.target_text = target_text
        self.other_text = other_text
        self.voice_folder = voice_folder
        self.speech_generator = SpeechGenerator()
        self.fitness_scorer = FitnessScorer(target_audio)

    def _load_voices(self) -> List[Tuple[str, torch.Tensor]]:
        """Load all voice tensors from the voice folder.

        Returns:
            List of (name, tensor) tuples
        """
        voices = []
        voice_files = list(Path(self.voice_folder).glob("*.pt"))

        for voice_path in voice_files:
            try:
                voice = safe_load_voice(str(voice_path))
                voices.append((voice_path.stem, voice))
            except Exception as e:
                print(f"Error loading {voice_path}: {e}")

        return voices

    def _score_voice(self, voice: torch.Tensor) -> Dict[str, Any]:
        """Score a single voice tensor.

        Args:
            voice: Voice tensor to evaluate

        Returns:
            Dictionary with scoring results
        """
        audio1 = self.speech_generator.generate_audio(self.target_text, voice)
        target_sim = self.fitness_scorer.target_similarity(audio1)

        audio2 = self.speech_generator.generate_audio(self.other_text, voice)
        results = self.fitness_scorer.hybrid_similarity(audio1, audio2, target_sim)
        results["voice"] = voice

        return results

    def top_performer_start(self, population_limit: int = 10) -> List[torch.Tensor]:
        """Find the top performing voices from available tensors.

        Args:
            population_limit: Maximum number of voices to return

        Returns:
            List of top performing voice tensors
        """
        voices = self._load_voices()
        if not voices:
            raise ValueError(f"No voice files found in {self.voice_folder}")

        print(f"Evaluating {len(voices)} voices...")
        results = []

        for name, voice in tqdm(voices, desc="Scoring voices"):
            try:
                score_result = self._score_voice(voice)
                score_result["name"] = name
                results.append(score_result)
            except Exception as e:
                print(f"Error scoring {name}: {e}")

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)

        # Print top performers
        print("\nTop performers:")
        for i, r in enumerate(results[:population_limit]):
            print(f"  {i+1}. {r['name']}: Score={r['score']:.2f}, "
                  f"Target={r['target_similarity']:.3f}, "
                  f"Self={r['self_similarity']:.3f}")

        return [r["voice"] for r in results[:population_limit]]

    def interpolate_search(self, population_limit: int = 10) -> List[torch.Tensor]:
        """Search for optimal voices using interpolation between top performers.

        This method first finds top performers, then creates interpolated
        variants between pairs of good voices to potentially find better
        starting positions in the voice tensor space.

        Args:
            population_limit: Maximum number of voices to return

        Returns:
            List of best voice tensors (including interpolated ones)
        """
        # First get top performers
        top_voices = self.top_performer_start(population_limit)

        if len(top_voices) < 2:
            return top_voices

        print("\nPerforming interpolation search...")
        os.makedirs(INTERPOLATED_DIR, exist_ok=True)

        all_results = []

        # Score original top voices
        for i, voice in enumerate(top_voices):
            result = self._score_voice(voice)
            result["name"] = f"original_{i}"
            all_results.append(result)

        # Interpolate between pairs
        alpha_values = np.arange(-1.5, 1.6, 0.25)

        pairs = []
        for i in range(len(top_voices)):
            for j in range(i + 1, len(top_voices)):
                pairs.append((i, j))

        for i, j in tqdm(pairs, desc="Interpolating"):
            voice1 = top_voices[i]
            voice2 = top_voices[j]

            for alpha in alpha_values:
                try:
                    interp_voice = interpolate(voice1, voice2, alpha)
                    result = self._score_voice(interp_voice)
                    result["name"] = f"interp_{i}_{j}_a{alpha:.2f}"
                    all_results.append(result)

                    # Save promising interpolations
                    if result["score"] > all_results[0]["score"]:
                        save_path = INTERPOLATED_DIR / f"{result['name']}_{result['score']:.2f}.pt"
                        torch.save(interp_voice, save_path)
                except Exception as e:
                    pass  # Skip failed interpolations

        # Sort by score
        all_results.sort(key=lambda x: x["score"], reverse=True)

        print("\nTop results after interpolation:")
        for i, r in enumerate(all_results[:population_limit]):
            print(f"  {i+1}. {r['name']}: Score={r['score']:.2f}, "
                  f"Target={r['target_similarity']:.3f}, "
                  f"Self={r['self_similarity']:.3f}")

        return [r["voice"] for r in all_results[:population_limit]]
