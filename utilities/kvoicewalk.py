"""Main random walk algorithm for voice cloning."""

import datetime
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import soundfile as sf
import torch
from tqdm import tqdm

from utilities.fitness_scorer import FitnessScorer
from utilities.initial_selector import InitialSelector
from utilities.path_router import OUT_DIR
from utilities.speech_generator import SpeechGenerator
from utilities.voice_generator import VoiceGenerator


class KVoiceWalk:
    """Random walk algorithm for voice cloning through tensor space exploration.

    This class implements a random walk optimization that iteratively mutates
    voice tensors to find one that closely matches a target voice. It uses
    a hybrid scoring system that balances target similarity, self-consistency,
    and audio quality.
    """

    def __init__(
        self,
        target_audio: Path,
        target_text: str,
        other_text: str,
        voice_folder: str,
        interpolate_start: bool = False,
        population_limit: int = 10,
        starting_voice: Optional[str] = None,
        output_name: str = "cloned_voice"
    ):
        """Initialize the voice walk optimizer.

        Args:
            target_audio: Path to the target audio file
            target_text: Text content of the target audio
            other_text: Different text for self-similarity testing
            voice_folder: Folder containing voice tensor files
            interpolate_start: Whether to use interpolation search for initial voices
            population_limit: Number of voices to use in initial selection
            starting_voice: Optional specific voice tensor to start from
            output_name: Base name for output files
        """
        self.target_audio = target_audio
        self.target_text = target_text
        self.other_text = other_text
        self.output_name = output_name

        print("Initializing KVoiceWalk...")

        # Initialize voice selector
        self.initial_selector = InitialSelector(
            str(target_audio),
            target_text,
            other_text,
            voice_folder=voice_folder
        )

        # Get initial voices
        print("\nSelecting initial voices...")
        if interpolate_start:
            voices = self.initial_selector.interpolate_search(population_limit)
        else:
            voices = self.initial_selector.top_performer_start(population_limit)

        # Initialize components
        self.speech_generator = SpeechGenerator()
        self.fitness_scorer = FitnessScorer(str(target_audio))
        self.voice_generator = VoiceGenerator(voices, starting_voice)
        self.starting_voice = self.voice_generator.starting_voice

        print("Initialization complete.")

    def random_walk(self, step_limit: int = 10000) -> Dict[str, Any]:
        """Execute the random walk optimization.

        Iteratively mutates the voice tensor, keeping improvements and
        discarding worse results. Saves checkpoints when improvements are found.

        Args:
            step_limit: Maximum number of iterations

        Returns:
            Dictionary with final results including best voice and scores
        """
        best_voice = self.starting_voice
        best_results = self.score_voice(best_voice)

        # Create output directory
        now = datetime.datetime.now()
        results_dir = Path(OUT_DIR) / f"{self.output_name}_{self.target_audio.stem}_{now.strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(results_dir, exist_ok=True)

        print(f"\nStarting random walk optimization...")
        print(f"Output directory: {results_dir}")
        print(f"Initial scores:")
        print(f"  Target Sim: {best_results['target_similarity']:.3f}")
        print(f"  Self Sim:   {best_results['self_similarity']:.3f}")
        print(f"  Feature Sim:{best_results['feature_similarity']:.3f}")
        print(f"  Score:      {best_results['score']:.2f}")
        print()

        pbar = tqdm(range(step_limit), desc="Random Walk")
        improvements = 0

        for i in pbar:
            # Random diversity factor for exploration
            diversity = random.uniform(0.01, 0.15)

            # Generate mutated voice
            voice = self.voice_generator.generate_voice(best_voice, diversity)

            # Quick rejection based on minimum similarity threshold
            min_similarity = best_results["target_similarity"] * 0.98
            voice_results = self.score_voice(voice, min_similarity)

            # Keep if better
            if voice_results["score"] > best_results["score"]:
                improvements += 1
                best_results = voice_results
                best_voice = voice

                # Update progress bar
                pbar.set_postfix({
                    "score": f"{best_results['score']:.2f}",
                    "target": f"{best_results['target_similarity']:.3f}",
                    "improvements": improvements
                })

                # Save checkpoint
                checkpoint_name = (
                    f"{self.output_name}_{i}_{best_results['score']:.2f}_"
                    f"{best_results['target_similarity']:.2f}_{self.target_audio.stem}"
                )
                torch.save(best_voice, results_dir / f"{checkpoint_name}.pt")
                sf.write(
                    results_dir / f"{checkpoint_name}.wav",
                    best_results["audio"],
                    24000
                )

                tqdm.write(
                    f"Step {i:>5}: Score={best_results['score']:.2f} "
                    f"Target={best_results['target_similarity']:.3f} "
                    f"Self={best_results['self_similarity']:.3f} "
                    f"Feature={best_results['feature_similarity']:.3f} "
                    f"Diversity={diversity:.3f}"
                )

        # Final summary
        print(f"\n{'='*60}")
        print(f"Random Walk Complete: {self.output_name}")
        print(f"{'='*60}")
        print(f"Total iterations:  {step_limit}")
        print(f"Total improvements:{improvements}")
        print(f"Final Score:       {best_results['score']:.2f}")
        print(f"Target Similarity: {best_results['target_similarity']:.3f}")
        print(f"Self Similarity:   {best_results['self_similarity']:.3f}")
        print(f"Feature Similarity:{best_results['feature_similarity']:.3f}")
        print(f"Output directory:  {results_dir}")
        print(f"{'='*60}")

        # Save final voice
        final_path = results_dir / f"{self.output_name}_final.pt"
        torch.save(best_voice, final_path)
        sf.write(
            results_dir / f"{self.output_name}_final.wav",
            best_results["audio"],
            24000
        )

        return {
            "voice": best_voice,
            "voice_path": final_path,
            "results_dir": results_dir,
            **best_results
        }

    def score_voice(
        self,
        voice: torch.Tensor,
        min_similarity: float = 0.0
    ) -> Dict[str, Any]:
        """Score a voice tensor against the target.

        Uses early rejection based on target similarity to speed up
        the optimization process.

        Args:
            voice: Voice tensor to evaluate
            min_similarity: Minimum target similarity to compute full score

        Returns:
            Dictionary with scores and generated audio
        """
        # Generate audio with target text
        audio = self.speech_generator.generate_audio(self.target_text, voice)
        target_similarity = self.fitness_scorer.target_similarity(audio)

        results: Dict[str, Any] = {"audio": audio}

        # Full scoring only if above threshold (early rejection)
        if target_similarity > min_similarity:
            # Generate second audio with different text for self-similarity
            audio2 = self.speech_generator.generate_audio(self.other_text, voice)
            results.update(
                self.fitness_scorer.hybrid_similarity(audio, audio2, target_similarity)
            )
        else:
            results["score"] = 0.0
            results["target_similarity"] = target_similarity
            results["self_similarity"] = 0.0
            results["feature_similarity"] = 0.0

        return results
