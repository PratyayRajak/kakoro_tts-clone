#!/usr/bin/env python3
"""
Voice Cloning for Kokoro TTS using Random Walk Algorithm

This tool generates custom voice style tensors for the Kokoro text-to-speech system
using a random walk algorithm paired with a hybrid scoring methodology.

Based on: https://github.com/RobViren/kvoicewalk

Usage:
    # Basic voice cloning
    python main.py --target_audio ./in/target.wav --target_text "Hello world"

    # With interpolation search for better starting point
    python main.py --target_audio ./in/target.wav --target_text "Hello world" --interpolate_start

    # Auto-transcribe target audio
    python main.py --target_audio ./in/target.wav --transcribe_start

    # Test a generated voice
    python main.py --test_voice ./out/voice.pt --target_text "Test text"
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from utilities.path_router import VOICES_DIR, OUT_DIR, TEXTS_DIR


def parse_args():
    parser = argparse.ArgumentParser(
        description="Voice cloning for Kokoro TTS using random walk optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Main operation modes
    mode_group = parser.add_argument_group("Operation Modes")
    mode_group.add_argument(
        "--target_audio",
        type=str,
        help="Path to target audio file (WAV, 24kHz mono preferred)"
    )
    mode_group.add_argument(
        "--target_text",
        type=str,
        help="Text content of the target audio (or path to .txt file)"
    )
    mode_group.add_argument(
        "--test_voice",
        type=str,
        help="Path to voice tensor (.pt) to test"
    )

    # Random walk options
    walk_group = parser.add_argument_group("Random Walk Options")
    walk_group.add_argument(
        "--interpolate_start",
        action="store_true",
        help="Use interpolation search for finding better starting voices"
    )
    walk_group.add_argument(
        "--step_limit",
        type=int,
        default=10000,
        help="Maximum number of random walk iterations (default: 10000)"
    )
    walk_group.add_argument(
        "--population_limit",
        type=int,
        default=10,
        help="Number of voices to use in initial selection (default: 10)"
    )
    walk_group.add_argument(
        "--starting_voice",
        type=str,
        help="Specific voice tensor to start random walk from"
    )
    walk_group.add_argument(
        "--voice_folder",
        type=str,
        default=str(VOICES_DIR),
        help=f"Folder containing voice tensors (default: {VOICES_DIR})"
    )

    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output_name",
        type=str,
        default="cloned_voice",
        help="Base name for output files (default: cloned_voice)"
    )

    # Utility options
    util_group = parser.add_argument_group("Utility Options")
    util_group.add_argument(
        "--transcribe_start",
        action="store_true",
        help="Auto-transcribe target audio before starting"
    )
    util_group.add_argument(
        "--transcribe_many",
        nargs="+",
        help="Transcribe multiple audio files"
    )
    util_group.add_argument(
        "--convert_audio",
        type=str,
        help="Convert audio file to 24kHz mono WAV"
    )
    util_group.add_argument(
        "--export_bin",
        type=str,
        help="Export voice tensors to compressed binary (.npz)"
    )
    util_group.add_argument(
        "--other_text",
        type=str,
        default="The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.",
        help="Alternative text for self-similarity testing"
    )

    return parser.parse_args()


def load_text(text_arg: str) -> str:
    """Load text from argument or file."""
    if text_arg.endswith(".txt") and Path(text_arg).exists():
        with open(text_arg, "r", encoding="utf-8") as f:
            return f.read().strip()
    return text_arg


def convert_audio(input_path: str):
    """Convert audio to 24kHz mono WAV."""
    from utilities.audio_processor import convert_to_wav_mono_24k
    output_path = convert_to_wav_mono_24k(input_path)
    print(f"Converted audio saved to: {output_path}")
    return output_path


def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio file to text."""
    from utilities.audio_processor import Transcriber
    transcriber = Transcriber()
    return transcriber.transcribe(audio_path)


def transcribe_many(audio_paths: list):
    """Transcribe multiple audio files."""
    from utilities.audio_processor import Transcriber
    transcriber = Transcriber()
    transcriber.transcribe_many(audio_paths)


def export_voices(output_path: str, voice_folder: str = str(VOICES_DIR)):
    """Export voice tensors to compressed binary."""
    voice_files = list(Path(voice_folder).glob("*.pt"))
    if not voice_files:
        print(f"No voice files found in {voice_folder}")
        return

    voices = {}
    for vf in voice_files:
        try:
            voice = torch.load(vf, weights_only=True)
            voices[vf.stem] = voice.numpy()
        except Exception as e:
            print(f"Error loading {vf}: {e}")

    np.savez_compressed(output_path, **voices)
    print(f"Exported {len(voices)} voices to {output_path}")


def test_voice(voice_path: str, text: str, output_name: str = "test_output"):
    """Generate audio using a voice tensor for testing."""
    from utilities.speech_generator import SpeechGenerator

    print(f"Loading voice: {voice_path}")
    voice = torch.load(voice_path, weights_only=True)

    print("Generating audio...")
    generator = SpeechGenerator()
    audio = generator.generate_audio(text, voice)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUT_DIR / f"{output_name}.wav"
    sf.write(output_path, audio, 24000)
    print(f"Audio saved to: {output_path}")


def run_voice_clone(args):
    """Run the voice cloning random walk."""
    from utilities.kvoicewalk import KVoiceWalk
    from utilities.audio_processor import convert_to_wav_mono_24k

    # Validate inputs
    if not args.target_audio:
        print("Error: --target_audio is required for voice cloning")
        sys.exit(1)

    target_audio = Path(args.target_audio)
    if not target_audio.exists():
        print(f"Error: Target audio not found: {target_audio}")
        sys.exit(1)

    # Convert audio if needed
    if not str(target_audio).endswith("_24k.wav"):
        print("Converting audio to 24kHz mono WAV...")
        target_audio = Path(convert_to_wav_mono_24k(str(target_audio)))

    # Get target text
    if args.transcribe_start:
        print("Transcribing target audio...")
        target_text = transcribe_audio(str(target_audio))
    elif args.target_text:
        target_text = load_text(args.target_text)
    else:
        # Check for existing transcription
        text_file = TEXTS_DIR / f"{target_audio.stem}.txt"
        if text_file.exists():
            target_text = load_text(str(text_file))
            print(f"Using existing transcription: {text_file}")
        else:
            print("Error: --target_text or --transcribe_start is required")
            sys.exit(1)

    # Check voice folder
    voice_folder = Path(args.voice_folder)
    if not voice_folder.exists():
        print(f"Error: Voice folder not found: {voice_folder}")
        print("Please download Kokoro voice tensors and place them in the voices folder.")
        sys.exit(1)

    voice_files = list(voice_folder.glob("*.pt"))
    if not voice_files:
        print(f"Error: No voice files (.pt) found in {voice_folder}")
        print("Please download Kokoro voice tensors from the model repository.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("Voice Cloning Configuration")
    print(f"{'='*60}")
    print(f"Target audio:      {target_audio}")
    print(f"Target text:       {target_text[:50]}..." if len(target_text) > 50 else f"Target text:       {target_text}")
    print(f"Voice folder:      {voice_folder} ({len(voice_files)} voices)")
    print(f"Interpolate start: {args.interpolate_start}")
    print(f"Step limit:        {args.step_limit}")
    print(f"Population limit:  {args.population_limit}")
    print(f"Output name:       {args.output_name}")
    print(f"{'='*60}\n")

    # Initialize and run
    walker = KVoiceWalk(
        target_audio=target_audio,
        target_text=target_text,
        other_text=args.other_text,
        voice_folder=str(voice_folder),
        interpolate_start=args.interpolate_start,
        population_limit=args.population_limit,
        starting_voice=args.starting_voice,
        output_name=args.output_name
    )

    results = walker.random_walk(step_limit=args.step_limit)

    print(f"\nVoice cloning complete!")
    print(f"Final voice tensor: {results['voice_path']}")
    print(f"Results directory:  {results['results_dir']}")

    return results


def main():
    args = parse_args()

    # Handle utility operations first
    if args.convert_audio:
        convert_audio(args.convert_audio)
        return

    if args.transcribe_many:
        transcribe_many(args.transcribe_many)
        return

    if args.export_bin:
        export_voices(args.export_bin, args.voice_folder)
        return

    # Test voice mode
    if args.test_voice:
        if not args.target_text:
            print("Error: --target_text is required for testing a voice")
            sys.exit(1)
        test_voice(args.test_voice, load_text(args.target_text), args.output_name)
        return

    # Voice cloning mode
    if args.target_audio:
        run_voice_clone(args)
        return

    # No valid operation specified
    print("Error: No operation specified.")
    print("Use --target_audio to clone a voice, --test_voice to test, or --help for more options.")
    sys.exit(1)


if __name__ == "__main__":
    main()
