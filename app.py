#!/usr/bin/env python3
"""
Voice Cloning UI for Kokoro TTS - ULTRA FAST VERSION
Uses only Resemblyzer similarity (no heavy feature extraction)
"""

import os
import random
from pathlib import Path
from typing import Optional, Tuple, List
import warnings

warnings.filterwarnings("ignore")

import gradio as gr
import numpy as np
import soundfile as sf
import torch

from utilities.path_router import VOICES_DIR, OUT_DIR, IN_DIR

# Lazy load heavy modules
_generator = None
_encoder = None


def get_generator():
    """Lazy load speech generator."""
    global _generator
    if _generator is None:
        from utilities.speech_generator import SpeechGenerator
        _generator = SpeechGenerator()
    return _generator


def get_encoder():
    """Lazy load voice encoder."""
    global _encoder
    if _encoder is None:
        from resemblyzer import VoiceEncoder
        _encoder = VoiceEncoder()
    return _encoder


def get_available_voices() -> list:
    voices = list(Path(VOICES_DIR).glob("*.pt"))
    return [v.stem for v in voices]


def get_cloned_voices() -> list:
    voices = []
    for folder in OUT_DIR.glob("*"):
        if folder.is_dir():
            for pt_file in folder.glob("*.pt"):
                voices.append(str(pt_file))
    for pt_file in OUT_DIR.glob("*.pt"):
        voices.append(str(pt_file))
    return voices


def fast_similarity(audio1: np.ndarray, audio2: np.ndarray) -> float:
    """Fast similarity using only Resemblyzer (no heavy feature extraction)."""
    from resemblyzer import preprocess_wav
    encoder = get_encoder()

    wav1 = preprocess_wav(audio1, source_sr=24000)
    wav2 = preprocess_wav(audio2, source_sr=24000)

    embed1 = encoder.embed_utterance(wav1)
    embed2 = encoder.embed_utterance(wav2)

    return float(np.inner(embed1, embed2))


def generate_speech(text: str, voice_path: str, speed: float = 1.0) -> Tuple[Optional[Tuple[int, np.ndarray]], str]:
    if not text.strip():
        return None, "Please enter some text."
    if not voice_path:
        return None, "Please select a voice."

    try:
        if voice_path.endswith(".pt"):
            voice = torch.load(voice_path, weights_only=True)
        else:
            voice = torch.load(VOICES_DIR / f"{voice_path}.pt", weights_only=True)

        generator = get_generator()
        audio = generator.generate_audio(text, voice, speed)
        return (24000, audio), "Speech generated successfully!"
    except Exception as e:
        return None, f"Error: {str(e)}"


def clone_voice_ultra_fast(
    audio_file,
    target_text: str,
    mode: str,
    step_limit: int,
    output_name: str,
    progress=gr.Progress()
) -> Tuple[Optional[Tuple[int, np.ndarray]], str, str]:
    """Ultra-fast voice cloning."""

    if audio_file is None:
        return None, "", "Please upload a target audio file."
    if not target_text.strip():
        return None, "", "Please enter the text spoken in the audio."

    voice_files = list(Path(VOICES_DIR).glob("*.pt"))
    if not voice_files:
        return None, "", f"No voice files found in {VOICES_DIR}"

    try:
        from resemblyzer import preprocess_wav
        from utilities.audio_processor import convert_to_wav_mono_24k

        log_lines = []
        log_lines.append(f"Mode: {mode}")

        # Save and convert audio
        IN_DIR.mkdir(parents=True, exist_ok=True)
        input_path = IN_DIR / "uploaded_target.wav"

        if isinstance(audio_file, tuple):
            sr, audio_data = audio_file
            sf.write(str(input_path), audio_data, sr)
        else:
            import shutil
            shutil.copy(audio_file, str(input_path))

        progress(0.05, desc="Converting audio...")
        converted_path = convert_to_wav_mono_24k(str(input_path))

        # Load target audio and get embedding
        progress(0.1, desc="Analyzing target voice...")
        target_audio, _ = sf.read(converted_path, dtype="float32")
        target_wav = preprocess_wav(converted_path, source_sr=24000)
        encoder = get_encoder()
        target_embed = encoder.embed_utterance(target_wav)

        generator = get_generator()

        # QUICK MATCH: Just find best voice
        if mode == "Quick Match (Instant)":
            progress(0.2, desc="Finding best match...")
            best_voice = None
            best_sim = 0
            best_name = ""

            for i, vf in enumerate(voice_files):
                progress(0.2 + 0.7 * (i / len(voice_files)), desc=f"Testing {vf.stem}...")
                try:
                    voice = torch.load(vf, weights_only=True)
                    audio = generator.generate_audio(target_text[:80], voice)
                    wav = preprocess_wav(audio, source_sr=24000)
                    embed = encoder.embed_utterance(wav)
                    sim = float(np.inner(embed, target_embed))

                    if sim > best_sim:
                        best_sim = sim
                        best_voice = voice
                        best_name = vf.stem
                except:
                    continue

            if best_voice is None:
                return None, "", "Could not find matching voice."

            log_lines.append(f"Best match: {best_name}")
            log_lines.append(f"Similarity: {best_sim:.1%}")

            OUT_DIR.mkdir(parents=True, exist_ok=True)
            final_path = OUT_DIR / f"{output_name}_matched.pt"
            torch.save(best_voice, final_path)

            sample_audio = generator.generate_audio(target_text[:200], best_voice)
            log_lines.append(f"Saved to: {final_path}")

            return (24000, sample_audio), str(final_path), "\n".join(log_lines)

        # OPTIMIZATION MODES
        progress(0.15, desc="Finding best starting voice...")

        # Quick search for top 3 voices
        voice_scores = []
        for i, vf in enumerate(voice_files[:20]):  # Only check first 20 for speed
            progress(0.15 + 0.15 * (i / 20), desc=f"Scanning {vf.stem}...")
            try:
                voice = torch.load(vf, weights_only=True)
                audio = generator.generate_audio(target_text[:60], voice)
                wav = preprocess_wav(audio, source_sr=24000)
                embed = encoder.embed_utterance(wav)
                sim = float(np.inner(embed, target_embed))
                voice_scores.append((vf, voice, sim))
            except:
                continue

        voice_scores.sort(key=lambda x: x[2], reverse=True)
        top_voices = voice_scores[:3]

        if not top_voices:
            return None, "", "Could not find suitable voices."

        best_file, best_voice, best_sim = top_voices[0]
        log_lines.append(f"Starting voice: {best_file.stem} ({best_sim:.1%})")

        # Calculate voice statistics for mutation
        all_voices = [v for _, v, _ in top_voices]
        stacked = torch.stack(all_voices, dim=0)
        voice_std = stacked.std(dim=0)

        # Optimization loop - FAST version
        if mode == "Light (Fast)":
            step_limit = min(step_limit, 100)
        elif mode == "Standard":
            step_limit = min(step_limit, 300)

        log_lines.append(f"Running {step_limit} optimization steps...")
        improvements = 0

        for i in range(step_limit):
            pct = 0.35 + 0.6 * (i / step_limit)
            progress(pct, desc=f"Step {i+1}/{step_limit} | Similarity: {best_sim:.1%}")

            # Mutate voice
            diversity = random.uniform(0.03, 0.10)
            noise = torch.randn_like(best_voice) * voice_std * diversity
            new_voice = best_voice + noise

            # Quick similarity check
            audio = generator.generate_audio(target_text[:80], new_voice)
            wav = preprocess_wav(audio, source_sr=24000)
            embed = encoder.embed_utterance(wav)
            new_sim = float(np.inner(embed, target_embed))

            if new_sim > best_sim:
                improvements += 1
                best_sim = new_sim
                best_voice = new_voice
                log_lines.append(f"Step {i}: {best_sim:.1%}")

        progress(0.98, desc="Saving...")

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        final_path = OUT_DIR / f"{output_name}_cloned.pt"
        torch.save(best_voice, final_path)

        sample_audio = generator.generate_audio(target_text[:200], best_voice)

        log_lines.append(f"\n--- Results ---")
        log_lines.append(f"Improvements: {improvements}")
        log_lines.append(f"Final Similarity: {best_sim:.1%}")
        log_lines.append(f"Saved to: {final_path}")

        return (24000, sample_audio), str(final_path), "\n".join(log_lines)

    except Exception as e:
        import traceback
        return None, "", f"Error: {str(e)}\n\n{traceback.format_exc()}"


def transcribe_audio(audio_file) -> str:
    if audio_file is None:
        return "Please upload an audio file first."
    try:
        from utilities.audio_processor import Transcriber
        IN_DIR.mkdir(parents=True, exist_ok=True)
        temp_path = IN_DIR / "temp_transcribe.wav"

        if isinstance(audio_file, tuple):
            sr, audio_data = audio_file
            sf.write(str(temp_path), audio_data, sr)
        else:
            import shutil
            shutil.copy(audio_file, str(temp_path))

        transcriber = Transcriber()
        return transcriber.transcribe(str(temp_path), save_text=False)
    except ImportError:
        return "Install faster-whisper: pip install faster-whisper"
    except Exception as e:
        return f"Error: {str(e)}"


def refresh_voices():
    return gr.update(choices=get_available_voices()), gr.update(choices=get_cloned_voices())


def create_ui():
    with gr.Blocks(title="Kokoro Voice Cloning") as app:
        gr.Markdown("""
        # Kokoro Voice Cloning (Fast)

        Clone voices quickly using Kokoro TTS.
        """)

        with gr.Tabs():
            with gr.TabItem("Clone Voice"):
                with gr.Row():
                    with gr.Column(scale=1):
                        audio_input = gr.Audio(
                            label="Target Audio",
                            type="filepath",
                            sources=["upload", "microphone"]
                        )

                        target_text = gr.Textbox(
                            label="Target Text (what is spoken in audio)",
                            placeholder="Enter the text...",
                            lines=2
                        )

                        transcribe_btn = gr.Button("Auto-Transcribe", variant="secondary")

                        mode_dropdown = gr.Dropdown(
                            choices=[
                                "Quick Match (Instant)",
                                "Light (Fast)",
                                "Standard",
                                "Deep (Slow)"
                            ],
                            value="Quick Match (Instant)",
                            label="Mode",
                            info="Quick=30sec, Light=1min, Standard=2-3min"
                        )

                        step_limit = gr.Slider(
                            minimum=50,
                            maximum=1000,
                            value=100,
                            step=50,
                            label="Steps (for optimization modes)",
                            visible=False
                        )

                        output_name = gr.Textbox(
                            label="Output Name",
                            value="my_voice"
                        )

                        clone_btn = gr.Button("Start Cloning", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        output_audio = gr.Audio(label="Output", type="numpy")
                        voice_path_output = gr.Textbox(label="Saved To", interactive=False)
                        log_output = gr.Textbox(label="Log", lines=10, interactive=False)

                def update_ui(mode):
                    if mode == "Quick Match (Instant)":
                        return gr.update(visible=False, value=0)
                    elif mode == "Light (Fast)":
                        return gr.update(visible=True, value=100)
                    elif mode == "Standard":
                        return gr.update(visible=True, value=300)
                    else:
                        return gr.update(visible=True, value=500)

                mode_dropdown.change(fn=update_ui, inputs=[mode_dropdown], outputs=[step_limit])
                transcribe_btn.click(fn=transcribe_audio, inputs=[audio_input], outputs=[target_text])
                clone_btn.click(
                    fn=clone_voice_ultra_fast,
                    inputs=[audio_input, target_text, mode_dropdown, step_limit, output_name],
                    outputs=[output_audio, voice_path_output, log_output]
                )

            with gr.TabItem("Test Voice"):
                with gr.Row():
                    with gr.Column():
                        stock_dropdown = gr.Dropdown(choices=get_available_voices(), label="Stock Voice")
                        cloned_dropdown = gr.Dropdown(choices=get_cloned_voices(), label="Cloned Voice")
                        refresh_btn = gr.Button("Refresh Lists")
                        test_text = gr.Textbox(label="Text", value="Hello, this is a voice test.", lines=2)
                        speed = gr.Slider(0.5, 2.0, 1.0, 0.1, label="Speed")
                        with gr.Row():
                            test_stock_btn = gr.Button("Test Stock")
                            test_cloned_btn = gr.Button("Test Cloned", variant="primary")

                    with gr.Column():
                        test_audio = gr.Audio(label="Output", type="numpy")
                        test_status = gr.Textbox(label="Status", interactive=False)

                refresh_btn.click(fn=refresh_voices, outputs=[stock_dropdown, cloned_dropdown])
                test_stock_btn.click(fn=generate_speech, inputs=[test_text, stock_dropdown, speed], outputs=[test_audio, test_status])
                test_cloned_btn.click(fn=generate_speech, inputs=[test_text, cloned_dropdown, speed], outputs=[test_audio, test_status])

            with gr.TabItem("Info"):
                gr.Markdown("""
                ## Speed Modes

                | Mode | Time | Use When |
                |------|------|----------|
                | **Quick Match** | ~30 sec | Find closest stock voice |
                | **Light** | ~1 min | Quick clone with 100 steps |
                | **Standard** | ~2-3 min | Better quality, 300 steps |
                | **Deep** | ~5+ min | Best quality, 500+ steps |

                ## Tips
                - Start with **Quick Match** - often good enough!
                - Use clear audio without background noise
                - 10-30 seconds of speech is ideal
                """)

    return app


if __name__ == "__main__":
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IN_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Voices: {len(list(VOICES_DIR.glob('*.pt')))} files")

    app = create_ui()
    app.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True)
