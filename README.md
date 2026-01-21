# Kokoro TTS Voice Cloning

Voice cloning system using random walk optimization for Kokoro text-to-speech.

Based on [kvoicewalk](https://github.com/RobViren/kvoicewalk) by Rob Viren.

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/PratyayRajak/kakoro_tts-clone.git
cd kakoro_tts-clone

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download voice files (REQUIRED - ~54 voice tensors)
python download_voices.py

# 4. Run the web UI
python app.py
```

Then open http://localhost:7860 in your browser.

## Speed Modes

| Mode | Time | Description |
|------|------|-------------|
| **Quick Match** | ~30 sec | Find best matching stock voice |
| **Light** | ~1 min | Light optimization (100 steps) |
| **Standard** | ~2-3 min | Better quality (300 steps) |
| **Deep** | ~5+ min | Best quality (500+ steps) |

## How It Works

1. Upload a target audio file (10-30 seconds of clear speech)
2. Enter the text spoken in the audio
3. Select a cloning mode
4. Click "Start Cloning"
5. Test your cloned voice in the "Test Voice" tab

The system uses **Resemblyzer** speaker embeddings to find and optimize voice tensors that sound similar to your target voice.

## Requirements

- Python 3.10-3.12
- ~500MB disk space for voice files
- GPU recommended but CPU works

## Project Structure

```
├── app.py                  # Web UI (Gradio)
├── main.py                 # CLI interface
├── download_voices.py      # Downloads 54 Kokoro voice tensors
├── requirements.txt        # Dependencies
├── utilities/
│   ├── speech_generator.py # Kokoro TTS wrapper
│   ├── fitness_scorer.py   # Voice similarity scoring
│   ├── voice_generator.py  # Tensor mutation
│   ├── audio_processor.py  # Audio conversion
│   └── ...
├── voices/                 # Voice tensors (.pt) - downloaded
├── in/                     # Input audio files
└── out/                    # Cloned voice outputs
```

## CLI Usage

```bash
# Clone a voice
python main.py --target_audio ./in/target.wav --target_text "Hello world"

# Test a cloned voice
python main.py --test_voice ./out/my_voice_cloned.pt --target_text "Test"
```

## Tips

- **Start with Quick Match** - often good enough!
- Use clear audio without background noise
- 10-30 seconds of speech works best
- Results are in an "uncanny valley" - similar but not identical

## Credits

- [kvoicewalk](https://github.com/RobViren/kvoicewalk) - Original implementation
- [Kokoro TTS](https://huggingface.co/hexgrad/Kokoro-82M) - TTS model
- [Resemblyzer](https://github.com/resemble-ai/Resemblyzer) - Speaker embeddings
