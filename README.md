# Voice Cloning for Kokoro TTS

Voice cloning system using random walk optimization to generate custom voice style tensors for the Kokoro text-to-speech system.

Based on [kvoicewalk](https://github.com/RobViren/kvoicewalk) by Rob Viren.

## How It Works

This tool uses a **random walk algorithm** to explore Kokoro's voice tensor space, finding tensors that produce speech similar to a target voice without retraining the model.

### Scoring System

The optimization uses three metrics combined via weighted harmonic mean:

1. **Target Similarity (48%)** - How similar the generated audio sounds to the target voice (using Resemblyzer speaker embeddings)
2. **Self-Similarity (50%)** - How consistent the voice is across different text inputs (prevents quality degradation)
3. **Feature Similarity (2%)** - Audio feature comparison to prevent convergence on perceptually poor results

## Installation

### Requirements
- Python 3.10-3.12
- CUDA-capable GPU recommended (but CPU works)

### Setup

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Optional: Install faster-whisper for auto-transcription
pip install faster-whisper
```

### Download Kokoro Voice Tensors

You need the Kokoro voice tensors to use as a starting population:

```bash
# Option 1: Download from HuggingFace
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('hexgrad/Kokoro-82M', local_dir='./kokoro_model')"

# Copy voice tensors to voices folder
# Voice tensors are .pt files in the model
```

Or manually download voice `.pt` files and place them in the `voices/` folder.

## Usage

### Basic Voice Cloning

```bash
# Clone a voice from an audio sample
python main.py --target_audio ./in/target.wav --target_text "The text spoken in the audio"

# With interpolation search for better starting point
python main.py --target_audio ./in/target.wav --target_text "Hello world" --interpolate_start

# Auto-transcribe target audio (requires faster-whisper)
python main.py --target_audio ./in/target.wav --transcribe_start
```

### Test a Generated Voice

```bash
python main.py --test_voice ./out/cloned_voice_final.pt --target_text "Test this voice"
```

### Audio Preparation

Target audio should be:
- **Format**: WAV (mono, 24kHz)
- **Duration**: 20-30 seconds recommended
- **Quality**: Clear speech, minimal background noise

```bash
# Convert audio to correct format (using ffmpeg)
ffmpeg -i input.mp3 -ar 24000 -ac 1 output.wav

# Or use built-in conversion
python main.py --convert_audio input.mp3
```

## Command Line Options

### Operation Modes
- `--target_audio PATH` - Target audio file to clone
- `--target_text TEXT` - Text content of target audio (or path to .txt file)
- `--test_voice PATH` - Test a voice tensor

### Random Walk Options
- `--interpolate_start` - Use interpolation search for better starting voices
- `--step_limit N` - Maximum iterations (default: 10000)
- `--population_limit N` - Initial voice candidates (default: 10)
- `--starting_voice PATH` - Specific voice tensor to start from
- `--voice_folder PATH` - Folder with voice tensors (default: ./voices)

### Output Options
- `--output_name NAME` - Base name for output files

### Utility Options
- `--transcribe_start` - Auto-transcribe target audio
- `--convert_audio PATH` - Convert audio to 24kHz mono WAV
- `--export_bin PATH` - Export voices to compressed binary

## Project Structure

```
tts/
├── main.py                 # CLI interface
├── requirements.txt        # Dependencies
├── utilities/
│   ├── path_router.py      # Directory configuration
│   ├── speech_generator.py # Kokoro TTS wrapper
│   ├── fitness_scorer.py   # Hybrid similarity scoring
│   ├── voice_generator.py  # Tensor mutation
│   ├── initial_selector.py # Voice selection & interpolation
│   ├── audio_processor.py  # Audio conversion & transcription
│   └── kvoicewalk.py       # Random walk algorithm
├── voices/                 # Voice tensor files (.pt)
├── in/                     # Input audio files
├── out/                    # Output files
├── interpolated/           # Interpolated voice tensors
└── texts/                  # Transcription files
```

## Expected Results

- **Baseline**: ~70% similarity with best stock voice
- **After optimization**: ~90% similarity
- **Typical improvements**: 20+ percentage points

Results occupy an "uncanny valley" of similarity rather than producing exact clones, due to limitations in Kokoro's architecture.

## Performance Tips

1. **Use interpolation start** (`--interpolate_start`) for better initial positions
2. **Longer target audio** (20-30 seconds) gives better results
3. **Clear speech** without music or background noise
4. **Run multiple sessions** - results have high variance
5. **GPU acceleration** significantly speeds up the process

## Troubleshooting

### "No voice files found"
Download Kokoro voice tensors and place `.pt` files in the `voices/` folder.

### "CUDA out of memory"
- Reduce batch sizes or use CPU
- Close other GPU applications

### Poor similarity scores
- Ensure target audio is clear and properly formatted
- Try different starting voices with `--starting_voice`
- Use `--interpolate_start` for better initial positions

## License

MIT License - See original [kvoicewalk](https://github.com/RobViren/kvoicewalk) repository.

## Credits

- Original implementation: [Rob Viren](https://github.com/RobViren/kvoicewalk)
- Kokoro TTS: [hexgrad](https://huggingface.co/hexgrad/Kokoro-82M)
- Resemblyzer: [resemble-ai](https://github.com/resemble-ai/Resemblyzer)
