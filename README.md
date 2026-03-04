# VibeVoice Server

A FastAPI-based TTS server using Microsoft's VibeVoice model for high-quality text-to-speech synthesis with voice cloning capabilities.

## Features

- High-quality text-to-speech synthesis
- Voice cloning from reference audio
- Support for multiple voice presets
- Voice conversion (change voice of existing audio)
- Docker support with RunPod compatibility
- RTX 5090 (Blackwell) GPU support

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/base_tts/` | TTS with default voice |
| `POST` | `/synthesize_speech/` | TTS with custom voice |
| `POST` | `/upload_audio/` | Upload reference audio for voice cloning |
| `POST` | `/change_voice/` | Voice conversion on existing audio |

### Endpoint Details

#### GET /base_tts/
```
GET /base_tts/?text=Hello%20world&speed=1.0
```
- `text` (required): Text to synthesize
- `speed` (optional, default=1.0): Speech speed (0.8-1.2)

#### POST /synthesize_speech/
```bash
curl -X POST http://localhost:7860/synthesize_speech/ \
  -F "text=Hello world" \
  -F "voice=my_voice" \
  -F "speed=1.0" \
  --output output.wav
```
- `text` (required): Text to synthesize
- `voice` (required): Voice label (must match uploaded audio)
- `speed` (optional, default=1.0): Speech speed (0.8-1.2)
- `diffusion_steps` (optional, default=20): Number of diffusion steps (higher = better quality, slower)
- `cfg_scale` (optional, default=1.3): Classifier-free guidance scale
- `seed` (optional, default=42): Random seed for reproducibility

#### POST /upload_audio/
Upload a reference audio file for voice cloning.
- `audio_file_label` (form): Label for the voice
- `file` (file): Audio file (wav, mp3, flac, ogg, max 5MB)

#### POST /change_voice/
Convert the voice of an existing audio file.
- `reference_speaker` (form): Voice label to use
- `file` (file): Audio file to convert

## Installation

### Option 1: Docker (Recommended)

Run the pre-built image:
```bash
docker run -p 7860:7860 \
  -v /path/to/models:/workspace/models/vibevoice \
  --gpus all \
  ghcr.io/ben73/vibevoice_server:latest
```

Models are automatically downloaded on first start. To persist models across container restarts, mount a volume to `/workspace/models/vibevoice`.

#### Building from source (optional)

```bash
docker build -t vibevoice_server .
```

### Option 2: Local Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Clone and install VibeVoice:
```bash
git clone https://github.com/vibevoice-community/VibeVoice.git
cd VibeVoice
pip install -e .
```

3. Download models:
```bash
./install_models.sh
```

4. Set environment variables:
```bash
export VIBEVOICE_MODEL_PATH=/path/to/VibeVoice-Large
export VIBEVOICE_TOKENIZER_PATH=/path/to/tokenizer
```

5. Run the server:
```bash
./start.sh
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VIBEVOICE_MODEL_PATH` | `/workspace/models/vibevoice/VibeVoice-Large` | Path to VibeVoice model |
| `VIBEVOICE_TOKENIZER_PATH` | `/workspace/models/vibevoice/tokenizer` | Path to Qwen tokenizer |

## Model Requirements

- **VibeVoice-Large**: ~18.7GB, requires ~20GB VRAM
- **Tokenizer**: Qwen2.5-1.5B tokenizer

For lower VRAM, consider using quantized models:
- `VibeVoice-Large-Q8`: ~12GB VRAM
- `VibeVoice-Large-Q4`: ~8GB VRAM

## Voice Cloning Tips

- Use clear audio with minimal background noise
- Recommended: 10-30 seconds of speech
- Audio is automatically resampled to 24kHz

## Notes

- **Speed parameter**: Clamped to 0.8-1.2 range
- **Voice cloning**: Uses audio prefill for natural voice reproduction
- **Voice conversion**: Uses Whisper for transcription (installed by default)

## License

MIT License (same as VibeVoice)
