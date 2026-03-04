import os
import time
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import Optional
import torchaudio
import soundfile as sf
from pydub import AudioSegment, silence
import re
import sys
import logging
import io
import magic

logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Model paths - configurable via environment variables
MODEL_PATH = os.environ.get("VIBEVOICE_MODEL_PATH", "/workspace/models/vibevoice/VibeVoice-Large")
TOKENIZER_PATH = os.environ.get("VIBEVOICE_TOKENIZER_PATH", "/workspace/models/vibevoice/tokenizer")

# Import VibeVoice - will be available after installing vibevoice package
from transformers import AutoTokenizer, AutoModel
import numpy as np

# Global model variables - will be loaded lazily or at startup
tokenizer = None
processor = None
model = None
model_loaded = False

# Voice embedding cache - stores processed voice data for repeated use
# Key: voice label, Value: dict with processed audio data and file mtime
voice_cache = {}
VOICE_CACHE_MAX_SIZE = 50  # Maximum number of voices to cache

def load_models():
    """Load models - can be called at startup or lazily on first request."""
    global tokenizer, processor, model, model_loaded
    
    if model_loaded:
        return True
    
    logging.info(f"Loading VibeVoice model from {MODEL_PATH}")
    logging.info(f"Loading tokenizer from {TOKENIZER_PATH}")
    
    try:
        from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
        from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
        
        # Check if paths are local directories
        tokenizer_is_local = os.path.isdir(TOKENIZER_PATH)
        model_is_local = os.path.isdir(MODEL_PATH)
        
        # Load tokenizer - use local_files_only if path is a local directory
        tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER_PATH, 
            trust_remote_code=True,
            local_files_only=tokenizer_is_local
        )
        
        # Load VibeVoice processor and model
        processor = VibeVoiceProcessor.from_pretrained(MODEL_PATH)
        model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map='auto',
            attn_implementation='sdpa'
        )
        model.eval()
        model_loaded = True
        logging.info(f"VibeVoice model loaded on {device}")
        return True
    except Exception as e:
        logging.error(f"Failed to load models: {e}")
        return False

# Try to load models at startup, but don't fail if they're not available
LAZY_LOAD = os.environ.get("LAZY_LOAD_MODELS", "false").lower() == "true"
if not LAZY_LOAD:
    try:
        load_models()
    except Exception as e:
        logging.warning(f"Models not loaded at startup: {e}. Will try lazy loading on first request.")

output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)

resources_dir = 'resources'
os.makedirs(resources_dir, exist_ok=True)

# Default voice settings
DEFAULT_VOICE = "default_en"
SAMPLE_RATE = 24000

# Copy default voice files if they exist in the model directory
voices_dir = os.path.join(os.path.dirname(MODEL_PATH), "voices")
if os.path.exists(voices_dir):
    import shutil
    for voice_file in os.listdir(voices_dir):
        if voice_file.endswith('.wav'):
            src = os.path.join(voices_dir, voice_file)
            dst = os.path.join(resources_dir, voice_file)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
                logging.info(f"Copied default voice: {voice_file}")


def convert_to_wav(input_path, output_path):
    """Convert any audio format to WAV using pydub."""
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1)  # Convert to mono
    audio = audio.set_frame_rate(SAMPLE_RATE)  # Set to expected sample rate
    audio.export(output_path, format='wav')


def detect_leading_silence(audio, silence_threshold=-42, chunk_size=10):
    """Detect silence at the beginning of the audio."""
    trim_ms = 0
    while audio[trim_ms:trim_ms + chunk_size].dBFS < silence_threshold and trim_ms < len(audio):
        trim_ms += chunk_size
    return trim_ms


def remove_silence_edges(audio, silence_threshold=-42):
    """Remove silence from the beginning and end of the audio."""
    start_trim = detect_leading_silence(audio, silence_threshold)
    end_trim = detect_leading_silence(audio.reverse(), silence_threshold)
    duration = len(audio)
    return audio[start_trim:duration - end_trim]


def get_file_mtime(filepath: str) -> float:
    """Get file modification time for cache invalidation."""
    try:
        return os.path.getmtime(filepath)
    except OSError:
        return 0.0


def process_reference_audio(reference_file: str) -> str:
    """Process reference audio: clip to ~15 seconds, remove silence edges."""
    temp_short_ref = f'{output_dir}/temp_short_ref.wav'
    aseg = AudioSegment.from_file(reference_file)

    # 1. try to find long silence for clipping
    non_silent_segs = silence.split_on_silence(
        aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=1000, seek_step=10
    )
    non_silent_wave = AudioSegment.silent(duration=0)
    for non_silent_seg in non_silent_segs:
        if len(non_silent_wave) > 6000 and len(non_silent_wave + non_silent_seg) > 15000:
            logging.info("Audio is over 15s, clipping short. (1)")
            break
        non_silent_wave += non_silent_seg

    # 2. try to find short silence for clipping if 1. failed
    if len(non_silent_wave) > 15000:
        non_silent_segs = silence.split_on_silence(
            aseg, min_silence_len=100, silence_thresh=-40, keep_silence=1000, seek_step=10
        )
        non_silent_wave = AudioSegment.silent(duration=0)
        for non_silent_seg in non_silent_segs:
            if len(non_silent_wave) > 6000 and len(non_silent_wave + non_silent_seg) > 15000:
                logging.info("Audio is over 15s, clipping short. (2)")
                break
            non_silent_wave += non_silent_seg

    aseg = non_silent_wave

    # 3. if no proper silence found for clipping
    if len(aseg) > 15000:
        aseg = aseg[:15000]
        logging.info("Audio is over 15s, clipping short. (3)")

    aseg = remove_silence_edges(aseg) + AudioSegment.silent(duration=50)
    aseg.export(temp_short_ref, format='wav')
    
    return temp_short_ref


def load_audio_for_cloning(audio_path: str) -> torch.Tensor:
    """Load and preprocess audio for voice cloning."""
    waveform, sr = torchaudio.load(audio_path)
    
    # Resample if needed
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        waveform = resampler(waveform)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    return waveform


def generate_speech(
    text: str,
    voice_audio_path: Optional[str] = None,
    voice_label: Optional[str] = None,
    speed: float = 1.0,
    diffusion_steps: int = 20,
    cfg_scale: float = 1.3,
    seed: int = 42,
) -> str:
    """Generate speech using VibeVoice model."""
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Map speed to voice_speed_factor (clamp to 0.8-1.2 range)
    voice_speed_factor = max(0.8, min(1.2, speed))
    if speed != voice_speed_factor:
        logging.warning(f"Speed {speed} clamped to {voice_speed_factor} (valid range: 0.8-1.2)")
    
    # Format text with speaker label for VibeVoice
    # Replace newlines with spaces to keep all text in a single speaker turn
    # The processor expects "Speaker X: text" format, one turn per line
    clean_text = text.replace('\n', ' ').replace('\r', ' ').strip()
    formatted_text = f"Speaker 1: {clean_text}"
    
    # Prepare inputs using VibeVoice processor
    if voice_audio_path and os.path.exists(voice_audio_path):
        # Check if we can use cached voice embeddings
        if voice_label and voice_label in voice_cache:
            cached = voice_cache[voice_label]
            file_mtime = get_file_mtime(voice_audio_path)
            
            if cached.get('mtime') == file_mtime and cached.get('path') == voice_audio_path:
                # Cache hit - use cached voice data
                logging.info(f"Using cached voice embedding for '{voice_label}'")
                cached_voice = cached['voice_inputs']
                
                # Process with voice to get correct input_ids structure
                # but we'll replace the speech tensors with cached versions
                inputs = processor(
                    text=[formatted_text],
                    voice_samples=[[voice_audio_path]],
                    padding=True,
                    return_tensors="pt",
                    return_attention_mask=True,
                )
                
                # Replace speech tensors with cached versions
                if cached_voice['speech_tensors'] is not None:
                    inputs['speech_tensors'] = cached_voice['speech_tensors'].clone()
                    inputs['speech_masks'] = cached_voice['speech_masks'].clone()
            else:
                # Cache miss or stale - process and cache
                logging.info(f"Processing and caching voice embedding for '{voice_label}'")
                inputs = processor(
                    text=[formatted_text],
                    voice_samples=[[voice_audio_path]],
                    padding=True,
                    return_tensors="pt",
                    return_attention_mask=True,
                )
                
                # Cache the voice data
                cached_voice_data = {
                    'speech_tensors': inputs.get('speech_tensors').clone() if inputs.get('speech_tensors') is not None else None,
                    'speech_masks': inputs.get('speech_masks').clone() if inputs.get('speech_masks') is not None else None,
                }
                
                # Manage cache size
                if len(voice_cache) >= VOICE_CACHE_MAX_SIZE:
                    oldest_key = next(iter(voice_cache))
                    del voice_cache[oldest_key]
                    logging.info(f"Voice cache full, removed oldest entry: '{oldest_key}'")
                
                voice_cache[voice_label] = {
                    'voice_inputs': cached_voice_data,
                    'mtime': file_mtime,
                    'path': voice_audio_path,
                }
        else:
            # No voice label or not in cache - process and optionally cache
            inputs = processor(
                text=[formatted_text],
                voice_samples=[[voice_audio_path]],
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )
            
            # Cache if we have a voice label
            if voice_label:
                logging.info(f"Processing and caching voice embedding for '{voice_label}'")
                file_mtime = get_file_mtime(voice_audio_path)
                cached_voice_data = {
                    'speech_tensors': inputs.get('speech_tensors').clone() if inputs.get('speech_tensors') is not None else None,
                    'speech_masks': inputs.get('speech_masks').clone() if inputs.get('speech_masks') is not None else None,
                }
                
                if len(voice_cache) >= VOICE_CACHE_MAX_SIZE:
                    oldest_key = next(iter(voice_cache))
                    del voice_cache[oldest_key]
                    logging.info(f"Voice cache full, removed oldest entry: '{oldest_key}'")
                
                voice_cache[voice_label] = {
                    'voice_inputs': cached_voice_data,
                    'mtime': file_mtime,
                    'path': voice_audio_path,
                }
    else:
        # No voice cloning - generate without voice samples
        inputs = processor(
            text=[formatted_text],
            voice_samples=[[]],
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
    
    # Move inputs to device
    inputs = inputs.to(device)
    
    # Set diffusion steps
    model.set_ddpm_inference_steps(num_steps=diffusion_steps)
    
    # Generate speech
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=None,
            cfg_scale=cfg_scale,
            tokenizer=processor.tokenizer,
            generation_config={'do_sample': False},
            verbose=False,
        )
    generation_time = time.time() - start_time
    
    # Get audio output from speech_outputs
    if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
        audio_output = outputs.speech_outputs[0]
    else:
        raise ValueError("No audio output generated")
    
    # Save audio using processor
    save_path = f'{output_dir}/output_synthesized.wav'
    processor.save_audio(audio_output, output_path=save_path)
    
    # Calculate audio duration and log generation stats
    audio_samples = audio_output.shape[-1] if len(audio_output.shape) > 0 else len(audio_output)
    audio_duration = audio_samples / SAMPLE_RATE
    logging.info(f"Generation completed in {generation_time:.2f}s (audio duration: {audio_duration:.2f}s, RTF: {generation_time/audio_duration:.2f}x)")
    
    # Apply speed adjustment if needed (post-processing)
    if voice_speed_factor != 1.0:
        audio_seg = AudioSegment.from_wav(save_path)
        # Speed up/slow down by changing frame rate then converting back
        adjusted = audio_seg._spawn(audio_seg.raw_data, overrides={
            "frame_rate": int(audio_seg.frame_rate * voice_speed_factor)
        }).set_frame_rate(SAMPLE_RATE)
        adjusted.export(save_path, format='wav')
    
    return save_path


@app.on_event("startup")
async def startup_event():
    """Warmup inference on startup."""
    try:
        test_text = "This is a test sentence generated by the VibeVoice API."
        voice = "demo_speaker0"
        await synthesize_speech(text=test_text, voice=voice)
        logging.info("Startup warmup complete")
    except Exception as e:
        logging.warning(f"Startup warmup failed (this may be normal if no demo voice exists): {e}")


@app.get("/base_tts/")
async def base_tts(text: str, speed: Optional[float] = 1.0):
    """
    Perform text-to-speech conversion using only the base speaker.
    """
    try:
        return await synthesize_speech(text=text, voice=DEFAULT_VOICE, speed=speed, diffusion_steps=20, cfg_scale=1.3, seed=42)
    except Exception as e:
        logging.error(f"Error in base_tts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/change_voice/")
async def change_voice(reference_speaker: str = Form(...), file: UploadFile = File(...)):
    """
    Change the voice of an existing audio file.
    """
    try:
        logging.info(f'changing voice to {reference_speaker}...')

        contents = await file.read()
        
        # Save the input audio temporarily
        input_path = f'{output_dir}/input_audio.wav'
        with open(input_path, 'wb') as f:
            f.write(contents)

        # Find the reference audio file
        matching_files = [f for f in os.listdir("resources") if f.startswith(str(reference_speaker))]
        if not matching_files:
            raise HTTPException(status_code=400, detail="No matching reference speaker found.")
        
        reference_file = f'resources/{matching_files[0]}'
        
        # Convert reference file to WAV if it's not already
        if not reference_file.lower().endswith('.wav'):
            ref_wav_path = f'{output_dir}/ref_converted.wav'
            convert_to_wav(reference_file, ref_wav_path)
            reference_file = ref_wav_path
        
        # For voice conversion, we need to transcribe the input audio first
        # Use Whisper for transcription
        try:
            import whisper
            whisper_model = whisper.load_model("base")
            result = whisper_model.transcribe(input_path)
            text = result["text"]
            logging.info(f"Transcribed text: {text}")
        except ImportError:
            # Fallback: if whisper not available, raise error
            raise HTTPException(
                status_code=500, 
                detail="Whisper not installed. Voice conversion requires ASR. Install with: pip install openai-whisper"
            )
        
        # Process reference audio
        processed_ref = process_reference_audio(reference_file)
        
        # Generate speech with the new voice
        save_path = generate_speech(
            text=text,
            voice_audio_path=processed_ref,
            voice_label=reference_speaker,
            speed=1.0,
        )

        result = StreamingResponse(open(save_path, 'rb'), media_type="audio/wav")
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload_audio/")
async def upload_audio(audio_file_label: str = Form(...), file: UploadFile = File(...)):
    """
    Upload an audio file for later use as the reference audio.
    """
    try:
        contents = await file.read()

        allowed_extensions = {'wav', 'mp3', 'flac', 'ogg'}
        max_file_size = 5 * 1024 * 1024  # 5MB

        if not file.filename.split('.')[-1] in allowed_extensions:
            return {"error": "Invalid file type. Allowed types are: wav, mp3, flac, ogg"}

        if len(contents) > max_file_size:
            return {"error": "File size is over limit. Max size is 5MB."}

        temp_file = io.BytesIO(contents)
        file_format = magic.from_buffer(temp_file.read(), mime=True)

        if 'audio' not in file_format:
            return {"error": "Invalid file content."}

        file_extension = file.filename.split('.')[-1]
        stored_file_name = f"{audio_file_label}.{file_extension}"

        with open(f"resources/{stored_file_name}", "wb") as f:
            f.write(contents)

        # Also create a WAV version for VibeVoice
        wav_path = f"resources/{audio_file_label}.wav"
        convert_to_wav(f"resources/{stored_file_name}", wav_path)

        return {"message": f"File {file.filename} uploaded successfully with label {audio_file_label}."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/synthesize_speech/")
async def synthesize_speech(
        text: str = Form(...),
        voice: str = Form(...),
        speed: Optional[float] = Form(1.0),
        diffusion_steps: Optional[int] = Form(20),
        cfg_scale: Optional[float] = Form(1.3),
        seed: Optional[int] = Form(42),
):
    """
    Synthesize speech from text using a specified voice and style.
    """
    start_time = time.time()
    try:
        logging.info(f'Generating speech for {voice}')

        # First try to find a WAV version
        matching_files = [f for f in os.listdir("resources") if f.startswith(voice) and f.lower().endswith('.wav')]
        
        # If no WAV found, try other formats and convert
        if not matching_files:
            matching_files = [f for f in os.listdir("resources") if f.startswith(voice)]
            if not matching_files:
                # No voice file found - use default/no voice cloning
                logging.warning(f"No matching voice found for '{voice}', using default voice")
                reference_file = None
            else:
                # Convert to WAV
                input_file = f'resources/{matching_files[0]}'
                wav_path = f'{output_dir}/ref_converted.wav'
                convert_to_wav(input_file, wav_path)
                reference_file = wav_path
        else:
            reference_file = f'resources/{matching_files[0]}'

        # Process reference audio if we have one
        if reference_file:
            processed_ref = process_reference_audio(reference_file)
        else:
            processed_ref = None
        
        # Generate speech
        save_path = generate_speech(
            text=text,
            voice_audio_path=processed_ref,
            voice_label=voice,
            speed=speed,
            diffusion_steps=diffusion_steps,
            cfg_scale=cfg_scale,
            seed=seed,
        )

        result = StreamingResponse(open(save_path, 'rb'), media_type="audio/wav")

        end_time = time.time()
        elapsed_time = end_time - start_time

        result.headers["X-Elapsed-Time"] = str(elapsed_time)
        result.headers["X-Device-Used"] = device

        # Add CORS headers
        result.headers["Access-Control-Allow-Origin"] = "*"
        result.headers["Access-Control-Allow-Credentials"] = "true"
        result.headers["Access-Control-Allow-Headers"] = "Origin, Content-Type, X-Amz-Date, Authorization, X-Api-Key, X-Amz-Security-Token, locale"
        result.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
