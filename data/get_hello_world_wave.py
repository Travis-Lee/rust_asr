#!/usr/bin/env python3
import os
import tempfile
from pydub import AudioSegment

# 1. Use eSpeak to generate temporary audio (default is WAV, but the sample rate might not meet requirements)
def generate_linux_tts(text, temp_wav_path):
    # Call eSpeak command-line tool to generate audio
    # Parameter description:
    # -v en: English voice
    # -s 150: speed (150 words per minute)
    # -w: output to file
    # --stdout: output to standard output, then redirect to file
    cmd = f'espeak -v en -s 150 "{text}" --stdout > "{temp_wav_path}"'
    os.system(cmd)
    # Check if the temporary file is generated
    if not os.path.exists(temp_wav_path):
        raise RuntimeError("eSpeak generation failed. Please make sure eSpeak is installed.")

# 2. Convert to 16kHz, 16-bit, mono WAV
text = "hello world"
output_wav = "hello_world_linux.wav"

# Create temporary file to store original TTS output
with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
    temp_wav_path = temp_file.name

# Generate original TTS audio
generate_linux_tts(text, temp_wav_path)

# Load temporary audio and convert format
audio = AudioSegment.from_file(temp_wav_path, format="wav")
# Set target format: 16kHz sample rate, 16-bit depth, mono
audio = audio.set_frame_rate(16000)       # Sample rate: 16000 Hz
audio = audio.set_sample_width(2)         # 16-bit depth (2 bytes)
audio = audio.set_channels(1)             # Mono channel
# Export to final WAV file
audio.export(output_wav, format="wav")

# Clean up temporary file
os.unlink(temp_wav_path)

print(f"Generated natural speech WAV file: {output_wav}")
print("Format check: 16kHz sample rate, 16-bit depth, mono (eSpeak voice)")

