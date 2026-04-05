"""OmniVoice TTS API server for Home Assistant."""

import argparse
import io
import logging
import os
import tempfile
from pathlib import Path

import torch
import torchaudio
from flask import Flask, jsonify, request, send_file

from omnivoice import OmniVoice

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
model = None


def load_model():
    """Load OmniVoice model."""
    global model
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device != "cpu" else torch.float32

    logger.info(f"Loading OmniVoice on {device} with {dtype}")
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice",
        device_map=device,
        dtype=dtype,
    )
    logger.info("Model loaded successfully")


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "model_loaded": model is not None})


@app.route("/tts", methods=["POST"])
def tts():
    """Generate TTS audio.

    Request JSON:
    - text: str (required) - Text to synthesize
    - ref_audio: str (optional) - Base64 encoded reference audio for cloning
    - ref_text: str (optional) - Transcript of reference audio
    - instruct: str (optional) - Voice design attributes (e.g., "female, british accent")
    - speed: float (optional) - Speaking rate (default 1.0)
    - num_steps: int (optional) - Diffusion steps (default 32)

    Returns: WAV audio file
    """
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' in request"}), 400

    text = data["text"]
    ref_audio_b64 = data.get("ref_audio")
    ref_text = data.get("ref_text")
    instruct = data.get("instruct")
    speed = data.get("speed", 1.0)
    num_steps = data.get("num_steps", 32)

    generate_kwargs = {
        "text": text,
        "num_step": num_steps,
        "speed": speed,
    }

    if ref_audio_b64:
        import base64
        import json as json_lib

        # Decode reference audio
        audio_bytes = base64.b64decode(ref_audio_b64)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            ref_audio_path = f.name

        generate_kwargs["ref_audio"] = ref_audio_path
        if ref_text:
            generate_kwargs["ref_text"] = ref_text
    elif instruct:
        generate_kwargs["instruct"] = instruct

    try:
        audio = model.generate(**generate_kwargs)

        # Save to buffer as WAV
        buffer = io.BytesIO()
        torchaudio.save(buffer, audio[0], 24000, format="wav")
        buffer.seek(0)

        return send_file(
            buffer,
            mimetype="audio/wav",
            as_attachment=False,
            download_name="output.wav"
        )
    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        if ref_audio_b64:
            os.unlink(ref_audio_path)


@app.route("/voices", methods=["GET"])
def list_voices():
    """List available voice presets from the reference audio directory."""
    voices_dir = Path("/data/voices")
    voices = []

    if voices_dir.exists():
        for f in voices_dir.glob("*.wav"):
            voices.append({
                "name": f.stem,
                "file": f.name,
                "has_transcript": (f.with_suffix(".txt")).exists()
            })

    return jsonify({"voices": voices})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    load_model()
    app.run(host=args.host, port=args.port)
