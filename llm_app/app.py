"""LLM AI Integration Dashboard for Home Assistant.

Provides a web dashboard and API for:
- AI Provider management (MiniMax, etc.)
- Voice cloning and generation (OmniVoice)
- TTS synthesis
"""

import argparse
import base64
import io
import json
import logging
import os
import tempfile
from pathlib import Path

import torch
import torchaudio
from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)

from omnivoice import OmniVoice

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

# Global state
config = {
    "providers": {},
    "voices": {},
}
omnivoice_model = None


# ============================================================================
# Initialization
# ============================================================================

def load_config():
    """Load configuration from options.json."""
    global config
    options_file = Path("/data/options.json")
    if options_file.exists():
        with open(options_file) as f:
            config.update(json.load(f))
    logger.info(f"Loaded config: {config}")


def load_omnivoice():
    """Load OmniVoice model."""
    global omnivoice_model
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device != "cpu" else torch.float32

    logger.info(f"Loading OmniVoice on {device}")
    try:
        omnivoice_model = OmniVoice.from_pretrained(
            "k2-fsa/OmniVoice",
            device_map=device,
            dtype=dtype,
        )
        logger.info("OmniVoice model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load OmniVoice: {e}")
        raise


# ============================================================================
# Web UI Routes
# ============================================================================

@app.route("/")
def index():
    """Main dashboard."""
    return render_template("index.html")


@app.route("/providers")
def providers_page():
    """AI Providers management page."""
    return render_template("providers.html")


@app.route("/voices")
def voices_page():
    """Voice management page."""
    return render_template("voices.html")


@app.route("/tts")
def tts_page():
    """TTS generation page."""
    return render_template("tts.html")


# ============================================================================
# API: Providers
# ============================================================================

@app.route("/api/providers", methods=["GET"])
def list_providers():
    """List configured AI providers."""
    return jsonify({"providers": config.get("providers", {})})


@app.route("/api/providers/<name>", methods=["GET"])
def get_provider(name):
    """Get provider configuration."""
    providers = config.get("providers", {})
    if name not in providers:
        return jsonify({"error": f"Provider '{name}' not found"}), 404
    return jsonify({"provider": providers[name]})


@app.route("/api/providers", methods=["POST"])
def add_provider():
    """Add or update an AI provider."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    name = data.get("name")
    if not name:
        return jsonify({"error": "Provider name is required"}), 400

    # Validate required fields
    provider_type = data.get("type")
    if provider_type == "minimax":
        if not data.get("api_key"):
            return jsonify({"error": "API key is required for MiniMax"}), 400

    if "providers" not in config:
        config["providers"] = {}
    config["providers"][name] = data
    save_config()

    return jsonify({"success": True, "provider": data})


@app.route("/api/providers/<name>", methods=["DELETE"])
def delete_provider(name):
    """Delete an AI provider."""
    if name not in config.get("providers", {}):
        return jsonify({"error": f"Provider '{name}' not found"}), 404

    del config["providers"][name]
    save_config()
    return jsonify({"success": True})


@app.route("/api/providers/<name>/test", methods=["POST"])
def test_provider(name):
    """Test an AI provider connection."""
    providers = config.get("providers", {})
    if name not in providers:
        return jsonify({"error": f"Provider '{name}' not found"}), 404

    provider = providers[name]

    if provider.get("type") == "minimax":
        return test_minimax_provider(provider)

    return jsonify({"error": f"Unknown provider type: {provider.get('type')}"}), 400


def test_minimax_provider(provider):
    """Test MiniMax provider connection."""
    import requests

    api_key = provider.get("api_key")
    url = "https://api.minimax.io/anthropic/v1/messages"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }

    data = {
        "model": provider.get("chat_model", "MiniMax-M2.7"),
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "Hi"}],
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        if response.status_code == 200:
            return jsonify({"success": True, "message": "Connection successful"})
        else:
            return jsonify({
                "success": False,
                "error": f"API error: {response.status_code}",
                "details": response.text[:200]
            }), 400
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


# ============================================================================
# API: Voices
# ============================================================================

@app.route("/api/voices", methods=["GET"])
def list_voices():
    """List all available voices."""
    voices_dir = Path("/data/voices")
    voices = []

    if voices_dir.exists():
        for f in sorted(voices_dir.glob("*.wav")):
            voice_info = {
                "name": f.stem,
                "file": f.name,
                "size": f.stat().st_size,
            }
            # Check for transcript
            transcript_file = f.with_suffix(".txt")
            if transcript_file.exists():
                voice_info["transcript"] = transcript_file.read_text().strip()
            voices.append(voice_info)

    return jsonify({"voices": voices})


@app.route("/api/voices", methods=["POST"])
def upload_voice():
    """Upload a voice reference audio."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    name = request.form.get("name", file.filename.rsplit(".", 1)[0])
    transcript = request.form.get("transcript", "")

    voices_dir = Path("/data/voices")
    voices_dir.mkdir(parents=True, exist_ok=True)

    # Save audio file
    audio_path = voices_dir / f"{name}.wav"
    file.save(audio_path)

    # Save transcript if provided
    if transcript:
        transcript_path = voices_dir / f"{name}.txt"
        transcript_path.write_text(transcript)

    return jsonify({"success": True, "voice": {
        "name": name,
        "file": audio_path.name,
    }})


@app.route("/api/voices/<name>", methods=["DELETE"])
def delete_voice(name):
    """Delete a voice."""
    voices_dir = Path("/data/voices")
    audio_path = voices_dir / f"{name}.wav"
    transcript_path = voices_dir / f"{name}.txt"

    deleted = []
    if audio_path.exists():
        audio_path.unlink()
        deleted.append(audio_path.name)
    if transcript_path.exists():
        transcript_path.unlink()
        deleted.append(transcript_path.name)

    if not deleted:
        return jsonify({"error": f"Voice '{name}' not found"}), 404

    return jsonify({"success": True, "deleted": deleted})


@app.route("/api/voices/<name>/download")
def download_voice(name):
    """Download a voice audio file."""
    voices_dir = Path("/data/voices")
    audio_path = voices_dir / f"{name}.wav"

    if not audio_path.exists():
        return jsonify({"error": f"Voice '{name}' not found"}), 404

    return send_file(audio_path, mimetype="audio/wav")


# ============================================================================
# API: TTS Generation
# ============================================================================

@app.route("/api/tts", methods=["POST"])
def generate_tts():
    """Generate TTS audio.

    Request JSON:
    - text: str (required) - Text to synthesize
    - voice: str (optional) - Voice preset name from /data/voices
    - ref_audio: str (optional) - Base64 encoded reference audio for cloning
    - ref_text: str (optional) - Transcript of reference audio
    - instruct: str (optional) - Voice design attributes
    - speed: float (optional) - Speaking rate
    - num_steps: int (optional) - Diffusion steps
    - provider: str (optional) - TTS provider (omnivoice, minimax)
    """
    if omnivoice_model is None:
        return jsonify({"error": "OmniVoice model not loaded"}), 500

    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' in request"}), 400

    text = data["text"]
    provider = data.get("provider", "omnivoice")

    if provider == "minimax":
        return generate_minimax_tts(data)

    # OmniVoice TTS
    return generate_omnivoice_tts(data)


def generate_omnivoice_tts(data):
    """Generate TTS using OmniVoice."""
    global omnivoice_model

    text = data["text"]
    voice_name = data.get("voice")
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

    # Voice cloning from preset
    if voice_name:
        voices_dir = Path("/data/voices")
        voice_path = voices_dir / f"{voice_name}.wav"
        if voice_path.exists():
            generate_kwargs["ref_audio"] = str(voice_path)
            transcript_path = voice_path.with_suffix(".txt")
            if transcript_path.exists():
                generate_kwargs["ref_text"] = transcript_path.read_text().strip()
        else:
            return jsonify({"error": f"Voice '{voice_name}' not found"}), 404
    elif ref_audio_b64:
        # Voice cloning from base64 audio
        audio_bytes = base64.b64decode(ref_audio_b64)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            generate_kwargs["ref_audio"] = f.name
            if ref_text:
                generate_kwargs["ref_text"] = ref_text
    elif instruct:
        # Voice design
        generate_kwargs["instruct"] = instruct

    try:
        audio = omnivoice_model.generate(**generate_kwargs)

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
        if ref_audio_b64 and "ref_audio" in generate_kwargs:
            os.unlink(generate_kwargs["ref_audio"])


def generate_minimax_tts(data):
    """Generate TTS using MiniMax API."""
    import requests

    # Find minimax provider
    providers = config.get("providers", {})
    minimax_provider = None
    for name, p in providers.items():
        if p.get("type") == "minimax":
            minimax_provider = p
            break

    if not minimax_provider:
        return jsonify({"error": "No MiniMax provider configured"}), 400

    text = data["text"]
    voice_id = data.get("voice_id", "English_PlayfulGirl")
    speed = data.get("speed", 1.0)

    url = "https://api.minimax.io/v1/t2a_v2"
    headers = {
        "Authorization": f"Bearer {minimax_provider.get('api_key')}",
    }

    payload = {
        "model": "speech-2.8-hd",
        "text": text,
        "voice_id": voice_id,
        "speed": speed,
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        if response.status_code == 200:
            result = response.json()
            audio_b64 = result.get("data", {}).get("audio")
            if audio_b64:
                audio_bytes = base64.b64decode(audio_b64)
                buffer = io.BytesIO(audio_bytes)
                return send_file(
                    buffer,
                    mimetype="audio/wav",
                    as_attachment=False,
                    download_name="output.wav"
                )
        return jsonify({
            "error": f"TTS failed: {response.status_code}",
            "details": response.text[:200]
        }), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================================
# API: Conversation
# ============================================================================

@app.route("/api/conversation", methods=["POST"])
def conversation():
    """Send a conversation message to an AI provider.

    Request JSON:
    - message: str (required) - User message
    - provider: str (optional) - Provider name (default: first configured)
    - system_prompt: str (optional) - Override system prompt
    """
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "Missing 'message' in request"}), 400

    message = data["message"]
    provider_name = data.get("provider")
    system_prompt = data.get("system_prompt")

    # Find provider
    providers = config.get("providers", {})
    if provider_name:
        if provider_name not in providers:
            return jsonify({"error": f"Provider '{provider_name}' not found"}), 404
        provider = providers[provider_name]
    else:
        # Use first available
        if not providers:
            return jsonify({"error": "No AI provider configured"}), 400
        provider = list(providers.values())[0]
        provider_name = list(providers.keys())[0]

    if provider.get("type") == "minimax":
        return conversation_minimax(message, provider, system_prompt)

    return jsonify({"error": f"Unknown provider type: {provider.get('type')}"}), 400


def conversation_minimax(message, provider, system_prompt=None):
    """Send conversation to MiniMax."""
    import requests

    url = "https://api.minimax.io/anthropic/v1/messages"
    headers = {
        "Authorization": f"Bearer {provider.get('api_key')}",
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }

    messages = []
    if system_prompt:
        messages.append({"role": "assistant", "content": system_prompt})
    messages.append({"role": "user", "content": message})

    payload = {
        "model": provider.get("chat_model", "MiniMax-M2.7"),
        "max_tokens": provider.get("max_tokens", 16000),
        "messages": messages,
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        if response.status_code == 200:
            result = response.json()
            return jsonify({
                "response": result.get("content", [{}])[0].get("text", ""),
                "provider": "minimax",
                "model": provider.get("chat_model"),
            })
        else:
            return jsonify({
                "error": f"API error: {response.status_code}",
                "details": response.text[:200]
            }), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================================
# Utilities
# ============================================================================

def save_config():
    """Save configuration to options.json."""
    options_file = Path("/data/options.json")
    options_file.parent.mkdir(parents=True, exist_ok=True)
    with open(options_file, "w") as f:
        json.dump(config, f, indent=2)
    logger.info("Config saved")


@app.route("/api/config", methods=["GET"])
def get_config():
    """Get current configuration."""
    return jsonify(config)


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "omnivoice_loaded": omnivoice_model is not None,
        "providers_count": len(config.get("providers", {})),
    })


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM AI Dashboard")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    load_config()
    load_omnivoice()

    app.run(host=args.host, port=args.port, debug=args.debug)
