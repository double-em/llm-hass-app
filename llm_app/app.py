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
import uuid
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

from omnivoice_client import OmniVoiceClient
from voice_presets import VoicePresetManager, VOICE_DESIGN_ATTRIBUTES, validate_instruct, build_instruct_string
from person import (
    create_person,
    delete_person,
    enroll_sample,
    get_person,
    identify_speaker,
    identify_speaker_from_base64,
    list_persons,
    verify_match,
    verify_match_from_base64,
)

from person_store import PersonStore
from enrollment import EnrollmentManager
from voiceprint import VoiceprintManager

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
omnivoice_client = None
person_store = None
enrollment_manager = None
voiceprint_manager = None


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
    global omnivoice_model, omnivoice_client
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device != "cpu" else torch.float32

    logger.info(f"Loading OmniVoice on {device}")
    try:
        omnivoice_model = OmniVoice.from_pretrained(
            "k2-fsa/OmniVoice",
            device_map=device,
            dtype=dtype,
        )
        omnivoice_client = OmniVoiceClient(omnivoice_model)
        logger.info("OmniVoice model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load OmniVoice: {e}")
        raise


def load_person_system():
    """Initialize the person system components."""
    global person_store, enrollment_manager, voiceprint_manager
    person_store = PersonStore()
    enrollment_manager = EnrollmentManager()
    voiceprint_manager = VoiceprintManager()
    logger.info("Person system initialized")


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


@app.route("/persons")
def persons_page():
    """Person enrollment page."""
    return render_template("persons.html")


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
    - text: str or list (required) - Text(s) to synthesize
    - voice: str (optional) - Voice preset name from /data/voices
    - ref_audio: str (optional) - Base64 encoded reference audio for cloning
    - ref_text: str (optional) - Transcript of reference audio
    - instruct: str (optional) - Voice design attributes
    - preset: str (optional) - Use saved voice preset
    - speed: float (optional) - Speaking rate
    - num_steps: int (optional) - Diffusion steps
    - provider: str (optional) - TTS provider (omnivoice, minimax)

    Query params:
    - download: 'true' returns file download
    - embed: 'true' returns base64 encoded audio
    - mode: 'clone', 'design', or 'auto' (default: auto)

    Returns:
    - Audio file (default) or JSON with metadata and optional base64 audio.
    """
    if omnivoice_model is None:
        return jsonify({"error": "OmniVoice model not loaded"}), 500

    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' in request"}), 400

    # Handle batch TTS
    is_batch = isinstance(data["text"], list)
    texts = data["text"] if is_batch else [data["text"]]

    provider = data.get("provider", "omnivoice")

    if provider == "minimax":
        return generate_minimax_tts(data)

    # OmniVoice TTS
    return generate_omnivoice_tts(data, texts, is_batch)


def generate_omnivoice_tts(data, texts, is_batch=False):
    """Generate TTS using OmniVoice.

    Args:
        data: Request JSON data.
        texts: List of texts to synthesize.
        is_batch: Whether this is a batch request.

    Returns:
        JSON response with audio(s) and metadata, or audio file.
    """
    global omnivoice_model, omnivoice_client

    voice_name = data.get("voice")
    ref_audio_b64 = data.get("ref_audio")
    ref_text = data.get("ref_text")
    instruct = data.get("instruct")
    preset_name = data.get("preset")
    speed = data.get("speed", 1.0)
    num_steps = data.get("num_steps", 32)
    mode = data.get("mode", "auto")

    # Query params
    download = request.args.get("download", "false").lower() == "true"
    embed = request.args.get("embed", "false").lower() == "true"

    # Handle preset mode
    if preset_name:
        return _generate_with_preset(texts, preset_name, speed, num_steps, download, embed, is_batch)

    # Handle regular generation
    results = []
    temp_files = []

    try:
        for text in texts:
            generate_kwargs = {
                "num_steps": num_steps,
                "speed": speed,
            }

            # Determine mode and set appropriate args
            if voice_name:
                # Voice cloning from preset
                voices_dir = Path("/data/voices")
                voice_path = voices_dir / f"{voice_name}.wav"
                if voice_path.exists():
                    generate_kwargs["ref_audio"] = str(voice_path)
                    transcript_path = voice_path.with_suffix(".txt")
                    if transcript_path.exists():
                        generate_kwargs["ref_text"] = transcript_path.read_text().strip()
                else:
                    return jsonify({"error": f"Voice '{voice_name}' not found"}), 404
                generate_kwargs["mode"] = "clone"
            elif ref_audio_b64:
                # Voice cloning from base64 audio
                audio_bytes = base64.b64decode(ref_audio_b64)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    f.write(audio_bytes)
                    generate_kwargs["ref_audio"] = f.name
                    temp_files.append(f.name)
                    if ref_text:
                        generate_kwargs["ref_text"] = ref_text
                generate_kwargs["mode"] = "clone"
            elif instruct:
                # Voice design
                generate_kwargs["instruct"] = instruct
                generate_kwargs["mode"] = "design"
            else:
                generate_kwargs["mode"] = "auto"

            audio = omnivoice_model.generate(text=text, **generate_kwargs)
            metadata = omnivoice_client.get_audio_metadata(audio)
            audio_bytes = omnivoice_client.audio_to_bytes(audio)

            result = {
                "text": text,
                "metadata": metadata,
            }
            if embed:
                result["audio"] = base64.b64encode(audio_bytes).decode("utf-8")
            else:
                result["audio"] = audio_bytes

            results.append(result)

        # Return response
        if download and not is_batch:
            # Single file download
            buffer = io.BytesIO(results[0]["audio"])
            return send_file(
                buffer,
                mimetype="audio/wav",
                as_attachment=True,
                download_name="output.wav"
            )

        if embed:
            return jsonify({
                "audios": results,
                "batch": is_batch,
            })

        # Return metadata with audio bytes (non-embed)
        if is_batch:
            return jsonify({
                "audios": results,
                "batch": True,
            })
        else:
            # Single non-embed: return audio file
            buffer = io.BytesIO(results[0]["audio"])
            return send_file(
                buffer,
                mimetype="audio/wav",
                as_attachment=download,
                download_name="output.wav"
            )

    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        for f in temp_files:
            try:
                os.unlink(f)
            except OSError:
                pass


def _generate_with_preset(texts, preset_name, speed, num_steps, download, embed, is_batch):
    """Generate TTS using a saved voice preset.

    Args:
        texts: List of texts to synthesize.
        preset_name: Name of the saved preset.
        speed: Speaking rate.
        num_steps: Diffusion steps.
        download: Whether to return as file download.
        embed: Whether to return base64 encoded.
        is_batch: Whether this is a batch request.

    Returns:
        JSON response with audio(s) and metadata, or audio file.
    """
    global omnivoice_client

    preset_manager = VoicePresetManager()
    preset = preset_manager.load_preset(preset_name)

    if preset is None:
        return jsonify({"error": f"Preset '{preset_name}' not found"}), 404

    results = []
    for text in texts:
        try:
            audio, preset_meta = omnivoice_client.generate_with_preset(
                text, preset_name, speed=speed, num_steps=num_steps
            )
            metadata = omnivoice_client.get_audio_metadata(audio)
            metadata["preset"] = preset_meta.get("name")
            metadata["preset_description"] = preset_meta.get("description", "")
            audio_bytes = omnivoice_client.audio_to_bytes(audio)

            result = {
                "text": text,
                "metadata": metadata,
            }
            if embed:
                result["audio"] = base64.b64encode(audio_bytes).decode("utf-8")
            else:
                result["audio"] = audio_bytes

            results.append(result)
        except Exception as e:
            logger.error(f"Preset TTS generation failed: {e}")
            return jsonify({"error": str(e)}), 500

    # Return response
    if download and not is_batch:
        buffer = io.BytesIO(results[0]["audio"])
        return send_file(
            buffer,
            mimetype="audio/wav",
            as_attachment=True,
            download_name=f"{preset_name}.wav"
        )

    if embed:
        return jsonify({
            "audios": results,
            "batch": is_batch,
            "preset": preset_name,
        })

    if is_batch:
        return jsonify({
            "audios": results,
            "batch": True,
            "preset": preset_name,
        })
    else:
        buffer = io.BytesIO(results[0]["audio"])
        return send_file(
            buffer,
            mimetype="audio/wav",
            as_attachment=download,
            download_name=f"{preset_name}.wav"
        )


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
# API: Persons - Voice Enrollment & Speaker Identification
# ============================================================================

@app.route("/api/persons", methods=["GET"])
def api_list_persons():
    """List all enrolled persons with enrollment status."""
    persons = person_store.list_persons()
    result = []
    for p in persons:
        p["enrollment_status"] = "complete" if p.get("enrollment_complete") else "pending"
        result.append(p)
    return jsonify({"persons": result})


@app.route("/api/persons", methods=["POST"])
def api_create_person():
    """Create a new person.

    JSON body:
    - name: str (required) - Person's display name

    Returns:
        Created person dict
    """
    data = request.get_json()
    if not data or not data.get("name"):
        return jsonify({"error": "Name is required"}), 400

    name = data["name"]
    person = person_store.create_person(name)
    return jsonify({"success": True, "person": person}), 201


@app.route("/api/persons/<person_id>", methods=["GET"])
def api_get_person(person_id):
    """Get person details with voice samples."""
    person = person_store.get_person(person_id)
    if not person:
        return jsonify({"error": f"Person '{person_id}' not found"}), 404

    # Get voice samples
    samples = person_store.get_voice_samples(person_id)
    person["voice_samples"] = samples

    # Get enrollment status if in progress
    enrollment = enrollment_manager.get_enrollment_for_person(person_id)
    if enrollment and enrollment["status"] == "in_progress":
        person["enrollment_status"] = "in_progress"
        person["enrollment_id"] = enrollment["id"]
        person["sample_count"] = len(enrollment["samples"])
    else:
        person["enrollment_status"] = "complete" if person.get("enrollment_complete") else "pending"

    return jsonify({"person": person})


@app.route("/api/persons/<person_id>", methods=["DELETE"])
def api_delete_person(person_id):
    """Delete a person and their voice samples."""
    person = person_store.get_person(person_id)
    if not person:
        return jsonify({"error": f"Person '{person_id}' not found"}), 404

    success = person_store.delete_person(person_id)
    return jsonify({"success": success})


@app.route("/api/persons/<person_id>/samples", methods=["POST"])
def api_add_sample(person_id):
    """Add voice sample to a person (multipart form).

    Form field:
    - audio: Audio file (required)
    - transcript: str (optional) - Transcript of the audio
    """
    person = person_store.get_person(person_id)
    if not person:
        return jsonify({"error": f"Person '{person_id}' not found"}), 404

    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    if not audio_file.filename:
        return jsonify({"error": "No file selected"}), 400

    transcript = request.form.get("transcript", "")

    # Save to samples directory
    samples_dir = Path("/data/samples") / person_id
    samples_dir.mkdir(parents=True, exist_ok=True)

    sample_path = samples_dir / f"{uuid.uuid4()}.wav"
    audio_file.save(sample_path)

    try:
        person_store.add_voice_sample(person_id, str(sample_path), transcript)
        samples = person_store.get_voice_samples(person_id)
        return jsonify({
            "success": True,
            "sample_count": len(samples),
            "samples": samples,
        })
    except Exception as e:
        logger.error(f"Failed to add sample: {e}")
        if sample_path.exists():
            sample_path.unlink()
        return jsonify({"error": str(e)}), 500


@app.route("/api/persons/<person_id>/status", methods=["GET"])
def api_person_status(person_id):
    """Get enrollment status for a person."""
    person = person_store.get_person(person_id)
    if not person:
        return jsonify({"error": f"Person '{person_id}' not found"}), 404

    # Check for active enrollment
    enrollment = enrollment_manager.get_enrollment_for_person(person_id)

    if enrollment:
        return jsonify({
            "person_id": person_id,
            "status": enrollment["status"],
            "enrollment_id": enrollment["id"],
            "sample_count": len(enrollment["samples"]),
            "min_required": enrollment_manager.MIN_SAMPLES,
            "recommended": enrollment_manager.RECOMMENDED_SAMPLES,
        })

    # No active enrollment, check if completed
    return jsonify({
        "person_id": person_id,
        "status": "enrolled" if person.get("enrollment_complete") else "not_enrolled",
        "voice_samples_count": len(person.get("voice_samples", [])),
        "has_voiceprint": person.get("voiceprint_path") is not None,
    })


@app.route("/api/identify-speaker", methods=["POST"])
def api_identify_speaker():
    """Identify speaker from audio file.

    Multipart form with:
    - audio: Audio file to identify (required)
    OR
    JSON body with:
    - audio: Base64 encoded audio (required)
    """
    # Check for multipart file upload
    if request.content_type and "multipart/form-data" in request.content_type:
        if "audio" not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        audio_file = request.files["audio"]
        if not audio_file.filename:
            return jsonify({"error": "No file selected"}), 400

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            audio_file.save(tmp.name)
            tmp_path = tmp.name

        try:
            result = identify_speaker(tmp_path)
            return jsonify(result)
        except Exception as e:
            logger.error(f"Speaker identification failed: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    # Check for JSON with base64 audio
    data = request.get_json()
    if data and "audio" in data:
        try:
            result = identify_speaker_from_base64(data["audio"])
            return jsonify(result)
        except Exception as e:
            logger.error(f"Speaker identification failed: {e}")
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "No audio provided"}), 400


@app.route("/api/verify-speaker", methods=["POST"])
def api_verify_speaker():
    """Verify audio matches an enrolled person.

    JSON body:
    - person_id: str (required) - ID of person to verify against
    - audio: Base64 encoded audio (required)
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    person_id = data.get("person_id")
    audio_b64 = data.get("audio")

    if not person_id:
        return jsonify({"error": "person_id is required"}), 400
    if not audio_b64:
        return jsonify({"error": "audio is required"}), 400

    person = get_person(person_id)
    if not person:
        return jsonify({"error": f"Person '{person_id}' not found"}), 404

    if not person.is_enrolled():
        return jsonify({"error": "Person is not enrolled"}), 400

    try:
        result = verify_match_from_base64(audio_b64, person)
        result["person"] = person.to_dict()
        return jsonify(result)
    except Exception as e:
        logger.error(f"Speaker verification failed: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# API: Voice Design
# ============================================================================

@app.route("/api/voice-design/attributes", methods=["GET"])
def get_voice_design_attributes():
    """Get supported voice design attributes.

    Returns:
        JSON with supported attributes for voice design:
        - gender: male, female
        - age: child, teenager, adult, elderly
        - pitch: very_low, low, medium, high, very_high
        - accent: various accents
        - style: various speaking styles
    """
    return jsonify({
        "attributes": VOICE_DESIGN_ATTRIBUTES,
        "example_instruct": "female, british accent, medium pitch, calm",
    })


@app.route("/api/voice-design/presets", methods=["GET"])
def list_voice_presets():
    """List saved voice presets.

    Returns:
        JSON with list of saved presets.
    """
    preset_manager = VoicePresetManager()
    presets = preset_manager.list_presets()
    return jsonify({"presets": presets})


@app.route("/api/voice-design/presets", methods=["POST"])
def create_voice_preset():
    """Create a new voice preset from generated audio.

    Request JSON:
    - audio: Base64 encoded audio (required)
    - name: str (required) - Unique preset name
    - description: str (optional)
    - ref_text: str (optional) - Transcript
    - instruct: str (optional) - Voice design instruction used
    - tags: list (optional) - Tags for categorization

    Returns:
        JSON with created preset metadata.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    name = data.get("name")
    if not name:
        return jsonify({"error": "name is required"}), 400

    audio_b64 = data.get("audio")
    if not audio_b64:
        return jsonify({"error": "audio is required"}), 400

    try:
        audio_bytes = base64.b64decode(audio_b64)
        import torchaudio as ta

        # Load audio from bytes
        buffer = io.BytesIO(audio_bytes)
        audio, sr = ta.load(buffer)

        preset_manager = VoicePresetManager()
        preset = preset_manager.save_preset(
            audio=audio,
            name=name,
            description=data.get("description", ""),
            ref_text=data.get("ref_text", ""),
            instruct=data.get("instruct", ""),
            tags=data.get("tags", []),
        )

        return jsonify({"success": True, "preset": preset}), 201
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Failed to create preset: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/voice-design/presets/<name>", methods=["GET"])
def get_voice_preset(name):
    """Get a specific voice preset.

    Returns:
        JSON with preset metadata.
    """
    preset_manager = VoicePresetManager()
    preset = preset_manager.load_preset(name)

    if preset is None:
        return jsonify({"error": f"Preset '{name}' not found"}), 404

    return jsonify({"preset": preset})


@app.route("/api/voice-design/presets/<name>", methods=["DELETE"])
def delete_voice_preset(name):
    """Delete a voice preset.

    Returns:
        JSON with deleted file paths.
    """
    preset_manager = VoicePresetManager()
    deleted = preset_manager.delete_preset(name)

    if not deleted:
        return jsonify({"error": f"Preset '{name}' not found"}), 404

    return jsonify({"success": True, "deleted": deleted})


@app.route("/api/voice-design/validate-instruct", methods=["POST"])
def validate_voice_instruct():
    """Validate a voice design instruct string.

    Request JSON:
    - instruct: str (required) - Instruct string to validate

    Returns:
        JSON with validation result and any issues.
    """
    data = request.get_json()
    if not data or "instruct" not in data:
        return jsonify({"error": "instruct string is required"}), 400

    instruct = data["instruct"]
    is_valid, issues = validate_instruct(instruct)

    return jsonify({
        "valid": is_valid,
        "instruct": instruct,
        "issues": issues,
    })


@app.route("/api/voice-design/build-instruct", methods=["POST"])
def build_voice_instruct():
    """Build an instruct string from individual attributes.

    Request JSON:
    - gender: str (optional)
    - age: str (optional)
    - pitch: str (optional)
    - accent: str (optional)
    - style: str (optional)

    Returns:
        JSON with built instruct string.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    instruct = build_instruct_string(
        gender=data.get("gender"),
        age=data.get("age"),
        pitch=data.get("pitch"),
        accent=data.get("accent"),
        style=data.get("style"),
    )

    return jsonify({
        "instruct": instruct,
        "attributes": data,
    })


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
    load_person_system()

    app.run(host=args.host, port=args.port, debug=args.debug)
