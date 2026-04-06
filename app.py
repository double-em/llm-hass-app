"""LLM AI Integration Dashboard for Home Assistant.

Provides a web dashboard and API for:
- AI Provider management (MiniMax, etc.)
- Voice cloning and generation (OmniVoice)
- TTS synthesis
"""

import argparse
import base64
import datetime
import hashlib
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

from memory import SessionStore, MessageStore, VectorStore, EmbeddingEngine
from memory.ha_assists import HAAssistsClient

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

# Memory system
session_store = None
message_store = None
vector_store = None
embedding_engine = None
ha_assists_client = None

# Voice cache
voice_cache_store = None


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


def load_memory_system():
    """Initialize the AI memory and vector memory components."""
    global session_store, message_store, vector_store, embedding_engine, ha_assists_client
    session_store = SessionStore()
    message_store = MessageStore()
    embedding_engine = EmbeddingEngine()
    vector_store = VectorStore()
    ha_assists_client = HAAssistsClient()
    logger.info("Memory system initialized")


def load_voice_cache():
    """Initialize the voice line cache."""
    global voice_cache_store
    from voice_cache_store import VoiceCacheStore
    voice_cache_store = VoiceCacheStore()
    logger.info("Voice cache initialized")


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


@app.route("/memory")
def memory_page():
    """AI Memory page."""
    return render_template("memory.html")


@app.route("/vector")
def vector_page():
    """Vector Memory page."""
    return render_template("vector.html")


@app.route("/ha-assists")
def ha_assists_page():
    """Home Assistant Assists integration page."""
    return render_template("ha-assists.html")


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
    global omnivoice_model, omnivoice_client, voice_cache_store

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

    # Check cache first if voice_cache_store is available
    use_cache = voice_cache_store is not None and not embed

    try:
        for text in texts:
            generate_kwargs = {
                "num_steps": num_steps,
                "speed": speed,
            }

            # Determine mode and set appropriate args
            cache_key = None
            if use_cache:
                # Build cache key from text + generation params
                key_parts = [text, str(speed), str(num_steps), mode]
                if voice_name:
                    key_parts.append(f"voice:{voice_name}")
                elif ref_audio_b64:
                    key_parts.append(f"ref_audio:{ref_audio_b64[:32]}")
                    if ref_text:
                        key_parts.append(f"ref_text:{ref_text}")
                elif instruct:
                    key_parts.append(f"instruct:{instruct}")
                cache_key = hashlib.sha256("|".join(key_parts).encode()).hexdigest()

                # Check cache
                cached = voice_cache_store.get(cache_key, include_audio=True)
                if cached:
                    logger.info(f"Voice cache hit for: {text[:50]}...")
                    result = {
                        "text": text,
                        "metadata": {
                            "provider": cached.provider,
                            "voice_name": cached.voice_name,
                            "speed": cached.speed,
                            "num_steps": cached.num_steps,
                            "audio_format": cached.audio_format,
                            "sample_rate": cached.sample_rate,
                            "duration_ms": cached.duration_ms,
                        },
                        "audio": cached.audio_blob,
                        "cached": True,
                    }
                    results.append(result)
                    continue

            # Generate new audio
            audio = omnivoice_model.generate(text=text, **generate_kwargs)
            metadata = omnivoice_client.get_audio_metadata(audio)
            audio_bytes = omnivoice_client.audio_to_bytes(audio)

            # Save to cache
            if use_cache and cache_key and audio_bytes:
                voice_cache_store.save(
                    text_hash=cache_key,
                    text=text,
                    audio_blob=audio_bytes,
                    provider="omnivoice",
                    voice_name=voice_name,
                    speed=speed,
                    num_steps=num_steps,
                    audio_format="wav",
                    sample_rate=16000,
                    duration_ms=metadata.get("duration_ms", 0),
                )
                logger.info(f"Saved to voice cache: {text[:50]}...")

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
    - session_id: str (optional) - Use existing session for context
    - create_session: bool (optional) - Auto-create session if not provided
    """
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "Missing 'message' in request"}), 400

    message = data["message"]
    provider_name = data.get("provider")
    system_prompt = data.get("system_prompt")
    session_id = data.get("session_id")
    create_session = data.get("create_session", False)

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

    # Handle session
    session = None
    if session_id:
        session = session_store.get_session(session_id)
        if not session:
            return jsonify({"error": f"Session '{session_id}' not found"}), 404
    elif create_session:
        session = session_store.create_session(provider=provider_name)

    if provider.get("type") == "minimax":
        return conversation_minimax(message, provider, system_prompt, session)

    return jsonify({"error": f"Unknown provider type: {provider.get('type')}"}), 400


def conversation_minimax(message, provider, system_prompt=None, session=None):
    """Send conversation to MiniMax with optional session context."""
    import requests

    url = "https://api.minimax.io/anthropic/v1/messages"
    headers = {
        "Authorization": f"Bearer {provider.get('api_key')}",
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }

    messages = []

    # Add conversation history from session if available
    if session:
        history = message_store.get_recent_messages(
            session["session_id"],
            max_tokens=provider.get("max_tokens", 4000)
        )
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})

    if system_prompt:
        messages.append({"role": "assistant", "content": system_prompt})

    # Add conversation history if provided
    if history:
        for msg in history[:-1]:  # Exclude the current message
            if msg.get("role") in ("user", "assistant"):
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

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
            response_text = result.get("content", [{}])[0].get("text", "")

            # Store messages in session if available
            if session:
                message_store.add_message(
                    session_id=session["session_id"],
                    role="user",
                    content=message,
                    metadata={"provider": "minimax", "model": provider.get("chat_model")}
                )
                message_store.add_message(
                    session_id=session["session_id"],
                    role="assistant",
                    content=response_text,
                    metadata={"provider": "minimax", "model": provider.get("chat_model")}
                )
                session_store.increment_message_count(session["session_id"])

            return jsonify({
                "response": response_text,
                "provider": "minimax",
                "model": provider.get("chat_model"),
                "session_id": session["session_id"] if session else None,
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
# API: Sessions - Conversation History
# ============================================================================

@app.route("/api/sessions", methods=["GET"])
def list_sessions():
    """List all conversation sessions.

    Query params:
    - limit: Max sessions to return (default 50)
    - offset: Number to skip (default 0)

    Returns:
        JSON with list of sessions.
    """
    limit = request.args.get("limit", 50, type=int)
    offset = request.args.get("offset", 0, type=int)

    sessions = session_store.list_sessions(limit=limit, offset=offset)
    total = session_store.get_session_count()

    return jsonify({
        "sessions": sessions,
        "total": total,
        "limit": limit,
        "offset": offset,
    })


@app.route("/api/sessions", methods=["POST"])
def create_session():
    """Create a new conversation session.

    JSON body:
    - name: Optional session name
    - provider: AI provider name (default: first configured)

    Returns:
        Created session dict.
    """
    data = request.get_json() or {}
    name = data.get("name")
    provider = data.get("provider", "minimax")

    # Use first configured provider if not specified
    if provider == "minimax" and not config.get("providers"):
        return jsonify({"error": "No AI provider configured"}), 400

    session = session_store.create_session(name=name, provider=provider)
    return jsonify({"success": True, "session": session}), 201


@app.route("/api/sessions/<session_id>", methods=["GET"])
def get_session(session_id):
    """Get a session with its messages.

    Returns:
        Session dict with messages.
    """
    session = session_store.get_session(session_id)
    if not session:
        return jsonify({"error": f"Session '{session_id}' not found"}), 404

    messages = message_store.get_messages(session_id)
    session["messages"] = messages

    return jsonify({"session": session})


@app.route("/api/sessions/<session_id>", methods=["DELETE"])
def delete_session(session_id):
    """Delete a session and its messages.

    Returns:
        Success status.
    """
    success = session_store.delete_session(session_id)
    if not success:
        return jsonify({"error": f"Session '{session_id}' not found"}), 404

    return jsonify({"success": True})


@app.route("/api/sessions/<session_id>/messages", methods=["GET"])
def get_session_messages(session_id):
    """Get messages for a session.

    Query params:
    - limit: Max messages to return (None for all)
    - offset: Number to skip (default 0)

    Returns:
        JSON with list of messages.
    """
    session = session_store.get_session(session_id)
    if not session:
        return jsonify({"error": f"Session '{session_id}' not found"}), 404

    limit = request.args.get("limit", type=int)
    offset = request.args.get("offset", 0, type=int)

    messages = message_store.get_messages(session_id, limit=limit, offset=offset)
    total = message_store.get_message_count(session_id)

    return jsonify({
        "messages": messages,
        "total": total,
        "limit": limit,
        "offset": offset,
    })


@app.route("/api/sessions/<session_id>/messages", methods=["POST"])
def add_session_message(session_id):
    """Add a message to a session.

    JSON body:
    - role: "user" or "assistant" (required)
    - content: Message content (required)
    - metadata: Optional metadata dict

    Returns:
        Created message dict.
    """
    session = session_store.get_session(session_id)
    if not session:
        return jsonify({"error": f"Session '{session_id}' not found"}), 404

    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    role = data.get("role")
    content = data.get("content")

    if role not in ("user", "assistant", "system"):
        return jsonify({"error": "Role must be 'user', 'assistant', or 'system'"}), 400
    if not content:
        return jsonify({"error": "Content is required"}), 400

    message = message_store.add_message(
        session_id=session_id,
        role=role,
        content=content,
        metadata=data.get("metadata")
    )

    # Update session message count
    session_store.increment_message_count(session_id)

    return jsonify({"success": True, "message": message}), 201


@app.route("/api/sessions/<session_id>", methods=["PATCH"])
def update_session(session_id):
    """Update a session (e.g., rename).

    JSON body:
    - name: New session name

    Returns:
        Updated session dict.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    session = session_store.update_session(session_id, data)
    if not session:
        return jsonify({"error": f"Session '{session_id}' not found"}), 404

    return jsonify({"success": True, "session": session})


# ============================================================================
# API: Memory - Vector Memory (Semantic Search)
# ============================================================================

@app.route("/api/memory", methods=["GET"])
def list_memory_entries():
    """List memory entries.

    Query params:
    - limit: Max entries to return (default 100)
    - offset: Number to skip (default 0)
    - source: Filter by source

    Returns:
        JSON with list of entries.
    """
    limit = request.args.get("limit", 100, type=int)
    offset = request.args.get("offset", 0, type=int)
    source = request.args.get("source")

    entries = vector_store.list_entries(limit=limit, offset=offset, source=source)

    return jsonify({
        "entries": entries,
        "limit": limit,
        "offset": offset,
    })


@app.route("/api/memory", methods=["POST"])
def add_memory_entry():
    """Add a new memory entry.

    JSON body:
    - content: Text content (required)
    - tags: Optional list of tags
    - source: Source (default: "manual")
    - related_session_id: Optional linked session

    Returns:
        Created entry dict.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    content = data.get("content")
    if not content:
        return jsonify({"error": "content is required"}), 400

    # Generate embedding
    embedding = embedding_engine.encode(content)

    entry = vector_store.add_entry(
        content=content,
        embedding=embedding,
        tags=data.get("tags"),
        source=data.get("source", "manual"),
        related_session_id=data.get("related_session_id"),
        metadata=data.get("metadata")
    )

    return jsonify({"success": True, "entry": entry}), 201


@app.route("/api/memory/search", methods=["POST"])
def search_memory():
    """Search memory entries by semantic similarity.

    JSON body:
    - query: Search query text (required)
    - limit: Max results (default 5)
    - threshold: Min similarity 0-1 (default 0.7)
    - tags: Optional filter by tags

    Returns:
        JSON with matching entries and similarity scores.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    query = data.get("query")
    if not query:
        return jsonify({"error": "query is required"}), 400

    # Generate query embedding
    query_embedding = embedding_engine.encode(query)

    results = vector_store.search(
        query_embedding=query_embedding,
        limit=data.get("limit", 5),
        threshold=data.get("threshold", 0.7),
        tags=data.get("tags"),
        source=data.get("source")
    )

    return jsonify({
        "results": results,
        "query": query,
        "total_results": len(results),
    })


@app.route("/api/memory/<entry_id>", methods=["GET"])
def get_memory_entry(entry_id):
    """Get a specific memory entry.

    Returns:
        Entry dict.
    """
    entries = vector_store.list_entries(limit=10000)
    for entry in entries:
        if entry.get("entry_id") == entry_id:
            return jsonify({"entry": entry})

    return jsonify({"error": f"Entry '{entry_id}' not found"}), 404


@app.route("/api/memory/<entry_id>", methods=["DELETE"])
def delete_memory_entry(entry_id):
    """Delete a memory entry.

    Returns:
        Success status.
    """
    success = vector_store.delete_entry(entry_id)
    if not success:
        return jsonify({"error": f"Entry '{entry_id}' not found"}), 404

    return jsonify({"success": True})


@app.route("/api/memory/stats", methods=["GET"])
def get_memory_stats():
    """Get vector memory statistics.

    Returns:
        JSON with stats (total entries, sources, tags, storage size).
    """
    stats = vector_store.get_stats()
    return jsonify(stats)


# ============================================================================
# API: HA Assists - Home Assistant Integration
# ============================================================================

@app.route("/api/ha/assists", methods=["POST"])
def process_ha_assist():
    """Process an HA Assist pipeline request.

    JSON body (HA Protocol format):
    - intent: Input text/intent
    - agent_id: Agent identifier
    - conversation_id: Optional conversation ID
    - language: Language code (en, zh, etc.)
    - text: Input text

    Returns:
        Response text.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    text = data.get("text") or data.get("intent", "")
    conversation_id = data.get("conversation_id")
    language = data.get("language", "en")

    result = ha_assists_client.process_assist_pipeline(
        text=text,
        conversation_id=conversation_id,
        language=language
    )

    if "error" in result:
        return jsonify(result), 400

    return jsonify(result)


@app.route("/api/ha/config", methods=["GET"])
def get_ha_config():
    """Get HA Assists configuration.

    Returns:
        Config dict (without sensitive token).
    """
    config = ha_assists_client.config
    return jsonify({
        "enabled": config.enabled,
        "ha_url": config.ha_url,
        "agent_id": config.agent_id,
        "capabilities": config.capabilities,
        "configured": bool(config.ha_url and config.ha_token),
    })


@app.route("/api/ha/config", methods=["POST"])
def update_ha_config():
    """Update HA Assists configuration.

    JSON body:
    - enabled: bool
    - ha_url: Home Assistant URL
    - ha_token: Long-lived access token
    - agent_id: Agent identifier
    - capabilities: List of capabilities

    Returns:
        Updated config.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    config = ha_assists_client.update_config(
        enabled=data.get("enabled"),
        ha_url=data.get("ha_url"),
        ha_token=data.get("ha_token"),
        agent_id=data.get("agent_id"),
        capabilities=data.get("capabilities")
    )

    return jsonify({
        "success": True,
        "enabled": config.enabled,
        "ha_url": config.ha_url,
        "agent_id": config.agent_id,
        "capabilities": config.capabilities,
    })


@app.route("/api/ha/test", methods=["POST"])
def test_ha_connection():
    """Test HA connection.

    Returns:
        Success status and message.
    """
    result = ha_assists_client.test_connection()
    if result.get("success"):
        return jsonify(result)
    else:
        return jsonify(result), 400


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
# API: AI Memory - Conversation Sessions
# ============================================================================

# In-memory session storage (replace with persistent storage in production)
memory_sessions = {}
memory_counter = 0


# ============================================================================
# API: Voice Line Cache
# ============================================================================


@app.route("/api/voice-cache", methods=["GET"])
def get_voice_cache_stats():
    """Get voice cache statistics."""
    if voice_cache_store is None:
        return jsonify({"error": "Voice cache not initialized"}), 500

    stats = voice_cache_store.get_stats()
    return jsonify(stats)


@app.route("/api/voice-cache", methods=["DELETE"])
def clear_voice_cache():
    """Clear all voice cache entries."""
    if voice_cache_store is None:
        return jsonify({"error": "Voice cache not initialized"}), 500

    count = voice_cache_store.clear_all()
    return jsonify({"success": True, "cleared": count})


@app.route("/api/voice-cache/find", methods=["GET"])
def find_voice_cache():
    """Find voice cache entries matching criteria."""
    if voice_cache_store is None:
        return jsonify({"error": "Voice cache not initialized"}), 500

    text = request.args.get("text")
    provider = request.args.get("provider")
    voice_name = request.args.get("voice_name")
    limit = int(request.args.get("limit", 50))

    entries = voice_cache_store.find(
        text=text,
        provider=provider,
        voice_name=voice_name,
        limit=limit,
    )

    return jsonify({
        "entries": [e.to_dict() for e in entries],
        "count": len(entries),
    })


@app.route("/api/voice-cache/<text_hash>", methods=["GET"])
def get_voice_cache_entry(text_hash):
    """Get a specific voice cache entry by text hash."""
    if voice_cache_store is None:
        return jsonify({"error": "Voice cache not initialized"}), 500

    include_audio = request.args.get("include_audio", "false").lower() == "true"
    entry = voice_cache_store.get(text_hash, include_audio=include_audio)

    if entry is None:
        return jsonify({"error": "Cache entry not found"}), 404

    result = entry.to_dict()
    if not include_audio:
        result["audio_blob"] = None

    return jsonify(result)


@app.route("/api/voice-cache/<text_hash>", methods=["DELETE"])
def invalidate_voice_cache(text_hash):
    """Invalidate (delete) a voice cache entry."""
    if voice_cache_store is None:
        return jsonify({"error": "Voice cache not initialized"}), 500

    success = voice_cache_store.invalidate(text_hash)
    if not success:
        return jsonify({"error": "Cache entry not found"}), 404

    return jsonify({"success": True})


@app.route("/api/memory/sessions", methods=["GET"])
def list_memory_sessions():
    """List all conversation sessions."""
    sessions = []
    for sid, session in memory_sessions.items():
        sessions.append({
            "id": sid,
            "title": session.get("title", "Untitled"),
            "created_at": session.get("created_at"),
            "updated_at": session.get("updated_at"),
            "message_count": len(session.get("messages", [])),
        })
    sessions.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return jsonify({"sessions": sessions})


@app.route("/api/memory/sessions", methods=["POST"])
def create_memory_session():
    """Create a new conversation session."""
    global memory_counter
    data = request.get_json() or {}

    memory_counter += 1
    sid = f"session_{memory_counter}"
    now = datetime.now().isoformat()

    memory_sessions[sid] = {
        "id": sid,
        "title": data.get("title") or f"Session {memory_counter}",
        "created_at": now,
        "updated_at": now,
        "messages": [],
        "provider": None,
    }

    return jsonify({"success": True, "session": memory_sessions[sid]}), 201


@app.route("/api/memory/sessions/<session_id>", methods=["GET"])
def get_memory_session(session_id):
    """Get a conversation session with messages."""
    session = memory_sessions.get(session_id)
    if not session:
        return jsonify({"error": f"Session '{session_id}' not found"}), 404

    return jsonify({
        "id": session["id"],
        "title": session.get("title"),
        "messages": session.get("messages", []),
        "created_at": session.get("created_at"),
        "updated_at": session.get("updated_at"),
    })


@app.route("/api/memory/sessions/<session_id>", methods=["DELETE"])
def clear_memory_session(session_id):
    """Clear all messages in a session."""
    if session_id not in memory_sessions:
        return jsonify({"error": f"Session '{session_id}' not found"}), 404

    memory_sessions[session_id]["messages"] = []
    memory_sessions[session_id]["updated_at"] = datetime.now().isoformat()
    return jsonify({"success": True})


@app.route("/api/memory/sessions/<session_id>/messages", methods=["POST"])
def add_session_message(session_id):
    """Add a message to a session and get AI response."""
    if session_id not in memory_sessions:
        return jsonify({"error": f"Session '{session_id}' not found"}), 404

    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "Message is required"}), 400

    session = memory_sessions[session_id]

    # Add user message
    session["messages"].append({
        "role": "user",
        "content": data["message"],
        "timestamp": datetime.now().isoformat(),
    })

    # Get AI response
    provider_name = data.get("provider") or (list(config.get("providers", {}).keys())[0] if config.get("providers") else None)

    if provider_name and provider_name in config.get("providers", {}):
        provider = config["providers"][provider_name]
        if provider.get("type") == "minimax":
            response_text = conversation_minimax(
                data["message"], provider, None, session["messages"]
            )
            if isinstance(response_text, tuple):
                response_data = response_text[0]
                response_text = response_data.get("response", "Error getting response")
    else:
        response_text = "No AI provider configured. Please add a provider in AI Providers."

    # Add assistant message
    session["messages"].append({
        "role": "assistant",
        "content": response_text,
        "timestamp": datetime.now().isoformat(),
    })

    session["updated_at"] = datetime.now().isoformat()

    return jsonify({
        "success": True,
        "messages": session["messages"],
        "response": response_text,
    })


@app.route("/api/memory/sessions/<session_id>/export", methods=["GET"])
def export_session(session_id):
    """Export a session as JSON."""
    if session_id not in memory_sessions:
        return jsonify({"error": f"Session '{session_id}' not found"}), 404

    return jsonify(memory_sessions[session_id])


# ============================================================================
# API: Vector Memory
# ============================================================================

vector_memories = []
vector_counter = 0


@app.route("/api/memory/stats", methods=["GET"])
def get_memory_stats():
    """Get vector memory statistics."""
    return jsonify({
        "total_memories": len(vector_memories),
        "total_facts": len([m for m in vector_memories if m.get("category") == "fact"]),
        "vector_dimensions": 1536,  # Typical for embedding models
    })


@app.route("/api/memory/recent", methods=["GET"])
def get_recent_memories():
    """Get recent vector memories."""
    limit = request.args.get("limit", 20, type=int)
    memories = sorted(vector_memories, key=lambda x: x.get("created_at", ""), reverse=True)[:limit]
    return jsonify({"memories": memories})


@app.route("/api/memory", methods=["GET"])
def list_memories():
    """List all vector memories."""
    return jsonify({"memories": vector_memories})


@app.route("/api/memory", methods=["POST"])
def add_memory():
    """Add a new vector memory."""
    global vector_counter
    data = request.get_json()

    if not data or "content" not in data:
        return jsonify({"error": "Content is required"}), 400

    vector_counter += 1
    memory = {
        "id": vector_counter,
        "content": data["content"],
        "category": data.get("category", "general"),
        "tags": data.get("tags", []),
        "created_at": datetime.now().isoformat(),
        "relevance": 1.0,
    }
    vector_memories.append(memory)

    return jsonify({"success": True, "memory": memory}), 201


@app.route("/api/memory/search", methods=["POST"])
def search_memories():
    """Search vector memories by semantic similarity."""
    data = request.get_json()

    if not data or "query" not in data:
        return jsonify({"error": "Query is required"}), 400

    query = data["query"]
    limit = data.get("limit", 10)
    threshold = data.get("threshold", 0.7)

    # Simple keyword matching for demo (replace with actual vector search)
    results = []
    query_words = set(query.lower().split())

    for mem in vector_memories:
        content_words = set(mem.get("content", "").lower().split())
        # Calculate simple Jaccard similarity
        intersection = len(query_words & content_words)
        union = len(query_words | content_words)
        similarity = intersection / union if union > 0 else 0

        if similarity >= threshold:
            results.append({
                "id": mem.get("id"),
                "content": mem.get("content"),
                "relevance": similarity,
                "category": mem.get("category"),
                "tags": mem.get("tags"),
                "metadata": {
                    "category": mem.get("category"),
                    "tags": mem.get("tags"),
                },
            })

    results.sort(key=lambda x: x["relevance"], reverse=True)
    return jsonify({"results": results[:limit]})


# ============================================================================
# API: Home Assistant Assists
# ============================================================================

ha_config = {
    "url": "",
    "token": "",
}


@app.route("/api/ha/status", methods=["GET"])
def ha_status():
    """Get Home Assistant connection status."""
    url = ha_config.get("url")
    token = ha_config.get("token")

    if not url or not token:
        return jsonify({
            "connected": False,
            "error": "HA not configured"
        })

    try:
        import requests
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(f"{url}/api/", headers=headers, timeout=5)

        if response.status_code == 200:
            version = response.json().get("version", "Unknown")
            return jsonify({
                "connected": True,
                "url": url,
                "version": version,
                "features": {
                    "conversation": True,
                    "voice": True,
                    "intents": True,
                    "sentences": True,
                }
            })
        else:
            return jsonify({
                "connected": False,
                "error": f"HTTP {response.status_code}"
            })
    except Exception as e:
        return jsonify({
            "connected": False,
            "error": str(e)
        })


@app.route("/api/ha/config", methods=["GET"])
def get_ha_config():
    """Get saved HA configuration (without token)."""
    return jsonify({
        "url": ha_config.get("url"),
        "token": "***" if ha_config.get("token") else "",
    })


@app.route("/api/ha/config", methods=["POST"])
def save_ha_config():
    """Save HA configuration."""
    global ha_config
    data = request.get_json()

    if data.get("url"):
        ha_config["url"] = data["url"].rstrip("/")
    if data.get("token"):
        ha_config["token"] = data["token"]

    save_ha_config_to_file()
    return jsonify({"success": True})


@app.route("/api/ha/test", methods=["POST"])
def test_ha_connection():
    """Test HA connection."""
    data = request.get_json()
    url = data.get("url", "").rstrip("/")
    token = data.get("token", "")

    if not url or not token:
        return jsonify({"success": False, "error": "URL and token are required"})

    try:
        import requests
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(f"{url}/api/", headers=headers, timeout=10)

        if response.status_code == 200:
            return jsonify({"success": True, "version": response.json().get("version")})
        else:
            return jsonify({"success": False, "error": f"HTTP {response.status_code}"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/ha/agents", methods=["GET"])
def list_ha_agents():
    """List HA assist agents."""
    if not ha_config.get("url") or not ha_config.get("token"):
        return jsonify({"agents": []})

    try:
        import requests
        headers = {"Authorization": f"Bearer {ha_config['token']}"}
        response = requests.get(
            f"{ha_config['url']}/api/conversation/agents",
            headers=headers,
            timeout=5
        )

        if response.status_code == 200:
            return jsonify({"agents": response.json()})
    except:
        pass

    return jsonify({"agents": []})


@app.route("/api/ha/conversations", methods=["GET"])
def list_ha_conversations():
    """List recent HA conversations."""
    limit = request.args.get("limit", 10, type=int)
    return jsonify({"conversations": []})


@app.route("/api/ha/converse", methods=["POST"])
def ha_converse():
    """Send conversation to HA assist agent."""
    data = request.get_json()

    if not ha_config.get("url") or not ha_config.get("token"):
        return jsonify({"error": "HA not configured"}), 400

    agent_id = data.get("agent_id")
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "Text is required"}), 400

    try:
        import requests
        headers = {
            "Authorization": f"Bearer {ha_config['token']}",
            "Content-Type": "application/json",
        }

        payload = {"text": text}
        if agent_id:
            payload["agent_id"] = agent_id

        response = requests.post(
            f"{ha_config['url']}/api/conversation/process",
            headers=headers,
            json=payload,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            return jsonify({
                "response": result.get("response", {}).get("speech", {}).get("plain", {}).get("text", ""),
                "conversation_id": result.get("conversation_id"),
            })
        else:
            return jsonify({"error": f"HTTP {response.status_code}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def save_ha_config_to_file():
    """Save HA config to file."""
    config_path = Path("/data/ha_config.json")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(ha_config, f)


def load_ha_config():
    """Load HA config from file."""
    global ha_config
    config_path = Path("/data/ha_config.json")
    if config_path.exists():
        with open(config_path) as f:
            ha_config.update(json.load(f))


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
        "sessions_count": session_store.get_session_count() if session_store else 0,
        "embedding_available": embedding_engine.is_available() if embedding_engine else False,
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
    load_memory_system()
    load_voice_cache()

    app.run(host=args.host, port=args.port, debug=args.debug)
