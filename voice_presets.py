"""Voice preset management for the LLM AI Dashboard."""

import json
from datetime import datetime
from pathlib import Path

import torchaudio

from logging_config import get_logger

logger = get_logger(__name__)

PRESETS_DIR = Path("/data/voice_presets")


class VoicePresetManager:
    """Manages voice presets stored as WAV + JSON metadata pairs.

    Storage format:
    - Audio: /data/voice_presets/<name>.wav
    - Metadata: /data/voice_presets/<name>.json
    """

    def __init__(self, presets_dir=None):
        """Initialize preset manager.

        Args:
            presets_dir: Custom presets directory path.
        """
        self.presets_dir = Path(presets_dir) if presets_dir else PRESETS_DIR
        self.presets_dir.mkdir(parents=True, exist_ok=True)

    def save_preset(self, audio, name, description="", ref_text="", instruct="", tags=None):
        """Save audio as a voice preset.

        Args:
            audio: Audio tensor to save.
            name: Unique name for the preset.
            description: Human-readable description.
            ref_text: Transcript of reference audio.
            instruct: Voice design instruction string.
            tags: List of tags for categorization.

        Returns:
            dict: Saved preset metadata.
        """
        audio_path = self.presets_dir / f"{name}.wav"
        metadata_path = self.presets_dir / f"{name}.json"

        if audio_path.exists():
            raise ValueError(f"Preset '{name}' already exists")

        # Save audio
        torchaudio.save(str(audio_path), audio[0] if audio.dim() == 2 else audio, 24000, format="wav")

        # Create metadata
        metadata = {
            "name": name,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "ref_text": ref_text,
            "instruct": instruct,
            "tags": tags or [],
            "ref_audio_path": str(audio_path),
        }

        # Save metadata
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved preset '{name}'")
        return metadata

    def load_preset(self, name):
        """Load a voice preset by name.

        Args:
            name: Preset name.

        Returns:
            dict: Preset metadata or None if not found.
        """
        metadata_path = self.presets_dir / f"{name}.json"
        if not metadata_path.exists():
            return None

        with open(metadata_path) as f:
            return json.load(f)

    def list_presets(self):
        """List all available presets.

        Returns:
            list: List of preset metadata dicts.
        """
        presets = []
        for metadata_path in sorted(self.presets_dir.glob("*.json")):
            try:
                with open(metadata_path) as f:
                    metadata = json.load(f)
                    presets.append(metadata)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load preset {metadata_path}: {e}")
        return presets

    def delete_preset(self, name):
        """Delete a voice preset.

        Args:
            name: Preset name to delete.

        Returns:
            list: Deleted file paths.
        """
        audio_path = self.presets_dir / f"{name}.wav"
        metadata_path = self.presets_dir / f"{name}.json"

        deleted = []
        if audio_path.exists():
            audio_path.unlink()
            deleted.append(str(audio_path))
        if metadata_path.exists():
            metadata_path.unlink()
            deleted.append(str(metadata_path))

        return deleted

    def update_preset(self, name, description=None, tags=None):
        """Update preset metadata.

        Args:
            name: Preset name.
            description: New description (optional).
            tags: New tags list (optional).

        Returns:
            dict: Updated metadata or None if preset not found.
        """
        metadata = self.load_preset(name)
        if metadata is None:
            return None

        if description is not None:
            metadata["description"] = description
        if tags is not None:
            metadata["tags"] = tags

        metadata_path = self.presets_dir / f"{name}.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return metadata


# Supported voice design attributes
VOICE_DESIGN_ATTRIBUTES = {
    "gender": ["male", "female"],
    "age": ["child", "teenager", "adult", "elderly"],
    "pitch": ["very_low", "low", "medium", "high", "very_high"],
    "accent": [
        "american", "british", "australian", "indian",
        "chinese", "japanese", "korean", "french", "german",
        "spanish", "italian", "brazilian", "russian",
    ],
    "style": [
        "neutral", "whisper", "cheerful", "calm",
        "serious", "excited", "sad", "romantic",
        "professional", "casual",
    ],
}


def build_instruct_string(gender=None, age=None, pitch=None, accent=None, style=None):
    """Build an instruct string from individual attributes.

    Args:
        gender: Gender value.
        age: Age value.
        pitch: Pitch value.
        accent: Accent value.
        style: Style value.

    Returns:
        str: Comma-separated instruct string.
    """
    parts = []
    if gender:
        parts.append(f"{gender}")
    if age:
        parts.append(f"{age}")
    if pitch:
        parts.append(f"{pitch} pitch")
    if accent:
        parts.append(f"{accent} accent")
    if style:
        parts.append(style)

    return ", ".join(parts)


def validate_instruct(instruct):
    """Validate an instruct string.

    Args:
        instruct: Instruct string to validate.

    Returns:
        tuple: (is_valid, list of issues)
    """
    issues = []
    parts = [p.strip().lower() for p in instruct.split(",")]

    for part in parts:
        found = False
        for attr, values in VOICE_DESIGN_ATTRIBUTES.items():
            if part in values:
                found = True
                break
            # Check partial matches for pitch/accent/style
            if attr == "pitch" and "pitch" in part:
                found = True
                break
            if attr == "accent" and "accent" in part:
                found = True
                break
        if not found and part:
            issues.append(f"Unknown attribute: '{part}'")

    return len(issues) == 0, issues
