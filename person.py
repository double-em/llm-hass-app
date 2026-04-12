"""Person management with voice enrollment and verification."""

import json
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np

from voice_engine import voice_encoder
from logging_config import get_logger

logger = get_logger(__name__)

# Paths
DATA_DIR = Path("/data")
VOICEPRINTS_DIR = DATA_DIR / "voiceprints"
PERSONS_FILE = DATA_DIR / "persons.json"

# Verification threshold
VERIFICATION_THRESHOLD = 0.75


class Person:
    """Person with voice enrollment."""

    def __init__(self, name, person_id=None, enrolled_samples=None, created_at=None):
        """Initialize a Person.

        Args:
            name: Person's display name
            person_id: Unique identifier (auto-generated if not provided)
            enrolled_samples: List of sample file paths
            created_at: ISO timestamp of creation
        """
        self.id = person_id or str(uuid.uuid4())[:8]
        self.name = name
        self.enrolled_samples = enrolled_samples or []
        self.created_at = created_at or datetime.utcnow().isoformat()

    @property
    def voiceprint_path(self):
        """Get the path to this person's voiceprint file."""
        return VOICEPRINTS_DIR / f"{self.id}.npy"

    def is_enrolled(self):
        """Check if person has a voiceprint enrolled."""
        return self.voiceprint_path.exists()

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "enrolled_samples": self.enrolled_samples,
            "created_at": self.created_at,
            "is_enrolled": self.is_enrolled(),
        }

    @classmethod
    def from_dict(cls, data):
        """Create Person from dictionary."""
        return cls(
            name=data["name"],
            person_id=data.get("id"),
            enrolled_samples=data.get("enrolled_samples", []),
            created_at=data.get("created_at"),
        )


def _ensure_dirs():
    """Ensure required directories exist."""
    VOICEPRINTS_DIR.mkdir(parents=True, exist_ok=True)


def _load_persons():
    """Load persons from JSON file."""
    if not PERSONS_FILE.exists():
        return []

    with open(PERSONS_FILE) as f:
        data = json.load(f)
        return [Person.from_dict(p) for p in data]


def _save_persons(persons):
    """Save persons to JSON file."""
    _ensure_dirs()
    with open(PERSONS_FILE, "w") as f:
        json.dump([p.to_dict() for p in persons], f, indent=2)


def _get_person_by_id(persons, person_id):
    """Find person by ID."""
    for p in persons:
        if p.id == person_id:
            return p
    return None


def list_persons():
    """List all enrolled persons."""
    return _load_persons()


def get_person(person_id):
    """Get a person by ID."""
    persons = _load_persons()
    return _get_person_by_id(persons, person_id)


def create_person(name, audio_samples=None):
    """Create a new person with optional initial enrollment.

    Args:
        name: Person's display name
        audio_samples: List of audio file paths for initial enrollment

    Returns:
        Created Person object
    """
    _ensure_dirs()

    person = Person(name=name)

    if audio_samples:
        for sample_path in audio_samples:
            _add_enrollment_sample(person, sample_path)

    persons = _load_persons()
    persons.append(person)
    _save_persons(persons)

    logger.info(f"Created person: {person.id} ({person.name})")
    return person


def delete_person(person_id):
    """Delete a person and their voiceprint.

    Args:
        person_id: ID of person to delete

    Returns:
        True if deleted, False if not found
    """
    persons = _load_persons()
    person = _get_person_by_id(persons, person_id)

    if not person:
        return False

    # Remove voiceprint file
    if person.voiceprint_path.exists():
        person.voiceprint_path.unlink()

    # Remove sample files
    for sample in person.enrolled_samples:
        sample_path = Path(sample)
        if sample_path.exists():
            sample_path.unlink()

    # Remove from persons list
    persons = [p for p in persons if p.id != person_id]
    _save_persons(persons)

    logger.info(f"Deleted person: {person_id}")
    return True


def enroll_sample(person_id, audio_path):
    """Add a voice sample to a person's enrollment.

    Args:
        person_id: ID of person to enroll
        audio_path: Path to audio sample file

    Returns:
        Updated Person object
    """
    persons = _load_persons()
    person = _get_person_by_id(persons, person_id)

    if not person:
        raise ValueError(f"Person not found: {person_id}")

    _add_enrollment_sample(person, audio_path)
    _save_persons(persons)

    logger.info(f"Enrolled sample for {person_id}: {audio_path}")
    return person


def _add_enrollment_sample(person, audio_path):
    """Add a sample and update voiceprint.

    Args:
        person: Person object
        audio_path: Path to audio file
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise ValueError(f"Audio file not found: {audio_path}")

    # Copy sample to voiceprints directory
    sample_dir = VOICEPRINTS_DIR / person.id
    sample_dir.mkdir(parents=True, exist_ok=True)

    sample_idx = len(person.enrolled_samples)
    sample_dest = sample_dir / f"sample_{sample_idx}.wav"
    shutil.copy2(audio_path, sample_dest)

    person.enrolled_samples.append(str(sample_dest))

    # Re-compute voiceprint from all samples
    _update_voiceprint(person)


def _update_voiceprint(person):
    """Update person's voiceprint from enrolled samples.

    Args:
        person: Person object with enrolled_samples
    """
    if not person.enrolled_samples:
        return

    # Compute embedding for each sample
    embeddings = []
    for sample_path in person.enrolled_samples:
        emb = voice_encoder.encode_voice(sample_path)
        embeddings.append(emb)

    # Average embeddings to create voiceprint
    voiceprint = np.mean(embeddings, axis=0)

    # Save voiceprint
    np.save(person.voiceprint_path, voiceprint)
    logger.info(f"Updated voiceprint for {person.id} from {len(embeddings)} samples")


def verify_match(audio_path, person):
    """Verify if audio matches a person's voiceprint.

    Args:
        audio_path: Path to audio file to verify
        person: Person object

    Returns:
        Dict with is_verified bool and score
    """
    if not person.is_enrolled():
        return {"is_verified": False, "score": 0.0, "error": "Person not enrolled"}

    result = voice_encoder.verify_speaker(audio_path, str(person.voiceprint_path))
    return result


def verify_match_from_base64(audio_b64, person):
    """Verify if base64 audio matches a person's voiceprint.

    Args:
        audio_b64: Base64 encoded audio string
        person: Person object

    Returns:
        Dict with is_verified bool and score
    """
    if not person.is_enrolled():
        return {"is_verified": False, "score": 0.0, "error": "Person not enrolled"}

    result = voice_encoder.verify_from_base64(audio_b64, str(person.voiceprint_path))
    return result


def identify_speaker(audio_path):
    """Identify speaker from audio against all enrolled persons.

    Args:
        audio_path: Path to audio file

    Returns:
        Dict with best_match (Person or None), score, is_match
    """
    persons = _load_persons()

    if not persons:
        return {"best_match": None, "score": 0.0, "is_match": False}

    best_match = None
    best_score = 0.0

    for person in persons:
        if not person.is_enrolled():
            continue

        result = voice_encoder.verify_speaker(audio_path, str(person.voiceprint_path))
        if result["score"] > best_score:
            best_score = result["score"]
            best_match = person

    is_match = best_score >= VERIFICATION_THRESHOLD

    return {
        "best_match": best_match.to_dict() if best_match else None,
        "score": round(best_score, 4),
        "is_match": is_match,
    }


def identify_speaker_from_base64(audio_b64):
    """Identify speaker from base64 audio against all enrolled persons.

    Args:
        audio_b64: Base64 encoded audio string

    Returns:
        Dict with best_match (Person dict or None), score, is_match
    """
    persons = _load_persons()

    if not persons:
        return {"best_match": None, "score": 0.0, "is_match": False}

    best_match = None
    best_score = 0.0

    for person in persons:
        if not person.is_enrolled():
            continue

        result = voice_encoder.verify_from_base64(audio_b64, str(person.voiceprint_path))
        if result["score"] > best_score:
            best_score = result["score"]
            best_match = person

    is_match = best_score >= VERIFICATION_THRESHOLD

    return {
        "best_match": best_match.to_dict() if best_match else None,
        "score": round(best_score, 4),
        "is_match": is_match,
    }
