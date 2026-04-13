"""Person store for managing person records and voice samples."""

import json
import logging
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)


class PersonStore:
    """Manages person records in /data/persons.json."""

    def __init__(self, data_dir: str = "/data"):
        self.data_dir = Path(data_dir)
        self.persons_file = self.data_dir / "persons.json"
        self._permission_error = False
        self._ensure_data_dir()

    def _ensure_data_dir(self):
        """Ensure data directory exists."""
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            if not self.persons_file.exists():
                self._save([])
        except PermissionError as e:
            logger.warning(f"Could not create {self.persons_file}: {e}. Using in-memory fallback.")
            self._permission_error = True

    def _load(self) -> list:
        """Load persons from JSON file."""
        try:
            with open(self.persons_file) as f:
                return json.load(f)
        except (PermissionError, FileNotFoundError) as e:
            logger.warning(f"Could not read {self.persons_file}: {e}. Returning empty list.")
            return []

    def _save(self, persons: list):
        """Save persons to JSON file."""
        if self._permission_error:
            return
        try:
            with open(self.persons_file, "w") as f:
                json.dump(persons, f, indent=2)
        except PermissionError as e:
            logger.warning(f"Could not write {self.persons_file}: {e}. Changes will not persist.")
            self._permission_error = True

    def create_person(self, name: str) -> dict:
        """Create a new person record.

        Args:
            name: Person's display name

        Returns:
            Created person dict
        """
        persons = self._load()
        person = {
            "id": str(uuid.uuid4()),
            "name": name,
            "created_at": self._timestamp(),
            "voice_samples": [],
            "voiceprint_path": None,
            "enrollment_complete": False,
        }
        persons.append(person)
        self._save(persons)
        return person

    def get_person(self, person_id: str) -> dict | None:
        """Get a person by ID.

        Args:
            person_id: Person UUID

        Returns:
            Person dict or None if not found
        """
        persons = self._load()
        for person in persons:
            if person["id"] == person_id:
                return person
        return None

    def list_persons(self) -> list:
        """List all persons.

        Returns:
            List of person dicts
        """
        return self._load()

    def delete_person(self, person_id: str) -> bool:
        """Delete a person and their voice samples.

        Args:
            person_id: Person UUID

        Returns:
            True if deleted, False if not found
        """
        persons = self._load()
        original_len = len(persons)
        persons = [p for p in persons if p["id"] != person_id]

        if len(persons) == original_len:
            return False

        # Delete voice sample files
        person = self.get_person(person_id)
        if person:
            for sample in person.get("voice_samples", []):
                sample_path = Path(sample["path"])
                if sample_path.exists():
                    sample_path.unlink()

            # Delete voiceprint if exists
            if person.get("voiceprint_path"):
                vp_path = Path(person["voiceprint_path"])
                if vp_path.exists():
                    vp_path.unlink()

        self._save(persons)
        return True

    def update_person(self, person_id: str, updates: dict) -> dict | None:
        """Update a person record.

        Args:
            person_id: Person UUID
            updates: Dict of fields to update

        Returns:
            Updated person dict or None if not found
        """
        persons = self._load()
        for i, person in enumerate(persons):
            if person["id"] == person_id:
                persons[i].update(updates)
                self._save(persons)
                return persons[i]
        return None

    def add_voice_sample(self, person_id: str, audio_path: str, transcript: str = None) -> dict | None:
        """Add a voice sample to a person.

        Args:
            person_id: Person UUID
            audio_path: Path to the audio file
            transcript: Optional transcript of the audio

        Returns:
            Updated person dict or None if person not found
        """
        persons = self._load()
        for i, person in enumerate(persons):
            if person["id"] == person_id:
                sample = {
                    "id": str(uuid.uuid4()),
                    "path": audio_path,
                    "transcript": transcript,
                    "created_at": self._timestamp(),
                }
                person["voice_samples"].append(sample)
                persons[i] = person
                self._save(persons)
                return person
        return None

    def get_voice_samples(self, person_id: str) -> list:
        """Get voice samples for a person.

        Args:
            person_id: Person UUID

        Returns:
            List of voice sample dicts
        """
        person = self.get_person(person_id)
        if not person:
            return []
        return person.get("voice_samples", [])

    def _timestamp(self) -> str:
        """Get current ISO timestamp."""
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()