"""Enrollment manager for voiceprint enrollment flow."""

import json
import uuid
from pathlib import Path


class EnrollmentManager:
    """Manages voice enrollment sessions."""

    MIN_SAMPLES = 3
    RECOMMENDED_SAMPLES = 5

    def __init__(self, data_dir: str = "/data"):
        self.data_dir = Path(data_dir)
        self.enrollments_dir = self.data_dir / "enrollments"
        self._ensure_dirs()

    def _ensure_dirs(self):
        """Ensure required directories exist."""
        self.enrollments_dir.mkdir(parents=True, exist_ok=True)

    def _get_enrollment_path(self, enrollment_id: str) -> Path:
        """Get path for enrollment state file."""
        return self.enrollments_dir / f"{enrollment_id}.json"

    def _load_enrollment(self, enrollment_id: str) -> dict | None:
        """Load enrollment state."""
        path = self._get_enrollment_path(enrollment_id)
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    def _save_enrollment(self, enrollment: dict):
        """Save enrollment state."""
        path = self._get_enrollment_path(enrollment["id"])
        with open(path, "w") as f:
            json.dump(enrollment, f, indent=2)

    def _timestamp(self) -> str:
        """Get current ISO timestamp."""
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()

    def start_enrollment(self, person_id: str) -> str:
        """Start a new enrollment session.

        Args:
            person_id: Person UUID to enroll

        Returns:
            Enrollment ID
        """
        enrollment_id = str(uuid.uuid4())
        enrollment = {
            "id": enrollment_id,
            "person_id": person_id,
            "status": "in_progress",
            "samples": [],
            "created_at": self._timestamp(),
            "updated_at": self._timestamp(),
        }
        self._save_enrollment(enrollment)
        return enrollment_id

    def add_sample(self, enrollment_id: str, audio_path: str) -> dict:
        """Add a voice sample to enrollment.

        Args:
            enrollment_id: Enrollment session ID
            audio_path: Path to the audio file

        Returns:
            Dict with sample_count and enrollment status

        Raises:
            ValueError: If enrollment not found or already completed
        """
        enrollment = self._load_enrollment(enrollment_id)
        if not enrollment:
            raise ValueError(f"Enrollment '{enrollment_id}' not found")

        if enrollment["status"] != "in_progress":
            raise ValueError(f"Enrollment is '{enrollment['status']}', cannot add samples")

        sample = {
            "id": str(uuid.uuid4()),
            "path": audio_path,
            "created_at": self._timestamp(),
        }
        enrollment["samples"].append(sample)
        enrollment["updated_at"] = self._timestamp()
        self._save_enrollment(enrollment)

        return {
            "sample_count": len(enrollment["samples"]),
            "min_required": self.MIN_SAMPLES,
            "recommended": self.RECOMMENDED_SAMPLES,
            "status": enrollment["status"],
        }

    def complete_enrollment(self, enrollment_id: str) -> dict:
        """Complete enrollment and trigger voiceprint extraction.

        Args:
            enrollment_id: Enrollment session ID

        Returns:
            Dict with enrollment results

        Raises:
            ValueError: If enrollment not found or insufficient samples
        """
        enrollment = self._load_enrollment(enrollment_id)
        if not enrollment:
            raise ValueError(f"Enrollment '{enrollment_id}' not found")

        if len(enrollment["samples"]) < self.MIN_SAMPLES:
            raise ValueError(
                f"Need at least {self.MIN_SAMPLES} samples, have {len(enrollment['samples'])}"
            )

        enrollment["status"] = "completed"
        enrollment["completed_at"] = self._timestamp()
        self._save_enrollment(enrollment)

        return {
            "enrollment_id": enrollment_id,
            "person_id": enrollment["person_id"],
            "sample_count": len(enrollment["samples"]),
            "status": enrollment["status"],
        }

    def cancel_enrollment(self, enrollment_id: str) -> bool:
        """Cancel and delete an enrollment session.

        Args:
            enrollment_id: Enrollment session ID

        Returns:
            True if cancelled, False if not found
        """
        enrollment = self._load_enrollment(enrollment_id)
        if not enrollment:
            return False

        # Delete sample files
        for sample in enrollment.get("samples", []):
            sample_path = Path(sample["path"])
            if sample_path.exists():
                sample_path.unlink()

        # Delete enrollment file
        path = self._get_enrollment_path(enrollment_id)
        path.unlink()
        return True

    def get_enrollment(self, enrollment_id: str) -> dict | None:
        """Get enrollment status.

        Args:
            enrollment_id: Enrollment session ID

        Returns:
            Enrollment dict or None if not found
        """
        return self._load_enrollment(enrollment_id)

    def get_enrollment_for_person(self, person_id: str) -> dict | None:
        """Get the most recent enrollment for a person.

        Args:
            person_id: Person UUID

        Returns:
            Enrollment dict or None if not found
        """
        if not self.enrollments_dir.exists():
            return None

        for path in self.enrollments_dir.glob("*.json"):
            with open(path) as f:
                enrollment = json.load(f)
                if enrollment["person_id"] == person_id:
                    return enrollment
        return None