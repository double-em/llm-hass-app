"""Tests for EnrollmentManager - session CRUD and history retrieval."""

import pytest
import json
import os
from pathlib import Path


class TestEnrollmentManagerCRUD:
    """Test enrollment session CRUD operations."""

    def test_start_enrollment_creates_session(self, enrollment_manager):
        """Test starting a new enrollment creates a session with correct fields."""
        person_id = "test-person-001"
        enrollment_id = enrollment_manager.start_enrollment(person_id)

        assert enrollment_id is not None
        assert len(enrollment_id) == 36  # UUID format

        enrollment = enrollment_manager.get_enrollment(enrollment_id)
        assert enrollment is not None
        assert enrollment["id"] == enrollment_id
        assert enrollment["person_id"] == person_id
        assert enrollment["status"] == "in_progress"
        assert enrollment["samples"] == []
        assert "created_at" in enrollment
        assert "updated_at" in enrollment

    def test_start_enrollment_multiple_sessions(self, enrollment_manager):
        """Test that multiple enrollment sessions can exist for different persons."""
        enrollment_id_1 = enrollment_manager.start_enrollment("person-1")
        enrollment_id_2 = enrollment_manager.start_enrollment("person-2")

        assert enrollment_id_1 != enrollment_id_2

        assert enrollment_manager.get_enrollment(enrollment_id_1)["person_id"] == "person-1"
        assert enrollment_manager.get_enrollment(enrollment_id_2)["person_id"] == "person-2"

    def test_add_sample_updates_enrollment(self, enrollment_manager, sample_wav_file):
        """Test adding a sample updates the enrollment correctly."""
        enrollment_id = enrollment_manager.start_enrollment("person-1")
        result = enrollment_manager.add_sample(enrollment_id, sample_wav_file)

        assert result["sample_count"] == 1
        assert result["status"] == "in_progress"
        assert result["min_required"] == 3
        assert result["recommended"] == 5

        enrollment = enrollment_manager.get_enrollment(enrollment_id)
        assert len(enrollment["samples"]) == 1
        assert enrollment["samples"][0]["path"] == sample_wav_file
        assert "id" in enrollment["samples"][0]
        assert "created_at" in enrollment["samples"][0]

    def test_add_sample_multiple_samples(self, enrollment_manager, sample_wav_file):
        """Test adding multiple samples counts correctly."""
        enrollment_id = enrollment_manager.start_enrollment("person-1")

        for i in range(5):
            result = enrollment_manager.add_sample(enrollment_id, sample_wav_file)
            assert result["sample_count"] == i + 1

        enrollment = enrollment_manager.get_enrollment(enrollment_id)
        assert len(enrollment["samples"]) == 5

    def test_add_sample_enrollment_not_found(self, enrollment_manager, sample_wav_file):
        """Test adding sample to non-existent enrollment raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            enrollment_manager.add_sample("fake-id-123", sample_wav_file)

    def test_add_sample_completed_enrollment_raises_error(self, enrollment_manager, sample_wav_file):
        """Test adding sample to completed enrollment raises ValueError."""
        enrollment_id = enrollment_manager.start_enrollment("person-1")

        # Add minimum samples and complete
        for _ in range(3):
            enrollment_manager.add_sample(enrollment_id, sample_wav_file)
        enrollment_manager.complete_enrollment(enrollment_id)

        # Try to add more samples
        with pytest.raises(ValueError, match="cannot add samples"):
            enrollment_manager.add_sample(enrollment_id, sample_wav_file)

    def test_complete_enrollment_success(self, enrollment_manager, sample_wav_file):
        """Test completing enrollment with sufficient samples."""
        enrollment_id = enrollment_manager.start_enrollment("person-1")

        for _ in range(3):
            enrollment_manager.add_sample(enrollment_id, sample_wav_file)

        result = enrollment_manager.complete_enrollment(enrollment_id)

        assert result["enrollment_id"] == enrollment_id
        assert result["person_id"] == "person-1"
        assert result["sample_count"] == 3
        assert result["status"] == "completed"

        enrollment = enrollment_manager.get_enrollment(enrollment_id)
        assert enrollment["status"] == "completed"
        assert "completed_at" in enrollment

    def test_complete_enrollment_insufficient_samples(self, enrollment_manager, sample_wav_file):
        """Test completing enrollment with insufficient samples raises ValueError."""
        enrollment_id = enrollment_manager.start_enrollment("person-1")
        enrollment_manager.add_sample(enrollment_id, sample_wav_file)

        with pytest.raises(ValueError, match="Need at least 3 samples"):
            enrollment_manager.complete_enrollment(enrollment_id)

    def test_complete_enrollment_not_found(self, enrollment_manager):
        """Test completing non-existent enrollment raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            enrollment_manager.complete_enrollment("fake-id-123")

    def test_cancel_enrollment_deletes_session(self, enrollment_manager, sample_wav_file):
        """Test cancelling enrollment removes the session."""
        enrollment_id = enrollment_manager.start_enrollment("person-1")
        enrollment_manager.add_sample(enrollment_id, sample_wav_file)

        result = enrollment_manager.cancel_enrollment(enrollment_id)
        assert result is True

        assert enrollment_manager.get_enrollment(enrollment_id) is None

    def test_cancel_enrollment_not_found(self, enrollment_manager):
        """Test cancelling non-existent enrollment returns False."""
        result = enrollment_manager.cancel_enrollment("fake-id-123")
        assert result is False

    def test_get_enrollment_not_found(self, enrollment_manager):
        """Test getting non-existent enrollment returns None."""
        assert enrollment_manager.get_enrollment("fake-id-123") is None


class TestEnrollmentManagerHistory:
    """Test enrollment history retrieval."""

    def test_get_enrollment_for_person_finds_active(self, enrollment_manager, sample_wav_file):
        """Test finding active enrollment for a person."""
        person_id = "person-with-enrollment"
        enrollment_id = enrollment_manager.start_enrollment(person_id)
        enrollment_manager.add_sample(enrollment_id, sample_wav_file)

        enrollment = enrollment_manager.get_enrollment_for_person(person_id)

        assert enrollment is not None
        assert enrollment["id"] == enrollment_id
        assert enrollment["person_id"] == person_id
        assert enrollment["status"] == "in_progress"

    def test_get_enrollment_for_person_not_enrolled(self, enrollment_manager):
        """Test getting enrollment for person with no enrollment returns None."""
        enrollment = enrollment_manager.get_enrollment_for_person("person-without-enrollment")
        assert enrollment is None

    def test_get_enrollment_for_person_completed_enrollment(self, enrollment_manager, sample_wav_file):
        """Test that completed enrollments can still be retrieved."""
        person_id = "person-completed"
        enrollment_id = enrollment_manager.start_enrollment(person_id)

        for _ in range(3):
            enrollment_manager.add_sample(enrollment_id, sample_wav_file)
        enrollment_manager.complete_enrollment(enrollment_id)

        enrollment = enrollment_manager.get_enrollment_for_person(person_id)
        assert enrollment is not None
        assert enrollment["status"] == "completed"

    def test_enrollment_persistence(self, temp_data_dir, sample_wav_file):
        """Test that enrollments persist across manager instances."""
        from enrollment import EnrollmentManager

        enrollment_id = EnrollmentManager(data_dir=temp_data_dir).start_enrollment("persistent-person")
        EnrollmentManager(data_dir=temp_data_dir).add_sample(enrollment_id, sample_wav_file)

        # New manager instance should find the enrollment
        manager2 = EnrollmentManager(data_dir=temp_data_dir)
        enrollment = manager2.get_enrollment(enrollment_id)

        assert enrollment is not None
        assert len(enrollment["samples"]) == 1


class TestEnrollmentManagerEdgeCases:
    """Test edge cases and error conditions."""

    def test_start_enrollment_empty_person_id(self, enrollment_manager):
        """Test that empty person ID is accepted (edge case - might be invalid)."""
        # Empty string might be valid or invalid depending on requirements
        # Current implementation should handle it
        enrollment_id = enrollment_manager.start_enrollment("")
        assert enrollment_id is not None

    def test_timestamps_are_iso_format(self, enrollment_manager):
        """Test that timestamps are in ISO format."""
        enrollment_id = enrollment_manager.start_enrollment("person-1")
        enrollment = enrollment_manager.get_enrollment(enrollment_id)

        # Check ISO format (contains T and Z)
        assert "T" in enrollment["created_at"]
        assert enrollment["created_at"].endswith("Z")

    def test_updated_at_changes_on_sample_add(self, enrollment_manager, sample_wav_file):
        """Test that updated_at changes when sample is added."""
        enrollment_id = enrollment_manager.start_enrollment("person-1")
        original_updated = enrollment_manager.get_enrollment(enrollment_id)["updated_at"]

        import time
        time.sleep(0.01)  # Ensure different timestamp

        enrollment_manager.add_sample(enrollment_id, sample_wav_file)
        updated_enrollment = enrollment_manager.get_enrollment(enrollment_id)

        assert updated_enrollment["updated_at"] != original_updated