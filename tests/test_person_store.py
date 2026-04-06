"""Tests for PersonStore - person CRUD operations."""

import pytest
import json
import os
from pathlib import Path


class TestPersonStoreCRUD:
    """Test person record CRUD operations."""

    def test_create_person_returns_person_dict(self, person_store):
        """Test creating a person returns correct dict structure."""
        person = person_store.create_person("Test Person")

        assert person is not None
        assert "id" in person
        assert person["name"] == "Test Person"
        assert "created_at" in person
        assert person["voice_samples"] == []
        assert person["voiceprint_path"] is None
        assert person["enrollment_complete"] is False

    def test_create_person_id_is_unique(self, person_store):
        """Test that created persons have unique IDs."""
        person1 = person_store.create_person("Person 1")
        person2 = person_store.create_person("Person 2")

        assert person1["id"] != person2["id"]

    def test_get_person_exists(self, person_store):
        """Test getting an existing person."""
        created = person_store.create_person("Get Test")
        retrieved = person_store.get_person(created["id"])

        assert retrieved is not None
        assert retrieved["id"] == created["id"]
        assert retrieved["name"] == "Get Test"

    def test_get_person_not_exists(self, person_store):
        """Test getting a non-existent person returns None."""
        result = person_store.get_person("nonexistent-id-123")
        assert result is None

    def test_list_persons_empty(self, person_store):
        """Test listing persons when none exist."""
        persons = person_store.list_persons()
        assert persons == []

    def test_list_persons_multiple(self, person_store):
        """Test listing multiple persons."""
        person_store.create_person("Person A")
        person_store.create_person("Person B")
        person_store.create_person("Person C")

        persons = person_store.list_persons()
        assert len(persons) == 3
        names = [p["name"] for p in persons]
        assert "Person A" in names
        assert "Person B" in names
        assert "Person C" in names

    def test_delete_person_exists(self, person_store):
        """Test deleting an existing person."""
        person = person_store.create_person("Delete Test")
        result = person_store.delete_person(person["id"])

        assert result is True
        assert person_store.get_person(person["id"]) is None

    def test_delete_person_not_exists(self, person_store):
        """Test deleting non-existent person returns False."""
        result = person_store.delete_person("nonexistent-id-123")
        assert result is False

    def test_update_person_exists(self, person_store):
        """Test updating a person."""
        person = person_store.create_person("Original Name")
        updated = person_store.update_person(person["id"], {"name": "Updated Name"})

        assert updated is not None
        assert updated["name"] == "Updated Name"
        assert person_store.get_person(person["id"])["name"] == "Updated Name"

    def test_update_person_not_exists(self, person_store):
        """Test updating non-existent person returns None."""
        result = person_store.update_person("nonexistent-id-123", {"name": "New Name"})
        assert result is None

    def test_update_person_multiple_fields(self, person_store):
        """Test updating multiple fields at once."""
        person = person_store.create_person("Multi Update")
        updates = {
            "name": "New Name",
            "enrollment_complete": True,
            "voiceprint_path": "/data/voiceprints/test.npy"
        }
        updated = person_store.update_person(person["id"], updates)

        assert updated["name"] == "New Name"
        assert updated["enrollment_complete"] is True
        assert updated["voiceprint_path"] == "/data/voiceprints/test.npy"


class TestPersonStoreVoiceSamples:
    """Test voice sample management."""

    def test_add_voice_sample(self, person_store, sample_wav_file):
        """Test adding a voice sample to a person."""
        person = person_store.create_person("Sample Test")
        updated = person_store.add_voice_sample(person["id"], sample_wav_file, "Test transcript")

        assert updated is not None
        assert len(updated["voice_samples"]) == 1
        assert updated["voice_samples"][0]["path"] == sample_wav_file
        assert updated["voice_samples"][0]["transcript"] == "Test transcript"

    def test_add_voice_sample_without_transcript(self, person_store, sample_wav_file):
        """Test adding a voice sample without transcript."""
        person = person_store.create_person("No Transcript Test")
        updated = person_store.add_voice_sample(person["id"], sample_wav_file)

        assert updated is not None
        assert len(updated["voice_samples"]) == 1
        assert updated["voice_samples"][0]["transcript"] is None

    def test_add_voice_sample_multiple(self, person_store, sample_wav_file):
        """Test adding multiple voice samples."""
        person = person_store.create_person("Multi Sample Test")

        for i in range(5):
            person_store.add_voice_sample(person["id"], sample_wav_file, f"Transcript {i}")

        updated = person_store.get_person(person["id"])
        assert len(updated["voice_samples"]) == 5

    def test_add_voice_sample_person_not_found(self, person_store, sample_wav_file):
        """Test adding sample to non-existent person returns None."""
        result = person_store.add_voice_sample("nonexistent-id", sample_wav_file)
        assert result is None

    def test_get_voice_samples_empty(self, person_store):
        """Test getting voice samples when none exist."""
        person = person_store.create_person("No Samples")
        samples = person_store.get_voice_samples(person["id"])

        assert samples == []

    def test_get_voice_samples_exists(self, person_store, sample_wav_file):
        """Test getting voice samples for a person."""
        person = person_store.create_person("Has Samples")
        person_store.add_voice_sample(person["id"], sample_wav_file, "First")
        person_store.add_voice_sample(person["id"], sample_wav_file, "Second")

        samples = person_store.get_voice_samples(person["id"])
        assert len(samples) == 2
        assert samples[0]["transcript"] == "First"
        assert samples[1]["transcript"] == "Second"

    def test_get_voice_samples_person_not_found(self, person_store):
        """Test getting samples for non-existent person returns empty list."""
        samples = person_store.get_voice_samples("nonexistent-id")
        assert samples == []


class TestPersonStorePersistence:
    """Test data persistence across instances."""

    def test_data_persists_across_instances(self, temp_data_dir):
        """Test that person data persists across PersonStore instances."""
        from person_store import PersonStore

        store1 = PersonStore(data_dir=temp_data_dir)
        store1.create_person("Persistent Person")

        store2 = PersonStore(data_dir=temp_data_dir)
        persons = store2.list_persons()

        assert len(persons) == 1
        assert persons[0]["name"] == "Persistent Person"

    def test_person_json_file_created(self, temp_data_dir):
        """Test that persons.json file is created."""
        from person_store import PersonStore

        PersonStore(data_dir=temp_data_dir).create_person("File Test")

        persons_file = Path(temp_data_dir) / "persons.json"
        assert persons_file.exists()

        with open(persons_file) as f:
            data = json.load(f)
            assert len(data) == 1
            assert data[0]["name"] == "File Test"


class TestPersonStoreEdgeCases:
    """Test edge cases and error conditions."""

    def test_create_person_empty_name(self, person_store):
        """Test creating person with empty name (edge case)."""
        # Empty name might be invalid - current implementation may or may not allow it
        person = person_store.create_person("")
        assert person["name"] == ""

    def test_create_person_unicode_name(self, person_store):
        """Test creating person with unicode characters."""
        person = person_store.create_person("日本語テスト")
        assert person["name"] == "日本語テスト"

    def test_id_is_string(self, person_store):
        """Test that person IDs are strings."""
        person = person_store.create_person("ID Test")
        assert isinstance(person["id"], str)

    def test_created_at_is_iso_format(self, person_store):
        """Test that created_at timestamp is in ISO format."""
        person = person_store.create_person("Timestamp Test")
        # Should contain T and end with Z (UTC)
        assert "T" in person["created_at"]
        assert person["created_at"].endswith("Z") or "+" in person["created_at"] or "-" in person["created_at"][-6:]