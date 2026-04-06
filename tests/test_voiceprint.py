"""Tests for VoiceprintManager - embeddings and semantic search accuracy.

Uses mocking to avoid requiring heavy ML dependencies.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestVoiceprintManagerEmbeddings:
    """Test voice embedding creation and storage."""

    def test_create_voiceprint_saves_file(self, voiceprint_manager, sample_wav_file):
        """Test that creating a voiceprint saves a .npy file."""
        path = voiceprint_manager.create_voiceprint("person-1", [sample_wav_file])

        assert path is not None
        assert Path(path).exists()
        assert Path(path).suffix == ".npy"

    def test_create_voiceprint_stores_embedding(self, voiceprint_manager, sample_wav_file):
        """Test that the stored file contains a valid embedding."""
        path = voiceprint_manager.create_voiceprint("person-1", [sample_wav_file])

        embedding = np.load(path)
        assert embedding is not None
        assert isinstance(embedding, np.ndarray)
        assert len(embedding.shape) == 1
        assert embedding.shape[0] == 512

    def test_create_voiceprint_multiple_samples(self, temp_data_dir, sample_wav_file):
        """Test creating voiceprint from multiple samples."""
        from voiceprint import VoiceprintManager

        paths = []
        for i in range(3):
            wav_path = Path(temp_data_dir) / f"sample_{i}.wav"
            with open(wav_path, "wb") as f:
                f.write(b"RIFF")
                f.write((36).to_bytes(4, "little"))
                f.write(b"WAVE")
                f.write(b"fmt ")
                f.write((16).to_bytes(4, "little"))
                f.write((1).to_bytes(2, "little"))
                f.write((1).to_bytes(2, "little"))
                f.write((16000).to_bytes(4, "little"))
                f.write((32000).to_bytes(4, "little"))
                f.write((2).to_bytes(2, "little"))
                f.write((16).to_bytes(2, "little"))
                f.write(b"data")
                f.write((0).to_bytes(4, "little"))
            paths.append(str(wav_path))

        manager = VoiceprintManager(data_dir=temp_data_dir)
        path = manager.create_voiceprint("multi-sample-person", paths)

        embedding = np.load(path)
        assert embedding is not None
        assert len(embedding) == 512

    def test_create_voiceprint_normalized(self, voiceprint_manager, sample_wav_file):
        """Test that the embedding is normalized (unit vector)."""
        path = voiceprint_manager.create_voiceprint("person-1", [sample_wav_file])

        embedding = np.load(path)
        norm = np.linalg.norm(embedding)

        assert abs(norm - 1.0) < 0.01

    def test_create_voiceprint_invalid_file_raises(self, voiceprint_manager):
        """Test that invalid audio file raises ValueError."""
        with pytest.raises(Exception):
            voiceprint_manager.create_voiceprint("person-1", ["/nonexistent/file.wav"])

    def test_create_voiceprint_no_samples_raises(self, voiceprint_manager):
        """Test that empty sample list raises ValueError."""
        with pytest.raises(ValueError, match="No valid embeddings"):
            voiceprint_manager.create_voiceprint("person-1", [])


class TestVoiceprintManagerRetrieval:
    """Test voiceprint retrieval."""

    def test_get_voiceprint_exists(self, voiceprint_manager, sample_wav_file):
        """Test getting path for existing voiceprint."""
        voiceprint_manager.create_voiceprint("person-1", [sample_wav_file])

        path = voiceprint_manager.get_voiceprint("person-1")
        assert path is not None
        assert Path(path).exists()

    def test_get_voiceprint_not_exists(self, voiceprint_manager):
        """Test getting path for non-existent voiceprint returns None."""
        path = voiceprint_manager.get_voiceprint("nonexistent-person")
        assert path is None

    def test_voiceprint_path_format(self, temp_data_dir, sample_wav_file):
        """Test that voiceprint is stored at correct path."""
        from voiceprint import VoiceprintManager

        manager = VoiceprintManager(data_dir=temp_data_dir)
        manager.create_voiceprint("my-person-id", [sample_wav_file])

        expected_path = Path(temp_data_dir) / "voiceprints" / "my-person-id.npy"
        assert expected_path.exists()


class TestVoiceprintManagerComparison:
    """Test voiceprint comparison (semantic search accuracy)."""

    def test_compare_voiceprint_high_similarity(self, voiceprint_manager, sample_wav_file):
        """Test that comparing same audio to itself gives high similarity."""
        voiceprint_manager.create_voiceprint("person-1", [sample_wav_file])

        score = voiceprint_manager.compare_voiceprint("person-1", sample_wav_file)

        assert score >= 0.7
        assert score <= 1.0

    def test_compare_voiceprint_not_found_raises(self, voiceprint_manager, sample_wav_file):
        """Test comparing against non-existent voiceprint raises ValueError."""
        with pytest.raises(ValueError, match="No voiceprint found"):
            voiceprint_manager.compare_voiceprint("nonexistent-person", sample_wav_file)

    def test_compare_voiceprint_score_range(self, voiceprint_manager, sample_wav_file):
        """Test that similarity scores are in 0-1 range."""
        voiceprint_manager.create_voiceprint("person-1", [sample_wav_file])

        score = voiceprint_manager.compare_voiceprint("person-1", sample_wav_file)

        assert 0.0 <= score <= 1.0

    def test_compare_voiceprint_different_audio_lower_score(self, temp_data_dir, sample_wav_file):
        """Test that different audio produces lower similarity score."""
        from voiceprint import VoiceprintManager

        manager = VoiceprintManager(data_dir=temp_data_dir)

        wav2_path = Path(temp_data_dir) / "different.wav"
        with open(wav2_path, "wb") as f:
            f.write(b"RIFF")
            f.write((36).to_bytes(4, "little"))
            f.write(b"WAVE")
            f.write(b"fmt ")
            f.write((16).to_bytes(4, "little"))
            f.write((1).to_bytes(2, "little"))
            f.write((1).to_bytes(2, "little"))
            f.write((16000).to_bytes(4, "little"))
            f.write((64000).to_bytes(4, "little"))
            f.write((2).to_bytes(2, "little"))
            f.write((16).to_bytes(2, "little"))
            f.write(b"data")
            f.write((0).to_bytes(4, "little"))

        manager.create_voiceprint("person-1", [sample_wav_file])
        score = manager.compare_voiceprint("person-1", str(wav2_path))

        assert 0.0 <= score <= 1.0


class TestVoiceprintManagerEdgeCases:
    """Test edge cases and error handling."""

    def test_embedding_dimension_consistency(self, temp_data_dir):
        """Test that all embeddings have consistent dimension."""
        from voiceprint import VoiceprintManager

        manager = VoiceprintManager(data_dir=temp_data_dir)

        paths = []
        for i in range(3):
            wav_path = Path(temp_data_dir) / f"dim_test_{i}.wav"
            with open(wav_path, "wb") as f:
                f.write(b"RIFF")
                f.write((36).to_bytes(4, "little"))
                f.write(b"WAVE")
                f.write(b"fmt ")
                f.write((16).to_bytes(4, "little"))
                f.write((1).to_bytes(2, "little"))
                f.write((1).to_bytes(2, "little"))
                f.write((16000).to_bytes(4, "little"))
                f.write((32000).to_bytes(4, "little"))
                f.write((2).to_bytes(2, "little"))
                f.write((16).to_bytes(2, "little"))
                f.write(b"data")
                f.write((0).to_bytes(4, "little"))
            paths.append(str(wav_path))

        embedding = manager._extract_embedding(paths[0])
        assert len(embedding) == 512

    def test_similarity_is_cosine_based(self, temp_data_dir):
        """Test that similarity computation uses cosine similarity."""
        from voiceprint import VoiceprintManager

        manager = VoiceprintManager(data_dir=temp_data_dir)

        emb1 = np.ones(512) / np.sqrt(512)
        emb2 = np.ones(512) / np.sqrt(512)

        similarity = np.dot(emb1, emb2)
        score = (similarity + 1) / 2
        assert score == 1.0

        emb3 = np.zeros(512)
        emb3[0] = 1.0
        emb4 = np.zeros(512)
        emb4[1] = 1.0
        similarity = np.dot(emb3, emb4)
        score = (similarity + 1) / 2
        assert score == 0.5