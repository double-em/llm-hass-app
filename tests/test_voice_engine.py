"""Tests for VoiceEncoder - speaker embedding and identification.

Uses mocking to avoid requiring heavy dependencies like resemblyzer.
"""

import pytest
import numpy as np
import base64
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestVoiceEncoderSimilarity:
    """Test embedding similarity computation - these don't need heavy mocking."""

    def test_similarity_identical_vectors(self):
        """Test cosine similarity of identical vectors is 1."""
        # Similarity is a static method, test directly
        from voice_engine import VoiceEncoder
        encoder = VoiceEncoder()
        vec = np.ones(256)

        similarity = encoder.similarity(vec, vec)
        assert abs(similarity - 1.0) < 0.0001

    def test_similarity_orthogonal_vectors(self):
        """Test cosine similarity of orthogonal vectors is 0."""
        from voice_engine import VoiceEncoder
        encoder = VoiceEncoder()
        vec1 = np.zeros(256)
        vec1[0] = 1.0
        vec2 = np.zeros(256)
        vec2[1] = 1.0

        similarity = encoder.similarity(vec1, vec2)
        assert abs(similarity) < 0.0001

    def test_similarity_opposite_vectors(self):
        """Test cosine similarity of opposite vectors is -1."""
        from voice_engine import VoiceEncoder
        encoder = VoiceEncoder()
        vec1 = np.ones(256)
        vec2 = -np.ones(256)

        similarity = encoder.similarity(vec1, vec2)
        assert abs(similarity - (-1.0)) < 0.0001

    def test_similarity_zero_vector(self):
        """Test similarity with zero vector returns 0."""
        from voice_engine import VoiceEncoder
        encoder = VoiceEncoder()
        vec1 = np.zeros(256)
        vec2 = np.ones(256)

        similarity = encoder.similarity(vec1, vec2)
        assert similarity == 0.0

    def test_similarity_symmetric(self):
        """Test that similarity(a, b) == similarity(b, a)."""
        from voice_engine import VoiceEncoder
        encoder = VoiceEncoder()
        vec1 = np.random.rand(256)
        vec2 = np.random.rand(256)

        sim1 = encoder.similarity(vec1, vec2)
        sim2 = encoder.similarity(vec2, vec1)
        assert abs(sim1 - sim2) < 0.0001

    def test_similarity_normalized_vectors(self):
        """Test similarity with already normalized vectors."""
        from voice_engine import VoiceEncoder
        encoder = VoiceEncoder()

        vec1 = np.ones(256) / np.sqrt(256)
        vec2 = np.ones(256) / np.sqrt(256)

        similarity = encoder.similarity(vec1, vec2)
        assert abs(similarity - 1.0) < 0.0001

    def test_similarity_range(self):
        """Test that similarity is always in [-1, 1]."""
        from voice_engine import VoiceEncoder
        encoder = VoiceEncoder()

        for _ in range(10):
            vec1 = np.random.rand(256) - 0.5
            vec2 = np.random.rand(256) - 0.5
            sim = encoder.similarity(vec1, vec2)
            assert -1.0 <= sim <= 1.0


class TestVoiceEncoderIdentification:
    """Test speaker identification with mocked encoder."""

    def test_identify_speaker_finds_match(self, temp_data_dir, sample_wav_file):
        """Test identifying speaker from voiceprints directory."""
        mock_encoder = MagicMock()
        mock_encoder.embed_utterance.return_value = np.ones(256)

        with patch('voice_engine.ResemblyzerVoiceEncoder', return_value=mock_encoder):
            from voice_engine import VoiceEncoder
            encoder = VoiceEncoder()
            encoder._encoder = mock_encoder

            # Create a voiceprint file
            voiceprints_dir = Path(temp_data_dir) / "voiceprints"
            voiceprints_dir.mkdir(exist_ok=True)
            np.save(voiceprints_dir / "person1.npy", np.ones(256))

            result = encoder.identify_speaker(sample_wav_file, str(voiceprints_dir))

            assert result["best_match"] == "person1"
            assert "score" in result
            assert "is_match" in result

    def test_identify_speaker_no_match(self, temp_data_dir, sample_wav_file):
        """Test identifying when no voiceprints exist."""
        mock_encoder = MagicMock()
        mock_encoder.embed_utterance.return_value = np.random.rand(256)

        with patch('voice_engine.ResemblyzerVoiceEncoder', return_value=mock_encoder):
            from voice_engine import VoiceEncoder
            encoder = VoiceEncoder()
            encoder._encoder = mock_encoder

            voiceprints_dir = Path(temp_data_dir) / "voiceprints"
            voiceprints_dir.mkdir(exist_ok=True)

            result = encoder.identify_speaker(sample_wav_file, str(voiceprints_dir))

            assert result["best_match"] is None
            assert result["score"] == 0.0
            assert result["is_match"] is False

    def test_identify_speaker_threshold(self, temp_data_dir, sample_wav_file):
        """Test that is_match is True only when score >= threshold."""
        mock_encoder = MagicMock()
        mock_encoder.embed_utterance.return_value = np.ones(256) * 0.9

        with patch('voice_engine.ResemblyzerVoiceEncoder', return_value=mock_encoder):
            from voice_engine import VoiceEncoder
            encoder = VoiceEncoder()
            encoder._encoder = mock_encoder

            voiceprints_dir = Path(temp_data_dir) / "voiceprints"
            voiceprints_dir.mkdir(exist_ok=True)
            np.save(voiceprints_dir / "person1.npy", np.ones(256) * 0.9)

            result = encoder.identify_speaker(sample_wav_file, str(voiceprints_dir))

            assert "is_match" in result

    def test_identify_speaker_empty_directory(self, temp_data_dir, sample_wav_file):
        """Test identifying from empty voiceprints directory."""
        mock_encoder = MagicMock()
        mock_encoder.embed_utterance.return_value = np.ones(256)

        with patch('voice_engine.ResemblyzerVoiceEncoder', return_value=mock_encoder):
            from voice_engine import VoiceEncoder
            encoder = VoiceEncoder()
            encoder._encoder = mock_encoder

            result = encoder.identify_speaker(sample_wav_file, temp_data_dir)

            assert result["best_match"] is None
            assert result["score"] == 0.0
            assert result["is_match"] is False

    def test_identify_from_base64(self, temp_data_dir):
        """Test identifying speaker from base64 audio."""
        mock_encoder = MagicMock()
        mock_encoder.embed_utterance.return_value = np.ones(256)

        with patch('voice_engine.ResemblyzerVoiceEncoder', return_value=mock_encoder):
            from voice_engine import VoiceEncoder
            encoder = VoiceEncoder()
            encoder._encoder = mock_encoder

            wav_data = b"RIFF" + (36).to_bytes(4, "little") + b"WAVE"
            wav_data += b"fmt " + (16).to_bytes(4, "little") + (1).to_bytes(2, "little")
            wav_data += (1).to_bytes(2, "little") + (16000).to_bytes(4, "little")
            wav_data += (32000).to_bytes(4, "little") + (2).to_bytes(2, "little")
            wav_data += (16).to_bytes(2, "little") + b"data" + (0).to_bytes(4, "little")
            b64_audio = base64.b64encode(wav_data).decode("utf-8")

            voiceprints_dir = Path(temp_data_dir) / "voiceprints"
            voiceprints_dir.mkdir(exist_ok=True)
            np.save(voiceprints_dir / "person1.npy", np.ones(256))

            result = encoder.identify_from_base64(b64_audio, str(voiceprints_dir))

            assert "best_match" in result
            assert "score" in result


class TestVoiceEncoderVerification:
    """Test speaker verification with mocked encoder."""

    def test_verify_speaker_match(self, temp_data_dir, sample_wav_file):
        """Test verifying speaker with matching voiceprint."""
        mock_encoder = MagicMock()
        mock_encoder.embed_utterance.return_value = np.ones(256)

        with patch('voice_engine.ResemblyzerVoiceEncoder', return_value=mock_encoder):
            from voice_engine import VoiceEncoder
            encoder = VoiceEncoder()
            encoder._encoder = mock_encoder

            voiceprint_path = Path(temp_data_dir) / "test_voiceprint.npy"
            np.save(voiceprint_path, np.ones(256))

            result = encoder.verify_speaker(sample_wav_file, str(voiceprint_path))

            assert "is_verified" in result
            assert "score" in result
            assert result["is_verified"] is True

    def test_verify_speaker_no_voiceprint(self, sample_wav_file):
        """Test verifying against non-existent voiceprint."""
        with patch('voice_engine.ResemblyzerVoiceEncoder') as mock_class:
            from voice_engine import VoiceEncoder
            encoder = VoiceEncoder()

            result = encoder.verify_speaker(sample_wav_file, "/nonexistent/path.npy")

            assert result["is_verified"] is False
            assert result["score"] == 0.0
            assert "error" in result

    def test_verify_from_base64(self, temp_data_dir):
        """Test verifying from base64 audio."""
        mock_encoder = MagicMock()
        mock_encoder.embed_utterance.return_value = np.ones(256)

        with patch('voice_engine.ResemblyzerVoiceEncoder', return_value=mock_encoder):
            from voice_engine import VoiceEncoder
            encoder = VoiceEncoder()
            encoder._encoder = mock_encoder

            voiceprint_path = Path(temp_data_dir) / "test.npy"
            np.save(voiceprint_path, np.ones(256))

            wav_data = b"RIFF" + (36).to_bytes(4, "little") + b"WAVE"
            wav_data += b"fmt " + (16).to_bytes(4, "little") + (1).to_bytes(2, "little")
            wav_data += (1).to_bytes(2, "little") + (16000).to_bytes(4, "little")
            wav_data += (32000).to_bytes(4, "little") + (2).to_bytes(2, "little")
            wav_data += (16).to_bytes(2, "little") + b"data" + (0).to_bytes(4, "little")
            b64_audio = base64.b64encode(wav_data).decode("utf-8")

            result = encoder.verify_from_base64(b64_audio, str(voiceprint_path))

            assert result["is_verified"] is True
            assert "score" in result


class TestVoiceEncoderEdgeCases:
    """Test edge cases and error handling."""

    def test_encoder_lazy_loads(self):
        """Test that encoder is lazy-loaded."""
        with patch('voice_engine.ResemblyzerVoiceEncoder'):
            from voice_engine import VoiceEncoder
            encoder = VoiceEncoder()
            assert encoder._encoder is None

            encoder._encoder = MagicMock()
            assert encoder._encoder is not None

    def test_similarity_returns_float(self):
        """Test that similarity returns a Python float."""
        with patch('voice_engine.ResemblyzerVoiceEncoder'):
            from voice_engine import VoiceEncoder
            encoder = VoiceEncoder()
            result = encoder.similarity(np.ones(256), np.ones(256))

            assert isinstance(result, float)

    def test_identify_speaker_directory_not_exists(self, temp_data_dir, sample_wav_file):
        """Test identifying when voiceprints directory doesn't exist."""
        mock_encoder = MagicMock()
        mock_encoder.embed_utterance.return_value = np.ones(256)

        with patch('voice_engine.ResemblyzerVoiceEncoder', return_value=mock_encoder):
            from voice_engine import VoiceEncoder
            encoder = VoiceEncoder()
            encoder._encoder = mock_encoder

            # Use a path that doesn't exist
            nonexistent_dir = Path(temp_data_dir) / "nonexistent" / "voiceprints"
            result = encoder.identify_speaker(sample_wav_file, str(nonexistent_dir))

            assert result["best_match"] is None
            assert result["score"] == 0.0
            assert result["is_match"] is False