"""Voice encoding and speaker embedding using Resemblyzer."""

import base64
import io
import os
import tempfile
from pathlib import Path

import numpy as np
import torchaudio
from resemblyzer import VoiceEncoder

from logging_config import get_logger

logger = get_logger(__name__)

# Similarity threshold for speaker identification
IDENTIFICATION_THRESHOLD = 0.75


class VoiceEncoder:
    """Voice encoder using Resemblyzer for speaker embeddings."""

    def __init__(self):
        """Initialize the voice encoder."""
        self._encoder = None

    @property
    def encoder(self):
        """Lazy-load the encoder."""
        if self._encoder is None:
            logger.info("Loading Resemblyzer voice encoder...")
            self._encoder = VoiceEncoder()
            logger.info("Resemblyzer voice encoder loaded")
        return self._encoder

    def _load_audio(self, audio_path):
        """Load audio file and return waveform.

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (waveform, sample_rate)
        """
        waveform, sample_rate = torchaudio.load(audio_path)

        # Resemblyzer expects single channel, convert if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample to 16000 Hz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        # Flatten to 1D for Resemblyzer
        waveform = waveform.squeeze().numpy()
        return waveform

    def _audio_from_base64(self, audio_b64):
        """Decode base64 audio to waveform.

        Args:
            audio_b64: Base64 encoded audio bytes

        Returns:
            Waveform as numpy array at 16kHz
        """
        audio_bytes = base64.b64decode(audio_b64)

        # Try loading as WAV from bytes
        try:
            waveform, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))

            # Convert stereo to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Resample to 16000 Hz
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                waveform = resampler(waveform)

            return waveform.squeeze().numpy()
        except Exception:
            # Fall back to raw bytes as WAV
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio_bytes)
                f.flush()
                waveform, sample_rate = torchaudio.load(f.name)
                os.unlink(f.name)

                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)

                if sample_rate != 16000:
                    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                    waveform = resampler(waveform)

                return waveform.squeeze().numpy()

    def encode_voice(self, audio_path):
        """Extract speaker embedding from an audio file.

        Args:
            audio_path: Path to audio file (WAV recommended)

        Returns:
            Embedding vector as numpy array
        """
        waveform = self._load_audio(audio_path)
        embedding = self.encoder.embed_utterance(waveform)
        return embedding

    def encode_from_base64(self, audio_b64):
        """Extract speaker embedding from base64 audio.

        Args:
            audio_b64: Base64 encoded audio string

        Returns:
            Embedding vector as numpy array
        """
        waveform = self._audio_from_base64(audio_b64)
        embedding = self.encoder.embed_utterance(waveform)
        return embedding

    def encode_from_file_or_base64(self, audio_source):
        """Extract speaker embedding from file path or base64.

        Args:
            audio_source: Either a file path (str/Path) or base64 audio string

        Returns:
            Embedding vector as numpy array
        """
        if isinstance(audio_source, (str, Path)) and Path(audio_source).exists():
            return self.encode_voice(audio_source)
        elif isinstance(audio_source, str):
            # Assume base64 if doesn't exist as file
            return self.encode_from_base64(audio_source)
        else:
            raise ValueError("audio_source must be a valid file path or base64 string")

    @staticmethod
    def similarity(embedding1, embedding2):
        """Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score (0 to 1)
        """
        # Normalize vectors
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        # Cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(similarity)

    def identify_speaker(self, audio_path, voiceprints_dir):
        """Identify the speaker from an audio file against known voiceprints.

        Args:
            audio_path: Path to audio file to identify
            voiceprints_dir: Directory containing .npy voiceprint files

        Returns:
            Dict with best_match (filename without ext), score, and is_match bool
        """
        # Extract embedding from input audio
        embedding = self.encode_voice(audio_path)

        voiceprints_dir = Path(voiceprints_dir)
        if not voiceprints_dir.exists():
            return {"best_match": None, "score": 0.0, "is_match": False}

        best_match = None
        best_score = 0.0

        for voiceprint_file in voiceprints_dir.glob("*.npy"):
            voiceprint = np.load(voiceprint_file)
            score = self.similarity(embedding, voiceprint)

            if score > best_score:
                best_score = score
                best_match = voiceprint_file.stem

        is_match = best_score >= IDENTIFICATION_THRESHOLD

        return {
            "best_match": best_match,
            "score": round(best_score, 4),
            "is_match": is_match,
        }

    def identify_from_base64(self, audio_b64, voiceprints_dir):
        """Identify speaker from base64 audio.

        Args:
            audio_b64: Base64 encoded audio string
            voiceprints_dir: Directory containing .npy voiceprint files

        Returns:
            Dict with best_match, score, and is_match
        """
        embedding = self.encode_from_base64(audio_b64)

        voiceprints_dir = Path(voiceprints_dir)
        if not voiceprints_dir.exists():
            return {"best_match": None, "score": 0.0, "is_match": False}

        best_match = None
        best_score = 0.0

        for voiceprint_file in voiceprints_dir.glob("*.npy"):
            voiceprint = np.load(voiceprint_file)
            score = self.similarity(embedding, voiceprint)

            if score > best_score:
                best_score = score
                best_match = voiceprint_file.stem

        is_match = best_score >= IDENTIFICATION_THRESHOLD

        return {
            "best_match": best_match,
            "score": round(best_score, 4),
            "is_match": is_match,
        }

    def verify_speaker(self, audio_path, voiceprint_path):
        """Verify if audio matches a specific voiceprint.

        Args:
            audio_path: Path to audio file to verify
            voiceprint_path: Path to reference voiceprint (.npy file)

        Returns:
            Dict with is_verified bool and score
        """
        if not Path(voiceprint_path).exists():
            return {"is_verified": False, "score": 0.0, "error": "Voiceprint not found"}

        embedding = self.encode_voice(audio_path)
        voiceprint = np.load(voiceprint_path)
        score = self.similarity(embedding, voiceprint)

        return {
            "is_verified": score >= IDENTIFICATION_THRESHOLD,
            "score": round(score, 4),
        }

    def verify_from_base64(self, audio_b64, voiceprint_path):
        """Verify if base64 audio matches a specific voiceprint.

        Args:
            audio_b64: Base64 encoded audio string
            voiceprint_path: Path to reference voiceprint (.npy file)

        Returns:
            Dict with is_verified bool and score
        """
        if not Path(voiceprint_path).exists():
            return {"is_verified": False, "score": 0.0, "error": "Voiceprint not found"}

        embedding = self.encode_from_base64(audio_b64)
        voiceprint = np.load(voiceprint_path)
        score = self.similarity(embedding, voiceprint)

        return {
            "is_verified": score >= IDENTIFICATION_THRESHOLD,
            "score": round(score, 4),
        }


# Global voice encoder instance
voice_encoder = VoiceEncoder()
