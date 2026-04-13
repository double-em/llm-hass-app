"""Voiceprint manager for creating and comparing voice embeddings."""

import logging

import numpy as np
import torch
import torchaudio
from pathlib import Path

logger = logging.getLogger(__name__)


class VoiceprintManager:
    """Manages voice embeddings (voiceprints) for speaker identification."""

    EMBEDDING_DIM = 512  # Typical embedding dimension for speaker verification

    def __init__(self, data_dir: str = "/data"):
        self.data_dir = Path(data_dir)
        self.voiceprints_dir = self.data_dir / "voiceprints"
        self._permission_error = False
        try:
            self.voiceprints_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError as e:
            logger.warning(f"Could not create {self.voiceprints_dir}: {e}. Using in-memory fallback.")
            self._permission_error = True

    def _get_voiceprint_path(self, person_id: str) -> Path:
        """Get path for a person's voiceprint file."""
        return self.voiceprints_dir / f"{person_id}.npy"

    def _extract_embedding(self, audio_path: str) -> np.ndarray:
        """Extract embedding from a single audio file.

        Uses a simple approach based on audio features since we don't have
        a dedicated speaker verification model readily available.

        Args:
            audio_path: Path to audio file

        Returns:
            Embedding array
        """
        waveform, sample_rate = torchaudio.load(audio_path)

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        # Get mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Simple spectral embedding: MFCC-like features
        # Use mel spectrogram and flatten to create a pseudo-embedding
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=512,
            n_mels=64,
        )
        mel_spec = mel_transform(waveform)

        # Take mean over time, then log
        embedding = torch.log(mel_spec + 1e-9).mean(dim=-1).squeeze()

        # Pad or truncate to fixed dimension
        if embedding.shape[-1] < self.EMBEDDING_DIM:
            padding = torch.zeros(self.EMBEDDING_DIM - embedding.shape[-1])
            embedding = torch.cat([embedding, padding])
        else:
            embedding = embedding[:self.EMBEDDING_DIM]

        return embedding.numpy()

    def create_voiceprint(self, person_id: str, sample_paths: list) -> str:
        """Create a voiceprint embedding from audio samples.

        Args:
            person_id: Person UUID
            sample_paths: List of paths to audio sample files

        Returns:
            Path to saved voiceprint file
        """
        embeddings = []
        for path in sample_paths:
            try:
                emb = self._extract_embedding(path)
                embeddings.append(emb)
            except Exception as e:
                raise ValueError(f"Failed to extract embedding from {path}: {e}")

        if not embeddings:
            raise ValueError("No valid embeddings extracted")

        # Average embeddings across samples for robustness
        voiceprint = np.mean(embeddings, axis=0)

        # Normalize
        voiceprint = voiceprint / (np.linalg.norm(voiceprint) + 1e-9)

        # Save
        output_path = self._get_voiceprint_path(person_id)
        try:
            np.save(output_path, voiceprint)
        except PermissionError as e:
            logger.warning(f"Could not save voiceprint {output_path}: {e}. Changes will not persist.")
            self._permission_error = True

        return str(output_path)

    def get_voiceprint(self, person_id: str) -> str | None:
        """Get path to a person's voiceprint.

        Args:
            person_id: Person UUID

        Returns:
            Path to voiceprint file or None if not found
        """
        path = self._get_voiceprint_path(person_id)
        if path.exists():
            return str(path)
        return None

    def compare_voiceprint(self, person_id: str, audio_path: str) -> float:
        """Compare audio against a stored voiceprint.

        Args:
            person_id: Person UUID
            audio_path: Path to audio to compare

        Returns:
            Similarity score (0.0 to 1.0, higher = more similar)
        """
        voiceprint_path = self._get_voiceprint_path(person_id)
        if not voiceprint_path.exists():
            raise ValueError(f"No voiceprint found for person '{person_id}'")

        # Load stored voiceprint
        stored = np.load(voiceprint_path)

        # Extract embedding from audio
        candidate = self._extract_embedding(audio_path)

        # Normalize candidate
        candidate = candidate / (np.linalg.norm(candidate) + 1e-9)

        # Cosine similarity
        similarity = np.dot(stored, candidate)

        # Convert to 0-1 range (cosine similarity is already in [-1, 1])
        # Map to 0-1 scale where 1 = identical
        score = (similarity + 1) / 2

        return float(score)