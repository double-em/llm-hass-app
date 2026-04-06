"""Embedding engine using sentence-transformers."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Default model
DEFAULT_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


class EmbeddingEngine:
    """Sentence-transformers embedding engine for vector memory."""

    _instance = None
    _model = None

    def __new__(cls, model_name: str = DEFAULT_MODEL):
        """Singleton pattern for embedding model."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_name: str = DEFAULT_MODEL):
        """Initialize embedding engine.

        Args:
            model_name: HuggingFace model name for embeddings.
        """
        if self._initialized and self._model is not None:
            return

        self.model_name = model_name
        self._load_model()
        self._initialized = True

    def _load_model(self):
        """Load sentence-transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Vector memory will use mock embeddings."
            )
            self._model = None

    def encode(self, text: str) -> list:
        """Encode text to embedding vector.

        Args:
            text: Text to encode.

        Returns:
            384-dimensional embedding vector as list.
        """
        if self._model is None:
            # Return mock embedding for testing
            return list(np.random.randn(EMBEDDING_DIM).astype(float))

        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def encode_batch(self, texts: list) -> list:
        """Encode multiple texts to embeddings.

        Args:
            texts: List of texts to encode.

        Returns:
            List of embedding vectors.
        """
        if self._model is None:
            # Return mock embeddings for testing
            return [
                list(np.random.randn(EMBEDDING_DIM).astype(float))
                for _ in texts
            ]

        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return [emb.tolist() for emb in embeddings]

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return EMBEDDING_DIM

    def is_available(self) -> bool:
        """Check if embedding model is loaded."""
        return self._model is not None