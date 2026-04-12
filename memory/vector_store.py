"""Vector memory store using ChromaDB for semantic search."""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from logging_config import get_logger

logger = get_logger(__name__)


class VectorStore:
    """ChromaDB-backed vector memory for semantic search."""

    def __init__(self, data_dir: str = "/data"):
        self.data_dir = Path(data_dir)
        self.vector_dir = self.data_dir / "vector_memory"
        self._ensure_dirs()
        self._chroma_client = None
        self._collection = None

    def _ensure_dirs(self):
        """Ensure required directories exist."""
        self.vector_dir.mkdir(parents=True, exist_ok=True)

    def _get_chroma_client(self):
        """Get or create ChromaDB client."""
        if self._chroma_client is None:
            try:
                import chromadb
                from chromadb.config import Settings
                self._chroma_client = chromadb.Client(
                    Settings(
                        persist_directory=str(self.vector_dir),
                        anonymized_telemetry=False
                    )
                )
            except ImportError:
                logger.warning("ChromaDB not installed. Vector search unavailable.")
                return None
        return self._chroma_client

    def _get_collection(self, collection_name: str = "memory"):
        """Get or create a ChromaDB collection."""
        client = self._get_chroma_client()
        if client is None:
            return None

        try:
            return client.get_or_create_collection(name=collection_name)
        except Exception:
            logger.exception("Failed to get collection")
            return None

    def _timestamp(self) -> str:
        """Get current ISO timestamp."""
        return datetime.now(timezone.utc).isoformat()

    def add_entry(
        self,
        content: str,
        embedding: list,
        tags: Optional[list] = None,
        source: str = "manual",
        related_session_id: Optional[str] = None,
        metadata: Optional[dict] = None
    ) -> dict:
        """Add a memory entry with embedding.

        Args:
            content: Text content to store.
            embedding: Pre-computed embedding vector.
            tags: Optional tags for categorization.
            source: Source of the entry ("conversation", "manual", "ha_event").
            related_session_id: Optional linked session ID.
            metadata: Additional metadata.

        Returns:
            Created entry dict with entry_id.
        """
        entry_id = str(uuid.uuid4())
        now = self._timestamp()

        entry = {
            "entry_id": entry_id,
            "content": content,
            "embedding": embedding,
            "tags": tags or [],
            "source": source,
            "related_session_id": related_session_id,
            "metadata": metadata or {},
            "created_at": now,
        }

        # Store in ChromaDB
        collection = self._get_collection()
        if collection:
            try:
                collection.add(
                    ids=[entry_id],
                    embeddings=[embedding],
                    documents=[content],
                    metadatas=[{
                        "tags": json.dumps(tags or []),
                        "source": source,
                        "related_session_id": related_session_id or "",
                        "created_at": now,
                    }]
                )
            except Exception:
                logger.exception("Failed to add to ChromaDB")

        # Save metadata to JSON sidecar
        metadata_file = self.vector_dir / f"{entry_id}.json"
        with open(metadata_file, "w") as f:
            json.dump(entry, f, indent=2)

        return entry

    def search(
        self,
        query_embedding: list,
        limit: int = 5,
        threshold: float = 0.7,
        tags: Optional[list] = None,
        source: Optional[str] = None
    ) -> list:
        """Search for similar entries.

        Args:
            query_embedding: Query embedding vector.
            limit: Maximum number of results.
            threshold: Minimum similarity score (0-1).
            tags: Optional filter by tags.
            source: Optional filter by source.

        Returns:
            List of matching entries with similarity scores.
        """
        collection = self._get_collection()
        if collection is None:
            return []

        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                include=["distances", "metadatas"]
            )

            matches = []
            for i, doc_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i]
                # Convert distance to similarity (ChromaDB uses cosine distance)
                similarity = 1.0 - distance

                if similarity >= threshold:
                    # Load full entry metadata
                    entry = self._load_entry(doc_id)
                    if entry:
                        # Apply tag filter if specified
                        if tags:
                            entry_tags = entry.get("tags", [])
                            if not any(tag in entry_tags for tag in tags):
                                continue

                        # Apply source filter if specified
                        if source and entry.get("source") != source:
                            continue

                        entry["similarity"] = round(similarity, 4)
                        matches.append(entry)

            return matches

        except Exception:
            logger.exception("Search failed")
            return []

    def delete_entry(self, entry_id: str) -> bool:
        """Delete a memory entry.

        Args:
            entry_id: Entry UUID.

        Returns:
            True if deleted, False if not found.
        """
        metadata_file = self.vector_dir / f"{entry_id}.json"
        if not metadata_file.exists():
            return False

        # Remove from ChromaDB
        collection = self._get_collection()
        if collection:
            try:
                collection.delete(ids=[entry_id])
            except Exception:
                logger.exception("Failed to delete from ChromaDB")

        # Remove metadata file
        metadata_file.unlink()
        return True

    def _load_entry(self, entry_id: str) -> Optional[dict]:
        """Load entry metadata."""
        metadata_file = self.vector_dir / f"{entry_id}.json"
        if not metadata_file.exists():
            return None
        with open(metadata_file) as f:
            return json.load(f)

    def list_entries(
        self,
        limit: int = 100,
        offset: int = 0,
        source: Optional[str] = None
    ) -> list:
        """List memory entries.

        Args:
            limit: Maximum entries to return.
            offset: Number to skip.
            source: Optional filter by source.

        Returns:
            List of entry dicts.
        """
        entries = []
        for metadata_file in self.vector_dir.glob("*.json"):
            try:
                with open(metadata_file) as f:
                    entry = json.load(f)
                    if source and entry.get("source") != source:
                        continue
                    entries.append(entry)
            except Exception:
                continue

        # Sort by created_at descending
        entries.sort(key=lambda e: e.get("created_at", ""), reverse=True)
        return entries[offset:offset + limit]

    def get_stats(self) -> dict:
        """Get vector memory statistics.

        Returns:
            Dict with total_entries, sources, tags, etc.
        """
        entries = self.list_entries(limit=10000)
        total = len(entries)

        sources = {}
        tags = set()
        for entry in entries:
            source = entry.get("source", "unknown")
            sources[source] = sources.get(source, 0) + 1
            tags.update(entry.get("tags", []))

        return {
            "total_entries": total,
            "sources": sources,
            "unique_tags": list(tags),
            "storage_size_bytes": self._get_storage_size(),
        }

    def _get_storage_size(self) -> int:
        """Get total storage size in bytes."""
        total = 0
        for f in self.vector_dir.glob("**/*"):
            if f.is_file():
                total += f.stat().st_size
        return total