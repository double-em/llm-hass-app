"""Voice line cache store for TTS caching."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from voice_cache import VoiceLineCache


class VoiceCacheStore:
    """Manages voice line cache entries stored in /data/voice_cache/."""

    def __init__(self, data_dir: str = "/data"):
        self.data_dir = Path(data_dir)
        self.cache_dir = self.data_dir / "voice_cache"
        self.index_file = self.cache_dir / "index.json"
        self._ensure_dirs()

    def _ensure_dirs(self):
        """Ensure required directories exist."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if not self.index_file.exists():
            self._save_index({})

    def _load_index(self) -> dict:
        """Load cache index."""
        with open(self.index_file) as f:
            return json.load(f)

    def _save_index(self, index: dict):
        """Save cache index."""
        with open(self.index_file, "w") as f:
            json.dump(index, f, indent=2)

    def _timestamp(self) -> str:
        """Get current ISO timestamp."""
        return datetime.now(timezone.utc).isoformat()

    def _get_cache_file(self, text_hash: str) -> Path:
        """Get path to cache file for a text hash."""
        return self.cache_dir / f"{text_hash}.json"

    def _load_entry(self, text_hash: str) -> Optional[VoiceLineCache]:
        """Load a cache entry by text hash."""
        cache_file = self._get_cache_file(text_hash)
        if not cache_file.exists():
            return None
        with open(cache_file) as f:
            data = json.load(f)
        return VoiceLineCache(**data)

    def _save_entry(self, entry: VoiceLineCache):
        """Save a cache entry."""
        cache_file = self._get_cache_file(entry.text_hash)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(entry.to_dict(), f, indent=2)

    def save(
        self,
        text_hash: str,
        text: str,
        audio_blob: Optional[bytes] = None,
        provider: str = "omnivoice",
        voice_name: Optional[str] = None,
        speed: float = 1.0,
        num_steps: int = 32,
        audio_format: str = "wav",
        sample_rate: int = 16000,
        duration_ms: int = 0,
    ) -> VoiceLineCache:
        """Save a voice line to cache.

        Args:
            text_hash: SHA256 hash of the input text.
            text: Original input text.
            audio_blob: Synthesized audio bytes (optional).
            provider: TTS provider used.
            voice_name: Voice preset name (if any).
            speed: Speaking rate.
            num_steps: Diffusion steps.
            audio_format: Audio format (wav, mp3, etc).
            sample_rate: Audio sample rate in Hz.
            duration_ms: Audio duration in milliseconds.

        Returns:
            Created VoiceLineCache entry.
        """
        entry = VoiceLineCache(
            text_hash=text_hash,
            text=text,
            audio_blob=audio_blob,
            provider=provider,
            voice_name=voice_name,
            speed=speed,
            num_steps=num_steps,
            audio_format=audio_format,
            sample_rate=sample_rate,
            duration_ms=duration_ms,
        )
        self._save_entry(entry)

        # Update index
        index = self._load_index()
        index[text_hash] = {
            "text": text,
            "provider": provider,
            "voice_name": voice_name,
            "created_at": entry.created_at.isoformat(),
        }
        self._save_index(index)

        return entry

    def get(self, text_hash: str, include_audio: bool = False) -> Optional[VoiceLineCache]:
        """Get a cache entry by text hash.

        Args:
            text_hash: SHA256 hash of the input text.
            include_audio: Whether to include audio_blob in returned entry.

        Returns:
            VoiceLineCache entry or None if not found.
        """
        entry = self._load_entry(text_hash)
        if entry:
            entry.access()
            self._save_entry(entry)
            if not include_audio:
                entry.audio_blob = None
        return entry

    def find(
        self,
        text: Optional[str] = None,
        provider: Optional[str] = None,
        voice_name: Optional[str] = None,
        limit: int = 50,
    ) -> list[VoiceLineCache]:
        """Find cache entries matching criteria.

        Args:
            text: Text substring to match (optional).
            provider: Filter by provider (optional).
            voice_name: Filter by voice name (optional).
            limit: Maximum entries to return.

        Returns:
            List of matching VoiceLineCache entries.
        """
        index = self._load_index()
        matches = []

        for text_hash, info in index.items():
            if text and text not in info.get("text", ""):
                continue
            if provider and info.get("provider") != provider:
                continue
            if voice_name and info.get("voice_name") != voice_name:
                continue

            entry = self._load_entry(text_hash)
            if entry:
                matches.append(entry)

        # Sort by last_accessed descending (most recent first)
        matches.sort(key=lambda e: e.last_accessed, reverse=True)
        return matches[:limit]

    def invalidate(self, text_hash: str) -> bool:
        """Remove a cache entry by text hash.

        Args:
            text_hash: SHA256 hash of the input text.

        Returns:
            True if deleted, False if not found.
        """
        cache_file = self._get_cache_file(text_hash)
        if not cache_file.exists():
            return False

        cache_file.unlink()

        # Update index
        index = self._load_index()
        if text_hash in index:
            del index[text_hash]
            self._save_index(index)

        return True

    def clear_all(self) -> int:
        """Clear all cache entries.

        Returns:
            Number of entries cleared.
        """
        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
            count += 1

        self._save_index({})
        return count

    def get_stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dict with cache statistics.
        """
        index = self._load_index()
        total_entries = len(index)
        total_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.json"))

        return {
            "total_entries": total_entries,
            "total_size_bytes": total_size,
        }
