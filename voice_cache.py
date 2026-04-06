"""Voice line cache data model for TTS caching."""

import dataclasses
from datetime import datetime
from typing import Optional


@dataclasses.dataclass
class VoiceLineCache:
    """Cache entry for a synthesized voice line.

    Attributes:
        text_hash: SHA256 hash of the input text.
        text: Original input text that was synthesized.
        audio_blob: The synthesized audio bytes (optional, for memory-cached retrieval).
        provider: TTS provider used (e.g., "omnivoice", "minimax").
        voice_name: Name of the voice preset used (if any).
        speed: Speaking rate used.
        num_steps: Diffusion steps used.
        created_at: Timestamp when the cache entry was created.
        last_accessed: Timestamp of last access.
        usage_count: Number of times this cache entry was retrieved.
        audio_format: Audio format (e.g., "wav", "mp3").
        sample_rate: Audio sample rate in Hz.
        duration_ms: Audio duration in milliseconds.
    """
    text_hash: str
    text: str
    audio_blob: Optional[bytes] = None
    provider: str = "omnivoice"
    voice_name: Optional[str] = None
    speed: float = 1.0
    num_steps: int = 32
    created_at: datetime = dataclasses.field(default_factory=datetime.utcnow)
    last_accessed: datetime = dataclasses.field(default_factory=datetime.utcnow)
    usage_count: int = 0
    audio_format: str = "wav"
    sample_rate: int = 16000
    duration_ms: int = 0

    def access(self) -> None:
        """Update last_accessed and increment usage_count."""
        self.last_accessed = datetime.utcnow()
        self.usage_count += 1

    @property
    def cache_key(self) -> str:
        """Generate a unique cache key for this entry."""
        return self.text_hash

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "text_hash": self.text_hash,
            "text": self.text,
            "provider": self.provider,
            "voice_name": self.voice_name,
            "speed": self.speed,
            "num_steps": self.num_steps,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "usage_count": self.usage_count,
            "audio_format": self.audio_format,
            "sample_rate": self.sample_rate,
            "duration_ms": self.duration_ms,
        }
