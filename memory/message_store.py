"""Message store for conversation messages."""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class MessageStore:
    """Manages conversation messages stored per session."""

    def __init__(self, data_dir: str = "/data"):
        self.data_dir = Path(data_dir)
        self.sessions_dir = self.data_dir / "sessions"
        self._permission_error = False

    def _timestamp(self) -> str:
        """Get current ISO timestamp."""
        return datetime.now(timezone.utc).isoformat()

    def _get_messages_file(self, session_id: str) -> Path:
        """Get path to messages file for a session."""
        return self.sessions_dir / f"{session_id}_messages.json"

    def _load_messages(self, session_id: str) -> list:
        """Load messages for a session."""
        messages_file = self._get_messages_file(session_id)
        if not messages_file.exists():
            return []
        try:
            with open(messages_file) as f:
                return json.load(f)
        except (PermissionError, FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Could not read {messages_file}: {e}. Returning empty list.")
            return []

    def _save_messages(self, session_id: str, messages: list):
        """Save messages for a session."""
        if self._permission_error:
            return
        messages_file = self._get_messages_file(session_id)
        try:
            messages_file.parent.mkdir(parents=True, exist_ok=True)
            with open(messages_file, "w") as f:
                json.dump(messages, f, indent=2)
        except PermissionError as e:
            logger.warning(f"Could not write {messages_file}: {e}. Changes will not persist.")
            self._permission_error = True

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[dict] = None
    ) -> dict:
        """Add a message to a session.

        Args:
            session_id: Session UUID.
            role: Message role ("user", "assistant", "system").
            content: Message content.
            metadata: Optional metadata (model, tokens, etc.).

        Returns:
            Created message dict.
        """
        message = {
            "message_id": str(uuid.uuid4()),
            "session_id": session_id,
            "role": role,
            "content": content,
            "timestamp": self._timestamp(),
            "metadata": metadata or {},
        }

        messages = self._load_messages(session_id)
        messages.append(message)
        self._save_messages(session_id, messages)

        return message

    def get_messages(
        self,
        session_id: str,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> list:
        """Get messages for a session.

        Args:
            session_id: Session UUID.
            limit: Maximum messages to return (None for all).
            offset: Number of messages to skip.

        Returns:
            List of message dicts.
        """
        messages = self._load_messages(session_id)

        # Sort by timestamp ascending (chronological order)
        messages.sort(key=lambda m: m.get("timestamp", ""))

        if offset > 0:
            messages = messages[offset:]
        if limit is not None:
            messages = messages[:limit]

        return messages

    def get_message_count(self, session_id: str) -> int:
        """Get number of messages in a session.

        Args:
            session_id: Session UUID.

        Returns:
            Number of messages.
        """
        messages = self._load_messages(session_id)
        return len(messages)

    def get_recent_messages(
        self,
        session_id: str,
        max_tokens: int = 4000
    ) -> list:
        """Get recent messages fitting within token budget.

        Args:
            session_id: Session UUID.
            max_tokens: Approximate max tokens (estimate ~4 chars per token).

        Returns:
            List of recent messages fitting token budget, oldest first.
        """
        messages = self._load_messages(session_id)
        messages.sort(key=lambda m: m.get("timestamp", ""))

        # Work backwards from the most recent
        selected = []
        total_chars = 0

        for msg in reversed(messages):
            msg_chars = len(msg.get("content", ""))
            if total_chars + msg_chars <= max_tokens * 4:
                selected.insert(0, msg)
                total_chars += msg_chars
            else:
                break

        return selected

    def delete_message(self, session_id: str, message_id: str) -> bool:
        """Delete a specific message.

        Args:
            session_id: Session UUID.
            message_id: Message UUID.

        Returns:
            True if deleted, False if not found.
        """
        messages = self._load_messages(session_id)
        original_len = len(messages)
        messages = [m for m in messages if m["message_id"] != message_id]

        if len(messages) == original_len:
            return False

        self._save_messages(session_id, messages)
        return True

    def clear_messages(self, session_id: str) -> bool:
        """Clear all messages for a session.

        Args:
            session_id: Session UUID.

        Returns:
            True if cleared, False if session not found.
        """
        messages_file = self._get_messages_file(session_id)
        if not messages_file.exists():
            return False

        try:
            messages_file.unlink()
        except PermissionError as e:
            logger.warning(f"Could not delete {messages_file}: {e}.")
            return False
        return True