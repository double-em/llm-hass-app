"""Conversation session store for AI Memory."""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class SessionStore:
    """Manages conversation sessions stored in /data/sessions/."""

    def __init__(self, data_dir: str = "/data"):
        self.data_dir = Path(data_dir)
        self.sessions_dir = self.data_dir / "sessions"
        self.index_file = self.sessions_dir / "index.json"
        self._permission_error = False
        self._ensure_dirs()

    def _ensure_dirs(self):
        """Ensure required directories exist."""
        try:
            self.sessions_dir.mkdir(parents=True, exist_ok=True)
            if not self.index_file.exists():
                self._save_index({})
        except PermissionError as e:
            logger.warning(f"Could not create {self.sessions_dir}: {e}. Using in-memory fallback.")
            self._permission_error = True

    def _load_index(self) -> dict:
        """Load session index."""
        try:
            with open(self.index_file) as f:
                return json.load(f)
        except (PermissionError, FileNotFoundError) as e:
            logger.warning(f"Could not read {self.index_file}: {e}. Returning empty index.")
            return {}

    def _save_index(self, index: dict):
        """Save session index."""
        if self._permission_error:
            return
        try:
            with open(self.index_file, "w") as f:
                json.dump(index, f, indent=2)
        except PermissionError as e:
            logger.warning(f"Could not write {self.index_file}: {e}. Changes will not persist.")
            self._permission_error = True

    def _timestamp(self) -> str:
        """Get current ISO timestamp."""
        return datetime.now(timezone.utc).isoformat()

    def create_session(
        self,
        name: Optional[str] = None,
        provider: str = "minimax"
    ) -> dict:
        """Create a new conversation session.

        Args:
            name: Optional user-friendly name. Auto-generated if not provided.
            provider: AI provider name for this session.

        Returns:
            Created session dict with session_id, name, provider, etc.
        """
        session_id = str(uuid.uuid4())
        now = self._timestamp()

        if not name:
            name = f"Session {session_id[:8]}"

        session = {
            "session_id": session_id,
            "name": name,
            "provider": provider,
            "created_at": now,
            "updated_at": now,
            "message_count": 0,
            "last_message_at": None,
        }

        # Save session file
        session_file = self.sessions_dir / f"{session_id}.json"
        try:
            with open(session_file, "w") as f:
                json.dump(session, f, indent=2)
        except PermissionError as e:
            logger.warning(f"Could not write session {session_file}: {e}.")

        # Update index
        index = self._load_index()
        index[session_id] = {
            "name": name,
            "provider": provider,
            "created_at": now,
        }
        self._save_index(index)

        return session

    def get_session(self, session_id: str) -> Optional[dict]:
        """Get a session by ID.

        Args:
            session_id: Session UUID.

        Returns:
            Session dict or None if not found.
        """
        session_file = self.sessions_dir / f"{session_id}.json"
        if not session_file.exists():
            return None
        try:
            with open(session_file) as f:
                return json.load(f)
        except (PermissionError, FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Could not read session {session_file}: {e}.")
            return None

    def list_sessions(
        self,
        limit: int = 50,
        offset: int = 0
    ) -> list:
        """List all sessions (newest first).

        Args:
            limit: Maximum number of sessions to return.
            offset: Number of sessions to skip.

        Returns:
            List of session dicts.
        """
        index = self._load_index()
        sessions = []
        for session_id in index.keys():
            session = self.get_session(session_id)
            if session:
                sessions.append(session)

        # Sort by updated_at descending
        sessions.sort(key=lambda s: s.get("updated_at", ""), reverse=True)
        return sessions[offset:offset + limit]

    def update_session(self, session_id: str, updates: dict) -> Optional[dict]:
        """Update a session.

        Args:
            session_id: Session UUID.
            updates: Dict of fields to update.

        Returns:
            Updated session dict or None if not found.
        """
        session = self.get_session(session_id)
        if not session:
            return None

        session.update(updates)
        session["updated_at"] = self._timestamp()

        session_file = self.sessions_dir / f"{session_id}.json"
        try:
            with open(session_file, "w") as f:
                json.dump(session, f, indent=2)
        except PermissionError as e:
            logger.warning(f"Could not write session {session_file}: {e}.")

        # Update index if name changed
        if "name" in updates:
            index = self._load_index()
            if session_id in index:
                index[session_id]["name"] = updates["name"]
                self._save_index(index)

        return session

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and its messages.

        Args:
            session_id: Session UUID.

        Returns:
            True if deleted, False if not found.
        """
        session_file = self.sessions_dir / f"{session_id}.json"
        if not session_file.exists():
            return False

        # Delete session file
        try:
            session_file.unlink()
        except PermissionError as e:
            logger.warning(f"Could not delete {session_file}: {e}.")

        # Delete associated messages file
        messages_file = self.sessions_dir / f"{session_id}_messages.json"
        if messages_file.exists():
            messages_file.unlink()

        # Update index
        index = self._load_index()
        if session_id in index:
            del index[session_id]
            self._save_index(index)

        return True

    def increment_message_count(self, session_id: str) -> Optional[dict]:
        """Increment message count and update timestamp.

        Args:
            session_id: Session UUID.

        Returns:
            Updated session dict or None if not found.
        """
        session = self.get_session(session_id)
        if not session:
            return None

        session["message_count"] = session.get("message_count", 0) + 1
        session["last_message_at"] = self._timestamp()
        session["updated_at"] = self._timestamp()

        session_file = self.sessions_dir / f"{session_id}.json"
        try:
            with open(session_file, "w") as f:
                json.dump(session, f, indent=2)
        except PermissionError as e:
            logger.warning(f"Could not write session {session_file}: {e}.")

        # Update index
        index = self._load_index()
        if session_id in index:
            index[session_id]["last_message_at"] = session["last_message_at"]
            self._save_index(index)

        return session

    def get_session_count(self) -> int:
        """Get total number of sessions.

        Returns:
            Number of sessions.
        """
        index = self._load_index()
        return len(index)