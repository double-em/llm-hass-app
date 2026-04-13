"""HA Assists integration for Home Assistant conversation agent."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

from logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class HAAssistsConfig:
    """Configuration for HA Assists integration."""
    enabled: bool = False
    ha_url: str = ""
    ha_token: str = ""
    agent_id: str = "llm_ai"
    capabilities: list = None

    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = ["tts", "stt", "conversation"]


class HAAssistsClient:
    """Client for Home Assistant Assist pipeline integration."""

    def __init__(self, data_dir: str = "/data"):
        self.data_dir = Path(data_dir)
        self.config_file = self.data_dir / "ha_assists_config.json"
        self._config: Optional[HAAssistsConfig] = None
        self._permission_error = False
        self._load_config()

    def _load_config(self):
        """Load HA Assists configuration."""
        try:
            if self.config_file.exists():
                with open(self.config_file) as f:
                    data = json.load(f)
                    self._config = HAAssistsConfig(**data)
            else:
                self._config = HAAssistsConfig()
        except (PermissionError, FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Could not read {self.config_file}: {e}. Using defaults.")
            self._config = HAAssistsConfig()

    def _save_config(self):
        """Save HA Assists configuration."""
        if self._permission_error:
            return
        try:
            with open(self.config_file, "w") as f:
                json.dump({
                    "enabled": self._config.enabled,
                    "ha_url": self._config.ha_url,
                    "ha_token": self._config.ha_token,
                    "agent_id": self._config.agent_id,
                    "capabilities": self._config.capabilities,
                }, f, indent=2)
        except PermissionError as e:
            logger.warning(f"Could not write {self.config_file}: {e}. Config changes will not persist.")
            self._permission_error = True

    @property
    def config(self) -> HAAssistsConfig:
        """Get current config."""
        return self._config

    def update_config(
        self,
        enabled: Optional[bool] = None,
        ha_url: Optional[str] = None,
        ha_token: Optional[str] = None,
        agent_id: Optional[str] = None,
        capabilities: Optional[list] = None
    ) -> HAAssistsConfig:
        """Update configuration.

        Returns:
            Updated config.
        """
        if enabled is not None:
            self._config.enabled = enabled
        if ha_url is not None:
            self._config.ha_url = ha_url
        if ha_token is not None:
            self._config.ha_token = ha_token
        if agent_id is not None:
            self._config.agent_id = agent_id
        if capabilities is not None:
            self._config.capabilities = capabilities

        self._save_config()
        return self._config

    def test_connection(self) -> dict:
        """Test HA connection.

        Returns:
            Dict with success status and message.
        """
        if not self._config.ha_url or not self._config.ha_token:
            return {"success": False, "error": "HA URL and token required"}

        try:
            headers = {
                "Authorization": f"Bearer {self._config.ha_token}",
                "Content-Type": "application/json",
            }
            response = requests.get(
                f"{self._config.ha_url}/api/",
                headers=headers,
                timeout=10
            )

            if response.status_code == 200:
                return {"success": True, "message": "Connection successful"}
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}",
                    "details": response.text[:200]
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def process_assist_pipeline(
        self,
        text: str,
        conversation_id: Optional[str] = None,
        language: str = "en"
    ) -> dict:
        """Process an assist pipeline request.

        Args:
            text: Input text/intent.
            conversation_id: Optional conversation ID for context.
            language: Language code (en, zh, etc.).

        Returns:
            Dict with response text.
        """
        if not self._config.enabled:
            return {"error": "HA Assists integration not enabled"}

        try:
            headers = {
                "Authorization": f"Bearer {self._config.ha_token}",
                "Content-Type": "application/json",
            }

            payload = {
                "intent": text,
                "agent_id": self._config.agent_id,
                "conversation_id": conversation_id or "",
                "language": language,
                "text": text,
            }

            response = requests.post(
                f"{self._config.ha_url}/api/conversation/process",
                headers=headers,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                return {"text": result.get("text", "")}
            else:
                return {
                    "error": f"HTTP {response.status_code}",
                    "details": response.text[:200]
                }

        except Exception as e:
            return {"error": str(e)}