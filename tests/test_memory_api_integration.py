"""Integration tests for AI Memory, Vector Memory, and HA Assists API endpoints.

Uses mocked external services to test the full pipeline end-to-end.
"""

import json
import tempfile
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

# Set testing environment before imports
os.environ["TESTING"] = "true"

from memory import SessionStore, MessageStore, VectorStore, EmbeddingEngine
from memory.ha_assists import HAAssistsClient, HAAssistsConfig


class TestSessionStoreAPI:
    """Test SessionStore API endpoints end-to-end."""

    def setup_method(self):
        """Set up test fixtures with isolated temp directories."""
        self.temp_dir = tempfile.mkdtemp()
        self.session_store = SessionStore(data_dir=self.temp_dir)
        self.message_store = MessageStore(data_dir=self.temp_dir)

    def test_create_session(self):
        """Test session creation."""
        session = self.session_store.create_session(name="Test Session", provider="minimax")

        assert session is not None
        assert "session_id" in session
        assert session["name"] == "Test Session"
        assert session["provider"] == "minimax"
        assert session["message_count"] == 0

    def test_get_session(self):
        """Test session retrieval."""
        created = self.session_store.create_session(name="Get Test")
        retrieved = self.session_store.get_session(created["session_id"])

        assert retrieved is not None
        assert retrieved["session_id"] == created["session_id"]
        assert retrieved["name"] == "Get Test"

    def test_get_nonexistent_session(self):
        """Test getting non-existent session returns None."""
        result = self.session_store.get_session("nonexistent-id")
        assert result is None

    def test_list_sessions(self):
        """Test session listing."""
        # Create multiple sessions
        self.session_store.create_session(name="Session 1")
        self.session_store.create_session(name="Session 2")
        self.session_store.create_session(name="Session 3")

        sessions = self.session_store.list_sessions(limit=10)
        assert len(sessions) == 3

    def test_list_sessions_with_pagination(self):
        """Test session listing with pagination."""
        for i in range(5):
            self.session_store.create_session(name=f"Page Session {i}")

        first_page = self.session_store.list_sessions(limit=2, offset=0)
        second_page = self.session_store.list_sessions(limit=2, offset=2)

        assert len(first_page) == 2
        assert len(second_page) == 2

    def test_update_session(self):
        """Test session update."""
        session = self.session_store.create_session(name="Original Name")
        updated = self.session_store.update_session(
            session["session_id"],
            {"name": "Updated Name"}
        )

        assert updated["name"] == "Updated Name"
        assert updated["session_id"] == session["session_id"]

    def test_delete_session(self):
        """Test session deletion."""
        session = self.session_store.create_session(name="To Delete")
        session_id = session["session_id"]

        # Add a message first
        self.message_store.add_message(session_id, "user", "Hello")

        result = self.session_store.delete_session(session_id)
        assert result is True

        # Verify deleted
        assert self.session_store.get_session(session_id) is None

    def test_increment_message_count(self):
        """Test message count increment."""
        session = self.session_store.create_session(name="Counter Test")

        for i in range(3):
            self.session_store.increment_message_count(session["session_id"])

        updated = self.session_store.get_session(session["session_id"])
        assert updated["message_count"] == 3

    def test_get_session_count(self):
        """Test session count."""
        self.session_store.create_session()
        self.session_store.create_session()
        self.session_store.create_session()

        count = self.session_store.get_session_count()
        assert count == 3


class TestMessageStoreAPI:
    """Test MessageStore API endpoints end-to-end."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.session_store = SessionStore(data_dir=self.temp_dir)
        self.message_store = MessageStore(data_dir=self.temp_dir)

    def test_add_message(self):
        """Test adding a message."""
        session = self.session_store.create_session()
        message = self.message_store.add_message(
            session_id=session["session_id"],
            role="user",
            content="Hello, world!"
        )

        assert message is not None
        assert message["role"] == "user"
        assert message["content"] == "Hello, world!"
        assert "message_id" in message
        assert "timestamp" in message

    def test_get_messages(self):
        """Test getting messages."""
        session = self.session_store.create_session()
        session_id = session["session_id"]

        self.message_store.add_message(session_id, "user", "First")
        self.message_store.add_message(session_id, "assistant", "Second")
        self.message_store.add_message(session_id, "user", "Third")

        messages = self.message_store.get_messages(session_id)
        assert len(messages) == 3

    def test_get_messages_with_limit(self):
        """Test getting messages with limit."""
        session = self.session_store.create_session()
        session_id = session["session_id"]

        for i in range(5):
            self.message_store.add_message(session_id, "user", f"Message {i}")

        messages = self.message_store.get_messages(session_id, limit=2)
        assert len(messages) == 2

    def test_get_messages_with_offset(self):
        """Test getting messages with offset."""
        session = self.session_store.create_session()
        session_id = session["session_id"]

        for i in range(4):
            self.message_store.add_message(session_id, "user", f"Message {i}")

        messages = self.message_store.get_messages(session_id, limit=2, offset=2)
        assert len(messages) == 2

    def test_get_message_count(self):
        """Test message count."""
        session = self.session_store.create_session()
        session_id = session["session_id"]

        self.message_store.add_message(session_id, "user", "1")
        self.message_store.add_message(session_id, "user", "2")
        self.message_store.add_message(session_id, "user", "3")

        count = self.message_store.get_message_count(session_id)
        assert count == 3

    def test_get_recent_messages(self):
        """Test getting recent messages within token budget."""
        session = self.session_store.create_session()
        session_id = session["session_id"]

        # Add messages with known lengths
        self.message_store.add_message(session_id, "user", "Short")
        self.message_store.add_message(session_id, "user", "Medium length message")
        self.message_store.add_message(session_id, "user", "A very long message that should be included in the context window")

        messages = self.message_store.get_recent_messages(session_id, max_tokens=1000)
        assert len(messages) > 0

    def test_delete_message(self):
        """Test deleting a message."""
        session = self.session_store.create_session()
        session_id = session["session_id"]

        msg = self.message_store.add_message(session_id, "user", "To delete")
        result = self.message_store.delete_message(session_id, msg["message_id"])

        assert result is True

    def test_clear_messages(self):
        """Test clearing all messages."""
        session = self.session_store.create_session()
        session_id = session["session_id"]

        self.message_store.add_message(session_id, "user", "1")
        self.message_store.add_message(session_id, "user", "2")

        result = self.message_store.clear_messages(session_id)
        assert result is True

        count = self.message_store.get_message_count(session_id)
        assert count == 0


class TestVectorStoreAPI:
    """Test VectorStore API endpoints end-to-end."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.vector_store = VectorStore(data_dir=self.temp_dir)

    @patch.object(VectorStore, '_get_chroma_client')
    def test_add_entry(self, mock_chroma):
        """Test adding a vector memory entry."""
        mock_collection = MagicMock()
        mock_chroma.return_value = MagicMock(get_or_create_collection=mock_collection)
        mock_collection.return_value = MagicMock()

        entry = self.vector_store.add_entry(
            content="Test memory content",
            embedding=[0.1] * 384,
            tags=["test", "integration"],
            source="test"
        )

        assert entry is not None
        assert "entry_id" in entry
        assert entry["content"] == "Test memory content"
        assert entry["tags"] == ["test", "integration"]

    @patch.object(VectorStore, '_get_chroma_client')
    def test_search(self, mock_chroma):
        """Test searching vector memory."""
        mock_collection = MagicMock()
        mock_chroma.return_value = MagicMock(get_or_create_collection=mock_collection)

        # Mock search results
        mock_collection.return_value.query.return_value = {
            "ids": [["entry1", "entry2"]],
            "distances": [[0.1, 0.3]],
            "metadatas": [[{"tags": "[]", "source": "test", "related_session_id": "", "created_at": "2024-01-01"}], []]
        }

        results = self.vector_store.search(
            query_embedding=[0.1] * 384,
            limit=5,
            threshold=0.7
        )

        assert isinstance(results, list)

    @patch.object(VectorStore, '_get_chroma_client')
    def test_delete_entry(self, mock_chroma):
        """Test deleting a vector memory entry."""
        mock_collection = MagicMock()
        mock_chroma.return_value = MagicMock(get_or_create_collection=mock_collection)
        mock_collection.return_value = MagicMock()

        # Add entry first
        entry = self.vector_store.add_entry(
            content="To delete",
            embedding=[0.1] * 384
        )

        # Delete it
        result = self.vector_store.delete_entry(entry["entry_id"])
        # The JSON file delete should work even if Chroma fails
        assert result is True

    @patch.object(VectorStore, '_get_chroma_client')
    def test_list_entries(self, mock_chroma):
        """Test listing vector memory entries."""
        mock_collection = MagicMock()
        mock_chroma.return_value = MagicMock(get_or_create_collection=mock_collection)
        mock_collection.return_value = MagicMock()

        # Add some entries
        for i in range(3):
            self.vector_store.add_entry(
                content=f"Memory {i}",
                embedding=[0.1] * 384,
                source="test"
            )

        entries = self.vector_store.list_entries(limit=10)
        assert len(entries) == 3

    @patch.object(VectorStore, '_get_chroma_client')
    def test_get_stats(self, mock_chroma):
        """Test getting vector memory stats."""
        mock_collection = MagicMock()
        mock_chroma.return_value = MagicMock(get_or_create_collection=mock_collection)
        mock_collection.return_value = MagicMock()

        self.vector_store.add_entry(
            content="Stat test 1",
            embedding=[0.1] * 384,
            tags=["test"],
            source="test"
        )
        self.vector_store.add_entry(
            content="Stat test 2",
            embedding=[0.1] * 384,
            source="manual"
        )

        stats = self.vector_store.get_stats()
        assert "total_entries" in stats
        assert "sources" in stats


class TestEmbeddingEngine:
    """Test EmbeddingEngine functionality."""

    def setup_method(self):
        """Reset singleton state for each test."""
        EmbeddingEngine._instance = None
        EmbeddingEngine._model = None

    def test_encode_with_mock_model(self):
        """Test encoding when model is not available."""
        engine = EmbeddingEngine()
        # Model won't be available in test env, will use mock
        embedding = engine.encode("Test text")

        assert isinstance(embedding, list)
        assert len(embedding) == 384

    def test_encode_batch(self):
        """Test batch encoding."""
        engine = EmbeddingEngine()
        texts = ["Text 1", "Text 2", "Text 3"]

        embeddings = engine.encode_batch(texts)

        assert len(embeddings) == 3
        for emb in embeddings:
            assert len(emb) == 384

    def test_dimension_property(self):
        """Test embedding dimension property."""
        engine = EmbeddingEngine()
        assert engine.dimension == 384

    def test_is_available(self):
        """Test model availability check."""
        engine = EmbeddingEngine()
        # In test env without sentence-transformers, should be False
        # (or True if mock is properly set up)
        assert isinstance(engine.is_available(), bool)


class TestHAAssistsClient:
    """Test HA Assists client integration."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.ha_client = HAAssistsClient(data_dir=self.temp_dir)

    def test_config_initialization(self):
        """Test HA Assists config initialization."""
        config = self.ha_client.config

        assert config is not None
        assert isinstance(config, HAAssistsConfig)
        assert config.enabled is False
        assert config.capabilities == ["tts", "stt", "conversation"]

    def test_update_config(self):
        """Test updating HA Assists configuration."""
        config = self.ha_client.update_config(
            enabled=True,
            ha_url="http://localhost:8123",
            ha_token="test_token_123",
            agent_id="test_agent"
        )

        assert config.enabled is True
        assert config.ha_url == "http://localhost:8123"
        assert config.ha_token == "test_token_123"
        assert config.agent_id == "test_agent"

    def test_update_config_partial(self):
        """Test partial config update."""
        self.ha_client.update_config(
            ha_url="http://localhost:8123",
            ha_token="test_token"
        )

        # Update only enabled
        config = self.ha_client.update_config(enabled=True)

        assert config.enabled is True
        assert config.ha_url == "http://localhost:8123"
        assert config.ha_token == "test_token"

    def test_test_connection_no_config(self):
        """Test connection test with no configuration."""
        result = self.ha_client.test_connection()

        assert result["success"] is False
        assert "URL and token required" in result["error"]

    @patch('requests.get')
    def test_test_connection_success(self, mock_get):
        """Test successful connection test."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        self.ha_client.update_config(
            ha_url="http://localhost:8123",
            ha_token="valid_token"
        )

        result = self.ha_client.test_connection()

        assert result["success"] is True
        assert result["message"] == "Connection successful"

    @patch('requests.get')
    def test_test_connection_failure(self, mock_get):
        """Test failed connection test."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_get.return_value = mock_response

        self.ha_client.update_config(
            ha_url="http://localhost:8123",
            ha_token="invalid_token"
        )

        result = self.ha_client.test_connection()

        assert result["success"] is False
        assert "401" in result["error"]

    @patch('requests.post')
    def test_process_assist_pipeline_not_enabled(self, mock_post):
        """Test processing when integration is not enabled."""
        self.ha_client.update_config(enabled=False)

        result = self.ha_client.process_assist_pipeline(text="Hello")

        assert "error" in result
        assert "not enabled" in result["error"]

    @patch('requests.post')
    def test_process_assist_pipeline_success(self, mock_post):
        """Test successful assist pipeline processing."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"text": "Hello! How can I help?"}
        mock_post.return_value = mock_response

        self.ha_client.update_config(
            enabled=True,
            ha_url="http://localhost:8123",
            ha_token="valid_token"
        )

        result = self.ha_client.process_assist_pipeline(
            text="Hello",
            conversation_id="conv_123",
            language="en"
        )

        assert "text" in result
        assert result["text"] == "Hello! How can I help?"
        mock_post.assert_called_once()

    @patch('requests.post')
    def test_process_assist_pipeline_api_error(self, mock_post):
        """Test assist pipeline with API error."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response

        self.ha_client.update_config(
            enabled=True,
            ha_url="http://localhost:8123",
            ha_token="valid_token"
        )

        result = self.ha_client.process_assist_pipeline(text="Hello")

        assert "error" in result
        assert "500" in result["error"]

    @patch('requests.post')
    def test_process_assist_pipeline_exception(self, mock_post):
        """Test assist pipeline with exception."""
        mock_post.side_effect = Exception("Connection refused")

        self.ha_client.update_config(
            enabled=True,
            ha_url="http://localhost:8123",
            ha_token="valid_token"
        )

        result = self.ha_client.process_assist_pipeline(text="Hello")

        assert "error" in result
        assert "Connection refused" in result["error"]

    def test_config_persistence(self):
        """Test that config persists across client instances."""
        self.ha_client.update_config(
            enabled=True,
            ha_url="http://localhost:8123",
            ha_token="persistent_token"
        )

        # Create new client instance
        new_client = HAAssistsClient(data_dir=self.temp_dir)

        assert new_client.config.enabled is True
        assert new_client.config.ha_url == "http://localhost:8123"
        assert new_client.config.ha_token == "persistent_token"


class TestIntegrationFlow:
    """End-to-end integration tests for complete workflows."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.session_store = SessionStore(data_dir=self.temp_dir)
        self.message_store = MessageStore(data_dir=self.temp_dir)

    def test_full_conversation_session_workflow(self):
        """Test complete conversation session workflow."""
        # Create session
        session = self.session_store.create_session(
            name="Conversation Test",
            provider="minimax"
        )
        session_id = session["session_id"]

        # Add user message
        user_msg = self.message_store.add_message(
            session_id=session_id,
            role="user",
            content="Hello, AI assistant!",
            metadata={"model": "MiniMax-M2.7"}
        )

        # Add assistant response
        assistant_msg = self.message_store.add_message(
            session_id=session_id,
            role="assistant",
            content="Hello! How can I assist you today?",
            metadata={"model": "MiniMax-M2.7"}
        )

        # Update session
        self.session_store.increment_message_count(session_id)
        self.session_store.increment_message_count(session_id)

        # Retrieve and verify
        retrieved_session = self.session_store.get_session(session_id)
        messages = self.message_store.get_messages(session_id)

        assert retrieved_session["message_count"] == 2
        assert len(messages) == 2
        assert messages[0]["content"] == "Hello, AI assistant!"
        assert messages[1]["content"] == "Hello! How can I assist you today?"

    def test_message_pagination_workflow(self):
        """Test message retrieval with pagination."""
        session = self.session_store.create_session()
        session_id = session["session_id"]

        # Add 10 messages
        for i in range(10):
            self.message_store.add_message(
                session_id=session_id,
                role="user",
                content=f"Message {i}"
            )

        # Test pagination
        page1 = self.message_store.get_messages(session_id, limit=5, offset=0)
        page2 = self.message_store.get_messages(session_id, limit=5, offset=5)

        assert len(page1) == 5
        assert len(page2) == 5

        # Verify ordering (chronological)
        assert page1[0]["content"] == "Message 0"
        assert page2[0]["content"] == "Message 5"

    def test_session_deletion_with_messages(self):
        """Test that deleting a session also handles messages."""
        session = self.session_store.create_session()
        session_id = session["session_id"]

        # Add messages
        self.message_store.add_message(session_id, "user", "Test 1")
        self.message_store.add_message(session_id, "assistant", "Test 2")

        # Delete session
        self.session_store.delete_session(session_id)

        # Verify session is gone
        assert self.session_store.get_session(session_id) is None

        # Verify messages file is cleaned up
        messages_file = Path(self.temp_dir) / "sessions" / f"{session_id}_messages.json"
        assert not messages_file.exists()

    def test_token_budget_enforcement(self):
        """Test that get_recent_messages respects token budget."""
        session = self.session_store.create_session()
        session_id = session["session_id"]

        # Add messages of varying lengths
        messages = [
            "Short.",
            "This is a medium length message.",
            "This is a much longer message that contains more content and should be prioritized in the token budget calculation."
        ]

        for msg in messages:
            self.message_store.add_message(session_id, "user", msg)

        # With very tight budget, should get fewer messages
        recent = self.message_store.get_recent_messages(session_id, max_tokens=50)
        total_chars = sum(len(m["content"]) for m in recent)

        # Should fit within budget
        assert total_chars <= 50 * 4  # ~4 chars per token


class TestHAAssistsPipelineIntegration:
    """Test HA Assists pipeline integration with mocked Home Assistant."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.ha_client = HAAssistsClient(data_dir=self.temp_dir)

    @patch('requests.post')
    @patch('requests.get')
    def test_full_assist_pipeline_flow(self, mock_get, mock_post):
        """Test complete assist pipeline flow with mocked HA."""
        # Mock successful connection
        mock_get.return_value = MagicMock(status_code=200)

        self.ha_client.update_config(
            enabled=True,
            ha_url="http://mock-ha:8123",
            ha_token="mock_token",
            agent_id="llm_ai"
        )

        # Mock assist pipeline response
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "text": "Processed response from LLM",
                "conversation_id": "conv_abc123"
            }
        )

        # Test the full flow
        result = self.ha_client.process_assist_pipeline(
            text="Turn on the lights",
            conversation_id="conv_abc123",
            language="en"
        )

        assert result["text"] == "Processed response from LLM"

        # Verify the request was made correctly
        call_args = mock_post.call_args
        assert "http://mock-ha:8123/api/conversation/process" in str(call_args)

    @patch('requests.post')
    def test_assist_pipeline_with_custom_agent(self, mock_post):
        """Test assist pipeline with custom agent ID."""
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"text": "Custom agent response"}
        )

        self.ha_client.update_config(
            enabled=True,
            ha_url="http://localhost:8123",
            ha_token="token",
            agent_id="custom_agent"
        )

        result = self.ha_client.process_assist_pipeline(text="Test")

        assert result["text"] == "Custom agent response"

        # Verify custom agent was in payload
        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["json"]["agent_id"] == "custom_agent"

    @patch('requests.post')
    def test_assist_pipeline_multilingual(self, mock_post):
        """Test assist pipeline with different languages."""
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"text": "Response in Chinese"}
        )

        self.ha_client.update_config(
            enabled=True,
            ha_url="http://localhost:8123",
            ha_token="token"
        )

        result = self.ha_client.process_assist_pipeline(
            text="打开灯",
            language="zh"
        )

        assert result["text"] == "Response in Chinese"

        # Verify language in payload
        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["json"]["language"] == "zh"

    def test_config_capabilities_update(self):
        """Test updating capabilities list."""
        self.ha_client.update_config(
            enabled=True,
            ha_url="http://localhost:8123",
            ha_token="token",
            capabilities=["tts", "conversation"]
        )

        config = self.ha_client.config
        assert config.capabilities == ["tts", "conversation"]

        # Update capabilities
        self.ha_client.update_config(capabilities=["stt", "tts", "conversation"])
        assert config.capabilities == ["stt", "tts", "conversation"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])