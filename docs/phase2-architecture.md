# Phase 2 Architecture: AI Memory & HA Assists Integration

## Overview

Phase 2 adds three major capabilities:
1. **AI Memory** - Conversation history and session context management
2. **Vector Memory** - Semantic search via embeddings
3. **HA Assists Preparation** - Home Assistant conversation agent integration

---

## 1. AI Memory (Conversation History & Sessions)

### Data Model

```python
class ConversationSession:
    session_id: str          # UUID
    name: str                # User-friendly name (auto-generated or custom)
    provider: str            # AI provider name
    created_at: datetime
    updated_at: datetime
    message_count: int
    last_message_at: datetime

class Message:
    message_id: str          # UUID
    session_id: str          # FK to ConversationSession
    role: str                # "user" | "assistant" | "system"
    content: str
    timestamp: datetime
    metadata: dict           # Optional: model used, tokens, etc.
```

### Storage
- **Location**: `/data/sessions/` directory
- **Format**: JSON files per session
- **Index**: `sessions/index.json` - master index of all sessions

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/sessions` | GET | List all sessions (paginated) |
| `/api/sessions` | POST | Create new session |
| `/api/sessions/<id>` | GET | Get session with messages |
| `/api/sessions/<id>` | DELETE | Delete session |
| `/api/sessions/<id>/messages` | GET | Get messages for session |
| `/api/sessions/<id>/messages` | POST | Add message to session |
| `/api/conversation` | POST | Send message (auto-creates session if needed) |

### Context Window Management
- Track token count per message (estimate ~4 chars per token)
- Respect provider's `max_tokens` limit
- Automatically summarize or prune older messages when approaching limit
- Configurable: `max_context_messages` per provider (default: 50)

### Frontend UI
- **Sessions Panel**: Sidebar showing list of sessions with last message preview
- **Session View**: Full conversation history with message bubbles
- **Session Controls**: Rename, delete, export, clear history

---

## 2. Vector Memory (Semantic Search)

### Data Model

```python
class MemoryEntry:
    entry_id: str           # UUID
    content: str            # Text content
    embedding: list[float]  # 384-dim vector (all-MiniLM-L6-v2)
    created_at: datetime
    tags: list[str]         # Optional categorization
    source: str              # "conversation", "manual", "ha_event"
    related_session_id: str # Optional FK to session
```

### Storage
- **Embeddings**: ChromaDB at `/data/vector_memory/`
- **Metadata**: JSON sidecar files

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/memory` | GET | List memory entries |
| `/api/memory` | POST | Add new memory entry |
| `/api/memory/search` | POST | Semantic search |
| `/api/memory/<id>` | DELETE | Delete entry |
| `/api/memory/stats` | GET | Get memory stats |

### Search Request/Response

```json
// POST /api/memory/search
{
  "query": "what was my password?",
  "limit": 5,
  "threshold": 0.7
}

// Response
{
  "results": [
    {
      "entry_id": "...",
      "content": "Password is abc123",
      "similarity": 0.89,
      "created_at": "...",
      "tags": ["passwords"]
    }
  ],
  "query": "what was my password?",
  "total_results": 1
}
```

### Dependencies
- `chromadb` - Vector database
- `sentence-transformers` - Embedding model (all-MiniLM-L6-v2)

### Frontend UI
- **Memory Dashboard**: Grid of memory entries with tags
- **Search Bar**: Prominent semantic search input
- **Add Memory**: Form to manually add entries
- **Stats Display**: Total entries, last updated, storage size

---

## 3. HA Assists Integration Preparation

### Goals
Prepare the app to act as a conversation agent for Home Assistant's Assist pipeline.

### Data Model

```python
class HAAssistsConfig:
    enabled: bool
    ha_url: str                    # Home Assistant URL
    ha_token: str                  # Long-lived access token
    agent_id: str                  # Unique agent identifier
    capabilities: list[str]        # ["tts", "stt", "conversation"]
```

### API Endpoints (HA Assists Protocol)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/ha/assists` | POST | ProcessAssistPipeline (HA Protocol) |
| `/api/ha/config` | GET/POST | HA integration settings |
| `/api/ha/test` | POST | Test HA connectivity |

### HA Assist Pipeline Request Format

```json
// POST /api/ha/assists
{
  "intent": "聊天",
  "agent_id": "llm_ai",
  "conversation_id": "uuid",
  "Language": "zh",
  "text": "打开客厅灯"
}

// Response
{"text": "已打开客厅灯"}
```

### WebSocket Preparation
- Add WebSocket endpoint `/ws/conversation` for streaming responses
- Support for real-time TTS playback in HA

### Frontend UI
- **HA Settings Page**: Configure HA URL, token, test connection
- **Status Indicator**: Show HA connectivity status
- **Agent Info**: Display agent_id, capabilities

---

## File Structure Additions

```
llm_app/
├── app.py                    # Main app (add new routes)
├── memory/
│   ├── __init__.py
│   ├── session_store.py     # ConversationSession CRUD
│   ├── message_store.py     # Message CRUD
│   ├── vector_store.py       # ChromaDB wrapper
│   └── embedding.py          # Sentence-transformers wrapper
├── templates/
│   ├── sessions.html         # NEW: Session management UI
│   ├── memory.html           # NEW: Vector memory UI
│   └── ha_settings.html      # NEW: HA integration settings
└── static/
    └── js/
        ├── sessions.js       # NEW: Session management
        └── memory.js         # NEW: Vector memory
```

---

## Implementation Order

1. **Session Store & API** (backend-dev) - Foundation for all memory features
2. **Message Store & API** (backend-dev) - Depends on session store
3. **Frontend Sessions UI** (frontend-dev) - Depends on session API
4. **Vector Memory Backend** (backend-dev) - Independent, can parallel
5. **Frontend Memory UI** (frontend-dev) - Depends on vector API
6. **HA Assists Backend** (backend-dev) - Independent
7. **HA Settings UI** (frontend-dev) - Depends on HA API
8. **Integration Tests** (qa-engineer) - Run after all components done
9. **CI/CD Updates** (devops) - Run anytime, parallel with dev

---

## Dependencies to Add

```txt
# requirements.txt additions
chromadb>=0.4.0
sentence-transformers>=2.2.0
websockets>=10.0
```

---

## Backward Compatibility

- All existing `/api/conversation` behavior preserved
- New endpoints are additive only
- Session can be disabled (backward compatible to single-message mode)
- HA Assists is opt-in configuration
