# LLM AI Dashboard for Home Assistant

A comprehensive Home Assistant integration providing AI provider management, voice cloning, and text-to-speech synthesis through an intuitive dashboard and API.

## Features

- **AI Provider Management**: Configure and manage AI providers (MiniMax, and more)
- **Voice Cloning**: Upload reference audio for voice cloning
- **Voice Design**: Generate speech using speaker attributes
- **Text-to-Speech**: Generate high-quality TTS using OmniVoice or MiniMax
- **Conversation**: Chat with AI providers directly from the dashboard
- **AI Memory**: Persistent conversation context and vector-based memory retrieval
- **Home Assistant Assist Integration**: Full pipeline support for HA Assist voice assistants

## Architecture

```
llm_ai_dashboard/         # Home Assistant Addon
├── app.py                 # Main Flask application
├── config.json             # Home Assistant addon configuration
├── Dockerfile              # Container build file
├── icon.png                # Addon icon
├── README.md               # This file
├── run.sh                  # Startup script
├── requirements.txt         # Python dependencies
├── const.py                # Constants
├── strings.json            # Localization strings (HA addon UI)
├── omnivoice_client.py    # OmniVoice API wrapper
├── person.py              # Voice enrollment/person management
├── person_store.py        # Person data persistence
├── enrollment.py          # Voice enrollment manager
├── voice_presets.py       # Voice preset management
├── voice_engine.py        # Voice synthesis engine
├── voice_cache.py         # Voice caching
├── voiceprint.py          # Voiceprint management
├── memory/
│   ├── __init__.py        # Memory system initialization
│   ├── session_store.py  # Conversation session storage
│   ├── message_store.py   # Message history with context window
│   ├── vector_store.py    # ChromaDB vector storage
│   ├── embedding.py       # Sentence-transformers embeddings
│   └── ha_assists.py      # HA Assist pipeline integration
├── templates/             # HTML templates
│   ├── layout.html
│   ├── index.html
│   ├── providers.html
│   ├── voices.html
│   ├── memory.html
│   ├── vector.html
│   ├── persons.html
│   └── tts.html
└── static/                # CSS and JS
    ├── css/style.css
    └── js/app.js
```

## Installation

### Option 1: Home Assistant Add-on Repository (Recommended)

1. Add this repository to Home Assistant:
   - Go to **Settings → System → Add-ons** (or click your user avatar → Add-ons)
   - Click **Add-on Store** (top right corner)
   - Click the three-dot menu → **Repositories**
   - Add: `https://github.com/double-em/llm-hass-app`
2. Find **LLM AI Dashboard** in the add-on list and click it
3. Click **Install**
4. Configure the addon (set your MiniMax API key, voice settings)
5. Start the addon
6. Access the dashboard at `http://<homeassistant>:8000`

### Option 2: Standalone Docker (Recommended for Local Development)

Two images are available:
- **`ghcr.io/double-em/llm-hass-app:cpu-latest`** — CPU-only (default, smaller ~3GB)
- **`ghcr.io/double-em/llm-hass-app:gpu-latest`** — CUDA GPU support (requires NVIDIA GPU ~5GB)

```bash
# CPU (default)
docker run -d -p 8000:8000 \
  -v $(pwd)/data:/data \
  -e MINIMAX_API_KEY=your_api_key \
  ghcr.io/double-em/llm-hass-app:cpu-latest

# GPU (requires NVIDIA runtime)
docker run -d -p 8000:8000 \
  --gpus all \
  -v $(pwd)/data:/data \
  -e MINIMAX_API_KEY=your_api_key \
  ghcr.io/double-em/llm-hass-app:gpu-latest

# Access at http://localhost:8000
```

**Building locally:**

```bash
# CPU build
docker build -f Dockerfile.cpu -t llm-ai-dashboard:cpu .
docker run -d -p 8000:8000 \
  -v $(pwd)/data:/data \
  -e MINIMAX_API_KEY=your_api_key \
  llm-ai-dashboard:cpu

# GPU build (requires NVIDIA GPU)
docker build -f Dockerfile.gpu -t llm-ai-dashboard:gpu .
docker run -d -p 8000:8000 \
  --gpus all \
  -v $(pwd)/data:/data \
  -e MINIMAX_API_KEY=your_api_key \
  llm-ai-dashboard:gpu
```

**With Docker Compose:**

```yaml
services:
  llm-ai-dashboard:
    image: ghcr.io/double-em/llm-hass-app:cpu-latest
    ports:
      - "8000:8000"
    volumes:
      - ./data:/data
    environment:
      - MINIMAX_API_KEY=your_api_key

  # For GPU support:
  # image: ghcr.io/double-em/llm-hass-app:gpu-latest
  # deploy:
  #   resources:
  #     reservations:
  #       devices:
  #         - driver: nvidia
  #           count: 1
  #           capabilities: [gpu]
```

**Data persistence:** Voice presets, persons, and memory are stored in `/data` inside the container. Mount a volume to persist data across restarts.

### Option 3: Manual Add-on Installation

1. Clone this repository to your Home Assistant server
2. Copy the `llm_app/` directory to `/addons/llm_ai_dashboard/`
3. Restart Home Assistant
4. The addon will appear in the Add-on Store

## Configuration

### Home Assistant Addon Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `default_provider` | string | `minimax` | Default AI provider |
| `default_voice_speed` | number | `1.0` | Default TTS speech speed |
| `default_diffusion_steps` | integer | `32` | Default diffusion steps for OmniVoice |

### config.yaml (Standalone)

```yaml
providers:
  minimax:
    api_key: "your_api_key_here"
    api_type: "minimax"

defaults:
  voice_speed: 1.0
  diffusion_steps: 32
  provider: "minimax"
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `MINIMAX_API_KEY` | MiniMax API key for TTS and conversation |
| `HA_URL` | Home Assistant URL (for Assist integration) |
| `HA_TOKEN` | Long-lived access token for HA API |

## API Endpoints

### Health & Status

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |

### AI Providers

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/providers` | GET | List configured AI providers |
| `/api/providers` | POST | Add or update a provider |
| `/api/providers/<name>` | DELETE | Delete a provider |
| `/api/providers/<name>/test` | POST | Test provider connection |

### Voice Presets

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/voices` | GET | List voice presets |
| `/api/voices` | POST | Upload a new voice preset |
| `/api/voices/<name>` | DELETE | Delete a voice preset |

### Text-to-Speech

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/tts` | POST | Generate TTS audio |

**POST /api/tts**

```json
{
  "text": "Hello, this is a test.",
  "provider": "omnivoice",
  "voice": "my-voice-preset",
  "instruct": "female, british accent",
  "speed": 1.0,
  "num_steps": 32
}
```

### Conversation

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/conversation` | POST | Send a conversation message |

**POST /api/conversation**

```json
{
  "message": "Hello, how are you?",
  "provider": "minimax"
}
```

### Persons & Enrollment

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/persons` | GET | List enrolled persons |
| `/api/persons` | POST | Create a new person |
| `/api/persons/<id>` | GET | Get person details |
| `/api/persons/<id>` | DELETE | Delete a person |
| `/api/enroll` | POST | Enroll a voice sample |
| `/api/identify` | POST | Identify speaker from audio |
| `/api/verify` | POST | Verify voice match |

### Memory

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/memory/sessions` | GET | List conversation sessions |
| `/api/memory/sessions/<id>` | GET | Get session messages |
| `/api/memory/search` | POST | Search vector memory |

## Voice Cloning with OmniVoice

OmniVoice supports 600+ languages and provides three voice modes:

1. **Voice Clone**: Use a reference audio file from `/data/voices/`
2. **Voice Design**: Describe the voice with attributes
3. **Auto Voice**: Let the model choose automatically

### Voice Design Attributes

- **Gender**: male, female
- **Age**: child, teenager, adult, elderly
- **Pitch**: very low, low, medium, high, very high
- **Accent**: american, british, australian, indian, etc.
- **Style**: whisper
- **Chinese dialects**: 四川话, 陕西话, etc.

Example: `female, british accent, medium pitch`

## Home Assistant Integration

### TTS in Automations

```yaml
service: rest_command.omnivoice_tts
data:
  text: "The front door is now unlocked"
```

### Conversation Agent

```yaml
conversation_agent:
  - name: llm_ai
    llm_ai:
      provider: minimax
```

### Assist Pipeline Integration

The integration supports the full Home Assistant Assist pipeline:

1. Configure a Voice Assistant in Home Assistant
2. Select "LLM AI" as the conversation agent
3. Use wake word or push-to-talk with the configured TTS voice

## MiniMax Integration

The app integrates with MiniMax AI for:
- **Conversation**: Natural language interactions
- **TTS**: High-quality voice synthesis (15+ English voices, 40+ Chinese voices)

Configure MiniMax in the dashboard or via the API with your API key from [MiniMax Platform](https://platform.minimax.io).

## Data Persistence

| Data | Location |
|------|----------|
| Configuration | `/data/options.json` |
| Voice presets | `/data/voices/` |
| Person data | `/data/persons/` |
| Vector memory | `/data/memory/` |

Voice files should be WAV format. Include a `.txt` file with the same name for the transcript.

## Development

### Requirements

- Python 3.10+
- torch >= 2.8.0
- torchaudio >= 2.8.0
- flask >= 3.0.0
- Home Assistant 2024.6+

### Local Development

```bash
cd llm_app
pip install -r requirements.txt
python app.py
```

### Testing

```bash
pytest llm_app/tests/
```

## License

MIT License - See LICENSE file for details.
