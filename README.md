# LLM AI Dashboard for Home Assistant

A comprehensive dashboard for managing AI providers and voice cloning TTS in Home Assistant.

## Features

- **AI Provider Management**: Configure and manage AI providers (MiniMax, and more)
- **Voice Cloning**: Upload reference audio for voice cloning
- **Voice Design**: Generate speech using speaker attributes
- **Text-to-Speech**: Generate high-quality TTS using OmniVoice or MiniMax
- **Conversation**: Chat with AI providers directly from the dashboard

## App Structure

```
llm_app/
├── Dockerfile          # Docker image with Flask + OmniVoice
├── config.yaml         # App configuration
├── requirements.txt    # Python dependencies
├── run.sh              # Startup script
├── app.py              # Main Flask application
├── templates/          # HTML templates
│   ├── layout.html
│   ├── index.html
│   ├── providers.html
│   ├── voices.html
│   └── tts.html
└── static/             # CSS and JS
    ├── css/style.css
    └── js/app.js
```

## Quick Start

1. Copy `llm_app/` to your Home Assistant `/addons` directory
2. Install the addon via Settings → Apps → App Store
3. Start the addon
4. Access the dashboard at `http://<homeassistant>:8000`

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/providers` | GET | List AI providers |
| `/api/providers` | POST | Add/update provider |
| `/api/providers/<name>` | DELETE | Delete provider |
| `/api/providers/<name>/test` | POST | Test provider connection |
| `/api/voices` | GET | List voice presets |
| `/api/voices` | POST | Upload voice |
| `/api/voices/<name>` | DELETE | Delete voice |
| `/api/tts` | POST | Generate TTS audio |
| `/api/conversation` | POST | Send conversation message |

### TTS Generation

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

**POST /api/conversation**

```json
{
  "message": "Hello, how are you?",
  "provider": "minimax"
}
```

## Voice Cloning with OmniVoice

OmniVoice supports 600+ languages and provides three voice modes:

1. **Voice Clone**: Use a reference audio file from `/data/voices/`
2. **Voice Design**: Describe the voice with attributes (e.g., "female, low pitch")
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

## MiniMax Integration

The app integrates with MiniMax AI for:
- **Conversation**: Natural language interactions
- **TTS**: High-quality voice synthesis (15+ English voices, 40+ Chinese voices)

Configure MiniMax in the dashboard or via the API with your API key from [MiniMax Platform](https://platform.minimax.io).

## Data Persistence

- Configuration: `/data/options.json`
- Voice presets: `/data/voices/`

Voice files should be WAV format. Include a `.txt` file with the same name for the transcript.
