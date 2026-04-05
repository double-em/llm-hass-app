# OmniVoice TTS for Home Assistant

Custom voice cloning and text-to-speech generation using [OmniVoice](https://github.com/k2-fsa/OmniVoice) for Home Assistant.

## Features

- **Voice Cloning**: Clone a voice from a short reference audio sample
- **Voice Design**: Generate speech using speaker attributes (gender, accent, pitch, etc.)
- **Auto Voice**: Generate speech with a randomly selected voice
- **600+ Languages**: Supports most languages for zero-shot TTS

## App Structure

```
llm_app/
├── Dockerfile      # Docker image with OmniVoice + Flask API
├── config.yaml     # App configuration and options schema
├── run.sh          # Startup script
└── api.py          # Flask API server wrapping OmniVoice
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/tts` | POST | Generate TTS audio |
| `/voices` | GET | List available voice presets |

### TTS Endpoint

**POST /tts**

Request body (JSON):
```json
{
  "text": "Hello, this is a test.",
  "ref_audio": "<base64-encoded-wav>",
  "ref_text": "Transcript of reference audio",
  "instruct": "female, british accent",
  "speed": 1.0,
  "num_steps": 32
}
```

- `text` (required): Text to synthesize
- `ref_audio` (optional): Base64-encoded WAV for voice cloning
- `ref_text` (optional): Transcript for reference audio
- `instruct` (optional): Voice design attributes
- `speed` (optional): Speaking rate (default: 1.0)
- `num_steps` (optional): Diffusion steps (default: 32)

Returns: WAV audio file

### Voice Presets

Place reference audio files in `/data/voices/` (e.g., `my_voice.wav`) with optional transcripts (`my_voice.txt`) to use them as voice presets.

## Home Assistant Integration

Add a `rest_command` to your Home Assistant configuration:

```yaml
rest_command:
  omnivoice_tts:
    url: "http://homeassistant:8000/tts"
    method: POST
    content_type: "application/json"
    payload: >
      {
        "text": "{{ states('sensor.tts_text') }}",
        "ref_audio": "{{ state_attr('input_text.voice_sample', 'b64_audio') }}",
        "instruct": "{{ states('input_select.voice_style') }}"
      }
```

## Setup

1. Copy `llm_app/` to your Home Assistant `/addons` directory
2. Install the addon via Settings → Apps → App Store
3. Start the addon
4. Access the API at `http://<ha>:8000`

## Voice Design Attributes

- **Gender**: male, female
- **Age**: child, teenager, adult, elderly
- **Pitch**: very low, low, medium, high, very high
- **Accent**: american, british, australian, indian, etc.
- **Style**: whisper
- **Chinese dialects**: 四川话, 陕西话, etc.

Example: `"female, british accent, medium pitch"`
