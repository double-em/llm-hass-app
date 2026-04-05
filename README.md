# LLM Home Assistant App

A Home Assistant App for LLM-powered automation.

## App Structure

```
llm_app/
├── Dockerfile      # Docker image for the app
├── config.yaml     # App configuration (name, description, options, schema)
└── run.sh          # Startup script
```

## Setup

1. Access your Home Assistant `/addons` directory via Samba or SSH
2. Copy the `llm_app` folder to your `/addons` directory
3. Open Home Assistant → Settings → Apps → App Store
4. Click "Check for updates" to see your app
5. Install and start the app

## Configuration

The app accepts the following options:

| Option | Type | Description |
|--------|------|-------------|
| model | string | LLM model to use (e.g., "claude") |
| api_key | string | API key for the LLM service |

## Development

Edit the files in `llm_app/` and restart the app after updating.

For more details, see the [Home Assistant App Tutorial](https://developers.home-assistant.io/docs/apps/tutorial/).
