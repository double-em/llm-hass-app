#!/usr/bin/with-contenv bashio

echo "Starting OmniVoice TTS API..."

# Load options
OPTIONS_FILE="/data/options.json"

if [ -f "$OPTIONS_FILE" ]; then
    MODEL=$(jq -r '.model // "k2-fsa/OmniVoice"' "$OPTIONS_FILE")
    DEVICE=$(jq -r '.device // "auto"' "$OPTIONS_FILE")
    DEFAULT_SPEED=$(jq -r '.default_speed // 1.0' "$OPTIONS_FILE")
    DEFAULT_STEPS=$(jq -r '.default_steps // 32' "$OPTIONS_FILE")

    echo "Config: model=$MODEL, device=$DEVICE, speed=$DEFAULT_SPEED, steps=$DEFAULT_STEPS"
fi

# Create voices directory if not exists
mkdir -p /data/voices

# Start the API server
cd /app
python3 api.py --host 0.0.0.0 --port 8000
