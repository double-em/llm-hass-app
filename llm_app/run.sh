#!/usr/bin/with-contenv bashio

echo "Starting LLM AI Dashboard..."

# Create directories
mkdir -p /data/voices

# Start the Flask app
cd /app
python3 app.py --host 0.0.0.0 --port 8000
