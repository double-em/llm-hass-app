#!/usr/bin/with-contenv bashio

echo "LLM App starting..."

# Read options
OPTIONS_FILE="/data/options.json"

if [ -f "$OPTIONS_FILE" ]; then
    echo "Loaded options from $OPTIONS_FILE"
    # Parse options and use them
    # Example: MODEL=$(jq -r '.model' "$OPTIONS_FILE")
fi

# Your LLM app logic here
echo "LLM App initialized!"
