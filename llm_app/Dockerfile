ARG BUILD_FROM
FROM $BUILD_FROM

# Install Python and dependencies
RUN apk add --no-cache python3 py3-pip bashio

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt /app/

# Install PyTorch (CPU version for HA addon)
RUN pip3 install --no-cache-dir torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu

# Install OmniVoice, Flask, and other dependencies
RUN pip3 install --no-cache-dir omnivoice flask requests

# Copy application files
COPY app.py /app/
COPY templates/ /app/templates/
COPY static/ /app/static/

# Create voices directory
RUN mkdir -p /data/voices

# Make run.sh executable
COPY run.sh /app/run.sh
RUN chmod a+x /app/run.sh

CMD [ "/app/run.sh" ]
