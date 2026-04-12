# Logging & Error Handling Design

## Overview

Add structured logging and comprehensive error handling so all errors — including startup crashes and unhandled exceptions — are written to stdout with timestamp, version, severity, and full stack traces. Logs flow to Docker/HA supervisor naturally.

## Log Format

```
%(asctime)s %(levelname)s %(version)s %(message)s
```

Example output:
```
2026-04-12T14:23:01.123Z INFO    1.0.1 app startup loaded config: {...}
2026-04-12T14:23:05.456Z ERROR   1.0.1 voice_engine OmniVoice load failed: CUDA not available
  File "app.py", line 106, in load_omnivoice
    voice_client = OmniVoice(...)
2026-04-12T14:23:06.789Z ERROR   1.0.1 app unhandled exception at /api/tts: PermissionError: [Errno 13] Permission denied
  File "app.py", line 500, in tts_handler
    ...
```

Fields: ISO timestamp (UTC, `Z` suffix), level (DEBUG/INFO/WARNING/ERROR), version string, message. Traceback lines follow error messages on subsequent lines indented by 2 spaces.

## Components

### `version.py` (new file)

Generated at Docker build time by the Dockerfile:

```python
__version__ = "1.0.1"
```

The Dockerfile adds this after copying source files:

```dockerfile
RUN echo '__version__ = "'"${VERSION}"'"' > /app/version.py
```

All Python files import it as: `from version import __version__`

### `logging_config.py` (new file)

Located at `/app/logging_config.py` (root of app, alongside `app.py`).

Provides:

1. **`get_logger(name: str) -> logging.Logger`** — factory that returns a logger with `version` field pre-attached via `LoggerAdapter`. All modules use this instead of `logging.getLogger(__name__)`.

2. **`configure_logging()`** — called once from `app.py` on startup. Configures root logger:
   - Level from `LOG_LEVEL` env var (default `INFO`)
   - `StreamHandler` writing to `sys.stdout`
   - Custom `Formatter` producing the format above

3. **`JSONFormatter`** — formats log records as `{"time": "...", "level": "...", "version": "...", "message": "..."}`. Used for structured output when `LOG_FORMAT=json` env var is set. Otherwise uses the plain text format above.

The `LoggerAdapter` pattern injects `version` into every log record without changing call sites.

### `app.py` changes

**Startup logging:**
- After `configure_logging()` call, log: `"app starting version={version}"`
- Log each initialization step (config, OmniVoice, person, memory, voice cache) with success/failure
- If any init step fails, log fatal and exit with non-zero code

**Global exception handler:**
```python
@app.errorhandler(Exception)
def handle_exception(e):
    logger = get_logger("app")
    logger.exception(f"unhandled exception at {request.path}: {type(e).__name__}: {e}")
    return jsonify({
        "error": str(e),
        "type": type(e).__name__,
        "traceback": traceback.format_exc()
    }), 500
```

**Existing route try/except blocks:**
- Replace `logging.getLogger(__name__)` with `get_logger(__name__)`
- Replace `logger.error(f"...")` with `logger.exception("...")` so traceback is included
- Keep existing error recovery logic (return error JSON, etc.)

**Import changes:**
- Remove `import logging` and `logging.basicConfig(...)`
- Add `from logging_config import configure_logging, get_logger`
- Add `from version import __version__`

### Per-module changes

All files that currently use `logging.getLogger(__name__)`:
- `voice_engine.py`
- `memory/ha_assists.py`
- `memory/vector_store.py`
- `memory/embedding.py`
- `person.py`
- `person_store.py`
- `enrollment.py`
- `omnivoice_client.py`
- `voiceprint.py`
- `voice_presets.py`
- `voice_cache.py`
- `voice_cache_store.py`
- `config_flow.py`

Changes per file:
- Replace `import logging` with `from logging_config import get_logger`
- Replace `logger = logging.getLogger(__name__)` with `logger = get_logger(__name__)`
- Replace `logger.error(f"...")` with `logger.exception("...")` for exceptions that include traceback
- For intentional "user made bad request" errors, keep `logger.warning(...)` or `logger.info(...)`

### `conftest.py` changes

If it uses logging, update to use `get_logger` pattern.

## Testing

- Existing tests should not break — logging config is only initialized when `app.py` runs, not on import
- `pytest tests/ -v` passes
- Manual test: start container, verify stdout shows startup logs with version, trigger an error, verify stack trace appears

## Dockerfile Changes

Add after copying application files:

```dockerfile
RUN echo '__version__ = "'"${VERSION}"'"' > /app/version.py
```

This writes the Docker `--build-arg VERSION` value into the Python file at build time.
