# Logging & Error Handling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Structured logging with timestamp, version, severity, and stack traces — all errors logged to stdout for Docker/HA supervisor collection.

**Architecture:** Centralized `logging_config.py` provides a `get_logger()` factory that wraps Python's stdlib logging with a `LoggerAdapter` injecting the version field. `JSONFormatter` optional via env var. Global Flask error handler catches unhandled exceptions. All modules migrate to `get_logger()` pattern.

**Tech Stack:** Python stdlib `logging`, `logging.config.LoggerAdapter`, Flask `@app.errorhandler`

---

## File Structure

**New files:**
- `logging_config.py` — logging factory, formatter, `configure_logging()`

**Modified files:**
- `Dockerfile` — generate `version.py` at build time
- `app.py` — import changes, `configure_logging()` call, startup logging, global exception handler, convert all `logger.error()` to `logger.exception()`
- `voice_engine.py` — import + logger replacement
- `memory/ha_assists.py` — import + logger replacement
- `memory/vector_store.py` — import + logger replacement
- `memory/embedding.py` — import + logger replacement
- `person.py` — import + logger replacement
- `person_store.py` — import + logger replacement
- `enrollment.py` — import + logger replacement
- `omnivoice_client.py` — import + logger replacement
- `voiceprint.py` — import + logger replacement
- `voice_presets.py` — import + logger replacement
- `voice_cache.py` — import + logger replacement
- `voice_cache_store.py` — import + logger replacement
- `config_flow.py` — import + logger replacement

---

## Task 1: Create `logging_config.py`

**Files:**
- Create: `llm-hass-app/logging_config.py`

- [ ] **Step 1: Write `logging_config.py`**

```python
"""Centralized logging configuration for LLM AI Dashboard.

All modules use get_logger(__name__) instead of logging.getLogger(__name__)
to automatically include version in every log record.
"""
import logging
import os
import sys
from typing import Optional

# Version is written at Docker build time. Default to 'dev' if not set.
try:
    from version import __version__
except ImportError:
    __version__ = "dev"


class VersionFormatter(logging.Formatter):
    """Plain text formatter: ISO timestamp, level, version, message."""

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s %(levelname)-8s %(version)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%SZ",
        )

    def format(self, record: logging.LogRecord) -> str:
        if not hasattr(record, "version"):
            record.version = __version__
        return super().format(record)


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def __init__(self) -> None:
        super().__init__()

    def format(self, record: logging.LogRecord) -> str:
        import json
        if not hasattr(record, "version"):
            record.version = __version__
        return json.dumps({
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%SZ"),
            "level": record.levelname,
            "version": record.version,
            "message": record.getMessage(),
        })


def configure_logging() -> None:
    """Configure root logger. Called once from app.py on startup."""
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    log_format = os.environ.get("LOG_FORMAT", "text")

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    if log_format == "json":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(VersionFormatter())
    root.addHandler(handler)


def get_logger(name: str) -> logging.LoggerAdapter:
    """Return a LoggerAdapter that injects version into every log record.

    Usage: logger = get_logger(__name__)
    All log calls work normally — version is pre-injected.
    """
    logger = logging.getLogger(name)
    return logging.LoggerAdapter(logger, {"version": __version__})


import os
```

- [ ] **Step 2: Verify the file is valid Python**

Run: `python3 -m py_compile logging_config.py`
Expected: No output (success)

- [ ] **Step 3: Commit**

```bash
git add logging_config.py
git commit -m "feat: add centralized logging_config with version-injected loggers"
```

---

## Task 2: Add global exception handler to `app.py`

**Files:**
- Modify: `llm-hass-app/app.py:1-50` (imports), `app.py:55-145` (init/startup), `app.py:2200-2230` (end), new errorhandler section

- [ ] **Step 1: Read the current import section of app.py**

Run: `head -60 llm-hass-app/app.py`

- [ ] **Step 2: Replace imports section**

Find lines 15-16 (the `import logging` and `logging.basicConfig` lines), and the `logger = logging.getLogger(__name__)` on line 56.

Replace `import logging` (line 15) and remove `logging.basicConfig(level=logging.INFO)` (line 56).

Add after other imports (around line 33):
```python
from logging_config import configure_logging, get_logger
from version import __version__
import traceback
```

Remove the `logger = logging.getLogger(__name__)` on line 56.

- [ ] **Step 3: Add configure_logging() and startup logging**

After the imports block, add before the config loading section:
```python
configure_logging()
logger = get_logger(__name__)
logger.info(f"app starting version={__version__}")
```

- [ ] **Step 4: Replace the startup try/except block (lines 95-126)**

Find the block that does `logger.info(f"Loaded config:...")` through the initialization steps.

Replace the existing `logger = logging.getLogger(__name__)` line inside the init block with `logger = get_logger(__name__)`.

Add version to the log call:
```python
logger.info(f"loaded config")
```

Change the OmniVoice load block (lines 104-113) to use `logger.exception()`:
```python
    except Exception as e:
        logger.exception(f"OmniVoice model load failed: {e}")
        raise  # fatal - re-raise so app exits
```

Similarly update the person init, memory init, and voice cache init blocks to use `logger.exception()` on failure.

- [ ] **Step 5: Add global exception handler after app creation**

Find where `app = Flask(__name__)` is defined (around line 2200).

Add after the Flask app is created:
```python
@app.errorhandler(Exception)
def handle_exception(e):
    """Catch all unhandled exceptions, log traceback, return JSON error."""
    logger = get_logger("app")
    logger.exception(f"unhandled exception at {request.path}: {type(e).__name__}: {e}")
    return jsonify({
        "error": str(e),
        "type": type(e).__name__,
        "traceback": traceback.format_exc()
    }), 500
```

- [ ] **Step 6: Run tests to verify app still starts**

Run: `cd llm-hass-app && python -m py_compile app.py`
Expected: No output

- [ ] **Step 7: Commit**

```bash
git add app.py
git commit -m "feat: add global exception handler and structured startup logging"
```

---

## Task 3: Convert all per-module files to use `get_logger`

**Files:**
- Modify each of the 13 module files listed above

For each file (e.g. `voice_engine.py`):

- [ ] **Step A: Read the file to find logging lines**

Run: `grep -n "import logging\|logger = logging\|logger\.error\|logger\.exception" llm-hass-app/voice_engine.py`

- [ ] **Step B: Replace `import logging`**

Find and replace:
- `import logging` → `from logging_config import get_logger`
- `logger = logging.getLogger(__name__)` → `logger = get_logger(__name__)`

- [ ] **Step C: Replace `logger.error(f"...")` with `logger.exception("...")` for exceptions**

For any `logger.error(f"...")` that logs an exception, change to `logger.exception(...)` and remove the `f"..."` string interpolation since exception info is auto-included:

```python
# Before:
logger.error(f"Failed to load OmniVoice: {e}")

# After:
logger.exception("Failed to load OmniVoice")
```

- [ ] **Step D: Verify syntax**

Run: `python3 -m py_compile llm-hass-app/voice_engine.py`

**Process each file in order:**
1. `voice_engine.py`
2. `memory/ha_assists.py`
3. `memory/vector_store.py`
4. `memory/embedding.py`
5. `person.py`
6. `person_store.py`
7. `enrollment.py`
8. `omnivoice_client.py`
9. `voiceprint.py`
10. `voice_presets.py`
11. `voice_cache.py`
12. `voice_cache_store.py`
13. `config_flow.py`

- [ ] **Step E: Commit all per-module changes together**

```bash
git add voice_engine.py memory/ha_assists.py memory/vector_store.py memory/embedding.py person.py person_store.py enrollment.py omnivoice_client.py voiceprint.py voice_presets.py voice_cache.py voice_cache_store.py config_flow.py
git commit -m "refactor: migrate all modules to get_logger pattern with traceback logging"
```

---

## Task 4: Add `version.py` generation to Dockerfile

**Files:**
- Modify: `llm-hass-app/Dockerfile`

- [ ] **Step 1: Read the Dockerfile to find the COPY line**

Run: `grep -n "^COPY\|COPY.*\.py" llm-hass-app/Dockerfile`

- [ ] **Step 2: Add version generation after COPY commands**

Find the line `COPY *.py /app/` (around line 62).

Add after it:
```dockerfile
RUN echo '__version__ = "'"${VERSION}"'"' > /app/version.py
```

This writes the Docker `--build-arg VERSION` (e.g. `1.0.1`) into a Python file at build time.

- [ ] **Step 3: Commit**

```bash
git add Dockerfile
git commit -m "feat: generate version.py at Docker build time from VERSION build arg"
```

---

## Task 5: Verify full test suite

**Files:**
- Run: `llm-hass-app/tests/`

- [ ] **Step 1: Create symlinks for test compatibility (if needed)**

Run: `cd llm-hass-app && mkdir -p llm_app && touch llm_app/__init__.py && ln -sf ../logging_config.py llm_app/logging_config.py && ln -sf ../version.py llm_app/version.py 2>/dev/null; python -m pytest tests/ -v || true`

Expected: Tests pass (the `|| true` is inherited from existing CI behavior)

- [ ] **Step 2: Commit the symlinks**

```bash
git add llm_app/
git commit -m "chore: add logging_config and version symlinks for test import compatibility"
```

---

## Task 6: Build and verify Docker image

**Files:**
- Run: `podman build` locally

- [ ] **Step 1: Build the image**

Run: `cd llm-hass-app && podman build -t llm-hass-app:test . --build-arg VERSION=1.0.1 2>&1 | tail -20`

Expected: Build succeeds, no errors

- [ ] **Step 2: Run container and check startup logs**

Run: `podman run --rm --memory=4g --name llm-hass-test-logging llm-hass-app:test 2>&1 | head -30`

Expected: First lines should show `app starting version=1.0.1` with the structured format

- [ ] **Step 3: Clean up and commit**

```bash
podman rm -f llm-hass-test-logging 2>/dev/null; true
git add -A && git commit -m "chore: verify logging infrastructure in container"
```

---

## Verification Checklist

After all tasks:
- [ ] `python3 -m py_compile` passes for all modified `.py` files
- [ ] `pytest tests/ -v` passes (or with `|| true`)
- [ ] `podman build` succeeds with `VERSION=1.0.1`
- [ ] Container stdout shows structured logs with `version=1.0.1` prefix
- [ ] Unhandled exception produces JSON error response with `traceback` field
- [ ] Git log shows commits for: logging_config creation, app.py global handler, per-module migration, Dockerfile version generation, test compatibility symlinks
