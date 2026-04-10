# CLAUDE.md

This file provides guidance to Claude Code when working on llm-hass-app.

## Docker Commands (Podman)

```bash
# Build image
podman build -t llm-hass-app:test .

# Run with memory limit (required - app loads large models)
podman run -d -p 8000:8000 --name llm-hass-test --memory=4g llm-hass-app:test

# Check logs (do this immediately if container exits)
podman logs llm-hass-test

# Clean up crashed/old containers (names persist after exit)
podman rm -f <container_id_or_name>

# Check if container is running
podman ps --filter "name=llm-hass-test" --format "{{.Names}} {{.Status}}"
```

## Important Notes

- **App startup takes ~2 minutes** - models download from HuggingFace on first load. Be patient before testing endpoints.
- **Always build and run container locally** before assuming work is done - don't skip this step.
- **Test endpoint:** `curl http://localhost:8000/api/health`
- **Podman HEALTHCHECK warning** is harmless - health check still works in practice.
- **Container exits silently on crash** - always check `podman logs` if health check fails.

## Import Conventions

Files in root (`/app/`) use local imports, NOT package prefix:
- `from voice_cache import VoiceLineCache` (correct)
- `from llm_app.voice_cache import VoiceLineCache` (wrong - will crash at runtime)

## Flask Route Gotchas

Function name = endpoint name. Duplicate function names cause `AssertionError: View function mapping is overwriting an existing endpoint function` at startup. Always rename duplicates (e.g., `_legacy` suffix).

## Libraries

- **resemblyzer** exports `VoiceEncoder`, NOT `ResemblyzerVoiceEncoder`
- **OmniVoice** loads on CPU by default via `device_map="cpu"`
