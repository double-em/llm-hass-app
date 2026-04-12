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

## HA Addon vs GHCR Image

The addon in `doubleem-hass-addons/llm_ai_dashboard/` has its own Dockerfile that clones and builds from source. It does NOT use the pre-built `ghcr.io/double-em/llm-hass-app` image. Both need to be kept in sync.

## GitHub Actions Versioning

- `release-please-action` with `release-type: python` requires `pyproject.toml`/`setup.py` — without them it always outputs `latest`. Use `git describe --tags --abbrev=0` instead.
- `fetch-depth: 0` required in checkout step for tags to be available to `git describe --tags`
- Bash `${var#prefix}` doesn't work in `sh` (GitHub Actions default shell) — use `sed 's/^v//'` to strip `v` prefix

## HA Addon Development Checklist

- **Always run image locally first** with `podman run --rm --memory=2g ghcr.io/double-em/llm-hass-app:X.Y.Z` to verify startup
- **Sync ALL of these together** or HA will fail to update: git tag + GHCR image build with `--build-arg VERSION` + GitHub release + addon config.yaml version + addon build.yaml LLM_VERSION
- **Alpine base causes "exec format error"** with PyTorch — use Debian slim (`python:3.12-slim`) instead
- **HA may mount /data with root ownership** — handle `PermissionError` on `/data/options.json` gracefully, don't crash on read
- **`podman system prune -a --force`** clears disk space when build fails with "no space left on device"
- **Build with `--platform linux/amd64`** otherwise Podman on Apple Silicon builds ARM64 images that HA can't run
