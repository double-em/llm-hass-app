# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.4.1] - 2026-06-23

### Fixed
- **Missing Docker image for v1.4.0.** The wrapper repo
  (`double-em/doubleem-hass-addons`) bumped the addon to `1.4.0` to
  ship the ingress fix, but the source tag was never pushed and the
  GitHub Actions image build never ran. As a result
  `ghcr.io/double-em/llm-hass-app:1.4.0` (and `:latest`) didn't exist
  on the registry, the supervisor's update attempts 404'd, and HA
  eventually dropped the addon from the sidebar entirely. This tag
  ships the actual image at `ghcr.io/double-em/llm-hass-app:1.4.1`
  (and `:latest`).
- **Home Assistant ingress support.** The dashboard previously assumed
  direct port access, so it was unreachable through the HA sidebar
  panel. The Flask app now reads `X-Ingress-Path` and rewrites
  `SCRIPT_NAME` + `PATH_INFO` so existing routes work, and templates
  use `url_for()` so links get the ingress prefix automatically.
- **JS `fetch()` calls** in inline templates now go through a small
  shim in `layout.html` that prepends the ingress prefix to absolute
  paths. No changes needed to the per-page scripts.

### Notes
- Pairs with `double-em/doubleem-hass-addons` 1.4.1 (HA addon). The
  `release.yml` `sync-ha-addon` job auto-bumps the wrapper to match
  this tag on push.
- HA supervisor pulls `ghcr.io/double-em/llm-hass-app:latest` and
  detects new digests via the addon store. After this image lands,
  re-adding the addon store entry (or hitting "Update" if it's still
  visible) should resolve the broken-update / disappeared-addon state.

## [1.0.1] - 2026-06-20

### Fixed
- **Home Assistant ingress support.** The dashboard previously assumed
  direct port access, so it was unreachable through the HA sidebar
  panel. The Flask app now reads `X-Ingress-Path` and rewrites
  `SCRIPT_NAME` + `PATH_INFO` so existing routes work, and templates
  use `url_for()` so links get the ingress prefix automatically.
- **JS `fetch()` calls** in inline templates now go through a small
  shim in `layout.html` that prepends the ingress prefix to absolute
  paths. No changes needed to the per-page scripts.

## [1.0.0] - 2026-04-05

### Added

#### AI Provider Management
- MiniMax AI provider integration for conversation and TTS
- Provider configuration via dashboard and API
- Provider connection testing

#### Voice Capabilities
- OmniVoice integration for high-quality voice synthesis (600+ languages)
- Voice cloning from reference audio files
- Voice design with customizable attributes (gender, age, pitch, accent, style)
- Auto voice mode for automatic voice selection
- Voice preset management (upload, list, delete)
- Voice caching system for performance optimization

#### TTS System
- REST API for TTS generation
- Configurable speech speed and diffusion steps
- Support for multiple voice providers (OmniVoice, MiniMax)
- 15+ English voices and 40+ Chinese voices via MiniMax

#### Conversation
- Real-time chat with AI providers
- Session-based conversation context
- Multiple provider support

#### Home Assistant Integration
- Full addon packaging for HA App Store
- Config flow for easy setup
- REST commands for TTS in automations
- Conversation agent for Assist pipeline
- Localized strings (English, Chinese)

#### AI Memory System
- Session and message storage
- Vector-based memory with embeddings
- Semantic search across conversation history
- ChromaDB-backed persistent storage

#### Person Management
- Voice enrollment system
- Speaker identification
- Voice verification
- Person store with persistent data

#### Voiceprint System
- Voiceprint extraction and management
- Speaker recognition capabilities

### Changed

- Migrated from standalone Flask app to Home Assistant integration
- Improved device detection (CUDA, MPS, CPU fallback)
- Updated requirements to use Home Assistant-compatible packages

### Fixed

- Voice preset loading from persistent storage
- Session initialization for conversation context

### Technical

#### Dependencies
- torch >= 2.8.0
- torchaudio >= 2.8.0
- flask >= 3.0.0
- omnivoice >= 0.1.0
- requests >= 2.31.0
- resemblyzer >= 0.1.0
- numpy >= 1.24.0
- chromadb >= 0.4.0
- sentence-transformers >= 2.2.0
- websockets >= 10.0

#### Architecture
- Modular design with separate managers (Enrollment, Voiceprint, Person, Voice)
- Memory system with pluggable storage backends
- HA Assist pipeline integration

---

## [0.1.0] - 2026-03-15

### Added
- Initial Flask-based dashboard
- Basic TTS functionality
- MiniMax API integration

[1.4.1]: https://github.com/double-em/llm-hass-app/releases/tag/v1.4.1
[1.0.1]: https://github.com/double-em/llm-hass-app/releases/tag/v1.0.1
[1.0.0]: https://github.com/double-em/llm-hass-app/releases/tag/v1.0.0
[0.1.0]: https://github.com/double-em/llm-hass-app/releases/tag/v0.1.0
