# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.1.0 (2026-04-09)


### Features

* add automatic semantic versioning with release-please ([b79699e](https://github.com/double-em/llm-hass-app/commit/b79699e72751a1ad52ef66eabe35d4d5aeb1054b))


### Bug Fixes

* correct release-please inputs and permissions ([828a5b5](https://github.com/double-em/llm-hass-app/commit/828a5b5f35a864e7687a5ab3e58892447a232340))

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

[1.0.0]: https://github.com/double-em/llm-hass-app/releases/tag/v1.0.0
[0.1.0]: https://github.com/double-em/llm-hass-app/releases/tag/v0.1.0
