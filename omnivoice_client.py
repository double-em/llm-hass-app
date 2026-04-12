"""OmniVoice client wrapper for the LLM AI Dashboard."""

import io
import tempfile
from pathlib import Path

import torch
import torchaudio

from logging_config import get_logger

logger = get_logger(__name__)


class OmniVoiceClient:
    """Wrapper for OmniVoice TTS model.

    Provides methods for generating speech in different modes:
    - clone: Voice cloning from reference audio
    - design: Voice design with attribute instructions
    - auto: Automatic voice synthesis
    """

    def __init__(self, model):
        """Initialize with an OmniVoice model instance.

        Args:
            model: An OmniVoice model instance from omnivoice package.
        """
        self.model = model
        self.device = next(model.parameters()).device if hasattr(model, 'parameters') else "cpu"
        self.sample_rate = 24000

    def generate_speech(self, text, mode="auto", **kwargs):
        """Generate speech audio.

        Args:
            text: Text to synthesize.
            mode: Generation mode - 'clone', 'design', or 'auto'.
            **kwargs: Mode-specific arguments:
                - clone: ref_audio or voice_name required
                - design: instruct string required
                - auto: no additional args

        Returns:
            torch.Tensor: Audio tensor with shape (channels, samples).
        """
        generate_kwargs = {
            "text": text,
            "num_step": kwargs.get("num_steps", 32),
            "speed": kwargs.get("speed", 1.0),
        }

        if mode == "clone":
            ref_audio = kwargs.get("ref_audio")
            voice_name = kwargs.get("voice_name")

            if voice_name:
                voices_dir = Path("/data/voices")
                voice_path = voices_dir / f"{voice_name}.wav"
                if not voice_path.exists():
                    raise ValueError(f"Voice '{voice_name}' not found")
                generate_kwargs["ref_audio"] = str(voice_path)
                transcript_path = voice_path.with_suffix(".txt")
                if transcript_path.exists():
                    generate_kwargs["ref_text"] = transcript_path.read_text().strip()
            elif ref_audio:
                # Handle both file path and base64
                if ref_audio.startswith("/") or ref_audio.startswith("data:"):
                    generate_kwargs["ref_audio"] = ref_audio
                else:
                    # Base64 encoded audio
                    import base64 as b64
                    audio_bytes = b64.b64decode(ref_audio)
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                        f.write(audio_bytes)
                        generate_kwargs["ref_audio"] = f.name

                if kwargs.get("ref_text"):
                    generate_kwargs["ref_text"] = kwargs["ref_text"]
            else:
                raise ValueError("clone mode requires ref_audio or voice_name")

        elif mode == "design":
            instruct = kwargs.get("instruct")
            if not instruct:
                raise ValueError("design mode requires instruct string")
            generate_kwargs["instruct"] = instruct

        elif mode == "auto":
            pass
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'clone', 'design', or 'auto'")

        audio = self.model.generate(**generate_kwargs)
        return audio

    def generate_with_preset(self, text, preset_name, **kwargs):
        """Generate speech using a saved voice preset.

        Args:
            text: Text to synthesize.
            preset_name: Name of the saved preset.
            **kwargs: Additional generation arguments (speed, num_steps).

        Returns:
            tuple: (audio tensor, preset metadata dict)
        """
        from .voice_presets import VoicePresetManager

        preset_manager = VoicePresetManager()
        preset = preset_manager.load_preset(preset_name)

        if preset is None:
            raise ValueError(f"Preset '{preset_name}' not found")

        generate_kwargs = {
            "num_steps": kwargs.get("num_steps", 32),
            "speed": kwargs.get("speed", 1.0),
        }

        # Use preset's reference audio for cloning
        if preset.get("ref_audio_path"):
            generate_kwargs["ref_audio"] = preset["ref_audio_path"]
            if preset.get("ref_text"):
                generate_kwargs["ref_text"] = preset["ref_text"]
        elif preset.get("instruct"):
            generate_kwargs["instruct"] = preset["instruct"]
        else:
            raise ValueError(f"Preset '{preset_name}' has no ref_audio or instruct")

        audio = self.model.generate(text=text, **generate_kwargs)
        return audio, preset

    def list_presets(self):
        """List available voice presets.

        Returns:
            list: List of preset info dicts with name, description, etc.
        """
        from .voice_presets import VoicePresetManager

        preset_manager = VoicePresetManager()
        return preset_manager.list_presets()

    def save_as_preset(self, audio, name, description="", ref_text="", instruct="", tags=None):
        """Save generated audio as a voice preset.

        Args:
            audio: Audio tensor to save.
            name: Unique name for the preset.
            description: Human-readable description.
            ref_text: Transcript of the reference audio.
            instruct: Voice design instruction string.
            tags: List of tags for categorization.

        Returns:
            dict: Saved preset metadata.
        """
        from .voice_presets import VoicePresetManager

        preset_manager = VoicePresetManager()
        return preset_manager.save_preset(
            audio=audio,
            name=name,
            description=description,
            ref_text=ref_text,
            instruct=instruct,
            tags=tags or [],
        )

    def audio_to_bytes(self, audio, sample_rate=None):
        """Convert audio tensor to WAV bytes.

        Args:
            audio: Audio tensor.
            sample_rate: Sample rate (defaults to 24000).

        Returns:
            bytes: WAV audio data.
        """
        if sample_rate is None:
            sample_rate = self.sample_rate

        buffer = io.BytesIO()
        torchaudio.save(buffer, audio[0] if audio.dim() == 2 else audio, sample_rate, format="wav")
        buffer.seek(0)
        return buffer.getvalue()

    def get_audio_metadata(self, audio, sample_rate=None):
        """Get metadata for an audio tensor.

        Args:
            audio: Audio tensor.
            sample_rate: Sample rate (defaults to 24000).

        Returns:
            dict: Metadata including duration and sample_rate.
        """
        if sample_rate is None:
            sample_rate = self.sample_rate

        num_samples = audio.shape[-1] if audio.dim() > 1 else audio.shape[-1]
        duration = num_samples / sample_rate

        return {
            "duration": round(duration, 2),
            "sample_rate": sample_rate,
            "num_samples": num_samples,
        }
