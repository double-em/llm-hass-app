"""Pytest configuration and fixtures."""

import sys
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import wave
import struct
import pytest

# Set testing environment before anything else
os.environ["TESTING"] = "true"


class MockTorchTensor:
    """Mock torch tensor that supports basic operations needed by the code."""

    def __init__(self, data, shape=(1, 16000)):
        self._data = np.array(data)
        self._shape = shape

    @property
    def shape(self):
        return self._shape

    def __getitem__(self, key):
        if isinstance(key, slice):
            return MockTorchTensor(self._data[key], shape=self._data[key].shape)
        return self._data[key]

    def __eq__(self, other):
        return MagicMock(return_value=False)

    def __ne__(self, other):
        return MagicMock(return_value=False)

    def __lt__(self, other):
        return MagicMock(return_value=False)

    def __gt__(self, other):
        return MagicMock(return_value=False)

    def __le__(self, other):
        return self

    def __ge__(self, other):
        return self

    def __mul__(self, other):
        return self

    def __div__(self, other):
        return self

    def __truediv__(self, other):
        return self

    def __rdiv__(self, other):
        return self

    def __rtruediv__(self, other):
        return self

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return MockTorchTensor(self._data + other, shape=self._shape)
        elif isinstance(other, MockTorchTensor):
            return MockTorchTensor(self._data + other._data, shape=self._shape)
        return self

    def __radd__(self, other):
        return MockTorchTensor(self._data + other, shape=self._shape)

    def mean(self, dim=None, keepdim=False):
        if dim is None:
            result = self._data.mean()
        else:
            result = self._data.mean(axis=dim, keepdims=keepdim)
        return MockTorchTensor(result, shape=(1,))

    def numpy(self):
        return self._data

    def squeeze(self, dim=None):
        return MockTorchTensor(self._data.squeeze(axis=dim) if dim else self._data.squeeze())

    def cat(self, other, dim=None):
        return MockTorchTensor(np.concatenate([self._data, other._data], axis=dim or 0))

    def __torch_function__(self, func, types, args=(), kwargs=None):
        return self


def _mock_torchaudio_load(path_or_buf):
    """Return mock waveform and sample rate as torch-like tensor."""
    waveform = np.zeros((1, 16000), dtype=np.float32)
    mock_tensor = MockTorchTensor(waveform, shape=(1, 16000))
    return (mock_tensor, 16000)


class MockMelSpectrogram:
    """Mock MelSpectrogram transform."""

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, waveform):
        # Return a mock tensor with appropriate shape for mel spectrogram
        time_steps = max(1, waveform.shape[-1] // 512 + 1)
        return MockTorchTensor(np.zeros((1, 64, time_steps)), shape=(1, 64, time_steps))


class MockResample:
    """Mock Resample transform."""

    def __init__(self, orig_freq, new_freq):
        pass

    def __call__(self, waveform):
        return waveform


mock_torchaudio = MagicMock()
mock_torchaudio.load = _mock_torchaudio_load
mock_torchaudio.transforms = MagicMock()
mock_torchaudio.transforms.Resample = MockResample
mock_torchaudio.transforms.MelSpectrogram = MockMelSpectrogram

mock_torch = MagicMock()
mock_torch.tensor = lambda *args, **kwargs: MockTorchTensor(args[0] if args else 0)
mock_torch.zeros = lambda *args, **kwargs: MockTorchTensor(np.zeros(args[0] if args else (1,)), shape=args[0] if args else (1,))
mock_torch.log = lambda x: x
mock_torch.cat = lambda tensors, dim=0: MockTorchTensor(np.concatenate([t._data for t in tensors], axis=dim))

# Install mocks
sys.modules['torch'] = mock_torch
sys.modules['torchaudio'] = mock_torchaudio
sys.modules['resemblyzer'] = MagicMock()
sys.modules['resemblyzer.api'] = MagicMock()
sys.modules['resemblyzer.voice_encoder'] = MagicMock()


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_data_dir(tmp_path):
    """Provide a temporary data directory for tests."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return str(data_dir)


def _create_sample_wav(path: str, duration_secs: float = 1.0, sample_rate: int = 16000):
    """Create a minimal WAV file for testing."""
    num_samples = int(duration_secs * sample_rate)
    with wave.open(path, 'wb') as wav_file:
        wav_file.setnchannels(1)  # mono
        wav_file.setsampwidth(2)   # 2 bytes per sample
        wav_file.setframerate(sample_rate)
        # Write simple silent audio
        for _ in range(num_samples):
            wav_file.writeframes(struct.pack('<h', 0))


@pytest.fixture
def sample_wav_file(tmp_path):
    """Provide a temporary sample WAV file for testing."""
    wav_path = tmp_path / "sample.wav"
    _create_sample_wav(str(wav_path))
    return str(wav_path)


@pytest.fixture
def person_store(temp_data_dir):
    """Provide a PersonStore instance with temp directory."""
    from person_store import PersonStore
    return PersonStore(data_dir=temp_data_dir)


@pytest.fixture
def enrollment_manager(temp_data_dir):
    """Provide an EnrollmentManager instance with temp directory."""
    from enrollment import EnrollmentManager
    return EnrollmentManager(data_dir=temp_data_dir)


@pytest.fixture
def voiceprint_manager(temp_data_dir):
    """Provide a VoiceprintManager instance with temp directory."""
    from voiceprint import VoiceprintManager
    return VoiceprintManager(data_dir=temp_data_dir)


@pytest.fixture
def mock_chroma():
    """Provide a mocked ChromaDB client."""
    mock = MagicMock()
    mock.collection = MagicMock()
    mock.collection.get = MagicMock(return_value={"ids": [], "embeddings": [], "documents": []})
    mock.collection.add = MagicMock()
    mock.collection.query = MagicMock(return_value={"ids": [], "documents": [], "distances": []})
    mock.collection.delete = MagicMock()
    return mock
