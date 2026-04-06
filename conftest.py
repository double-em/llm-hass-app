"""Pytest configuration and fixtures."""

import sys
import os
from unittest.mock import MagicMock
import numpy as np

# Set testing environment before anything else
os.environ["TESTING"] = "true"


class MockTensor(np.ndarray):
    """Mock tensor that behaves like a torch Tensor with numpy() method."""

    def numpy(self):
        return np.array(self)


def _mock_torchaudio_load(path_or_buf):
    """Return mock waveform and sample rate as torch-like tensor."""
    # Create a mock tensor that has .numpy() method and shape operations
    waveform = np.zeros((1, 16000), dtype=np.float32)
    mock_tensor = waveform.view(MockTensor)

    # Add shape property that torch expects
    type(mock_tensor).shape = property(lambda self: (1, 16000))

    return (mock_tensor, 16000)


mock_torchaudio = MagicMock()
mock_torchaudio.load = _mock_torchaudio_load
mock_torchaudio.transforms = MagicMock()
mock_torchaudio.transforms.Resample = MagicMock(return_value=MagicMock())
mock_torchaudio.transforms.MelSpectrogram = MagicMock(return_value=MagicMock())

mock_torch = MagicMock()

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