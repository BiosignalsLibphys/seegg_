"""
Data handling modules for ECoG GAN.

This module provides flexible data loaders that can handle various data formats:
- Dictionary format with subject-based organization
- 2D arrays: [signals, samples]
- 3D arrays: [signals, channels, samples]
- Variable number of channels
"""

from .loaders import ECoGDataLoader, DataFormatDetector
from .preprocessors import DataPreprocessor, DataNormalizer, SignalResampler

__all__ = [
    'ECoGDataLoader',
    'DataFormatDetector',
    'DataPreprocessor',
    'DataNormalizer',
    'SignalResampler'
]
