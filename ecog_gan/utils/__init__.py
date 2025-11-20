"""
Utility functions for ECoG GAN.

This module provides various utility functions including device management,
configuration loading, path utilities, and visualization tools.
"""

from .device import setup_device, cleanup_memory
from .config import load_config, save_config, merge_configs
from .paths import create_output_directories, get_timestamp_folder
from .visualization import plot_signals, plot_spectrograms, plot_correlation_matrix

__all__ = [
    'setup_device',
    'cleanup_memory',
    'load_config',
    'save_config', 
    'merge_configs',
    'create_output_directories',
    'get_timestamp_folder',
    'plot_signals',
    'plot_spectrograms',
    'plot_correlation_matrix'
]
