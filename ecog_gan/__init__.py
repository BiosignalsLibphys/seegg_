"""
Multichannel ECoG GAN with Spatio-Temporal Attention

A PyTorch implementation of a Generative Adversarial Network for generating 
synthetic multichannel electrocorticography (ECoG) signals.
"""

__version__ = "1.0.0"
__author__ = "Nianfei Ao"
__email__ = "a553379103@gmail.com"

from .models import Generator, WindowCritic
from .data import ECoGDataLoader
from .training import Trainer
from .utils import load_config, setup_device

__all__ = [
    'Generator',
    'WindowCritic', 
    'ECoGDataLoader',
    'Trainer',
    'load_config',
    'setup_device'
]
