"""
Model architectures for ECoG GAN.

This module contains the Generator, Critic, and attention mechanism implementations
optimized for multichannel ECoG signal generation.
"""

from .generator import Generator
from .critic import WindowCritic
from .attention import (
    TemporalAE, SpatialAE, LearnedPE, TemporalAttBuilder, SpatialAttBuilder,
    ConditionalSpatialAE, SingleChannelSpatialAttention
)

__all__ = [
    'Generator',
    'WindowCritic',
    'TemporalAE',
    'SpatialAE',
    'LearnedPE',
    'TemporalAttBuilder',
    'SpatialAttBuilder',
    'ConditionalSpatialAE',
    'SingleChannelSpatialAttention'
]
