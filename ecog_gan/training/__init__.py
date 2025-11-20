"""
Training utilities for ECoG GAN.

This module provides training loops, loss functions, and monitoring utilities
for training the ECoG GAN model.
"""

from .trainer_xai import Trainer, GANTrainer
from .losses import WGANGPLoss, compute_gradient_penalty
from .monitoring import TrainingMonitor, MetricsTracker

__all__ = [
    'Trainer',
    'GANTrainer', 
    'WGANGPLoss',
    'compute_gradient_penalty',
    'TrainingMonitor',
    'MetricsTracker'
]
