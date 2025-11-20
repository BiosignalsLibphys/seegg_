"""
Main trainer class for ECoG GAN.

This module provides the main training loop and utilities for training
the ECoG GAN model with various configurations and monitoring.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple
import logging

from .losses import WGANGPLoss, get_loss_function
from .monitoring import TrainingMonitor
from ..utils.device import setup_device, cleanup_memory
from ..models import Generator, WindowCritic

logger = logging.getLogger(__name__)


class Trainer:
    """Base trainer class for ECoG GAN."""
    
    def __init__(self, 
                 generator: nn.Module,
                 critic: nn.Module,
                 config: Dict[str, Any],
                 device: Optional[torch.device] = None):
        """
        Initialize trainer.
        
        Args:
            generator: Generator model
            critic: Critic model
            config: Training configuration
            device: Device to train on (auto-detected if None)
        """
        self.generator = generator
        self.critic = critic
        self.config = config
        self.device = device or setup_device(config.get('gpu_id', 0))
        
        # Move models to device
        self.generator.to(self.device)
        self.critic.to(self.device)
        
        # Initialize training components
        self.setup_optimizers()
        self.setup_schedulers()
        self.setup_loss_function()
        self.setup_monitoring()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        
        logger.info(f"Trainer initialized on device: {self.device}")
    
    def setup_optimizers(self):
        """Setup optimizers for generator and critic."""
        opt_config = self.config.get('optimizer', {})
        
        # Generator optimizer
        gen_config = opt_config.get('generator', {})
        self.optimizer_G = optim.Adam(
            self.generator.parameters(),
            lr=gen_config.get('lr', 0.0002),
            betas=gen_config.get('betas', (0.5, 0.999))
        )
        
        # Critic optimizer
        critic_config = opt_config.get('critic', {})
        self.optimizer_C = optim.Adam(
            self.critic.parameters(),
            lr=critic_config.get('lr', 0.0002),
            betas=critic_config.get('betas', (0.5, 0.999))
        )
    
    def setup_schedulers(self):
        """Setup learning rate schedulers."""
        sched_config = self.config.get('scheduler', {})
        
        if sched_config.get('use_scheduler', False):
            self.scheduler_G = optim.lr_scheduler.StepLR(
                self.optimizer_G,
                step_size=sched_config.get('step_size', 100),
                gamma=sched_config.get('gamma', 0.9)
            )
            self.scheduler_C = optim.lr_scheduler.StepLR(
                self.optimizer_C,
                step_size=sched_config.get('step_size', 100),
                gamma=sched_config.get('gamma', 0.9)
            )
        else:
            self.scheduler_G = None
            self.scheduler_C = None
    
    def setup_loss_function(self):
        """Setup loss function."""
        loss_config = self.config.get('loss', {'type': 'wgan_gp'})
        self.loss_fn = get_loss_function(loss_config)
    
    def setup_monitoring(self):
        """Setup training monitoring."""
        monitor_config = self.config.get('monitoring', {})
        output_dir = monitor_config.get('output_dir', './outputs')
        
        self.monitor = TrainingMonitor(
            output_dir=output_dir,
            save_frequency=monitor_config.get('save_frequency', 10)
        )
    
    def train_step_critic(self, real_batch: torch.Tensor) -> Dict[str, float]:
        """
        Perform one training step for the critic.
        
        Args:
            real_batch: Batch of real data
            
        Returns:
            Dictionary of loss components
        """
        self.optimizer_C.zero_grad()
        
        batch_size = real_batch.size(0)
        latent_dim = self.config.get('model', {}).get('generator', {}).get('latent_dim', 100)
        
        # Generate fake samples
        z = torch.randn(batch_size, latent_dim, device=self.device)
        fake_batch = self.generator(z).detach()  # Detach to avoid generator gradients
        
        # Compute critic loss
        critic_loss, loss_components = self.loss_fn.critic_loss(
            self.critic, real_batch, fake_batch, self.device
        )
        
        # Backward pass
        critic_loss.backward()
        
        # Compute gradient norm
        grad_norm_C = self.compute_gradient_norm(self.critic)
        loss_components['grad_norm_critic'] = grad_norm_C
        
        # Update critic
        self.optimizer_C.step()
        
        return loss_components
    
    def train_step_generator(self, batch_size: int) -> Dict[str, float]:
        """
        Perform one training step for the generator.
        
        Args:
            batch_size: Size of the batch
            
        Returns:
            Dictionary of loss components
        """
        self.optimizer_G.zero_grad()
        
        latent_dim = self.config.get('model', {}).get('generator', {}).get('latent_dim', 100)
        
        # Generate fake samples
        z = torch.randn(batch_size, latent_dim, device=self.device)
        fake_batch = self.generator(z)
        
        # Compute generator loss
        generator_loss, loss_components = self.loss_fn.generator_loss(
            self.critic, fake_batch
        )
        
        # Backward pass
        generator_loss.backward()
        
        # Compute gradient norm
        grad_norm_G = self.compute_gradient_norm(self.generator)
        loss_components['grad_norm_generator'] = grad_norm_G
        
        # Update generator
        self.optimizer_G.step()
        
        return loss_components
    
    def compute_gradient_norm(self, model: nn.Module) -> float:
        """Compute gradient norm for a model."""
        total_norm = 0.0
        param_count = 0
        
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                param_count += 1
        
        if param_count == 0:
            return 0.0
        
        return (total_norm ** 0.5)
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, Any]:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Epoch metrics
        """
        self.generator.train()
        self.critic.train()
        
        epoch_metrics = {
            'generator_losses': [],
            'critic_losses': [],
            'grad_norms_G': [],
            'grad_norms_C': []
        }
        
        critic_iterations = self.config.get('training', {}).get('critic_iterations', 5)
        
        for batch_idx, real_batch in enumerate(dataloader):
            real_batch = real_batch.to(self.device)
            batch_size = real_batch.size(0)
            
            # Collect metrics for synchronized plotting
            batch_critic_losses = []
            batch_critic_grad_norms = []
            batch_gradient_penalties = []

            # Train critic multiple times
            for _ in range(critic_iterations):
                critic_metrics = self.train_step_critic(real_batch)
                batch_critic_losses.append(critic_metrics['critic_loss'])
                batch_critic_grad_norms.append(critic_metrics['grad_norm_critic'])
                batch_gradient_penalties.append(critic_metrics.get('gradient_penalty', 0.0))

                # Update gradient_penalty
                self.monitor.update_batch({'gradient_penalty': critic_metrics.get('gradient_penalty', 0.0)})

                # Still aggregate for epoch statistics
                epoch_metrics['critic_losses'].append(critic_metrics['critic_loss'])
                epoch_metrics['grad_norms_C'].append(critic_metrics['grad_norm_critic'])

            # Train generator
            generator_metrics = self.train_step_generator(batch_size)
            epoch_metrics['generator_losses'].append(generator_metrics['generator_loss'])
            epoch_metrics['grad_norms_G'].append(generator_metrics['grad_norm_generator'])

            # Create synchronized batch metrics (one point per batch, not per critic iteration)
            synchronized_metrics = {
                'critic_loss': sum(batch_critic_losses) / len(batch_critic_losses),  # Average critic loss
                'generator_loss': generator_metrics['generator_loss'],
                'grad_norm_critic': sum(batch_critic_grad_norms) / len(batch_critic_grad_norms),  # Average critic grad norm
                'grad_norm_generator': generator_metrics['grad_norm_generator']
            }

            # Update monitoring with synchronized metrics (once per batch)
            self.monitor.update_batch(synchronized_metrics)


            self.global_step += 1  # One step per batch (synchronized)
        
        # Compute epoch statistics
        epoch_stats = {
            'epoch': epoch,
            'generator_loss': sum(epoch_metrics['generator_losses']) / len(epoch_metrics['generator_losses']),
            'critic_loss': sum(epoch_metrics['critic_losses']) / len(epoch_metrics['critic_losses']),
            'grad_norm_generator': sum(epoch_metrics['grad_norms_G']) / len(epoch_metrics['grad_norms_G']),
            'grad_norm_critic': sum(epoch_metrics['grad_norms_C']) / len(epoch_metrics['grad_norms_C']),
            'lr_generator': self.optimizer_G.param_groups[0]['lr'],
            'lr_critic': self.optimizer_C.param_groups[0]['lr']
        }
        
        return epoch_stats
    
    def train(self, dataloader: DataLoader, num_epochs: int):
        """
        Main training loop.
        
        Args:
            dataloader: Training data loader
            num_epochs: Number of epochs to train
        """
        self.monitor.start_training(num_epochs)
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Train for one epoch
            epoch_stats = self.train_epoch(dataloader, epoch)
            
            # Update learning rate schedulers
            if self.scheduler_G:
                self.scheduler_G.step()
            if self.scheduler_C:
                self.scheduler_C.step()
            
            # Add timing information
            epoch_time = time.time() - epoch_start_time
            epoch_stats['epoch_time'] = epoch_time
            
            # Update monitoring
            self.monitor.update_epoch(epoch, epoch_stats)
            
            # Save checkpoints
            if epoch % self.config.get('checkpoint_frequency', 50) == 0:
                self.save_checkpoint(epoch)
            
            # Cleanup memory
            cleanup_memory(self.device)
            
            self.current_epoch = epoch
        
        self.monitor.finish_training()
    
    def save_checkpoint(self, epoch: int):
        """Save model checkpoints."""
        checkpoint_dir = os.path.join(self.monitor.output_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save generator
        generator_path = os.path.join(checkpoint_dir, f'generator_epoch_{epoch}.pth')
        torch.save(self.generator.state_dict(), generator_path)
        
        # Save critic
        critic_path = os.path.join(checkpoint_dir, f'critic_epoch_{epoch}.pth')
        torch.save(self.critic.state_dict(), critic_path)
        
        # Save training state
        state_path = os.path.join(checkpoint_dir, f'training_state_epoch_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_C_state_dict': self.optimizer_C.state_dict(),
            'scheduler_G_state_dict': self.scheduler_G.state_dict() if self.scheduler_G else None,
            'scheduler_C_state_dict': self.scheduler_C.state_dict() if self.scheduler_C else None,
            'config': self.config
        }, state_path)
        
        logger.info(f"Checkpoint saved at epoch {epoch}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_C.load_state_dict(checkpoint['optimizer_C_state_dict'])
        
        if self.scheduler_G and checkpoint.get('scheduler_G_state_dict'):
            self.scheduler_G.load_state_dict(checkpoint['scheduler_G_state_dict'])
        if self.scheduler_C and checkpoint.get('scheduler_C_state_dict'):
            self.scheduler_C.load_state_dict(checkpoint['scheduler_C_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    def generate_samples(self, n_samples: int) -> torch.Tensor:
        """Generate synthetic samples."""
        return self.generator.generate_samples(n_samples, self.device)


class GANTrainer(Trainer):
    """Specialized trainer for GAN training with additional features."""
    
    def __init__(self, *args, **kwargs):
        """Initialize GAN trainer."""
        super().__init__(*args, **kwargs)
        
        # Additional GAN-specific setup
        self.setup_additional_losses()
    
    def setup_additional_losses(self):
        """Setup additional loss functions for improved training."""
        loss_config = self.config.get('loss', {})
        
        # Feature matching loss
        if loss_config.get('use_feature_matching', False):
            from .losses import FeatureMatchingLoss
            self.feature_matching_loss = FeatureMatchingLoss(
                weight=loss_config.get('feature_matching_weight', 1.0)
            )
        else:
            self.feature_matching_loss = None
        
        # Spectral loss
        if loss_config.get('use_spectral_loss', False):
            from .losses import SpectralLoss
            self.spectral_loss = SpectralLoss(
                weight=loss_config.get('spectral_loss_weight', 1.0)
            )
        else:
            self.spectral_loss = None
