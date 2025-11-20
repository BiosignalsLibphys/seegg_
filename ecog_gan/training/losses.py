"""
Loss functions for ECoG GAN training.

This module implements various loss functions including WGAN-GP loss
with gradient penalty computation.
"""

import torch
import torch.nn as nn
from typing import Tuple


def compute_gradient_penalty(critic: nn.Module, 
                           real_samples: torch.Tensor, 
                           fake_samples: torch.Tensor,
                           device: torch.device) -> torch.Tensor:
    """
    Compute gradient penalty for WGAN-GP.
    
    Args:
        critic: Critic/discriminator model
        real_samples: Real data samples
        fake_samples: Generated fake samples
        device: Device to compute on
        
    Returns:
        Gradient penalty value
    """
    batch_size = real_samples.size(0)
    
    # Random interpolation factor
    alpha = torch.rand(batch_size, 1, 1, device=device)
    
    # Interpolate between real and fake samples
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    
    # Get critic output for interpolated samples
    d_interpolates = critic(interpolates)
    
    # Create fake labels for gradient computation
    fake_labels = torch.ones(d_interpolates.size(), device=device, requires_grad=False)
    
    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake_labels,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Reshape gradients and compute penalty
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty


class WGANGPLoss:
    """Wasserstein GAN with Gradient Penalty loss."""
    
    def __init__(self, lambda_gp: float = 10.0):
        """
        Initialize WGAN-GP loss.
        
        Args:
            lambda_gp: Gradient penalty coefficient
        """
        self.lambda_gp = lambda_gp
    
    def critic_loss(self, 
                   critic: nn.Module,
                   real_samples: torch.Tensor,
                   fake_samples: torch.Tensor,
                   device: torch.device) -> Tuple[torch.Tensor, dict]:
        """
        Compute critic loss with gradient penalty.
        
        Args:
            critic: Critic model
            real_samples: Real data samples
            fake_samples: Generated samples (detached)
            device: Device to compute on
            
        Returns:
            Loss value and loss components dictionary
        """
        # Get critic outputs
        real_output = critic(real_samples)
        fake_output = critic(fake_samples)
        
        # Wasserstein distance
        wasserstein_distance = -torch.mean(real_output) + torch.mean(fake_output)
        
        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(critic, real_samples, fake_samples, device)
        
        # Total critic loss
        critic_loss = wasserstein_distance + self.lambda_gp * gradient_penalty
        
        # Loss components for monitoring
        loss_components = {
            'critic_loss': critic_loss.item(),
            'wasserstein_distance': wasserstein_distance.item(),
            'gradient_penalty': gradient_penalty.item(),
            'real_output_mean': torch.mean(real_output).item(),
            'fake_output_mean': torch.mean(fake_output).item()
        }
        
        return critic_loss, loss_components
    
    def generator_loss(self, 
                      critic: nn.Module,
                      fake_samples: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Compute generator loss.
        
        Args:
            critic: Critic model
            fake_samples: Generated samples
            
        Returns:
            Loss value and loss components dictionary
        """
        fake_output = critic(fake_samples)
        generator_loss = -torch.mean(fake_output)
        
        loss_components = {
            'generator_loss': generator_loss.item(),
            'fake_output_mean': torch.mean(fake_output).item()
        }
        
        return generator_loss, loss_components


class AdversarialLoss:
    """Standard adversarial loss (alternative to WGAN-GP)."""
    
    def __init__(self, loss_type: str = 'bce'):
        """
        Initialize adversarial loss.
        
        Args:
            loss_type: Type of loss ('bce', 'mse', 'hinge')
        """
        self.loss_type = loss_type
        
        if loss_type == 'bce':
            self.criterion = nn.BCEWithLogitsLoss()
        elif loss_type == 'mse':
            self.criterion = nn.MSELoss()
        elif loss_type == 'hinge':
            self.criterion = None  # Implemented manually
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def critic_loss(self,
                   critic: nn.Module,
                   real_samples: torch.Tensor,
                   fake_samples: torch.Tensor,
                   device: torch.device) -> Tuple[torch.Tensor, dict]:
        """Compute critic loss for standard adversarial training."""
        batch_size = real_samples.size(0)
        
        # Labels
        real_labels = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)
        
        # Get outputs
        real_output = critic(real_samples)
        fake_output = critic(fake_samples)
        
        if self.loss_type == 'hinge':
            # Hinge loss
            real_loss = torch.mean(torch.relu(1.0 - real_output))
            fake_loss = torch.mean(torch.relu(1.0 + fake_output))
            critic_loss = real_loss + fake_loss
        else:
            # BCE or MSE loss
            real_loss = self.criterion(real_output, real_labels)
            fake_loss = self.criterion(fake_output, fake_labels)
            critic_loss = real_loss + fake_loss
        
        loss_components = {
            'critic_loss': critic_loss.item(),
            'real_loss': real_loss.item(),
            'fake_loss': fake_loss.item(),
            'real_output_mean': torch.mean(real_output).item(),
            'fake_output_mean': torch.mean(fake_output).item()
        }
        
        return critic_loss, loss_components
    
    def generator_loss(self,
                      critic: nn.Module,
                      fake_samples: torch.Tensor,
                      device: torch.device) -> Tuple[torch.Tensor, dict]:
        """Compute generator loss for standard adversarial training."""
        batch_size = fake_samples.size(0)
        real_labels = torch.ones(batch_size, 1, device=device)
        
        fake_output = critic(fake_samples)
        
        if self.loss_type == 'hinge':
            generator_loss = -torch.mean(fake_output)
        else:
            generator_loss = self.criterion(fake_output, real_labels)
        
        loss_components = {
            'generator_loss': generator_loss.item(),
            'fake_output_mean': torch.mean(fake_output).item()
        }
        
        return generator_loss, loss_components


class FeatureMatchingLoss:
    """Feature matching loss for improved training stability."""
    
    def __init__(self, weight: float = 1.0):
        """
        Initialize feature matching loss.
        
        Args:
            weight: Weight for feature matching loss
        """
        self.weight = weight
        self.mse_loss = nn.MSELoss()
    
    def __call__(self,
                real_features: torch.Tensor,
                fake_features: torch.Tensor) -> torch.Tensor:
        """
        Compute feature matching loss.
        
        Args:
            real_features: Features from real samples
            fake_features: Features from fake samples
            
        Returns:
            Feature matching loss
        """
        return self.weight * self.mse_loss(fake_features, real_features.detach())


class SpectralLoss:
    """Spectral loss for frequency domain matching."""
    
    def __init__(self, weight: float = 1.0):
        """
        Initialize spectral loss.
        
        Args:
            weight: Weight for spectral loss
        """
        self.weight = weight
        self.mse_loss = nn.MSELoss()
    
    def __call__(self,
                real_samples: torch.Tensor,
                fake_samples: torch.Tensor) -> torch.Tensor:
        """
        Compute spectral loss using FFT.
        
        Args:
            real_samples: Real signal samples
            fake_samples: Generated signal samples
            
        Returns:
            Spectral loss
        """
        # Compute FFT
        real_fft = torch.fft.fft(real_samples, dim=-1)
        fake_fft = torch.fft.fft(fake_samples, dim=-1)
        
        # Compute power spectral density
        real_psd = torch.abs(real_fft) ** 2
        fake_psd = torch.abs(fake_fft) ** 2
        
        # MSE loss in frequency domain
        spectral_loss = self.mse_loss(fake_psd, real_psd)
        
        return self.weight * spectral_loss


def get_loss_function(loss_config: dict):
    """
    Factory function to create loss functions based on configuration.

    Args:
        loss_config: Loss configuration dictionary

    Returns:
        Loss function instance
    """
    loss_type = loss_config.get('type', 'wgan_gp')

    if loss_type == 'wgan_gp':
        return WGANGPLoss(lambda_gp=loss_config.get('lambda_gp', 10.0))
    elif loss_type == 'adversarial':
        return AdversarialLoss(loss_type=loss_config.get('adversarial_type', 'bce'))
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
