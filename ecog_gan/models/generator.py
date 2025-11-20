"""
Generator model for ECoG GAN.

This module implements the Generator architecture with transposed convolutions
and optional spatial attention mechanisms for synthetic ECoG signal generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from .attention import SpatialAE, ConditionalSpatialAE, SingleChannelSpatialAttention


class Generator(nn.Module):
    """
    Generator model for creating synthetic ECoG signals.
    
    The generator uses transposed convolutions to upsample from a latent vector
    to the target signal shape, with optional spatial attention mechanisms.
    """
    
    def __init__(self, 
                 latent_dim: int, 
                 out_channels: int, 
                 target_shape: Tuple[int, int, int],
                 use_attention: bool = False,
                 attention_config: Optional[dict] = None):
        """
        Initialize Generator.

        Args:
            latent_dim: Dimension of the latent (z) vector
            out_channels: Base channel size for internal layers
            target_shape: Desired output shape (batch, channels, samples)
            use_attention: Whether to use spatial attention
            attention_config: Configuration for attention mechanism
                - spatial_attention_type: 'conditional', 'embedding', or 'standard' (default: 'conditional')
                - embedding_dim: For embedding approach (default: 8)
                - num_heads: Number of attention heads (default: 4)
                - dropout: Dropout probability (default: 0.1)
        """
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.target_shape = target_shape
        self.target_height = target_shape[1]  # Number of channels
        self.target_width = target_shape[2]   # Number of samples
        self.use_attention = use_attention
        
        # Calculate initial dimensions for upsampling
        self.initial_height = max(1, self.target_height // 10)
        self.initial_width = max(1, self.target_width // 64)
        
        # Main generator network
        self.model = self._build_generator()
        
        # Optional spatial attention with different approaches
        if use_attention:
            att_config = attention_config or {}
            spatial_type = att_config.get('spatial_attention_type', 'conditional')
            num_heads = att_config.get('num_heads', 4)
            dropout = att_config.get('dropout', 0.1)

            if spatial_type == 'conditional':
                # Approach 1: Conditional spatial attention (skip for single channel)
                self.spatial_attention = ConditionalSpatialAE(
                    sample_shape=(1, self.target_height, self.target_width),
                    num_heads=num_heads,
                    dropout=dropout
                )
            elif spatial_type == 'embedding':
                # Approach 2: Feature embedding for single channel
                embedding_dim = att_config.get('embedding_dim', 8)
                self.spatial_attention = SingleChannelSpatialAttention(
                    time_points=self.target_width,
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    dropout=dropout
                )
            else:  # 'standard'
                # Original approach (may fail for single channel)
                self.spatial_attention = SpatialAE(
                    sample_shape=(1, self.target_height, self.target_width),
                    num_heads=num_heads,
                    dropout=dropout
                )
        
        self._initialize_weights()

    def _build_generator(self) -> nn.Sequential:
        """Build the main generator network."""
        layers = []
        
        # Initial linear layer to project latent vector
        initial_size = self.out_channels * 8 * self.initial_width
        layers.extend([
            nn.Linear(self.latent_dim, initial_size),
            nn.LeakyReLU(0.2, inplace=True),
            # Reshape to [batch, channels, width] for 1D convolutions
            nn.Unflatten(dim=1, unflattened_size=(self.out_channels * 8, self.initial_width))
        ])
        
        # Transposed convolution layers for upsampling
        conv_layers = [
            # Layer 1: Upsample temporal dimension
            (self.out_channels * 8, self.out_channels * 6, 4, 2, 1),
            # Layer 2
            (self.out_channels * 6, self.out_channels * 4, 4, 2, 1),
            # Layer 3
            (self.out_channels * 4, self.out_channels * 2, 4, 2, 1),
            # Layer 4
            (self.out_channels * 2, self.out_channels, 4, 2, 1),
            # Layer 5
            (self.out_channels, self.out_channels // 2, 4, 2, 1),
            # Final layer to target channels
            (self.out_channels // 2, self.target_height, 4, 2, 1)
        ]
        
        for i, (in_ch, out_ch, kernel, stride, padding) in enumerate(conv_layers):
            layers.append(nn.ConvTranspose1d(in_ch, out_ch, kernel, stride, padding, bias=False))
            
            # Add batch norm and activation for all layers except the last
            if i < len(conv_layers) - 1:
                layers.extend([
                    nn.BatchNorm1d(out_ch),
                    nn.LeakyReLU(0.2, inplace=True)
                ])
            else:
                # Final activation
                layers.append(nn.Tanh())
        
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through generator.
        
        Args:
            x: Latent vector of shape [batch_size, latent_dim]
            
        Returns:
            Generated signals of shape [batch_size, channels, samples]
        """
        # Pass through main generator network
        x = self.model(x)
        
        # Apply spatial attention if enabled
        if self.use_attention:
            x = self.spatial_attention(x)
        
        # Interpolate to exact target width if needed
        if x.size(-1) != self.target_width:
            x = F.interpolate(
                x,
                size=self.target_width,
                mode='linear',
                align_corners=False
            )
        
        return x

    def generate_samples(self, 
                        n_samples: int, 
                        device: torch.device,
                        return_latent: bool = False) -> torch.Tensor:
        """
        Generate synthetic samples.
        
        Args:
            n_samples: Number of samples to generate
            device: Device to generate samples on
            return_latent: Whether to also return the latent vectors
            
        Returns:
            Generated samples, optionally with latent vectors
        """
        self.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim, device=device)
            samples = self(z)
            
            if return_latent:
                return samples, z
            return samples

    def get_model_info(self) -> dict:
        """Get information about the model architecture."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': 'Generator',
            'latent_dim': self.latent_dim,
            'out_channels': self.out_channels,
            'target_shape': self.target_shape,
            'use_attention': self.use_attention,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'initial_dimensions': (self.initial_height, self.initial_width)
        }


class AdaptiveGenerator(Generator):
    """
    Adaptive Generator that can handle variable input/output dimensions.
    
    This version automatically adapts to different channel numbers and
    sequence lengths based on the target shape.
    """
    
    def __init__(self, 
                 latent_dim: int, 
                 out_channels: int, 
                 target_shape: Tuple[int, int, int],
                 adaptive_layers: bool = True,
                 **kwargs):
        """
        Initialize Adaptive Generator.
        
        Args:
            latent_dim: Dimension of the latent vector
            out_channels: Base channel size for internal layers
            target_shape: Target output shape
            adaptive_layers: Whether to use adaptive layer sizing
            **kwargs: Additional arguments for parent class
        """
        self.adaptive_layers = adaptive_layers
        super().__init__(latent_dim, out_channels, target_shape, **kwargs)
    
    def _build_generator(self) -> nn.Sequential:
        """Build adaptive generator network."""
        if not self.adaptive_layers:
            return super()._build_generator()
        
        # Adaptive layer sizing based on target dimensions
        layers = []
        
        # Calculate optimal initial dimensions
        target_samples = self.target_width
        target_channels = self.target_height
        
        # Determine number of upsampling layers needed
        num_layers = max(4, int(torch.log2(torch.tensor(target_samples / 16.0)).ceil().item()))
        
        # Initial projection
        initial_width = max(1, target_samples // (2 ** num_layers))
        initial_size = self.out_channels * 8 * initial_width
        
        layers.extend([
            nn.Linear(self.latent_dim, initial_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Unflatten(dim=1, unflattened_size=(self.out_channels * 8, initial_width))
        ])
        
        # Adaptive transposed convolution layers
        current_channels = self.out_channels * 8
        
        for i in range(num_layers):
            next_channels = max(target_channels, current_channels // 2)
            
            # Final layer outputs target channels
            if i == num_layers - 1:
                next_channels = target_channels
            
            layers.append(nn.ConvTranspose1d(
                current_channels, next_channels, 
                kernel_size=4, stride=2, padding=1, bias=False
            ))
            
            if i < num_layers - 1:
                layers.extend([
                    nn.BatchNorm1d(next_channels),
                    nn.LeakyReLU(0.2, inplace=True)
                ])
            else:
                layers.append(nn.Tanh())
            
            current_channels = next_channels
        
        return nn.Sequential(*layers)

    def get_adaptive_info(self) -> dict:
        """Get information about adaptive architecture."""
        info = self.get_model_info()
        info.update({
            'adaptive_layers': self.adaptive_layers,
            'architecture_type': 'Adaptive Generator'
        })
        return info
