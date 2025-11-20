"""
Critic (Discriminator) model for ECoG GAN.

This module implements the WindowCritic architecture that processes signals
in overlapping windows with temporal and spatial attention mechanisms.
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from typing import Tuple, Optional
from .attention import TemporalAE, SpatialAE, LearnedPE, ConditionalSpatialAE, SingleChannelSpatialAttention


class WindowCritic(nn.Module):
    """
    Window-based critic that processes signals in overlapping windows.
    
    The critic uses temporal and spatial attention mechanisms to analyze
    local patterns in ECoG signals and provides a global assessment.
    """
    
    def __init__(self, 
                 time_window: float, 
                 fs: int, 
                 channels: int, 
                 embedding_dim: int,
                 max_window_nums: int = 11, 
                 use_PE: bool = False,
                 attention_config: Optional[dict] = None):
        """
        Initialize WindowCritic.
        
        Args:
            time_window: Duration of each window in seconds
            fs: Sampling frequency
            channels: Number of input channels
            embedding_dim: Dimension of window embeddings
            max_window_nums: Maximum number of windows for positional encoding
            use_PE: Whether to use positional encoding
            attention_config: Configuration for attention mechanisms
                - spatial_attention_type: 'conditional', 'embedding', or 'standard' (default: 'conditional')
                - embedding_dim: For embedding approach (default: 8)
                - num_heads: Number of attention heads (default: 1)
                - dropout: Dropout probability (default: 0.1)
        """
        super(WindowCritic, self).__init__()
        self.window_size = int(time_window * fs)
        self.channels = channels
        self.embedding_dim = embedding_dim
        self.stride = int(self.window_size * 0.50)  # 50% overlap
        self.max_window_nums = max_window_nums
        self.use_PE = use_PE
        
        # Default attention configuration
        att_config = attention_config or {}
        spatial_type = att_config.get('spatial_attention_type', 'conditional')
        num_heads = att_config.get('num_heads', 1)
        dropout = att_config.get('dropout', 0.1)

        # Temporal attention (always works for any number of channels)
        # For temporal attention, num_heads must be compatible with channels
        temporal_heads = min(num_heads, channels)
        self.temporal_ae = TemporalAE(
            sample_shape=(1, channels, self.window_size),
            num_heads=temporal_heads,
            dropout=dropout
        )

        # Spatial attention with different approaches
        if spatial_type == 'conditional':
            # Approach 1: Conditional spatial attention (skip for single channel)
            self.spatial_ae = ConditionalSpatialAE(
                sample_shape=(1, channels, self.window_size),
                num_heads=num_heads,
                dropout=dropout
            )
        elif spatial_type == 'embedding':
            # Approach 2: Feature embedding for single channel
            embedding_dim = att_config.get('embedding_dim', 8)
            self.spatial_ae = SingleChannelSpatialAttention(
                time_points=self.window_size,
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                dropout=dropout
            )
        else:  # 'standard'
            # Original approach (may fail for single channel)
            self.spatial_ae = SpatialAE(
                sample_shape=(1, channels, self.window_size),
                num_heads=num_heads,
                dropout=dropout
            )

        # Feature fusion and pooling
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fuse = spectral_norm(nn.Linear(channels * 2, embedding_dim))

        # Positional encoding for window sequences
        if use_PE:
            self.positional_encoding = LearnedPE(
                embedding_dim=embedding_dim, 
                max_len=max_window_nums
            )

        # Global encoder over windows
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=4, 
            dim_feedforward=256, 
            batch_first=True
        )
        self.global_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Final classification layer
        self.final = spectral_norm(nn.Linear(embedding_dim, 1))

    def window_extraction(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract overlapping windows from input signals.
        
        Args:
            x: Input tensor of shape [batch, channels, samples]
            
        Returns:
            Windows tensor of shape [batch, num_windows, channels, window_size]
        """
        batch_size, channels, total_samples = x.shape
        
        # Calculate number of windows
        num_windows = (total_samples - self.window_size) // self.stride + 1
        
        # Extract windows
        windows = []
        for i in range(0, total_samples - self.window_size + 1, self.stride):
            window = x[:, :, i:i + self.window_size]  # [batch, channels, window_size]
            windows.append(window)
        
        # Stack windows: [batch, num_windows, channels, window_size]
        return torch.stack(windows, dim=1)

    def process_window(self, window: torch.Tensor) -> torch.Tensor:
        """
        Process a single window through attention mechanisms.
        
        Args:
            window: Window tensor of shape [batch, channels, window_size]
            
        Returns:
            Window features of shape [batch, embedding_dim]
        """
        # Apply temporal and spatial attention
        temporal_features = self.temporal_ae(window)
        spatial_features = self.spatial_ae(window)
        
        # Concatenate features along channel dimension
        fused_features = torch.cat([temporal_features, spatial_features], dim=1)
        
        # Global average pooling
        pooled_features = self.pool(fused_features).squeeze(-1)  # [batch, channels*2]
        
        # Project to embedding dimension
        window_embedding = self.fuse(pooled_features)  # [batch, embedding_dim]
        
        return window_embedding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through WindowCritic.
        
        Args:
            x: Input tensor of shape [batch, channels, samples]
            
        Returns:
            Critic output of shape [batch, 1]
        """
        batch_size, channels, samples = x.shape
        
        # Extract windows
        windows = self.window_extraction(x)  # [batch, num_windows, channels, window_size]
        batch_size, num_windows, channels, window_size = windows.shape
        
        # Process each window
        window_features = []
        for i in range(num_windows):
            window = windows[:, i]  # [batch, channels, window_size]
            features = self.process_window(window)
            window_features.append(features)
        
        # Stack window features: [batch, num_windows, embedding_dim]
        window_features = torch.stack(window_features, dim=1)
        
        # Apply positional encoding if enabled
        if self.use_PE:
            window_features = self.positional_encoding(window_features)
        
        # Global processing across windows
        global_features = self.global_encoder(window_features)  # [batch, num_windows, embedding_dim]
        
        # Aggregate across windows (mean pooling)
        aggregated_features = global_features.mean(dim=1)  # [batch, embedding_dim]
        
        # Final classification
        output = self.final(aggregated_features)  # [batch, 1]
        
        return output

    def get_model_info(self) -> dict:
        """Get information about the model architecture."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': 'WindowCritic',
            'window_size': self.window_size,
            'channels': self.channels,
            'embedding_dim': self.embedding_dim,
            'stride': self.stride,
            'max_window_nums': self.max_window_nums,
            'use_PE': self.use_PE,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }


class AdaptiveCritic(WindowCritic):
    """
    Adaptive Critic that can handle variable input dimensions.
    
    This version automatically adapts to different channel numbers and
    sequence lengths.
    """
    
    def __init__(self, 
                 time_window: float, 
                 fs: int, 
                 channels: int, 
                 embedding_dim: int,
                 adaptive_stride: bool = True,
                 min_windows: int = 3,
                 **kwargs):
        """
        Initialize Adaptive Critic.
        
        Args:
            time_window: Duration of each window in seconds
            fs: Sampling frequency
            channels: Number of input channels
            embedding_dim: Dimension of window embeddings
            adaptive_stride: Whether to adapt stride based on input length
            min_windows: Minimum number of windows to extract
            **kwargs: Additional arguments for parent class
        """
        self.adaptive_stride = adaptive_stride
        self.min_windows = min_windows
        super().__init__(time_window, fs, channels, embedding_dim, **kwargs)
    
    def window_extraction(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adaptive window extraction that adjusts stride based on input length.
        
        Args:
            x: Input tensor of shape [batch, channels, samples]
            
        Returns:
            Windows tensor of shape [batch, num_windows, channels, window_size]
        """
        if not self.adaptive_stride:
            return super().window_extraction(x)
        
        batch_size, channels, total_samples = x.shape
        
        # Adaptive stride calculation
        if total_samples <= self.window_size:
            # If input is shorter than window size, use the entire signal
            return x.unsqueeze(1)  # [batch, 1, channels, samples]
        
        # Calculate adaptive stride to ensure minimum number of windows
        max_stride = (total_samples - self.window_size) // (self.min_windows - 1)
        adaptive_stride = min(self.stride, max_stride)
        adaptive_stride = max(1, adaptive_stride)  # Ensure stride is at least 1
        
        # Extract windows with adaptive stride
        windows = []
        for i in range(0, total_samples - self.window_size + 1, adaptive_stride):
            window = x[:, :, i:i + self.window_size]
            windows.append(window)
            
            # Stop if we have enough windows
            if len(windows) >= self.max_window_nums:
                break
        
        return torch.stack(windows, dim=1)
    
    def get_adaptive_info(self) -> dict:
        """Get information about adaptive architecture."""
        info = self.get_model_info()
        info.update({
            'adaptive_stride': self.adaptive_stride,
            'min_windows': self.min_windows,
            'architecture_type': 'Adaptive Critic'
        })
        return info
