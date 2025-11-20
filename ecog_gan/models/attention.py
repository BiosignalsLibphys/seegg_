"""
Attention mechanisms for ECoG GAN models.

This module implements various attention mechanisms including temporal and spatial
attention with learned positional encoding for better sequence modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class LearnedPE(nn.Module):
    """Learned positional encoding for sequences."""
    
    def __init__(self, embedding_dim: int, max_len: int):
        """
        Initialize learned positional encoding.
        
        Args:
            embedding_dim: Dimension of embeddings
            max_len: Maximum sequence length
        """
        super(LearnedPE, self).__init__()
        self.encoding = nn.Parameter(torch.zeros(max_len, embedding_dim))
        nn.init.xavier_uniform_(self.encoding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply positional encoding to input.
        
        Args:
            x: Input tensor of shape [batch, seq_len, embedding_dim]
            
        Returns:
            Input with positional encoding added
        """
        return x + self.encoding[:x.size(1), :].unsqueeze(0)


class TemporalAE(nn.Module):
    """Temporal attention encoder with feedforward network."""
    
    def __init__(self, sample_shape: Tuple[int, int, int], num_heads: int, dropout: float):
        """
        Initialize temporal attention encoder.
        
        Args:
            sample_shape: Shape of input samples (batch, channels, time_points)
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.T_points = sample_shape[2]
        self.N_channels = sample_shape[1]
        self.attn_axis = 'temporal'
        
        # Positional encoding
        self.positional_encoding = LearnedPE(
            max_len=self.T_points, 
            embedding_dim=self.N_channels
        )

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=self.N_channels, 
            num_heads=num_heads, 
            batch_first=True
        )
        self.relu = nn.ReLU()

        self.ffn = nn.Sequential(
            nn.Linear(self.N_channels, self.N_channels * 2),
            nn.ReLU(),
            nn.Linear(self.N_channels * 2, self.N_channels),
            nn.ReLU()
        )

        self.layer_norm = nn.LayerNorm(self.N_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through temporal attention.
        
        Args:
            x: Input tensor of shape [batch, channels, time_points]
            
        Returns:
            Output tensor of shape [batch, channels, time_points]
        """
        # Transpose to [batch, time_points, channels] for attention
        t0 = x.transpose(1, 2)
        
        # Add positional encoding
        t1 = self.positional_encoding(t0)
        
        # Multi-head attention
        attn_output, attn_weights = self.multihead_attn(t1, t1, t1)
        # Expose attention weights for hooks (shape: B x T x T)
        self.last_attn = attn_weights.detach()
        
        # Residual connection with layer norm and dropout
        t2 = self.layer_norm(self.dropout(t1 + self.relu(attn_output)))
        
        # Feedforward network with residual connection
        t3 = self.layer_norm(self.dropout(t2 + self.ffn(t2)))

        # Transpose back to [batch, channels, time_points]
        return t3.transpose(1, 2)


class SpatialAE(nn.Module):
    """Spatial attention encoder with feedforward network."""
    
    def __init__(self, sample_shape: Tuple[int, int, int], num_heads: int, dropout: float):
        """
        Initialize spatial attention encoder.
        
        Args:
            sample_shape: Shape of input samples (batch, channels, time_points)
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.T_points = sample_shape[2]
        self.N_channels = sample_shape[1]
        self.attn_axis = 'channel'
        
        # Positional encoding for channels
        self.positional_encoding = LearnedPE(
            max_len=self.N_channels, 
            embedding_dim=self.T_points
        )

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=self.T_points, 
            num_heads=num_heads, 
            batch_first=True
        )
        self.relu = nn.ReLU()

        self.ffn = nn.Sequential(
            nn.Linear(self.T_points, self.T_points * 2),
            nn.ReLU(),
            nn.Linear(self.T_points * 2, self.T_points), 
            nn.ReLU()
        )

        self.layer_norm = nn.LayerNorm(self.T_points)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through spatial attention.
        
        Args:
            x: Input tensor of shape [batch, channels, time_points]
            
        Returns:
            Output tensor of shape [batch, channels, time_points]
        """
        # Input is already [batch, channels, time_points] - perfect for spatial attention
        
        # Add positional encoding
        t1 = self.positional_encoding(x)
        
        # Multi-head attention
        attn_output, attn_weights = self.multihead_attn(t1, t1, t1)
        # Expose attention weights for hooks (shape: B x C x C)
        self.last_attn = attn_weights.detach()
        
        # Residual connection with layer norm and dropout
        t2 = self.layer_norm(self.dropout(t1 + self.relu(attn_output)))
        
        # Feedforward network with residual connection
        t3 = self.layer_norm(self.dropout(t2 + self.ffn(t2)))

        return t3


class TemporalAttBuilder(nn.Module):
    """Simplified temporal attention builder without FFN."""
    
    def __init__(self, sample_shape: Tuple[int, int, int], num_heads: int, dropout: float):
        """
        Initialize temporal attention builder.
        
        Args:
            sample_shape: Shape of input samples (batch, channels, time_points)
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.T_points = sample_shape[2]
        self.N_channels = sample_shape[1]
        
        # Positional encoding
        self.positional_encoding = LearnedPE(
            max_len=self.T_points, 
            embedding_dim=self.N_channels
        )

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=self.N_channels, 
            num_heads=num_heads, 
            batch_first=True
        )
    
        self.layer_norm = nn.LayerNorm(self.N_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through temporal attention builder.
        
        Args:
            x: Input tensor of shape [batch, channels, time_points]
            
        Returns:
            Output tensor of shape [batch, channels, time_points]
        """
        # Transpose to [batch, time_points, channels] for attention
        t0 = x.transpose(1, 2)
        
        # Add positional encoding
        t1 = self.positional_encoding(t0)
        
        # Multi-head attention
        attn_output, _ = self.multihead_attn(t1, t1, t1)
        
        # Residual connection with layer norm and dropout
        t2 = self.layer_norm(t1 + self.dropout(attn_output))

        # Transpose back to [batch, channels, time_points]
        return t2.transpose(1, 2)


class SpatialAttBuilder(nn.Module):
    """Simplified spatial attention builder without FFN."""
    
    def __init__(self, sample_shape: Tuple[int, int, int], num_heads: int, dropout: float):
        """
        Initialize spatial attention builder.
        
        Args:
            sample_shape: Shape of input samples (batch, channels, time_points)
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.T_points = sample_shape[2]
        self.N_channels = sample_shape[1]
        
        # Positional encoding for channels
        self.positional_encoding = LearnedPE(
            max_len=self.N_channels, 
            embedding_dim=self.T_points
        )

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=self.T_points, 
            num_heads=num_heads, 
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(self.T_points)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through spatial attention builder.
        
        Args:
            x: Input tensor of shape [batch, channels, time_points]
            
        Returns:
            Output tensor of shape [batch, channels, time_points]
        """
        # Add positional encoding
        t1 = self.positional_encoding(x)
        
        # Multi-head attention
        attn_output, _ = self.multihead_attn(t1, t1, t1)
        
        # Residual connection with layer norm and dropout
        t2 = self.layer_norm(t1 + self.dropout(attn_output))

        return t2


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (alternative to learned PE)."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Initialize sinusoidal positional encoding.

        Args:
            d_model: Model dimension
            dropout: Dropout probability
            max_len: Maximum sequence length
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply sinusoidal positional encoding.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embedding_dim)

        Returns:
            Input with positional encoding added
        """
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)


class ConditionalSpatialAE(nn.Module):
    """
    Conditional spatial attention that only applies when channels > 1.
    For single-channel data, it acts as an identity function.
    """

    def __init__(self, sample_shape: Tuple[int, int, int], num_heads: int, dropout: float):
        """
        Initialize conditional spatial attention.

        Args:
            sample_shape: Shape of input samples (batch, channels, time_points)
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.T_points = sample_shape[2]
        self.N_channels = sample_shape[1]
        self.use_attention = self.N_channels > 1
        self.attn_axis = 'channel'

        if self.use_attention:
            # Only create attention layers if we have multiple channels
            self.positional_encoding = LearnedPE(
                max_len=self.N_channels,
                embedding_dim=self.T_points
            )

            self.multihead_attn = nn.MultiheadAttention(
                embed_dim=self.T_points,
                num_heads=min(num_heads, self.N_channels),  # Ensure num_heads <= channels
                batch_first=True
            )
            self.relu = nn.ReLU()

            self.ffn = nn.Sequential(
                nn.Linear(self.T_points, self.T_points * 2),
                nn.ReLU(),
                nn.Linear(self.T_points * 2, self.T_points),
                nn.ReLU()
            )

            self.layer_norm = nn.LayerNorm(self.T_points)
            self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through conditional spatial attention.

        Args:
            x: Input tensor of shape [batch, channels, time_points]

        Returns:
            Output tensor of shape [batch, channels, time_points]
        """
        if not self.use_attention:
            # For single channel, just return input unchanged
            return x

        # Multi-channel: apply spatial attention
        t1 = self.positional_encoding(x)
        attn_output, attn_weights = self.multihead_attn(t1, t1, t1)
        # Expose attention weights for hooks (shape: B x C x C)
        self.last_attn = attn_weights.detach()
        t2 = self.layer_norm(self.dropout(t1 + self.relu(attn_output)))
        t3 = self.layer_norm(self.dropout(t2 + self.ffn(t2)))
        return t3


class SingleChannelSpatialAttention(nn.Module):
    """
    Spatial attention for single-channel data using feature embedding.
    Embeds single channel to multiple features, applies spatial attention, then projects back.
    """

    def __init__(self, time_points: int, embedding_dim: int = 8, num_heads: int = 4, dropout: float = 0.1):
        """
        Initialize single-channel spatial attention with feature embedding.

        Args:
            time_points: Number of time points in the signal
            embedding_dim: Number of feature channels to embed to
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.time_points = time_points
        self.embedding_dim = embedding_dim
        self.attn_axis = 'channel'

        # Ensure num_heads is compatible with time_points (not embedding_dim for spatial attention)
        # For spatial attention, embed_dim in MultiheadAttention is time_points, not embedding_dim
        if time_points % num_heads != 0:
            # Adjust num_heads to be compatible
            num_heads = min(num_heads, time_points)
            for h in range(num_heads, 0, -1):
                if time_points % h == 0:
                    num_heads = h
                    break

        # Embed single channel to multiple feature channels
        self.feature_embedding = nn.Conv1d(1, embedding_dim, kernel_size=1, bias=False)

        # Apply spatial attention on embedded features
        # Note: For spatial attention, the attention is over channels (embedding_dim)
        # but the embed_dim for MultiheadAttention is time_points
        self.positional_encoding = LearnedPE(
            max_len=embedding_dim,
            embedding_dim=time_points
        )

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=time_points,
            num_heads=num_heads,
            batch_first=True
        )
        self.relu = nn.ReLU()

        self.ffn = nn.Sequential(
            nn.Linear(time_points, time_points * 2),
            nn.ReLU(),
            nn.Linear(time_points * 2, time_points),
            nn.ReLU()
        )

        self.layer_norm = nn.LayerNorm(time_points)
        self.dropout = nn.Dropout(dropout)

        # Project back to single channel
        self.output_projection = nn.Conv1d(embedding_dim, 1, kernel_size=1, bias=False)

        # Initialize weights
        nn.init.xavier_uniform_(self.feature_embedding.weight)
        nn.init.xavier_uniform_(self.output_projection.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through single-channel spatial attention.

        Args:
            x: Input tensor of shape [batch, 1, time_points]

        Returns:
            Output tensor of shape [batch, 1, time_points]
        """
        # Embed to multiple feature channels
        embedded = self.feature_embedding(x)  # [batch, embedding_dim, time_points]

        # Apply spatial attention manually (similar to SpatialAE)
        # Add positional encoding
        t1 = self.positional_encoding(embedded)

        # Multi-head attention
        attn_output, attn_weights = self.multihead_attn(t1, t1, t1)
        # Expose attention weights for hooks (shape: B x E x E where E=embedding_dim)
        self.last_attn = attn_weights.detach()

        # Residual connection with layer norm and dropout
        t2 = self.layer_norm(self.dropout(t1 + self.relu(attn_output)))

        # Feedforward network with residual connection
        t3 = self.layer_norm(self.dropout(t2 + self.ffn(t2)))

        # Project back to single channel
        output = self.output_projection(t3)  # [batch, 1, time_points]

        # Add residual connection
        return output + x
