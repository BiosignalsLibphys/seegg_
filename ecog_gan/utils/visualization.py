"""
Visualization utilities for ECoG GAN.

This module provides functions for plotting signals, spectrograms,
correlation matrices, and other visualizations for ECoG data analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from typing import Union, Optional, Tuple, List
import torch
import logging

logger = logging.getLogger(__name__)

# Set matplotlib style
plt.style.use('default')
sns.set_palette("husl")


def plot_signals(signals: Union[np.ndarray, torch.Tensor],
                sampling_rate: int = 512,
                channels: Optional[List[str]] = None,
                time_range: Optional[Tuple[float, float]] = None,
                title: str = "ECoG Signals",
                figsize: Tuple[int, int] = (12, 8),
                save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot multichannel ECoG signals.
    
    Args:
        signals: Signal data of shape [channels, samples] or [samples, channels]
        sampling_rate: Sampling rate in Hz
        channels: List of channel names
        time_range: Optional time range to plot (start, end) in seconds
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save the plot
        
    Returns:
        matplotlib Figure object
    """
    if isinstance(signals, torch.Tensor):
        signals = signals.detach().cpu().numpy()
    
    # Ensure signals are [channels, samples]
    if signals.shape[0] > signals.shape[1]:
        signals = signals.T
    
    n_channels, n_samples = signals.shape
    time = np.arange(n_samples) / sampling_rate
    
    # Apply time range if specified
    if time_range is not None:
        start_idx = int(time_range[0] * sampling_rate)
        end_idx = int(time_range[1] * sampling_rate)
        time = time[start_idx:end_idx]
        signals = signals[:, start_idx:end_idx]
    
    fig, axes = plt.subplots(n_channels, 1, figsize=figsize, sharex=True)
    if n_channels == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        ax.plot(time, signals[i], linewidth=0.8)
        
        # Set channel label
        if channels and i < len(channels):
            ax.set_ylabel(channels[i])
        else:
            ax.set_ylabel(f'Ch {i+1}')
        
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Signal plot saved to {save_path}")
    
    return fig


def plot_spectrograms(signals: Union[np.ndarray, torch.Tensor],
                     sampling_rate: int = 512,
                     nperseg: int = 256,
                     noverlap: Optional[int] = None,
                     channels: Optional[List[str]] = None,
                     title: str = "ECoG Spectrograms",
                     figsize: Tuple[int, int] = (12, 8),
                     save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot spectrograms for multichannel ECoG signals.
    
    Args:
        signals: Signal data of shape [channels, samples]
        sampling_rate: Sampling rate in Hz
        nperseg: Length of each segment for STFT
        noverlap: Number of points to overlap between segments
        channels: List of channel names
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save the plot
        
    Returns:
        matplotlib Figure object
    """
    if isinstance(signals, torch.Tensor):
        signals = signals.detach().cpu().numpy()
    
    # Ensure signals are [channels, samples]
    if signals.shape[0] > signals.shape[1]:
        signals = signals.T
    
    n_channels, n_samples = signals.shape
    
    if noverlap is None:
        noverlap = nperseg // 2
    
    # Calculate number of rows and columns for subplots
    n_cols = min(4, n_channels)
    n_rows = (n_channels + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_channels == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_channels):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        
        # Compute spectrogram
        f, t, Sxx = signal.spectrogram(
            signals[i], 
            fs=sampling_rate,
            nperseg=nperseg,
            noverlap=noverlap
        )
        
        # Plot spectrogram
        im = ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud')
        
        # Set labels and title
        if channels and i < len(channels):
            ax.set_title(channels[i])
        else:
            ax.set_title(f'Channel {i+1}')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Power (dB)')
    
    # Hide empty subplots
    for i in range(n_channels, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        if n_rows > 1:
            axes[row, col].set_visible(False)
        else:
            axes[col].set_visible(False)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Spectrogram plot saved to {save_path}")
    
    return fig


def plot_correlation_matrix(signals: Union[np.ndarray, torch.Tensor],
                          channels: Optional[List[str]] = None,
                          method: str = 'pearson',
                          title: str = "Channel Correlation Matrix",
                          figsize: Tuple[int, int] = (10, 8),
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot correlation matrix between channels.
    
    Args:
        signals: Signal data of shape [channels, samples]
        channels: List of channel names
        method: Correlation method ('pearson', 'spearman', 'kendall')
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save the plot
        
    Returns:
        matplotlib Figure object
    """
    if isinstance(signals, torch.Tensor):
        signals = signals.detach().cpu().numpy()
    
    # Ensure signals are [channels, samples]
    if signals.shape[0] > signals.shape[1]:
        signals = signals.T
    
    # Compute correlation matrix
    if method == 'pearson':
        corr_matrix = np.corrcoef(signals)
    else:
        from scipy.stats import spearmanr, kendalltau
        if method == 'spearman':
            corr_matrix, _ = spearmanr(signals, axis=1)
        elif method == 'kendall':
            n_channels = signals.shape[0]
            corr_matrix = np.zeros((n_channels, n_channels))
            for i in range(n_channels):
                for j in range(n_channels):
                    corr_matrix[i, j], _ = kendalltau(signals[i], signals[j])
        else:
            raise ValueError(f"Unknown correlation method: {method}")
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    
    # Set ticks and labels
    n_channels = signals.shape[0]
    if channels and len(channels) == n_channels:
        ax.set_xticks(range(n_channels))
        ax.set_yticks(range(n_channels))
        ax.set_xticklabels(channels, rotation=45, ha='right')
        ax.set_yticklabels(channels)
    else:
        ax.set_xticks(range(0, n_channels, max(1, n_channels // 10)))
        ax.set_yticks(range(0, n_channels, max(1, n_channels // 10)))
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation Coefficient')
    
    # Add correlation values as text
    if n_channels <= 20:  # Only show text for small matrices
        for i in range(n_channels):
            for j in range(n_channels):
                text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)
    
    ax.set_title(f'{title} ({method.capitalize()})')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Correlation matrix plot saved to {save_path}")
    
    return fig


def plot_power_spectrum(signals: Union[np.ndarray, torch.Tensor],
                       sampling_rate: int = 512,
                       channels: Optional[List[str]] = None,
                       freq_range: Optional[Tuple[float, float]] = None,
                       title: str = "Power Spectral Density",
                       figsize: Tuple[int, int] = (12, 6),
                       save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot power spectral density for multichannel signals.
    
    Args:
        signals: Signal data of shape [channels, samples]
        sampling_rate: Sampling rate in Hz
        channels: List of channel names
        freq_range: Optional frequency range to plot (low, high) in Hz
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save the plot
        
    Returns:
        matplotlib Figure object
    """
    if isinstance(signals, torch.Tensor):
        signals = signals.detach().cpu().numpy()
    
    # Ensure signals are [channels, samples]
    if signals.shape[0] > signals.shape[1]:
        signals = signals.T
    
    n_channels, n_samples = signals.shape
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for i in range(n_channels):
        # Compute power spectral density
        f, Pxx = signal.welch(signals[i], fs=sampling_rate, nperseg=min(1024, n_samples//4))
        
        # Apply frequency range if specified
        if freq_range is not None:
            mask = (f >= freq_range[0]) & (f <= freq_range[1])
            f = f[mask]
            Pxx = Pxx[mask]
        
        # Plot PSD
        if channels and i < len(channels):
            label = channels[i]
        else:
            label = f'Channel {i+1}'
        
        ax.semilogy(f, Pxx, label=label, alpha=0.8)
    
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power Spectral Density')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Power spectrum plot saved to {save_path}")
    
    return fig


def compare_signals(real_signals: Union[np.ndarray, torch.Tensor],
                   fake_signals: Union[np.ndarray, torch.Tensor],
                   sampling_rate: int = 512,
                   channel_idx: int = 0,
                   time_range: Optional[Tuple[float, float]] = None,
                   title: str = "Real vs Generated Signals",
                   figsize: Tuple[int, int] = (12, 6),
                   save_path: Optional[str] = None) -> plt.Figure:
    """
    Compare real and generated signals.
    
    Args:
        real_signals: Real signal data
        fake_signals: Generated signal data
        sampling_rate: Sampling rate in Hz
        channel_idx: Channel index to compare
        time_range: Optional time range to plot
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save the plot
        
    Returns:
        matplotlib Figure object
    """
    if isinstance(real_signals, torch.Tensor):
        real_signals = real_signals.detach().cpu().numpy()
    if isinstance(fake_signals, torch.Tensor):
        fake_signals = fake_signals.detach().cpu().numpy()
    
    # Ensure signals are [channels, samples]
    if real_signals.shape[0] > real_signals.shape[1]:
        real_signals = real_signals.T
    if fake_signals.shape[0] > fake_signals.shape[1]:
        fake_signals = fake_signals.T
    
    # Extract specific channel
    real_signal = real_signals[channel_idx]
    fake_signal = fake_signals[channel_idx]
    
    n_samples = min(len(real_signal), len(fake_signal))
    time = np.arange(n_samples) / sampling_rate
    
    # Apply time range if specified
    if time_range is not None:
        start_idx = int(time_range[0] * sampling_rate)
        end_idx = int(time_range[1] * sampling_rate)
        time = time[start_idx:end_idx]
        real_signal = real_signal[start_idx:end_idx]
        fake_signal = fake_signal[start_idx:end_idx]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Plot real signal
    ax1.plot(time, real_signal, color='blue', alpha=0.8, linewidth=0.8)
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Real Signal')
    ax1.grid(True, alpha=0.3)
    
    # Plot generated signal
    ax2.plot(time, fake_signal, color='red', alpha=0.8, linewidth=0.8)
    ax2.set_ylabel('Amplitude')
    ax2.set_xlabel('Time (s)')
    ax2.set_title('Generated Signal')
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(f'{title} - Channel {channel_idx + 1}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Signal comparison plot saved to {save_path}")
    
    return fig


def plot_training_progress(losses_G: List[float],
                         losses_C: List[float],
                         title: str = "Training Progress",
                         figsize: Tuple[int, int] = (12, 5),
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot training losses over time.
    
    Args:
        losses_G: Generator losses
        losses_C: Critic losses
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save the plot
        
    Returns:
        matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot losses
    ax1.plot(losses_G, label='Generator', color='#FFD43B', alpha=0.8)
    ax1.plot(losses_C, label='Critic', color='#4B8BBE', alpha=0.8)
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Losses')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot smoothed losses
    if len(losses_G) > 10:
        window = min(50, len(losses_G) // 10)
        smooth_G = np.convolve(losses_G, np.ones(window)/window, mode='valid')
        smooth_C = np.convolve(losses_C, np.ones(window)/window, mode='valid')
        
        ax2.plot(smooth_G, label='Generator (smoothed)', color='#FFD43B')
        ax2.plot(smooth_C, label='Critic (smoothed)', color='#4B8BBE')
        ax2.set_xlabel('Iterations')
        ax2.set_ylabel('Loss')
        ax2.set_title('Smoothed Training Losses')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training progress plot saved to {save_path}")
    
    return fig


