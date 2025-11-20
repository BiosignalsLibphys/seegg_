"""
Data preprocessing utilities for ECoG signals.

This module provides various preprocessing functions including normalization,
filtering, and data augmentation techniques.
"""

import numpy as np
import torch
from typing import Union, Tuple, Optional, Dict, Any
from scipy import signal
import logging

logger = logging.getLogger(__name__)


class DataNormalizer:
    """Normalize ECoG data using various strategies."""
    
    def __init__(self, method: str = 'zscore', axis: Optional[int] = None):
        """
        Initialize normalizer.
        
        Args:
            method: Normalization method ('zscore', 'minmax', 'robust', 'global')
            axis: Axis along which to normalize (None for global normalization)
        """
        self.method = method
        self.axis = axis
        self.stats = {}
        
    def fit(self, data: Union[np.ndarray, torch.Tensor]) -> 'DataNormalizer':
        """
        Fit normalizer to data.
        
        Args:
            data: Input data to fit normalizer
            
        Returns:
            Self for method chaining
        """
        if isinstance(data, torch.Tensor):
            data = data.numpy()
            
        if self.method == 'zscore':
            self.stats['mean'] = np.mean(data, axis=self.axis, keepdims=True)
            self.stats['std'] = np.std(data, axis=self.axis, keepdims=True)
            # Avoid division by zero
            self.stats['std'] = np.where(self.stats['std'] == 0, 1, self.stats['std'])
            
        elif self.method == 'minmax':
            self.stats['min'] = np.min(data, axis=self.axis, keepdims=True)
            self.stats['max'] = np.max(data, axis=self.axis, keepdims=True)
            # Avoid division by zero
            range_val = self.stats['max'] - self.stats['min']
            self.stats['range'] = np.where(range_val == 0, 1, range_val)
            
        elif self.method == 'robust':
            self.stats['median'] = np.median(data, axis=self.axis, keepdims=True)
            self.stats['mad'] = np.median(np.abs(data - self.stats['median']), 
                                        axis=self.axis, keepdims=True)
            # Avoid division by zero
            self.stats['mad'] = np.where(self.stats['mad'] == 0, 1, self.stats['mad'])
            
        elif self.method == 'global':
            # Global statistics across all dimensions
            self.stats['mean'] = np.mean(data)
            self.stats['std'] = np.std(data)
            if self.stats['std'] == 0:
                self.stats['std'] = 1
                
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
            
        logger.info(f"Normalizer fitted with method '{self.method}', stats: {self.stats}")
        return self
    
    def transform(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Transform data using fitted normalizer.
        
        Args:
            data: Data to transform
            
        Returns:
            Normalized data
        """
        if not self.stats:
            raise ValueError("Normalizer not fitted. Call fit() first.")
            
        is_tensor = isinstance(data, torch.Tensor)
        if is_tensor:
            device = data.device
            data = data.cpu().numpy()
            
        if self.method == 'zscore':
            normalized = (data - self.stats['mean']) / self.stats['std']
        elif self.method == 'minmax':
            normalized = 2 * (data - self.stats['min']) / self.stats['range'] - 1
        elif self.method == 'robust':
            normalized = (data - self.stats['median']) / (1.4826 * self.stats['mad'])
        elif self.method == 'global':
            normalized = (data - self.stats['mean']) / self.stats['std']
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
            
        if is_tensor:
            normalized = torch.tensor(normalized, dtype=torch.float32).to(device)
            
        return normalized
    
    def fit_transform(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Fit normalizer and transform data in one step."""
        return self.fit(data).transform(data)
    
    def inverse_transform(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Inverse transform normalized data back to original scale.
        
        Args:
            data: Normalized data to inverse transform
            
        Returns:
            Data in original scale
        """
        if not self.stats:
            raise ValueError("Normalizer not fitted. Call fit() first.")
            
        is_tensor = isinstance(data, torch.Tensor)
        if is_tensor:
            device = data.device
            data = data.cpu().numpy()
            
        if self.method == 'zscore':
            original = data * self.stats['std'] + self.stats['mean']
        elif self.method == 'minmax':
            original = data * self.stats['range'] + self.stats['min']
        elif self.method == 'robust':
            original = data * (1.4826 * self.stats['mad']) + self.stats['median']
        elif self.method == 'global':
            original = data * self.stats['std'] + self.stats['mean']
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
            
        if is_tensor:
            original = torch.tensor(original, dtype=torch.float32).to(device)
            
        return original


class SignalResampler:
    """Resample ECoG signals to different sampling rates."""

    def __init__(self, original_fs: float, target_fs: float, method: str = 'scipy'):
        """
        Initialize signal resampler.

        Args:
            original_fs: Original sampling frequency in Hz
            target_fs: Target sampling frequency in Hz
            method: Resampling method ('scipy', 'decimate')
        """
        self.original_fs = original_fs
        self.target_fs = target_fs
        self.method = method
        self.downsample_factor = original_fs / target_fs

        if self.downsample_factor < 1:
            raise ValueError(f"Cannot upsample: original_fs ({original_fs}) < target_fs ({target_fs})")

        logger.info(f"SignalResampler initialized: {original_fs}Hz → {target_fs}Hz (factor: {self.downsample_factor:.2f})")

    def resample(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Resample signal data.

        Args:
            data: Input signal data. Can be:
                - 1D: [samples]
                - 2D: [signals, samples] or [channels, samples]
                - 3D: [batch, channels, samples]

        Returns:
            Resampled signal data with same number of dimensions
        """
        is_tensor = isinstance(data, torch.Tensor)
        if is_tensor:
            device = data.device
            data = data.cpu().numpy()

        original_shape = data.shape

        # Calculate target number of samples
        original_samples = original_shape[-1]
        target_samples = int(original_samples * self.target_fs / self.original_fs)

        logger.info(f"Resampling from {original_samples} to {target_samples} samples")

        if data.ndim == 1:
            # 1D signal
            resampled_data = self._resample_1d(data, target_samples)

        elif data.ndim == 2:
            # 2D: [signals/channels, samples]
            resampled_data = np.zeros((original_shape[0], target_samples), dtype=data.dtype)
            for i in range(original_shape[0]):
                resampled_data[i] = self._resample_1d(data[i], target_samples)

        elif data.ndim == 3:
            # 3D: [batch, channels, samples]
            resampled_data = np.zeros((original_shape[0], original_shape[1], target_samples), dtype=data.dtype)
            for i in range(original_shape[0]):
                for j in range(original_shape[1]):
                    resampled_data[i, j] = self._resample_1d(data[i, j], target_samples)
        else:
            raise ValueError(f"Unsupported data dimensionality: {data.ndim}D")

        if is_tensor:
            resampled_data = torch.tensor(resampled_data, dtype=torch.float32).to(device)

        return resampled_data

    def _resample_1d(self, signal_1d: np.ndarray, target_samples: int) -> np.ndarray:
        """Resample a 1D signal."""
        if self.method == 'scipy':
            # Use scipy.signal.resample for high-quality resampling
            return signal.resample(signal_1d, target_samples)

        elif self.method == 'decimate':
            # Use scipy.signal.decimate for integer downsampling factors
            if not self.downsample_factor.is_integer():
                logger.warning(f"Decimate method requires integer factor, got {self.downsample_factor}. Using scipy method.")
                return signal.resample(signal_1d, target_samples)

            factor = int(self.downsample_factor)
            return signal.decimate(signal_1d, factor, ftype='iir')

        else:
            raise ValueError(f"Unknown resampling method: {self.method}")


class DataPreprocessor:
    """Comprehensive data preprocessing pipeline."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize preprocessor with configuration.
        
        Args:
            config: Preprocessing configuration dictionary
        """
        self.config = config
        self.normalizer = None
        self.resampler = None
        self.fitted = False

        # Option B: Lazy initialization. Do not construct normalizer/resampler here.
        # They will be created during fit/transform if their apply flags are enabled.
    
    def fit(self, data: Union[np.ndarray, torch.Tensor]) -> 'DataPreprocessor':
        """
        Fit preprocessor to data.
        
        Args:
            data: Training data to fit preprocessor
            
        Returns:
            Self for method chaining
        """
        # Lazy-create and fit normalizer only if enabled
        if ('normalization' in self.config and self.config['normalization'] is not None and
            self.config['normalization'].get('apply_normalization', True)):
            if self.normalizer is None:
                norm_config = self.config['normalization']
                self.normalizer = DataNormalizer(
                    method=norm_config.get('method', 'zscore'),
                    axis=norm_config.get('axis', None)
                )
            self.normalizer.fit(data)
            
        self.fitted = True
        logger.info("Preprocessor fitted to data")
        return self
    
    def transform(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Transform data using fitted preprocessor.
        
        Args:
            data: Data to transform
            
        Returns:
            Preprocessed data
        """
        if not self.fitted:
            logger.warning("Preprocessor not fitted. Fitting to current data.")
            self.fit(data)
            
        processed_data = data

        # Apply resampling if enabled (lazy-create resampler). Do first.
        if ('resampling' in self.config and self.config['resampling'] is not None and
            self.config['resampling'].get('apply_resampling', True)):
            if self.resampler is None:
                resample_config = self.config['resampling']
                self.resampler = SignalResampler(
                    original_fs=resample_config['original_fs'],
                    target_fs=resample_config['target_fs'],
                    method=resample_config.get('method', 'scipy')
                )
            processed_data = self.resampler.resample(processed_data)

        # Apply filtering if specified
        if ('filtering' in self.config and self.config['filtering'] is not None and
            self.config['filtering'].get('apply_filtering', True)):
            processed_data = self._apply_filtering(processed_data)

        # Apply normalization if enabled and normalizer exists
        if ('normalization' in self.config and self.config['normalization'] is not None and
            self.config['normalization'].get('apply_normalization', True) and
            self.normalizer is not None):
            processed_data = self.normalizer.transform(processed_data)

        # Apply data augmentation if specified
        if ('augmentation' in self.config and self.config['augmentation'] is not None and
            self.config['augmentation'].get('apply_augmentation', True)):
            processed_data = self._apply_augmentation(processed_data)

        return processed_data
    
    def fit_transform(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Fit preprocessor and transform data in one step."""
        return self.fit(data).transform(data)
    
    def _apply_filtering(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Apply filtering to data."""
        filter_config = self.config['filtering']
        
        is_tensor = isinstance(data, torch.Tensor)
        if is_tensor:
            device = data.device
            data = data.cpu().numpy()
            
        # Apply notch filter if specified
        if 'notch_freq' in filter_config:
            fs = filter_config.get('sampling_rate', 512)
            notch_freq = filter_config['notch_freq']
            quality_factor = filter_config.get('quality_factor', 30)
            
            # Apply notch filter to each channel
            for i in range(data.shape[-2]):  # Assuming [..., channels, samples]
                b, a = signal.iirnotch(notch_freq, quality_factor, fs)
                data[..., i, :] = signal.filtfilt(b, a, data[..., i, :])
                
        # Apply bandpass filter if specified
        if 'bandpass' in filter_config:
            fs = filter_config.get('sampling_rate', 512)
            low_freq, high_freq = filter_config['bandpass']
            
            # Apply bandpass filter to each channel
            sos = signal.butter(4, [low_freq, high_freq], btype='band', fs=fs, output='sos')
            for i in range(data.shape[-2]):
                data[..., i, :] = signal.sosfiltfilt(sos, data[..., i, :])
                
        if is_tensor:
            data = torch.tensor(data, dtype=torch.float32).to(device)
            
        return data
    
    def _apply_augmentation(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Apply data augmentation techniques."""
        aug_config = self.config['augmentation']
        
        # Add noise if specified
        if 'noise_level' in aug_config:
            noise_level = aug_config['noise_level']
            if isinstance(data, torch.Tensor):
                noise = torch.randn_like(data) * noise_level
                data = data + noise
            else:
                noise = np.random.randn(*data.shape) * noise_level
                data = data + noise
                
        # Apply time shifting if specified
        if 'time_shift' in aug_config:
            max_shift = aug_config['time_shift']
            if isinstance(data, torch.Tensor):
                shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
                if shift != 0:
                    data = torch.roll(data, shift, dims=-1)
            else:
                shift = np.random.randint(-max_shift, max_shift + 1)
                if shift != 0:
                    data = np.roll(data, shift, axis=-1)
                    
        return data
