"""
Flexible data loaders for various ECoG data formats.

This module provides data loaders that can automatically detect and handle
different data formats including dictionaries, 2D arrays, and 3D arrays
with variable numbers of channels.
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Union, Tuple, Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class DataFormatDetector:
    """Automatically detect the format of input data."""
    
    @staticmethod
    def detect_format(data: Any) -> str:
        """
        Detect the format of input data.
        
        Args:
            data: Input data of unknown format
            
        Returns:
            str: Format type ('dict', 'array_2d', 'array_3d', 'list', 'unknown')
        """
        if isinstance(data, dict):
            return 'dict'
        elif isinstance(data, (list, tuple)):
            if len(data) > 0:
                first_item = data[0]
                if isinstance(first_item, np.ndarray):
                    if first_item.ndim == 1:
                        return 'list_1d'
                    elif first_item.ndim == 2:
                        return 'list_2d'
                    else:
                        return 'list_nd'
            return 'list'
        elif isinstance(data, np.ndarray):
            if data.ndim == 2:
                return 'array_2d'
            elif data.ndim == 3:
                return 'array_3d'
            elif data.ndim == 1:
                return 'array_1d'
            else:
                return 'array_nd'
        else:
            return 'unknown'
    
    @staticmethod
    def get_data_info(data: Any) -> Dict[str, Any]:
        """
        Get detailed information about the data format and shape.
        
        Args:
            data: Input data
            
        Returns:
            Dict containing format info, shapes, and statistics
        """
        format_type = DataFormatDetector.detect_format(data)
        info = {
            'format': format_type,
            'type': type(data).__name__,
            'total_samples': 0,
            'shapes': [],
            'channels_info': {},
            'sample_lengths': []
        }
        
        if format_type == 'dict':
            info['num_subjects'] = len(data)
            info['subject_keys'] = list(data.keys())
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    info['shapes'].append(value.shape)
                    if value.ndim >= 2:
                        info['total_samples'] += value.shape[0]
                        if value.ndim == 3:  # [windows, channels, samples]
                            info['channels_info'][key] = value.shape[1]
                            info['sample_lengths'].extend([value.shape[2]] * value.shape[0])
                        elif value.ndim == 2:  # [channels, samples] or [windows, samples]
                            # Assume [channels, samples] if second dim is much larger
                            if value.shape[1] > value.shape[0] * 10:
                                info['channels_info'][key] = value.shape[0]
                                info['sample_lengths'].append(value.shape[1])
                                info['total_samples'] = 1
                            else:  # [windows, samples]
                                info['channels_info'][key] = 1
                                info['sample_lengths'].extend([value.shape[1]] * value.shape[0])
                                
        elif format_type in ['array_2d', 'array_3d']:
            info['shape'] = data.shape
            if format_type == 'array_2d':
                # Could be [signals, samples] or [channels, samples]
                info['total_samples'] = data.shape[0]
                info['sample_length'] = data.shape[1]
            elif format_type == 'array_3d':
                # [signals, channels, samples]
                info['total_samples'] = data.shape[0]
                info['num_channels'] = data.shape[1]
                info['sample_length'] = data.shape[2]
                
        elif format_type.startswith('list'):
            info['num_items'] = len(data)
            if len(data) > 0:
                sample_shapes = [item.shape if hasattr(item, 'shape') else len(item) for item in data[:5]]
                info['sample_shapes'] = sample_shapes
                info['total_samples'] = len(data)
        
        return info


class ECoGDataset(Dataset):
    """PyTorch Dataset for ECoG data with flexible format support."""
    
    def __init__(self, data: Any, seq_len: int, transform: Optional[callable] = None):
        """
        Initialize ECoG dataset.
        
        Args:
            data: Input data in various formats
            seq_len: Expected sequence length
            transform: Optional transform to apply to samples
        """
        self.seq_len = seq_len
        self.transform = transform
        self.samples = []
        
        # Process data based on format
        self._process_data(data)
        
        logger.info(f"Dataset initialized with {len(self.samples)} samples")
        if len(self.samples) > 0:
            logger.info(f"Sample shape: {self.samples[0].shape}")
    
    def _process_data(self, data: Any):
        """Process input data into standardized format."""
        format_type = DataFormatDetector.detect_format(data)
        
        if format_type == 'dict':
            self._process_dict_data(data)
        elif format_type == 'array_2d':
            self._process_2d_array(data)
        elif format_type == 'array_3d':
            self._process_3d_array(data)
        elif format_type.startswith('list'):
            self._process_list_data(data)
        else:
            raise ValueError(f"Unsupported data format: {format_type}")
    
    def _process_dict_data(self, data: Dict):
        """Process dictionary format data."""
        for subject_id, subject_data in data.items():
            if not isinstance(subject_data, np.ndarray):
                logger.warning(f"Skipping subject {subject_id}: not a numpy array")
                continue
                
            if subject_data.ndim == 2:
                # Could be [channels, samples] or [windows, samples]
                if subject_data.shape[1] == self.seq_len:
                    # Likely [channels, samples]
                    self.samples.append(torch.tensor(subject_data, dtype=torch.float32))
                elif subject_data.shape[1] > self.seq_len:
                    # Likely [channels, samples] - need to segment
                    self._segment_long_signal(subject_data)
                else:
                    # Likely [windows, samples] - check each window
                    for window_idx in range(subject_data.shape[0]):
                        window_data = subject_data[window_idx]
                        if len(window_data) == self.seq_len:
                            # Add channel dimension for single channel
                            window_tensor = torch.tensor(window_data.reshape(1, -1), dtype=torch.float32)
                            self.samples.append(window_tensor)
                        else:
                            logger.warning(f"Window {window_idx} of subject {subject_id} has length {len(window_data)}, expected {self.seq_len}")
                            
            elif subject_data.ndim == 3:
                # [windows, channels, samples]
                for window_idx in range(subject_data.shape[0]):
                    window_data = subject_data[window_idx]
                    if window_data.shape[1] == self.seq_len:
                        self.samples.append(torch.tensor(window_data, dtype=torch.float32))
                    else:
                        logger.warning(f"Window {window_idx} of subject {subject_id} has length {window_data.shape[1]}, expected {self.seq_len}")
    
    def _process_2d_array(self, data: np.ndarray):
        """Process 2D array data."""
        if data.shape[1] == self.seq_len:
            # [signals, samples] - each row is a signal
            for i in range(data.shape[0]):
                signal = data[i].reshape(1, -1)  # Add channel dimension
                self.samples.append(torch.tensor(signal, dtype=torch.float32))
        elif data.shape[0] == self.seq_len:
            # [samples, signals] - transpose needed
            data = data.T
            for i in range(data.shape[0]):
                signal = data[i].reshape(1, -1)
                self.samples.append(torch.tensor(signal, dtype=torch.float32))
        else:
            # Might be [channels, samples] for a single multichannel signal
            if data.shape[1] > data.shape[0] * 10:  # Heuristic: samples >> channels
                self.samples.append(torch.tensor(data, dtype=torch.float32))
            else:
                logger.warning(f"Unclear 2D array format with shape {data.shape}")
    
    def _process_3d_array(self, data: np.ndarray):
        """Process 3D array data [signals, channels, samples]."""
        for i in range(data.shape[0]):
            signal = data[i]  # [channels, samples]
            if signal.shape[1] == self.seq_len:
                self.samples.append(torch.tensor(signal, dtype=torch.float32))
            else:
                logger.warning(f"Signal {i} has length {signal.shape[1]}, expected {self.seq_len}")
    
    def _process_list_data(self, data: List):
        """Process list format data."""
        for i, item in enumerate(data):
            if isinstance(item, np.ndarray):
                if item.ndim == 1:
                    if len(item) == self.seq_len:
                        signal = item.reshape(1, -1)  # Add channel dimension
                        self.samples.append(torch.tensor(signal, dtype=torch.float32))
                elif item.ndim == 2:
                    if item.shape[1] == self.seq_len:
                        self.samples.append(torch.tensor(item, dtype=torch.float32))
                    elif item.shape[0] == self.seq_len:
                        # Transpose if needed
                        self.samples.append(torch.tensor(item.T, dtype=torch.float32))
    
    def _segment_long_signal(self, signal: np.ndarray):
        """Segment long signals into seq_len chunks."""
        if signal.ndim == 2:
            channels, samples = signal.shape
            num_segments = samples // self.seq_len
            for i in range(num_segments):
                start_idx = i * self.seq_len
                end_idx = start_idx + self.seq_len
                segment = signal[:, start_idx:end_idx]
                self.samples.append(torch.tensor(segment, dtype=torch.float32))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        sample = self.samples[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample


class ECoGDataLoader:
    """High-level data loader for ECoG data with automatic format detection."""
    
    def __init__(self, data_path: str, seq_len: int, batch_size: int = 32, 
                 shuffle: bool = True, num_workers: int = 0, transform: Optional[callable] = None):
        """
        Initialize ECoG data loader.
        
        Args:
            data_path: Path to data file
            seq_len: Expected sequence length
            batch_size: Batch size for training
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            transform: Optional transform to apply
        """
        self.data_path = data_path
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.transform = transform
        
        # Load and analyze data
        self.data = self._load_data()
        self.data_info = DataFormatDetector.get_data_info(self.data)
        
        logger.info(f"Data format detected: {self.data_info['format']}")
        logger.info(f"Data info: {self.data_info}")
        
        # Create dataset
        self.dataset = ECoGDataset(self.data, seq_len, transform)
        
    def _load_data(self) -> Any:
        """Load data from file."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        file_ext = os.path.splitext(self.data_path)[1].lower()
        
        if file_ext == '.pkl':
            with open(self.data_path, 'rb') as f:
                return pickle.load(f)
        elif file_ext == '.npy':
            return np.load(self.data_path, allow_pickle=True)
        elif file_ext in ['.csv']:
            return pd.read_csv(self.data_path).values
        else:
            # Try pickle first, then numpy
            try:
                return pd.read_pickle(self.data_path)
            except:
                try:
                    return np.load(self.data_path, allow_pickle=True)
                except:
                    raise ValueError(f"Unsupported file format: {file_ext}")
    
    def get_dataloader(self) -> DataLoader:
        """Get PyTorch DataLoader."""
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers
        )
    
    def get_data_info(self) -> Dict[str, Any]:
        """Get information about the loaded data."""
        return self.data_info
    
    def get_sample_shape(self) -> Tuple[int, ...]:
        """Get the shape of a single sample."""
        if len(self.dataset) > 0:
            return self.dataset[0].shape
        return (0,)

    def __len__(self) -> int:
        """Get number of samples in dataset."""
        return len(self.dataset)

    def __iter__(self):
        """Make the data loader iterable."""
        return iter(self.get_dataloader())
