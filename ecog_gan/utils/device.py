"""
Device management utilities for ECoG GAN.

This module provides functions for device detection, setup, and memory management
across different hardware platforms (CPU, CUDA, MPS).
"""

import torch
import gc
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def setup_device(gpu_id: Optional[int] = None) -> torch.device:
    """
    Setup and return the appropriate device for training.
    
    Args:
        gpu_id: GPU ID to use (None for auto-detection)
        
    Returns:
        torch.device: The device to use for training
    """
    # Check for MPS (Apple Silicon) availability
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("Using MPS (Apple Silicon) device")
        return device
    
    # Check for CUDA availability
    elif torch.cuda.is_available():
        if gpu_id is not None:
            if gpu_id >= torch.cuda.device_count():
                logger.warning(f"GPU {gpu_id} not available. Using GPU 0 instead.")
                gpu_id = 0
            torch.cuda.set_device(gpu_id)
        else:
            gpu_id = 0
        
        device = torch.device(f'cuda:{gpu_id}')
        
        # Configure CUDA settings for better performance
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        
        logger.info(f"Using CUDA device: {device}")
        logger.info(f"GPU Name: {torch.cuda.get_device_name(gpu_id)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(gpu_id).total_memory / 1e9:.1f} GB")
        
        return device
    
    # Fallback to CPU
    else:
        device = torch.device('cpu')
        logger.info("Using CPU device")
        return device


def cleanup_memory(device: torch.device):
    """
    Clean up memory by clearing caches and running garbage collection.
    
    Args:
        device: Device to clean up memory for
    """
    # Clear gradients
    torch.autograd.set_grad_enabled(False)
    torch.autograd.set_grad_enabled(True)
    torch.autograd.set_detect_anomaly(False)
    
    # Device-specific cleanup
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.empty_cache()
        torch.mps.synchronize()
    
    # General cleanup
    gc.collect()
    
    logger.debug("Memory cleanup completed")


def get_device_info(device: torch.device) -> dict:
    """
    Get detailed information about the device.
    
    Args:
        device: Device to get information for
        
    Returns:
        Dictionary containing device information
    """
    info = {
        'device_type': device.type,
        'device_index': device.index if device.index is not None else 0
    }
    
    if device.type == 'cuda':
        info.update({
            'cuda_available': True,
            'cuda_version': torch.version.cuda,
            'cudnn_version': torch.backends.cudnn.version(),
            'gpu_count': torch.cuda.device_count(),
            'gpu_name': torch.cuda.get_device_name(device.index or 0),
            'gpu_memory_total': torch.cuda.get_device_properties(device.index or 0).total_memory,
            'gpu_memory_allocated': torch.cuda.memory_allocated(device.index or 0),
            'gpu_memory_cached': torch.cuda.memory_reserved(device.index or 0)
        })
    elif device.type == 'mps':
        info.update({
            'mps_available': True,
            'mps_memory_allocated': torch.mps.current_allocated_memory(),
        })
    else:
        info.update({
            'cuda_available': torch.cuda.is_available(),
            'mps_available': torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
        })
    
    return info


def set_memory_fraction(fraction: float, device: torch.device):
    """
    Set memory fraction for GPU devices.
    
    Args:
        fraction: Fraction of memory to use (0.0 to 1.0)
        device: Device to set memory fraction for
    """
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(fraction, device.index or 0)
        logger.info(f"Set CUDA memory fraction to {fraction}")
    else:
        logger.warning(f"Memory fraction setting not supported for device type: {device.type}")


def monitor_memory_usage(device: torch.device) -> dict:
    """
    Monitor current memory usage.
    
    Args:
        device: Device to monitor
        
    Returns:
        Dictionary with memory usage statistics
    """
    usage = {}
    
    if device.type == 'cuda':
        usage = {
            'allocated': torch.cuda.memory_allocated(device.index or 0),
            'cached': torch.cuda.memory_reserved(device.index or 0),
            'max_allocated': torch.cuda.max_memory_allocated(device.index or 0),
            'max_cached': torch.cuda.max_memory_reserved(device.index or 0)
        }
        
        # Convert to MB for readability
        for key in usage:
            usage[key] = usage[key] / (1024 ** 2)
            
    elif device.type == 'mps':
        usage = {
            'allocated': torch.mps.current_allocated_memory() / (1024 ** 2),
            'driver_allocated': torch.mps.driver_allocated_memory() / (1024 ** 2)
        }
    
    return usage


def optimize_for_inference(model: torch.nn.Module, device: torch.device):
    """
    Optimize model for inference.
    
    Args:
        model: Model to optimize
        device: Device the model is on
    """
    model.eval()
    
    # Disable gradient computation
    for param in model.parameters():
        param.requires_grad = False
    
    # Enable inference optimizations
    if device.type == 'cuda':
        # Enable TensorRT optimizations if available
        try:
            model = torch.jit.optimize_for_inference(model)
            logger.info("Applied TensorRT optimizations")
        except:
            logger.debug("TensorRT optimizations not available")
    
    logger.info("Model optimized for inference")
    return model


class MemoryTracker:
    """Context manager for tracking memory usage."""
    
    def __init__(self, device: torch.device, name: str = "operation"):
        """
        Initialize memory tracker.
        
        Args:
            device: Device to track memory for
            name: Name of the operation being tracked
        """
        self.device = device
        self.name = name
        self.start_memory = None
        self.end_memory = None
    
    def __enter__(self):
        """Start memory tracking."""
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
            self.start_memory = torch.cuda.memory_allocated(self.device.index or 0)
        elif self.device.type == 'mps':
            self.start_memory = torch.mps.current_allocated_memory()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End memory tracking and report usage."""
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
            self.end_memory = torch.cuda.memory_allocated(self.device.index or 0)
        elif self.device.type == 'mps':
            self.end_memory = torch.mps.current_allocated_memory()
        
        if self.start_memory is not None and self.end_memory is not None:
            memory_diff = (self.end_memory - self.start_memory) / (1024 ** 2)  # Convert to MB
            logger.debug(f"Memory usage for {self.name}: {memory_diff:.2f} MB")
    
    def get_memory_diff(self) -> float:
        """Get memory difference in MB."""
        if self.start_memory is not None and self.end_memory is not None:
            return (self.end_memory - self.start_memory) / (1024 ** 2)
        return 0.0


def check_memory_requirements(model: torch.nn.Module, 
                            batch_size: int, 
                            input_shape: tuple,
                            device: torch.device) -> dict:
    """
    Estimate memory requirements for a model.
    
    Args:
        model: Model to check
        batch_size: Batch size to use
        input_shape: Shape of input tensor (without batch dimension)
        device: Device to check for
        
    Returns:
        Dictionary with memory estimates
    """
    # Create dummy input
    dummy_input = torch.randn(batch_size, *input_shape, device=device)
    
    with MemoryTracker(device, "model_forward") as tracker:
        with torch.no_grad():
            _ = model(dummy_input)
    
    forward_memory = tracker.get_memory_diff()
    
    # Estimate backward pass memory (roughly 2x forward pass)
    backward_memory = forward_memory * 2
    
    # Model parameters memory
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
    
    return {
        'forward_pass_mb': forward_memory,
        'backward_pass_mb': backward_memory,
        'parameters_mb': param_memory,
        'total_estimated_mb': forward_memory + backward_memory + param_memory
    }
