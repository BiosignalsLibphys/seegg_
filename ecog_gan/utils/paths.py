"""
Path utilities for ECoG GAN.

This module provides utilities for creating output directories,
managing file paths, and organizing training outputs.
"""

import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


def get_timestamp_folder() -> Tuple[str, str]:
    """
    Create a timestamp-based folder name.
    
    Returns:
        Tuple of (folder_name, timestamp)
    """
    now = datetime.now()
    timestamp = now.strftime("%d%m_%H%M")  # Format: DDMM_HHMM
    folder_name = f"ecog_gan_{timestamp}"
    return folder_name, timestamp


def create_output_directories(base_dir: Union[str, Path], 
                            timestamp: Optional[str] = None,
                            subdirs: Optional[list] = None) -> Dict[str, str]:
    """
    Create organized output directory structure.
    
    Args:
        base_dir: Base output directory
        timestamp: Optional timestamp string
        subdirs: Optional list of subdirectories to create
        
    Returns:
        Dictionary mapping directory names to paths
    """
    if timestamp is None:
        _, timestamp = get_timestamp_folder()
    
    base_dir = Path(base_dir)
    output_dir = base_dir / f"ecog_gan_{timestamp}"
    
    # Default subdirectories
    if subdirs is None:
        subdirs = [
            "checkpoints",
            "generated_data", 
            "losses",
            "plots",
            "logs",
            "metrics",
            "configs",
            "samples"
        ]
    
    # Create directories
    paths = {"base": str(output_dir)}
    
    for subdir in subdirs:
        subdir_path = output_dir / subdir
        subdir_path.mkdir(parents=True, exist_ok=True)
        paths[subdir] = str(subdir_path)
    
    logger.info(f"Created output directories in {output_dir}")
    return paths


def get_model_paths(output_dir: Union[str, Path], 
                   timestamp: str,
                   epoch: Optional[int] = None) -> Dict[str, str]:
    """
    Get standardized paths for model files.
    
    Args:
        output_dir: Base output directory
        timestamp: Timestamp string
        epoch: Optional epoch number for versioning
        
    Returns:
        Dictionary of model file paths
    """
    output_dir = Path(output_dir)
    
    if epoch is not None:
        suffix = f"_epoch_{epoch}_{timestamp}"
    else:
        suffix = f"_{timestamp}"
    
    paths = {
        "generator": str(output_dir / "checkpoints" / f"generator{suffix}.pth"),
        "critic": str(output_dir / "checkpoints" / f"critic{suffix}.pth"),
        "training_state": str(output_dir / "checkpoints" / f"training_state{suffix}.pth"),
        "config": str(output_dir / "configs" / f"config{suffix}.yaml"),
        "metadata": str(output_dir / f"metadata{suffix}.json")
    }
    
    return paths


def get_data_paths(output_dir: Union[str, Path], 
                  timestamp: str) -> Dict[str, str]:
    """
    Get standardized paths for data files.
    
    Args:
        output_dir: Base output directory
        timestamp: Timestamp string
        
    Returns:
        Dictionary of data file paths
    """
    output_dir = Path(output_dir)
    
    paths = {
        "generated_data": str(output_dir / "generated_data" / f"synthetic_signals_{timestamp}.pkl"),
        "losses": str(output_dir / "losses" / f"losses_{timestamp}.pkl"),
        "metrics": str(output_dir / "metrics" / f"metrics_{timestamp}.pkl"),
        "gradients": str(output_dir / "metrics" / f"gradients_{timestamp}.pkl")
    }
    
    return paths


def get_plot_paths(output_dir: Union[str, Path], 
                  timestamp: str,
                  epoch: Optional[int] = None) -> Dict[str, str]:
    """
    Get standardized paths for plot files.
    
    Args:
        output_dir: Base output directory
        timestamp: Timestamp string
        epoch: Optional epoch number
        
    Returns:
        Dictionary of plot file paths
    """
    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots"
    
    if epoch is not None:
        suffix = f"_epoch_{epoch}_{timestamp}"
    else:
        suffix = f"_{timestamp}"
    
    paths = {
        "losses": str(plots_dir / f"losses{suffix}.png"),
        "gradient_norms": str(plots_dir / f"gradient_norms{suffix}.png"),
        "learning_rates": str(plots_dir / f"learning_rates{suffix}.png"),
        "gradient_penalty": str(plots_dir / f"gradient_penalty{suffix}.png"),
        "sample_signals": str(plots_dir / f"sample_signals{suffix}.png"),
        "spectrograms": str(plots_dir / f"spectrograms{suffix}.png")
    }
    
    return paths


def ensure_directory_exists(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_relative_path(path: Union[str, Path], 
                     base: Union[str, Path]) -> str:
    """
    Get relative path from base directory.
    
    Args:
        path: Target path
        base: Base directory
        
    Returns:
        Relative path string
    """
    try:
        return str(Path(path).relative_to(Path(base)))
    except ValueError:
        # If paths are not relative, return absolute path
        return str(Path(path).absolute())


def find_latest_checkpoint(checkpoint_dir: Union[str, Path],
                          model_type: str = "generator") -> Optional[str]:
    """
    Find the latest checkpoint file.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        model_type: Type of model ("generator" or "critic")
        
    Returns:
        Path to latest checkpoint or None if not found
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        return None
    
    # Find all checkpoint files for the model type
    pattern = f"{model_type}_epoch_*.pth"
    checkpoints = list(checkpoint_dir.glob(pattern))
    
    if not checkpoints:
        return None
    
    # Sort by modification time and return the latest
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    return str(latest)


def clean_old_files(directory: Union[str, Path], 
                   pattern: str,
                   keep_latest: int = 5) -> int:
    """
    Clean old files matching a pattern, keeping only the latest N files.
    
    Args:
        directory: Directory to clean
        pattern: File pattern to match
        keep_latest: Number of latest files to keep
        
    Returns:
        Number of files deleted
    """
    directory = Path(directory)
    
    if not directory.exists():
        return 0
    
    # Find all matching files
    files = list(directory.glob(pattern))
    
    if len(files) <= keep_latest:
        return 0
    
    # Sort by modification time (newest first)
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    
    # Delete old files
    files_to_delete = files[keep_latest:]
    deleted_count = 0
    
    for file_path in files_to_delete:
        try:
            file_path.unlink()
            deleted_count += 1
            logger.debug(f"Deleted old file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to delete {file_path}: {e}")
    
    if deleted_count > 0:
        logger.info(f"Cleaned {deleted_count} old files from {directory}")
    
    return deleted_count


def get_file_size(path: Union[str, Path]) -> int:
    """
    Get file size in bytes.
    
    Args:
        path: File path
        
    Returns:
        File size in bytes
    """
    return Path(path).stat().st_size


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"


def get_directory_size(directory: Union[str, Path]) -> int:
    """
    Get total size of all files in a directory.
    
    Args:
        directory: Directory path
        
    Returns:
        Total size in bytes
    """
    directory = Path(directory)
    total_size = 0
    
    for file_path in directory.rglob('*'):
        if file_path.is_file():
            total_size += file_path.stat().st_size
    
    return total_size


class PathManager:
    """Manage paths for a training session."""
    
    def __init__(self, base_dir: Union[str, Path], timestamp: Optional[str] = None):
        """
        Initialize path manager.
        
        Args:
            base_dir: Base output directory
            timestamp: Optional timestamp string
        """
        if timestamp is None:
            _, timestamp = get_timestamp_folder()
        
        self.timestamp = timestamp
        self.base_dir = Path(base_dir)
        self.output_dir = self.base_dir / f"ecog_gan_{timestamp}"
        
        # Create directory structure
        self.paths = create_output_directories(base_dir, timestamp)
    
    def get_model_path(self, model_type: str, epoch: Optional[int] = None) -> str:
        """Get path for model file."""
        model_paths = get_model_paths(self.output_dir, self.timestamp, epoch)
        return model_paths[model_type]
    
    def get_data_path(self, data_type: str) -> str:
        """Get path for data file."""
        data_paths = get_data_paths(self.output_dir, self.timestamp)
        return data_paths[data_type]
    
    def get_plot_path(self, plot_type: str, epoch: Optional[int] = None) -> str:
        """Get path for plot file."""
        plot_paths = get_plot_paths(self.output_dir, self.timestamp, epoch)
        return plot_paths[plot_type]
    
    def cleanup_old_checkpoints(self, keep_latest: int = 5) -> int:
        """Clean up old checkpoint files."""
        checkpoint_dir = Path(self.paths["checkpoints"])
        
        deleted = 0
        for model_type in ["generator", "critic", "training_state"]:
            pattern = f"{model_type}_epoch_*.pth"
            deleted += clean_old_files(checkpoint_dir, pattern, keep_latest)
        
        return deleted
    
    def get_summary(self) -> Dict[str, str]:
        """Get summary of paths and sizes."""
        summary = {
            "output_directory": str(self.output_dir),
            "timestamp": self.timestamp,
            "total_size": format_file_size(get_directory_size(self.output_dir))
        }
        
        for name, path in self.paths.items():
            if Path(path).exists():
                size = get_directory_size(path)
                summary[f"{name}_size"] = format_file_size(size)
        
        return summary


def safe_filename(filename: str) -> str:
    """
    Create a safe filename by removing/replacing invalid characters.

    Args:
        filename: Original filename

    Returns:
        Safe filename string
    """
    import re
    # Remove or replace invalid characters
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove multiple consecutive underscores
    safe_name = re.sub(r'_+', '_', safe_name)
    # Remove leading/trailing underscores
    safe_name = safe_name.strip('_')
    return safe_name
