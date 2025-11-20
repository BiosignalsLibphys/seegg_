"""
Training monitoring and metrics tracking for ECoG GAN.

This module provides utilities for monitoring training progress, tracking metrics,
and logging training information.
"""

import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
import torch

logger = logging.getLogger(__name__)


class MetricsTracker:
    """Track and store training metrics."""
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.metrics = {
            'losses_generator': [],
            'losses_critic': [],
            'gradient_norms_generator': [],
            'gradient_norms_critic': [],
            'learning_rates_generator': [],
            'learning_rates_critic': [],
            'gradient_penalties': [],
            'epoch_times': [],
            'batch_times': []
        }
        self.epoch_stats = []
        self.batch_stats = []
    
    def update_batch_metrics(self, batch_metrics: Dict[str, float]):
        """Update metrics for a single batch."""
        # Map the actual metric names to the expected names
        key_mapping = {
            'critic_loss': 'losses_critic',
            'generator_loss': 'losses_generator',
            'grad_norm_critic': 'gradient_norms_critic',
            'grad_norm_generator': 'gradient_norms_generator',
            'gradient_penalty': 'gradient_penalties'
        }

        for key, value in batch_metrics.items():
            mapped_key = key_mapping.get(key, key)
            if mapped_key in self.metrics:
                self.metrics[mapped_key].append(value)

        self.batch_stats.append({
            'timestamp': datetime.now().isoformat(),
            **batch_metrics
        })
    
    def update_epoch_metrics(self, epoch_metrics: Dict[str, Any]):
        """Update metrics for a complete epoch."""
        # Extract learning rates from epoch metrics and add to tracking
        if 'lr_generator' in epoch_metrics:
            self.metrics['learning_rates_generator'].append(epoch_metrics['lr_generator'])
        if 'lr_critic' in epoch_metrics:
            self.metrics['learning_rates_critic'].append(epoch_metrics['lr_critic'])

        # Add epoch time if available
        if 'epoch_time' in epoch_metrics:
            self.metrics['epoch_times'].append(epoch_metrics['epoch_time'])

        self.epoch_stats.append({
            'timestamp': datetime.now().isoformat(),
            **epoch_metrics
        })
    
    def get_latest_metrics(self, n: int = 100) -> Dict[str, List]:
        """Get the latest n metrics."""
        latest = {}
        for key, values in self.metrics.items():
            latest[key] = values[-n:] if len(values) >= n else values
        return latest
    
    def get_epoch_summary(self, epoch: int) -> Dict[str, Any]:
        """Get summary statistics for a specific epoch."""
        if epoch < len(self.epoch_stats):
            return self.epoch_stats[epoch]
        return {}
    
    def save_metrics(self, filepath: str):
        """Save metrics to file."""
        data = {
            'metrics': self.metrics,
            'epoch_stats': self.epoch_stats,
            'batch_stats': self.batch_stats,
            'saved_at': datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Metrics saved to {filepath}")
    
    def load_metrics(self, filepath: str):
        """Load metrics from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.metrics = data.get('metrics', {})
        self.epoch_stats = data.get('epoch_stats', [])
        self.batch_stats = data.get('batch_stats', [])
        
        logger.info(f"Metrics loaded from {filepath}")


class TrainingMonitor:
    """Monitor training progress and generate visualizations."""
    
    def __init__(self, output_dir: str, save_frequency: int = 10):
        """
        Initialize training monitor.
        
        Args:
            output_dir: Directory to save monitoring outputs
            save_frequency: Frequency (in epochs) to save monitoring data
        """
        self.output_dir = output_dir
        self.save_frequency = save_frequency
        self.metrics_tracker = MetricsTracker()
        
        # Create output directories
        self.plots_dir = os.path.join(output_dir, 'plots')
        self.logs_dir = os.path.join(output_dir, 'logs')
        self.metrics_dir = os.path.join(output_dir, 'metrics')
        
        for dir_path in [self.plots_dir, self.logs_dir, self.metrics_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Training state
        self.start_time = None
        self.current_epoch = 0
        self.total_epochs = 0
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_file = os.path.join(self.logs_dir, 'training.log')
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    def start_training(self, total_epochs: int):
        """Mark the start of training."""
        self.start_time = datetime.now()
        self.total_epochs = total_epochs
        self.current_epoch = 0
        
        logger.info("="*80)
        logger.info("Training Started".center(80, "="))
        logger.info("="*80)
        logger.info(f"Total epochs: {total_epochs}")
        logger.info(f"Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    def update_batch(self, batch_metrics: Dict[str, float]):
        """Update metrics for a batch."""
        self.metrics_tracker.update_batch_metrics(batch_metrics)
    
    def update_epoch(self, epoch: int, epoch_metrics: Dict[str, Any]):
        """Update metrics for an epoch."""
        self.current_epoch = epoch
        self.metrics_tracker.update_epoch_metrics(epoch_metrics)
        
        # Log epoch summary
        self.log_epoch_summary(epoch, epoch_metrics)
        
        # Save metrics periodically
        if epoch % self.save_frequency == 0:
            self.save_training_state(epoch)
            self.generate_plots(epoch)
    
    def log_epoch_summary(self, epoch: int, metrics: Dict[str, Any]):
        """Log summary for an epoch."""
        logger.info("-" * 80)
        logger.info(f"Epoch [{epoch+1}/{self.total_epochs}]")
        
        # Log losses
        if 'generator_loss' in metrics:
            logger.info(f"Generator Loss: {metrics['generator_loss']:.6f}")
        if 'critic_loss' in metrics:
            logger.info(f"Critic Loss: {metrics['critic_loss']:.6f}")
        
        # Log gradient norms
        if 'grad_norm_generator' in metrics:
            logger.info(f"Generator Grad Norm: {metrics['grad_norm_generator']:.6f}")
        if 'grad_norm_critic' in metrics:
            logger.info(f"Critic Grad Norm: {metrics['grad_norm_critic']:.6f}")
        
        # Log learning rates
        if 'lr_generator' in metrics:
            logger.info(f"Generator LR: {metrics['lr_generator']:.2e}")
        if 'lr_critic' in metrics:
            logger.info(f"Critic LR: {metrics['lr_critic']:.2e}")
        
        # Time estimation
        if self.start_time and epoch > 0:
            elapsed = datetime.now() - self.start_time
            avg_epoch_time = elapsed.total_seconds() / (epoch + 1)
            remaining_epochs = self.total_epochs - epoch - 1
            estimated_remaining = timedelta(seconds=avg_epoch_time * remaining_epochs)
            
            logger.info(f"Elapsed: {self.format_time(elapsed.total_seconds())}")
            logger.info(f"Estimated remaining: {self.format_time(estimated_remaining.total_seconds())}")
    
    def save_training_state(self, epoch: int):
        """Save current training state."""
        # Save metrics
        metrics_file = os.path.join(self.metrics_dir, f'metrics_epoch_{epoch}.pkl')
        self.metrics_tracker.save_metrics(metrics_file)
        
        # Save training metadata
        metadata = {
            'current_epoch': epoch,
            'total_epochs': self.total_epochs,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'save_time': datetime.now().isoformat(),
            'output_dir': self.output_dir
        }
        
        metadata_file = os.path.join(self.metrics_dir, f'metadata_epoch_{epoch}.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def generate_plots(self, epoch: int):
        """Generate training plots."""
        metrics = self.metrics_tracker.metrics
        
        # Loss plot
        if metrics['losses_generator'] and metrics['losses_critic']:
            self.plot_losses(epoch)
        
        # Gradient norms plot
        if metrics['gradient_norms_generator'] and metrics['gradient_norms_critic']:
            self.plot_gradient_norms(epoch)
        
        # Learning rates plot
        if metrics['learning_rates_generator'] and metrics['learning_rates_critic']:
            self.plot_learning_rates(epoch)
        
        # Gradient penalty plot
        if metrics['gradient_penalties']:
            self.plot_gradient_penalties(epoch)
    
    def plot_losses(self, epoch: int):
        """Plot training losses."""
        plt.figure(figsize=(10, 6))

        gen_losses = self.metrics_tracker.metrics['losses_generator']
        critic_losses = self.metrics_tracker.metrics['losses_critic']

        # Plot raw losses with transparency
        plt.plot(gen_losses, label='Generator Loss', color='#FFD43B', alpha=0.6)
        plt.plot(critic_losses, label='Critic Loss', color='#4B8BBE', alpha=0.6)

        # Add smoothed lines if we have enough data
        if len(gen_losses) > 20:
            window = min(50, len(gen_losses) // 10)
            smooth_gen = np.convolve(gen_losses, np.ones(window)/window, mode='valid')
            smooth_critic = np.convolve(critic_losses, np.ones(window)/window, mode='valid')

            plt.plot(range(window-1, len(gen_losses)), smooth_gen,
                    label='Generator (smoothed)', color='#FFD43B', linewidth=2)
            plt.plot(range(window-1, len(critic_losses)), smooth_critic,
                    label='Critic (smoothed)', color='#4B8BBE', linewidth=2)

        plt.xlabel('Iterations (Synchronized Batches)')
        plt.ylabel('Loss')
        plt.title('Training Losses')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Remove top and right spines
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f'losses_epoch_{epoch}.png'), dpi=300)
        plt.close()
    
    def plot_gradient_norms(self, epoch: int):
        """Plot gradient norms."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics_tracker.metrics['gradient_norms_generator'], 
                label='Generator Grad Norm', color='#FFD43B')
        plt.plot(self.metrics_tracker.metrics['gradient_norms_critic'], 
                label='Critic Grad Norm', color='#4B8BBE')
        plt.xlabel('Iterations')
        plt.ylabel('Gradient Norm')
        plt.title('Gradient Norms')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f'gradient_norms_epoch_{epoch}.png'), dpi=300)
        plt.close()
    
    def plot_learning_rates(self, epoch: int):
        """Plot learning rates."""
        plt.figure(figsize=(10, 6))
        epochs = list(range(len(self.metrics_tracker.metrics['learning_rates_generator'])))
        plt.plot(epochs, self.metrics_tracker.metrics['learning_rates_generator'], 
                label='Generator LR', color='#FFD43B')
        plt.plot(epochs, self.metrics_tracker.metrics['learning_rates_critic'], 
                label='Critic LR', color='#4B8BBE')
        plt.xlabel('Epochs')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rates')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f'learning_rates_epoch_{epoch}.png'), dpi=300)
        plt.close()
    
    def plot_gradient_penalties(self, epoch: int):
        """Plot gradient penalties."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics_tracker.metrics['gradient_penalties'], 
                color='#4B8BBE', alpha=0.7)
        plt.xlabel('Iterations')
        plt.ylabel('Gradient Penalty')
        plt.title('Gradient Penalty Over Training')
        plt.grid(True, alpha=0.3)
        
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f'gradient_penalty_epoch_{epoch}.png'), dpi=300)
        plt.close()
    
    @staticmethod
    def format_time(seconds: float) -> str:
        """Format seconds into human readable time."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}h {minutes:02d}m {secs:02d}s"
    
    def finish_training(self):
        """Mark the end of training."""
        if self.start_time:
            total_time = datetime.now() - self.start_time
            logger.info("="*80)
            logger.info("Training Completed".center(80, "="))
            logger.info("="*80)
            logger.info(f"Total training time: {self.format_time(total_time.total_seconds())}")
            logger.info(f"Average time per epoch: {self.format_time(total_time.total_seconds() / max(1, self.current_epoch))}")
        
        # Final save
        self.save_training_state(self.current_epoch)
        self.generate_plots(self.current_epoch)
