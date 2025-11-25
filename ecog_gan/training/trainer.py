"""
Main trainer class for ECoG GAN.

This module provides the main training loop and utilities for training
the ECoG GAN model with various configurations and monitoring.
"""

import os
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple
import logging
from collections import defaultdict

from .losses import WGANGPLoss, get_loss_function
from .monitoring import TrainingMonitor
from ..utils.device import setup_device, cleanup_memory
from ..models import Generator, WindowCritic

logger = logging.getLogger(__name__)


class Trainer:
    """Base trainer class for ECoG GAN."""
    
    def __init__(self, 
                 generator: nn.Module,
                 critic: nn.Module,
                 config: Dict[str, Any],
                 device: Optional[torch.device] = None):
        """
        Initialize trainer.
        
        Args:
            generator: Generator model
            critic: Critic model
            config: Training configuration
            device: Device to train on (auto-detected if None)
        """
        self.generator = generator
        self.critic = critic
        self.config = config
        self.device = device or setup_device(config.get('gpu_id', 0))
        
        # Move models to device
        self.generator.to(self.device)
        self.critic.to(self.device)
        
        # Initialize training components
        self.setup_optimizers()
        self.setup_schedulers()
        self.setup_loss_function()
        self.setup_monitoring()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0

        # Explainable AI components
        self._xai = {
            'attn_buffers': [],      # all attention matrices (Q,K) for backward-compat
            'attn_temporal': [],     # explicitly-labeled temporal attention (Q,K)
            'attn_channel': [],      # explicitly-labeled channel attention (Q,K)
            'attn_handles': [],      # hook handles
        }
        self._xai_defaults()

        self._register_attention_hooks()  # collect TAE/attention weights during forward
        
        logger.info(f"Trainer initialized on device: {self.device}")
    
    def setup_optimizers(self):
        """Setup optimizers for generator and critic."""
        opt_config = self.config.get('optimizer', {})
        
        # Generator optimizer
        gen_config = opt_config.get('generator', {})
        self.optimizer_G = optim.Adam(
            self.generator.parameters(),
            lr=gen_config.get('lr', 0.0002),
            betas=gen_config.get('betas', (0.5, 0.999))
        )
        
        # Critic optimizer
        critic_config = opt_config.get('critic', {})
        self.optimizer_C = optim.Adam(
            self.critic.parameters(),
            lr=critic_config.get('lr', 0.0002),
            betas=critic_config.get('betas', (0.5, 0.999))
        )
    
    def setup_schedulers(self):
        """Setup learning rate schedulers."""
        sched_config = self.config.get('scheduler', {})
        
        if sched_config.get('use_scheduler', False):
            self.scheduler_G = optim.lr_scheduler.StepLR(
                self.optimizer_G,
                step_size=sched_config.get('step_size', 100),
                gamma=sched_config.get('gamma', 0.9)
            )
            self.scheduler_C = optim.lr_scheduler.StepLR(
                self.optimizer_C,
                step_size=sched_config.get('step_size', 100),
                gamma=sched_config.get('gamma', 0.9)
            )
        else:
            self.scheduler_G = None
            self.scheduler_C = None
    
    def setup_loss_function(self):
        """Setup loss function."""
        loss_config = self.config.get('loss', {'type': 'wgan_gp'})
        self.loss_fn = get_loss_function(loss_config)
    
    def setup_monitoring(self):
        """Setup training monitoring."""
        monitor_config = self.config.get('monitoring', {})
        output_dir = monitor_config.get('output_dir', './outputs')
        
        self.monitor = TrainingMonitor(
            output_dir=output_dir,
            save_frequency=monitor_config.get('save_frequency', 10)
        )
    
    def train_step_critic(self, real_batch: torch.Tensor) -> Dict[str, float]:
        """
        Perform one training step for the critic.
        
        Args:
            real_batch: Batch of real data
            
        Returns:
            Dictionary of loss components
        """
        self.optimizer_C.zero_grad()
        
        batch_size = real_batch.size(0)
        latent_dim = self.config.get('model', {}).get('generator', {}).get('latent_dim', 100)
        
        # Generate fake samples
        z = torch.randn(batch_size, latent_dim, device=self.device)
        fake_batch = self.generator(z).detach()  # Detach to avoid generator gradients
        
        # Compute critic loss
        critic_loss, loss_components = self.loss_fn.critic_loss(
            self.critic, real_batch, fake_batch, self.device
        )
        
        # Backward pass
        critic_loss.backward()
        
        # Compute gradient norm
        grad_norm_C = self.compute_gradient_norm(self.critic)
        loss_components['grad_norm_critic'] = grad_norm_C
        
        # Update critic
        self.optimizer_C.step()
        
        return loss_components
    
    def train_step_generator(self, batch_size: int) -> Dict[str, float]:
        """
        Perform one training step for the generator.
        
        Args:
            batch_size: Size of the batch
            
        Returns:
            Dictionary of loss components
        """
        self.optimizer_G.zero_grad()
        
        latent_dim = self.config.get('model', {}).get('generator', {}).get('latent_dim', 100)
        
        # Generate fake samples
        z = torch.randn(batch_size, latent_dim, device=self.device)
        fake_batch = self.generator(z)
        
        # Compute generator loss
        generator_loss, loss_components = self.loss_fn.generator_loss(
            self.critic, fake_batch
        )
        
        # Backward pass
        generator_loss.backward()
        
        # Compute gradient norm
        grad_norm_G = self.compute_gradient_norm(self.generator)
        loss_components['grad_norm_generator'] = grad_norm_G
        
        # Update generator
        self.optimizer_G.step()
        
        return loss_components
    
    def compute_gradient_norm(self, model: nn.Module) -> float:
        """Compute gradient norm for a model."""
        total_norm = 0.0
        param_count = 0
        
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                param_count += 1
        
        if param_count == 0:
            return 0.0
        
        return (total_norm ** 0.5)
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, Any]:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Epoch metrics
        """
        self.generator.train()
        self.critic.train()
        
        epoch_metrics = {
            'generator_losses': [],
            'critic_losses': [],
            'grad_norms_G': [],
            'grad_norms_C': []
        }

        epoch_sals_real = defaultdict(list)
        epoch_sals_fake = defaultdict(list)
        epoch_occ = defaultdict(list)
        
        critic_iterations = self.config.get('training', {}).get('critic_iterations', 5)
        
        for batch_idx, real_batch in enumerate(dataloader):
            real_batch = real_batch.to(self.device)
            batch_size = real_batch.size(0)
            # XAI: periodic probes
            xai_cfg = self.config['xai']
            # 1) Gradient-based spectral saliency (real & fake)
            if xai_cfg['grad_spectral_every'] and (batch_idx % xai_cfg['grad_spectral_every'] == 0):
                sal_real = self._spectral_saliency(real_batch)
                for k, v in sal_real.items():
                    epoch_sals_real[k].append(v)

                latent_dim = self.config.get('model', {}).get('generator', {}).get('latent_dim', 100)
                z = torch.randn(batch_size, latent_dim, device=self.device)
                fake = self.generator(z)
                sal_fake = self._spectral_saliency(fake)
                for k, v in sal_fake.items():
                    epoch_sals_fake[k].append(v)

            # 2) Spectral occlusion (real; small subset for speed)
            if xai_cfg['occlusion_every'] and (batch_idx % xai_cfg['occlusion_every'] == 0):
                xs = real_batch[: min(8, batch_size)]
                occ = self._spectral_occlusion(xs)
                for k, v in occ.items():
                    epoch_occ[k].append(v)

            # 3) Time-domain saliency stats (real)
            if xai_cfg['time_saliency_every'] and (batch_idx % xai_cfg['time_saliency_every'] == 0):
                sal_curve, sal_max, sal_mean = self._time_saliency(real_batch)
                epoch_metrics.setdefault('time_sal_max', []).append(sal_max)
                epoch_metrics.setdefault('time_sal_mean', []).append(sal_mean)

            # 4) STFT Grad-CAM (preview scalar)
            if xai_cfg['stft_gradcam_every'] and (batch_idx % xai_cfg['stft_gradcam_every'] == 0):
                H = self._stft_gradcam(real_batch[: min(4, batch_size)])
                epoch_metrics.setdefault('stft_gradcam_mean', []).append(float(H.mean().item()))

            # 5) Channel x Band heatmap (real & fake; saved as image)
            if xai_cfg['channel_band_every'] and (batch_idx % xai_cfg['channel_band_every'] == 0):
                self._save_channel_band_heatmap_pair(real_batch[: min(8, batch_size)], epoch)

            # Collect metrics for synchronized plotting
            batch_critic_losses = []
            batch_critic_grad_norms = []
            batch_gradient_penalties = []

            # Train critic multiple times
            for _ in range(critic_iterations):
                critic_metrics = self.train_step_critic(real_batch)
                batch_critic_losses.append(critic_metrics['critic_loss'])
                batch_critic_grad_norms.append(critic_metrics['grad_norm_critic'])
                batch_gradient_penalties.append(critic_metrics.get('gradient_penalty', 0.0))

                # Update gradient_penalty
                self.monitor.update_batch({'gradient_penalty': critic_metrics.get('gradient_penalty', 0.0)})

                # Still aggregate for epoch statistics
                epoch_metrics['critic_losses'].append(critic_metrics['critic_loss'])
                epoch_metrics['grad_norms_C'].append(critic_metrics['grad_norm_critic'])

            # Train generator
            generator_metrics = self.train_step_generator(batch_size)
            epoch_metrics['generator_losses'].append(generator_metrics['generator_loss'])
            epoch_metrics['grad_norms_G'].append(generator_metrics['grad_norm_generator'])

            # Create synchronized batch metrics (one point per batch, not per critic iteration)
            synchronized_metrics = {
                'critic_loss': sum(batch_critic_losses) / len(batch_critic_losses),  # Average critic loss
                'generator_loss': generator_metrics['generator_loss'],
                'grad_norm_critic': sum(batch_critic_grad_norms) / len(batch_critic_grad_norms),  # Average critic grad norm
                'grad_norm_generator': generator_metrics['grad_norm_generator']
            }

            # Update monitoring with synchronized metrics (once per batch)
            self.monitor.update_batch(synchronized_metrics)


            self.global_step += 1  # One step per batch (synchronized)
        
        # Compute epoch statistics
        epoch_stats = {
            'epoch': epoch,
            'generator_loss': sum(epoch_metrics['generator_losses']) / len(epoch_metrics['generator_losses']),
            'critic_loss': sum(epoch_metrics['critic_losses']) / len(epoch_metrics['critic_losses']),
            'grad_norm_generator': sum(epoch_metrics['grad_norms_G']) / len(epoch_metrics['grad_norms_G']),
            'grad_norm_critic': sum(epoch_metrics['grad_norms_C']) / len(epoch_metrics['grad_norms_C']),
            'lr_generator': self.optimizer_G.param_groups[0]['lr'],
            'lr_critic': self.optimizer_C.param_groups[0]['lr']
        }

        # Add means for time-saliency and STFT Grad-CAM if present
        if epoch_metrics.get('time_sal_max'):
            epoch_stats['xai/time_saliency_max'] = float(
                sum(epoch_metrics['time_sal_max']) / len(epoch_metrics['time_sal_max']))
        if epoch_metrics.get('time_sal_mean'):
            epoch_stats['xai/time_saliency_mean'] = float(
                sum(epoch_metrics['time_sal_mean']) / len(epoch_metrics['time_sal_mean']))
        if epoch_metrics.get('stft_gradcam_mean'):
            epoch_stats['xai/stft_gradcam_mean'] = float(
                sum(epoch_metrics['stft_gradcam_mean']) / len(epoch_metrics['stft_gradcam_mean']))

        # Average and re-normalize per-band saliency/occlusion
        def _mean_and_norm(bucket):
            out = {}
            for k, vals in bucket.items():
                if vals:
                    out[k] = float(sum(vals) / len(vals))
            s = sum(out.values()) + 1e-9
            for k in list(out.keys()):
                out[k] = out[k] / s
            return out

        sal_r_mean = _mean_and_norm(epoch_sals_real)  # keys like xai/saliency/alpha
        sal_f_mean = _mean_and_norm(epoch_sals_fake)
        occ_mean = _mean_and_norm(epoch_occ)  # keys like xai/occlusion_drop/alpha

        # Write to epoch_stats using *_epoch suffix
        for k, v in sal_r_mean.items():
            epoch_stats[f'{k}_real_epoch'] = v
        for k, v in sal_f_mean.items():
            epoch_stats[f'{k}_fake_epoch'] = v
        for k, v in occ_mean.items():
            epoch_stats[f'{k}_epoch'] = v

        # Attention rollout entropy (once per epoch) — write into epoch_stats (NOT update_batch)
        if self.config['xai']['rollout_every'] and (epoch % self.config['xai']['rollout_every'] == 0):
            ent = self._attention_rollout_entropy()
            if ent is not None:
                epoch_stats['xai/attn_entropy'] = ent
            # Save attention heatmaps snapshot
            self._save_attention_heatmaps(epoch)
        self._clear_attention_buffers()

        return epoch_stats

    def train(self, dataloader: DataLoader, num_epochs: int):
        """
        Main training loop.
        
        Args:
            dataloader: Training data loader
            num_epochs: Number of epochs to train
        """
        self.monitor.start_training(num_epochs)
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Train for one epoch
            epoch_stats = self.train_epoch(dataloader, epoch)
            
            # Update learning rate schedulers
            if self.scheduler_G:
                self.scheduler_G.step()
            if self.scheduler_C:
                self.scheduler_C.step()
            
            # Add timing information
            epoch_time = time.time() - epoch_start_time
            epoch_stats['epoch_time'] = epoch_time
            
            # Update monitoring
            self.monitor.update_epoch(epoch, epoch_stats)
            
            # Save checkpoints
            if epoch % self.config.get('checkpoint_frequency', 50) == 0:
                self.save_checkpoint(epoch)
            
            # Cleanup memory
            cleanup_memory(self.device)
            
            self.current_epoch = epoch
        
        self.monitor.finish_training()
    
    def save_checkpoint(self, epoch: int):
        """Save model checkpoints."""
        checkpoint_dir = os.path.join(self.monitor.output_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save generator
        generator_path = os.path.join(checkpoint_dir, f'generator_epoch_{epoch}.pth')
        torch.save(self.generator.state_dict(), generator_path)
        
        # Save critic
        critic_path = os.path.join(checkpoint_dir, f'critic_epoch_{epoch}.pth')
        torch.save(self.critic.state_dict(), critic_path)
        
        # Save training state
        state_path = os.path.join(checkpoint_dir, f'training_state_epoch_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_C_state_dict': self.optimizer_C.state_dict(),
            'scheduler_G_state_dict': self.scheduler_G.state_dict() if self.scheduler_G else None,
            'scheduler_C_state_dict': self.scheduler_C.state_dict() if self.scheduler_C else None,
            'config': self.config
        }, state_path)
        
        logger.info(f"Checkpoint saved at epoch {epoch}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_C.load_state_dict(checkpoint['optimizer_C_state_dict'])
        
        if self.scheduler_G and checkpoint.get('scheduler_G_state_dict'):
            self.scheduler_G.load_state_dict(checkpoint['scheduler_G_state_dict'])
        if self.scheduler_C and checkpoint.get('scheduler_C_state_dict'):
            self.scheduler_C.load_state_dict(checkpoint['scheduler_C_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    def generate_samples(self, n_samples: int) -> torch.Tensor:
        """Generate synthetic samples."""
        return self.generator.generate_samples(n_samples, self.device)

    # ---- add to class body ----
    def _xai_defaults(self):
        # Reasonable defaults; override via config['xai']
        x = self.config.setdefault('xai', {})
        x.setdefault('bands', [
            ('delta', 0.5, 4.0),
            ('theta', 4.0, 8.0),
            ('alpha', 8.0, 13.0),
            ('beta', 13.0, 30.0),
            ('gamma', 30.0, 80.0),
            ('ripple', 80.0, 250.0),
            ('fast ripple', 250.0, 500.0),
        ])
        x.setdefault('grad_spectral_every', 20)  # batches
        x.setdefault('occlusion_every', 60)  # batches
        x.setdefault('time_saliency_every', 20)  # batches
        x.setdefault('stft_gradcam_every', 100)  # batches
        x.setdefault('rollout_every', 1)  # epochs
        x.setdefault('stft', {'n_fft': 256, 'hop': 64, 'win': 256})
        x.setdefault('channel_band_every', 50)  # batches

    def _register_attention_hooks(self):
        """Attach forward hooks to modules that expose attention weights."""

        def hook_fn(module, inp, out):
            attn = None
            if isinstance(out, (tuple, list)) and len(out) >= 2:
                attn = out[1]
            elif isinstance(out, dict) and 'attn_weights' in out:
                attn = out['attn_weights']
            elif hasattr(module, 'last_attn'):
                attn = module.last_attn
            if attn is not None:
                with torch.no_grad():
                    A = attn
                    # normalize common shapes to (Q,K)
                    if A.dim() == 4:  # (B, H, Q, K)
                        A = A.mean(dim=(0, 1))  # -> (Q, K)
                    elif A.dim() == 3:  # (B, Q, K) or (H, Q, K) → average first dim
                        A = A.mean(dim=0)  # -> (Q, K)
                    # print(f"Collected Attention Shape: {A.shape}")
                    A = A.detach().cpu()
                    self._xai['attn_buffers'].append(A)

                    # If module carries explicit axis annotation, route to labeled buckets
                    axis = getattr(module, 'attn_axis', None)
                    if axis == 'temporal':
                        self._xai['attn_temporal'].append(A)
                    elif axis == 'channel':
                        self._xai['attn_channel'].append(A)

        for name, m in self.critic.named_modules():
            if isinstance(m, torch.nn.MultiheadAttention) or 'attn' in name.lower() or 'transformer' in name.lower():
                h = m.register_forward_hook(hook_fn)
                self._xai['attn_handles'].append(h)

    def _clear_attention_buffers(self):
        self._xai['attn_buffers'].clear()
        self._xai['attn_temporal'].clear()
        self._xai['attn_channel'].clear()


    def _fs_and_bands(self):
        fs = self.config.get('data', {}).get('fs', 2048)
        bands = self.config.get('xai', {}).get('bands')
        return fs, bands

    @torch.no_grad()
    def _band_masks(self, T, fs, bands):
        freqs = torch.fft.rfftfreq(T, d=1.0 / fs).to(self.device)
        masks = [((freqs >= f0) & (freqs < f1)) for _, f0, f1 in bands]
        return freqs, masks

    def _spectral_saliency(self, x_time: torch.Tensor) -> dict:
        """
        Gradient-based spectral saliency per band (normalized).
        x_time: (B, C, T)
        """
        fs, bands = self._fs_and_bands()
        x = x_time.detach().clone().requires_grad_(True)
        score = self.critic(x).mean()
        self.critic.zero_grad(set_to_none=True)
        score.backward(retain_graph=False)
        grad = x.grad  # (B, C, T)

        sal_time = (grad.abs() * x.abs()).mean(dim=1)  # (B, T)
        sal_freq = torch.fft.rfft(sal_time, dim=-1).abs().mean(dim=0)  # (F,)
        _, band_masks = self._band_masks(x.shape[-1], fs, bands)

        out = {}
        total = sal_freq.sum().item() + 1e-9
        for (name, _, _), mask in zip(bands, band_masks):
            out[f'xai/saliency/{name}'] = (sal_freq[mask].sum().item()) / total
        return out

    def _band_notch(self, x: torch.Tensor, fs: float, f0: float, f1: float) -> torch.Tensor:
        """Zero the band [f0,f1) in rFFT magnitude and iFFT back."""
        X = torch.fft.rfft(x, dim=-1)
        freqs = torch.fft.rfftfreq(x.shape[-1], d=1.0 / fs).to(x.device)
        mask = ((freqs >= f0) & (freqs < f1)).view(1, 1, -1)
        X_masked = X * (~mask)
        return torch.fft.irfft(X_masked, n=x.shape[-1], dim=-1)

    @torch.no_grad()
    def _spectral_occlusion(self, x_time: torch.Tensor) -> dict:
        """Relative drop in critic score when each band is notched (normalized)."""
        fs, bands = self._fs_and_bands()
        base = self.critic(x_time).mean().item()
        drops = {}
        for name, f0, f1 in bands:
            xn = self._band_notch(x_time, fs, f0, f1)
            drop = max(base - self.critic(xn).mean().item(), 0.0)
            drops[f'xai/occlusion_drop/{name}'] = drop
        s = sum(drops.values()) + 1e-9
        for k in drops:
            drops[k] /= s
        return drops

    def _time_saliency(self, x_time: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        """
        Time-domain saliency on (B,C,T): grad*input aggregated over channels & batch.
        Returns: sal_curve(T,), max, mean (floats)
        """
        x = x_time.detach().clone().requires_grad_(True)
        y = self.critic(x).mean()
        self.critic.zero_grad(set_to_none=True);
        y.backward()
        sal_t = (x.grad.abs() * x.abs()).mean(dim=1).mean(dim=0)  # (T,)
        return sal_t.detach(), float(sal_t.max().item()), float(sal_t.mean().item())

    def _attention_rollout_entropy(self) -> Optional[float]:
        """Simple entropy over averaged attention (Q,K) → distribution over K."""
        import torch.nn.functional as F
        if not self._xai['attn_buffers']:
            return None
        # Mix of shapes possible (e.g., temporal T x T and channel C x C). Resize to common grid.
        buffers = self._xai['attn_buffers']
        sizes = [b.shape for b in buffers]
        target_Q = max(s[0] for s in sizes)
        target_K = max(s[1] for s in sizes)
        resized = []
        for b in buffers:
            if b.shape != (target_Q, target_K):
                b_rs = F.interpolate(b.unsqueeze(0).unsqueeze(0), size=(target_Q, target_K),
                                     mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
            else:
                b_rs = b
            resized.append(b_rs)
        A = torch.stack(resized)  # (N, Q, K)
        A = A.clamp(min=1e-9)
        A = A / A.sum(dim=-1, keepdim=True)  # row-normalize
        A_mean = A.mean(dim=(0,))  # (Q, K)
        p = A_mean.mean(dim=0)  # (K,) average over queries
        p = p / p.sum()
        ent = float(-(p * (p.add(1e-12).log())).sum().item())
        return ent

    def _save_attention_heatmaps(self, epoch: int) -> None:
        """Save mean attention heatmaps for temporal vs channel attention.
        Heuristic: larger maps are temporal (based on T), smaller are channel (based on C).
        """
        if not self._xai['attn_buffers']:
            return
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import torch.nn.functional as F

        # Prefer explicitly labeled buffers; fallback to heuristic split if empty
        temporal = list(self._xai['attn_temporal'])
        channel = list(self._xai['attn_channel'])
        if not temporal and not channel:
            buffers = self._xai['attn_buffers']
            if not buffers:
                return
            # Split by size threshold (median of max dim)
            max_dims = [max(b.shape) for b in buffers]
            thr = sorted(max_dims)[len(max_dims) // 2]
            temporal = [b for b in buffers if max(b.shape) >= thr]
            channel = [b for b in buffers if max(b.shape) < thr]

        def _resize_stack_mean(tensors):
            if not tensors:
                return None
            sizes = [t.shape for t in tensors]
            tq = max(s[0] for s in sizes)
            tk = max(s[1] for s in sizes)
            out = []
            for t in tensors:
                if t.shape != (tq, tk):
                    tr = F.interpolate(t.unsqueeze(0).unsqueeze(0), size=(tq, tk),
                                       mode='bilinear', align_corners=False).squeeze()
                else:
                    tr = t
                # normalize for visualization
                tr = tr - tr.min()
                denom = tr.max().clamp_min(1e-9)
                tr = tr / denom
                out.append(tr)
            return torch.stack(out).mean(0)

        Ht = _resize_stack_mean(temporal)
        Hc = _resize_stack_mean(channel)

        plots_dir = os.path.join(self.monitor.output_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)

        # Enforce canonical sizes and labels: temporal (T x T), channel (C x C)
        C = getattr(self.critic, 'channels', None)
        T = getattr(self.critic, 'window_size', None)

        if Ht is not None:
            if T is not None:
                if Ht.shape != (T, T):
                    Ht = F.interpolate(Ht.unsqueeze(0).unsqueeze(0), size=(T, T),
                                       mode='bilinear', align_corners=False).squeeze(0).squeeze(0)

            # Save data to .npy file ---
            np_path_t = os.path.join(plots_dir, f'attn_temporal_epoch_{epoch}.npy')
            np.save(np_path_t, Ht.numpy())

            plt.figure(figsize=(5.5, 4.5))
            plt.imshow(Ht.numpy(), aspect='auto', origin='lower', cmap='YlGnBu', vmin=0.0, vmax=1.0)
            cbar = plt.colorbar()
            plt.title('Mean Temporal Attention')
            # Axis ticks: [1, T/4, T/2, 3T/4, T]
            if T is not None and T >= 4:
                xticks = [0, T//4 - 1 if T//4>0 else 0, T//2 - 1 if T//2>0 else 0, (3*T)//4 - 1 if (3*T)//4>0 else 0, T-1]
                xlabels = [1, T//4, T//2, (3*T)//4, T]
                plt.xticks(xticks, xlabels)
                plt.yticks(xticks, xlabels)
                plt.xlabel('Time index')
                plt.ylabel('Time index')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'attn_temporal_epoch_{epoch}.png'))
            plt.close()
        if Hc is not None:
            if C is not None:
                if Hc.shape != (C, C):
                    Hc = F.interpolate(Hc.unsqueeze(0).unsqueeze(0), size=(C, C),
                                       mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
            # Save data to .npy file ---
            np_path_c = os.path.join(plots_dir, f'attn_channel_epoch_{epoch}.npy')
            np.save(np_path_c, Hc.numpy())

            plt.figure(figsize=(5.5, 4.5))
            plt.imshow(Hc.numpy(), aspect='auto', origin='lower', cmap='YlGnBu', vmin=0.0, vmax=1.0)
            cbar = plt.colorbar()
            plt.title('Mean Channel Attention')
            # Axis ticks/labels: CH1..CHC
            if C is not None and C <= 64:
                ticks = list(range(C))
                labels = [f'CH{i+1}' for i in range(C)]
                plt.xticks(ticks, labels, rotation=45, ha='right')
                plt.yticks(ticks, labels)
                plt.xlabel('Channel')
                plt.ylabel('Channel')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'attn_channel_epoch_{epoch}.png'))
            plt.close()

    def _stft_gradcam(self, x_time: torch.Tensor) -> torch.Tensor:
        """
        STFT Grad-CAM style map:
          1) time saliency per-sample -> (T,)
          2) STFT magnitude of input -> (F, T)
          3) weight spectrogram by normalized time-saliency
        Returns: heatmap (F, T) averaged over batch.
        """
        fs, _ = self._fs_and_bands()
        B, C, T = x_time.shape
        # 1) time saliency per sample
        x = x_time.detach().clone().requires_grad_(True)
        y = self.critic(x).mean()
        self.critic.zero_grad(set_to_none=True);
        y.backward()
        sal = (x.grad.abs() * x.abs()).mean(dim=1)  # (B, T)
        sal = sal / (sal.amax(dim=-1, keepdim=True) + 1e-9)

        # 2) STFT magnitude
        st = self.config['xai']['stft']
        n_fft, hop, win = st['n_fft'], st['hop'], st['win']
        window = torch.hann_window(win, device=self.device)
        heatmaps = []
        for i in range(B):
            xi = x_time[i:i + 1].mean(dim=1)  # (1,T) avg over channels
            S = torch.stft(xi, n_fft=n_fft, hop_length=hop, win_length=win,
                           window=window, center=True, return_complex=True)  # (1,F,Tf)
            H = S.abs().squeeze(0)  # (F, Tf)
            # match lengths (simple crop/pad)
            Tf = H.shape[-1]
            t = sal[i]
            if Tf != t.shape[0]:
                # interpolate time-saliency to Tf
                t_new = torch.nn.functional.interpolate(t.view(1, 1, -1), size=Tf, mode='linear',
                                                        align_corners=False).view(-1)
            else:
                t_new = t
            Hn = H / (H.amax() + 1e-9)
            heatmaps.append(Hn * t_new)
        Hmean = torch.stack(heatmaps).mean(dim=0)  # (F, Tf)
        return Hmean.detach().cpu()

    def _channel_band_matrix(self, x_time: torch.Tensor) -> Tuple[list, torch.Tensor]:
        """
        Compute channel x frequency-band saliency using grad*input.
        Returns: (band_names, heatmap) with heatmap shape (C, num_bands), CPU tensor normalized per channel.
        """
        fs, bands = self._fs_and_bands()
        band_names = [name for name, _, _ in bands]
        B, C, T = x_time.shape

        x = x_time.detach().clone().requires_grad_(True)
        y = self.critic(x).mean()
        self.critic.zero_grad(set_to_none=True)
        y.backward()

        # saliency per channel over time, averaged over batch
        sal_ct = (x.grad.abs() * x.abs()).mean(dim=0)  # (C, T)
        # go to frequency domain per channel
        S_cf = torch.fft.rfft(sal_ct, dim=-1).abs()  # (C, F)
        freqs = torch.fft.rfftfreq(T, d=1.0 / fs).to(self.device)

        # integrate magnitude within each band
        chan_band = []  # list of (C,)
        for _, f0, f1 in bands:
            mask = ((freqs >= f0) & (freqs < f1)).view(1, -1)
            val = (S_cf * mask).sum(dim=1)  # (C,)
            chan_band.append(val)
        M = torch.stack(chan_band, dim=1)  # (C, num_bands)

        # normalize per channel for interpretability
        M = M / (M.sum(dim=1, keepdim=True) + 1e-9)
        return band_names, M.detach().cpu()

    def _save_channel_band_heatmap(self, x_time: torch.Tensor, epoch: int) -> None:
        """Save channel x band heatmap image for a given batch of real data."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        band_names, M = self._channel_band_matrix(x_time)
        plots_dir = os.path.join(self.monitor.output_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)

        plt.figure(figsize=(max(6, len(band_names) * 0.8), max(6, M.shape[0] * 0.15)))
        plt.imshow(M.numpy(), aspect='auto', origin='lower', cmap='magma')
        plt.colorbar(label='Relative saliency')
        plt.xticks(ticks=list(range(len(band_names))), labels=band_names, rotation=45, ha='right')
        plt.yticks(ticks=list(range(M.shape[0])), labels=[f'ch{c+1}' for c in range(M.shape[0])])
        plt.xlabel('Frequency band')
        plt.ylabel('Channel')
        plt.title('Channel × Band saliency (grad*input)')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'chan_band_epoch_{epoch}.png'))
        plt.close()

    def _save_channel_band_heatmap_pair(self, x_real: torch.Tensor, epoch: int) -> None:
        """Save channel×band heatmaps for real and fake, plus a combined side-by-side figure."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Real
        band_names, M_real = self._channel_band_matrix(x_real)

        # Fake with matching batch size
        B = x_real.shape[0]
        latent_dim = self.config.get('model', {}).get('generator', {}).get('latent_dim', 100)
        z = torch.randn(B, latent_dim, device=self.device)
        x_fake = self.generator(z).detach()
        _, M_fake = self._channel_band_matrix(x_fake)

        plots_dir = os.path.join(self.monitor.output_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)

        # M_real and M_fake are CPU tensors from _channel_band_matrix
        np.save(os.path.join(plots_dir, f'chan_band_real_epoch_{epoch}.npy'), M_real.numpy())
        np.save(os.path.join(plots_dir, f'chan_band_fake_epoch_{epoch}.npy'), M_fake.numpy())
        # Save the band names too, as they are the labels for the x-axis
        np.save(os.path.join(plots_dir, f'chan_band_names_epoch_{epoch}.npy'), np.array(band_names))
        # --- END ADDED ---

        # Save separate
        def _save_single(M, suffix, cmap):
            plt.figure(figsize=(max(6, len(band_names) * 0.8), max(6, M.shape[0] * 0.15)))
            plt.imshow(M.numpy(), aspect='auto', origin='lower', cmap=cmap, vmin=0.0, vmax=1.0)
            plt.colorbar(label='Relative saliency (0-1)')
            plt.xticks(ticks=list(range(len(band_names))), labels=band_names, rotation=45, ha='right')
            plt.yticks(ticks=list(range(M.shape[0])), labels=[f'ch{c+1}' for c in range(M.shape[0])])
            plt.xlabel('Frequency band')
            plt.ylabel('Channel')
            plt.title(f'Channel × Band saliency ({suffix})')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'chan_band_{suffix}_epoch_{epoch}.png'))
            plt.close()

        # Use the same light colormap for both real and fake for comparability
        light_cmap = 'YlOrRd'
        _save_single(M_real, 'real', light_cmap)
        _save_single(M_fake, 'fake', light_cmap)

        # Save combined side-by-side
        fig, axes = plt.subplots(1, 2, figsize=(max(10, len(band_names) * 1.2), max(6, M_real.shape[0] * 0.18)))
        im0 = axes[0].imshow(M_real.numpy(), aspect='auto', origin='lower', cmap=light_cmap, vmin=0.0, vmax=1.0)
        axes[0].set_title('Real')
        axes[0].set_xticks(list(range(len(band_names))))
        axes[0].set_xticklabels(band_names, rotation=45, ha='right')
        axes[0].set_yticks(list(range(M_real.shape[0])))
        axes[0].set_yticklabels([f'ch{c+1}' for c in range(M_real.shape[0])])
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(M_fake.numpy(), aspect='auto', origin='lower', cmap=light_cmap, vmin=0.0, vmax=1.0)
        axes[1].set_title('Fake')
        axes[1].set_xticks(list(range(len(band_names))))
        axes[1].set_xticklabels(band_names, rotation=45, ha='right')
        axes[1].set_yticks(list(range(M_fake.shape[0])))
        axes[1].set_yticklabels([f'ch{c+1}' for c in range(M_fake.shape[0])])
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        fig.suptitle('Channel × Band saliency: Real vs Fake')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'chan_band_real_fake_epoch_{epoch}.png'))
        plt.close(fig)

class GANTrainer(Trainer):
    """Specialized trainer for GAN training with additional features."""
    
    def __init__(self, *args, **kwargs):
        """Initialize GAN trainer."""
        super().__init__(*args, **kwargs)
        
        # Additional GAN-specific setup
        self.setup_additional_losses()
    
    def setup_additional_losses(self):
        """Setup additional loss functions for improved training."""
        loss_config = self.config.get('loss', {})
        
        # Feature matching loss
        if loss_config.get('use_feature_matching', False):
            from .losses import FeatureMatchingLoss
            self.feature_matching_loss = FeatureMatchingLoss(
                weight=loss_config.get('feature_matching_weight', 1.0)
            )
        else:
            self.feature_matching_loss = None
        
        # Spectral loss
        if loss_config.get('use_spectral_loss', False):
            from .losses import SpectralLoss
            self.spectral_loss = SpectralLoss(
                weight=loss_config.get('spectral_loss_weight', 1.0)
            )
        else:
            self.spectral_loss = None
