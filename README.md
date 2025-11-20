# Attention-Guided iEEG Synthesis With DCWGAN-GP

A PyTorch implementation of the DCWGAN-GP (Deep Convolutional Wasserstein GAN with Gradient Penalty) for generating synthetic intracranial EEG (iEEG/ECoG) signals with attention mechanisms, as described in "Attention-Guided iEEG Synthesis With DCWGAN-GP for Enhanced Epilepsy Surgery Planning" .

## Overview

This project implements the DCWGAN-GP architecture specifically designed for iEEG/ECoG signal generation for epilepsy surgery planning, featuring:

- **Generator**: Deep convolutional architecture with attention-guided synthesis
- **Critic**: Wasserstein GAN critic with gradient penalty for stable training
- **Attention Mechanisms**: Spatio-temporal attention for capturing brain signal patterns
- **DCWGAN-GP Training**: Wasserstein GAN with gradient penalty for enhanced stability
- **Epilepsy Focus**: Optimized for generating realistic iEEG signals for surgical planning

## Features

- 🧠 **iEEG-Specific Architecture**: Designed specifically for intracranial EEG signal generation for epilepsy surgery planning
- 🎯 **Attention-Guided Synthesis**: Advanced spatio-temporal attention mechanisms for realistic brain signal generation
- ⚡ **DCWGAN-GP Framework**: Wasserstein GAN with gradient penalty for stable and high-quality synthesis
- 📊 **Comprehensive Monitoring**: Real-time training monitoring with plots and metrics
- 🔧 **Medical Data Processing**: Built-in preprocessing optimized for clinical iEEG data
- 🚀 **GPU Acceleration**: Optimized for CUDA-enabled training on medical datasets

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM

### Quick Install

1. Clone the repository:
```bash
git clone <repository-url>
cd ST_ECoG
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package in development mode:
```bash
pip install -e .
```

### Manual Installation

If you prefer to install dependencies manually:

```bash
pip install torch torchvision torchaudio
pip install numpy pandas scipy matplotlib seaborn
pip install pyyaml scikit-learn
```

## Quick Start

### Basic Training

1. **Prepare your data**: Place your iEEG/ECoG data in pickle format with shape `(samples, channels, time_points)`

2. **Configure training**: Edit `configs/default_config.yaml` or use the provided training script:
```python
python training_sample.py
```

3. **Monitor training**: Check the output directory for:
   - Loss plots (`plots/losses_epoch_*.png`)
   - Gradient norm plots (`plots/gradient_norms_epoch_*.png`)
   - Generated samples (`generated_data/synthetic_samples.pkl`)

### Custom Training

```python
from ecog_gan import Generator, WindowCritic, ECoGDataLoader, Trainer, load_config

# Load configuration
config = load_config('configs/default_config.yaml')

# Initialize DCWGAN-GP models
generator = Generator(
    latent_dim=128,
    out_channels=64,
    target_shape=(1, 64, 1536),  # (batch, channels, samples)
    use_attention=True  # Attention-guided synthesis
)

critic = WindowCritic(
    time_window=0.5,
    fs=512,
    channels=64,
    embedding_dim=64,
    use_PE=True  # Positional encoding for temporal patterns
)

# Load data
data_loader = ECoGDataLoader(
    data_path='your_data.pkl',
    seq_len=1536,
    batch_size=64
)

# Initialize trainer
trainer = Trainer(generator, critic, config)

# Start training
trainer.train(data_loader.get_dataloader(), num_epochs=100)
```

## Configuration

The training process is highly configurable through YAML files. Key configuration sections:

### Model Architecture
```yaml
model:
  generator:
    latent_dim: 128
    out_channels: 64
    use_attention: true
  critic:
    time_window: 0.5
    embedding_dim: 64
    use_PE: true
```

### Training Parameters
```yaml
training:
  num_epochs: 100
  critic_iterations: 5
  checkpoint_frequency: 50
```

### Data Processing
```yaml
data:
  seq_len: 1536
  sampling_rate: 512
  batch_size: 64
  preprocessing:
    normalization:
      method: "zscore"
    filtering:
      apply_filtering: true
      bandpass: [1, 100]
```

## Data Format

The system expects iEEG/ECoG data in the following format:

- **File format**: Pickle (.pkl) files
- **Data structure**: Dictionary with subject/session keys
- **Shape**: `(samples, channels, time_points)`
- **Data type**: NumPy arrays with float32 precision

Example data structure:
```python
{
    'subject_001': np.array([...]),  # Shape: (n_samples, n_channels, n_timepoints)
    'subject_002': np.array([...]),
    # ...
}
```

## Architecture Details

### Generator
- **Input**: Latent vector (128D by default)
- **Architecture**: Transposed convolutions with upsampling
- **Attention**: Optional spatial attention with multiple variants
- **Output**: Synthetic iEEG signals for epilepsy surgery planning

### WindowCritic
- **Input**: Real or synthetic iEEG signals
- **Architecture**: Window-based processing with attention
- **Attention**: Temporal and spatial attention mechanisms
- **Output**: Realism score for each window

### Attention Mechanisms
- **Spatial Attention**: Captures channel-wise relationships
- **Temporal Attention**: Models time-dependent patterns
- **Positional Encoding**: Learned position embeddings
- **Conditional Attention**: Context-aware attention

## Monitoring and Visualization

The training process includes comprehensive monitoring:

- **Loss Tracking**: Generator and critic losses over time
- **Gradient Monitoring**: Gradient norms and clipping statistics
- **Learning Rate Scheduling**: LR decay visualization
- **Sample Generation**: Periodic synthetic sample generation
- **Model Checkpoints**: Automatic model saving

## Performance Tips

1. **GPU Memory**: Adjust batch size based on available GPU memory
2. **Data Loading**: Use multiple workers for faster data loading
3. **Mixed Precision**: Enable for faster training on modern GPUs
4. **Gradient Clipping**: Tune gradient penalty weight for stability

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size
   - Enable gradient checkpointing
   - Use mixed precision training

2. **Training Instability**:
   - Adjust learning rates
   - Increase gradient penalty weight
   - Use different optimizer settings

3. **Data Loading Errors**:
   - Check data format and shape
   - Verify file paths in configuration
   - Ensure proper data preprocessing

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Citation

If you use this code in your research, please cite:

```bibtex
@article{ao2024attention,
  title={Attention-Guided iEEG Synthesis With DCWGAN-GP for Enhanced Epilepsy Surgery Planning},
  author={},
  journal={},
  year={2025},
  publisher={}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
