#!/usr/bin/env python3
"""
Simple training script for VSCode execution.
Customize the parameters below and run directly in VSCode.
"""

import os
import sys
import numpy as np
import pickle
import logging
import json
from pathlib import Path

# Add the project to path
sys.path.append(str(Path(__file__).parent))

from ecog_gan import Generator, WindowCritic, ECoGDataLoader, Trainer, load_config
from ecog_gan.utils import setup_device, create_output_directories
from ecog_gan.data.preprocessors import DataPreprocessor

# Configuration parameters
DATA_PATH = "./global_norm_3s_512hz.pkl"
CONFIG_PATH = "./ST_ECoG/configs/default_config.yaml"
OUTPUT_DIR = "./training_results"  
NUM_EPOCHS = 1000
BATCH_SIZE = 64
GPU_ID = 0
CREATE_SAMPLE_DATA = False

Use_PE = True
GENERATOR_LR = 5e-5 
CRITIC_LR = 1e-5 

N_CRITIC = 5


def main():
    """Main training function."""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    print("Starting ECoG GAN Training")
    print("=" * 50)
    
    # Check if data file exists
    if not os.path.exists(DATA_PATH):
        print(f"Data file not found: {DATA_PATH}")
        print("Set CREATE_SAMPLE_DATA = True to create sample data")
        return 1
    
    # Load configuration
    try:
        config = load_config(CONFIG_PATH)
        print(f"Configuration loaded: {CONFIG_PATH}")
    except Exception as e:
        print(f"Failed to load configuration: {e}")
        return 1
    
    # Override config with script parameters
    config['training']['num_epochs'] = NUM_EPOCHS
    config['data']['batch_size'] = BATCH_SIZE
    # Note: monitoring output_dir will be set later to use timestamped directory

    # Optimizer overrides: trainer reads from config['optimizer']
    if 'optimizer' not in config:
        config['optimizer'] = {}
    if 'generator' not in config['optimizer']:
        config['optimizer']['generator'] = {}
    if 'critic' not in config['optimizer']:
        config['optimizer']['critic'] = {}
    config['optimizer']['generator']['lr'] = GENERATOR_LR
    config['optimizer']['generator']['betas'] = (0.5, 0.999)
    config['optimizer']['critic']['lr'] = CRITIC_LR
    config['optimizer']['critic']['betas'] = (0.5, 0.999)
    config['training']['critic_iterations'] = N_CRITIC
    config['model']['critic']['use_PE'] = Use_PE

    # Training stability improvements
    config['training']['gradient_clipping'] = {
        'generator': 1.0,
        'critic': 5.0
    }

    # Scheduler overrides: trainer reads from top-level config['scheduler']
    config['scheduler'] = {
        'use_scheduler': True,
        'type': 'step',
        'step_size': 100,
        'gamma': 0.9
    }

    # Loss configuration
    config['loss']['gradient_penalty_weight'] = 10.0
    config['loss']['drift_penalty'] = 0.001

    if GPU_ID is not None:
        config['device']['gpu_id'] = GPU_ID
    
    # Print final merged configuration before running
    print("\n==== Final merged configuration (pre-run) ====")
    try:
        logger.info(f"Combined config: {json.dumps(config, indent=2, ensure_ascii=False)}")
    except Exception as e:
        print(f"Warning: failed to pretty-print config: {e}")
        print(config)

    # Setup device
    device = setup_device(config['device'].get('gpu_id'))
    print(f"Using device: {device}")
    
    # Create output directories
    output_paths = create_output_directories(OUTPUT_DIR)
    print(f"Output directories created: {output_paths['base']}")

    # Use the timestamped directory for monitoring instead of parent directory
    timestamped_output_dir = output_paths['base']
    config['monitoring']['output_dir'] = timestamped_output_dir
    print(f"Monitoring will save to: {timestamped_output_dir}")

    # Setup preprocessing (lazy/conditional based on apply flags)
    preprocessor = None
    if 'preprocessing' in config['data'] and config['data']['preprocessing'] is not None:
        prep_cfg = config['data']['preprocessing']
        norm_enabled = bool(prep_cfg.get('normalization', {}).get('apply_normalization', True)) if prep_cfg.get('normalization') is not None else False
        resample_enabled = bool(prep_cfg.get('resampling', {}).get('apply_resampling', True)) if prep_cfg.get('resampling') is not None else False
        filter_enabled = bool(prep_cfg.get('filtering', {}).get('apply_filtering', True)) if prep_cfg.get('filtering') is not None else False
        augment_enabled = bool(prep_cfg.get('augmentation', {}).get('apply_augmentation', True)) if prep_cfg.get('augmentation') is not None else False

        any_enabled = norm_enabled or resample_enabled or filter_enabled or augment_enabled
        if any_enabled:
            print("\n Setting up preprocessing...")
            enabled_steps = []
            if resample_enabled: enabled_steps.append('resampling')
            if filter_enabled: enabled_steps.append('filtering')
            if norm_enabled: enabled_steps.append('normalization')
            if augment_enabled: enabled_steps.append('augmentation')
            print(f"   - Enabled steps: {enabled_steps}")
            preprocessor = DataPreprocessor(prep_cfg)
        else:
            print("\n Preprocessing is configured but all apply flags are disabled. Skipping.")

    # Load data
    print("\n Loading training data...")
    try:
        data_loader = ECoGDataLoader(
            data_path=DATA_PATH,
            seq_len=config['data']['seq_len'],
            batch_size=config['data']['batch_size'],
            shuffle=config['data']['shuffle'],
            num_workers=config['data']['num_workers']
        )
        
        data_info = data_loader.get_data_info()
        sample_shape = data_loader.get_sample_shape()

        # Apply preprocessing if configured and any step enabled
        if preprocessor:
            print("\n Applying preprocessing...")
            # Get raw data from loader
            raw_data = data_loader.data
            raw_data = np.concatenate(list(raw_data.values()), axis=0)
            print(f"   - Raw data shape: {raw_data.shape}")
            # print(f"   - Raw data range: [{raw_data.min():.3f}, {raw_data.max():.3f}]")

            # Apply preprocessing
            processed_data = preprocessor.fit_transform(raw_data)
            print(f"   - Processed data shape: {processed_data.shape}")
            # print(f"   - Processed data range: [{processed_data.min():.3f}, {processed_data.max():.3f}]")

            # Replace data in loader
            data_loader.data = processed_data
            # Recreate dataset with processed data
            data_loader.dataset = data_loader.dataset.__class__(processed_data, data_loader.seq_len, data_loader.transform)
            print(" Preprocessing applied!")

        print(f" Data loaded successfully!")
        print(f"   - Format: {data_info['format']}")
        print(f"   - Number of samples: {len(data_loader)}")
        print(f"   - Sample shape: {sample_shape}")
        
    except Exception as e:
        print(f" Failed to load data: {e}")
        return 1
    
    # Initialize models
    print("\n Initializing models...")
    try:
        target_shape = (1, sample_shape[0], sample_shape[1])
        
        generator = Generator(
            latent_dim=config['model']['generator']['latent_dim'],
            out_channels=config['model']['generator']['out_channels'],
            target_shape=target_shape,
            use_attention=config['model']['generator']['use_attention'],
            attention_config=config['model']['generator']['attention_config']
        )
        
        critic = WindowCritic(
            time_window=config['model']['critic']['time_window'],
            fs=config['data']['sampling_rate'],
            channels=sample_shape[0],
            embedding_dim=config['model']['critic']['embedding_dim'],
            max_window_nums=config['model']['critic']['max_window_nums'],
            use_PE=config['model']['critic']['use_PE'],
            attention_config=config['model']['critic']['attention_config']
        )
        
        print(f" Models initialized!")
        print(f"   - Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
        print(f"   - Critic parameters: {sum(p.numel() for p in critic.parameters()):,}")
        
    except Exception as e:
        print(f" Failed to initialize models: {e}")
        return 1
    
    # Initialize trainer
    print("\n Initializing trainer...")
    try:
        trainer = Trainer(
            generator=generator,
            critic=critic,
            config=config,
            device=device
        )
        print(" Trainer initialized!")
        
    except Exception as e:
        print(f" Failed to initialize trainer: {e}")
        return 1
    
    # Start training
    print(f"\n Starting training for {NUM_EPOCHS} epochs...")
    print("This may take a while depending on your hardware.")
    print("You can monitor progress in the output directory.")
    
    try:
        trainer.train(
            dataloader=data_loader.get_dataloader(),
            num_epochs=NUM_EPOCHS
        )
        
        print("\n Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n Training interrupted by user")
        return 1
    except Exception as e:
        print(f"\n Training failed: {e}")
        return 1
    
    # Generate sample data
    print("\n Generating synthetic samples...")
    try:
        n_samples = config['data']['generation_number']
        synthetic_data = trainer.generate_samples(n_samples)
        
        # Save synthetic data
        output_file = os.path.join(output_paths['generated_data'], 'synthetic_samples.pkl')
        with open(output_file, 'wb') as f:
            pickle.dump(synthetic_data.cpu().numpy(), f)
        
        print(f" Generated {n_samples} synthetic samples")
        print(f"   - Shape: {synthetic_data.shape}")
        print(f"   - Saved to: {output_file}")
        
    except Exception as e:
        print(f" Failed to generate samples: {e}")
        return 1
    
    print("\n Training pipeline completed successfully!")
    print(f" Results saved in: {timestamped_output_dir}")
    print(f" Parent directory: {OUTPUT_DIR}")
    print(f"   Check: {timestamped_output_dir}/plots/")
    print("   - losses_epoch_*.png")
    print("   - gradient_norms_epoch_*.png")
    print("   - learning_rates_epoch_*.png")

    return 0

if __name__ == '__main__':
    exit_code = main()
    print(f"\nExiting with code: {exit_code}")
