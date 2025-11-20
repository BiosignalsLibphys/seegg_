"""
Configuration management utilities for ECoG GAN.

This module provides functions for loading, saving, and managing
configuration files in YAML and JSON formats.
"""

import os
import yaml
import json
from typing import Dict, Any, Optional, Union
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file format is not supported
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    file_ext = config_path.suffix.lower()
    
    try:
        with open(config_path, 'r') as f:
            if file_ext in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            elif file_ext == '.json':
                config = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {file_ext}")
        
        logger.info(f"Configuration loaded from {config_path}")
        return config
    
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}")
        raise


def save_config(config: Dict[str, Any], 
                config_path: Union[str, Path],
                format: str = 'yaml') -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary to save
        config_path: Path to save configuration file
        format: File format ('yaml' or 'json')
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(config_path, 'w') as f:
            if format.lower() in ['yaml', 'yml']:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            elif format.lower() == 'json':
                json.dump(config, f, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Configuration saved to {config_path}")
    
    except Exception as e:
        logger.error(f"Error saving configuration to {config_path}: {e}")
        raise


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries.
    Later configs override earlier ones.
    
    Args:
        *configs: Configuration dictionaries to merge
        
    Returns:
        Merged configuration dictionary
    """
    merged = {}
    
    for config in configs:
        merged = _deep_merge(merged, config)
    
    return merged


def _deep_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary (takes precedence)
        
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def validate_config(config: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    """
    Validate configuration against a schema.
    
    Args:
        config: Configuration to validate
        schema: Schema to validate against
        
    Returns:
        True if valid, False otherwise
    """
    try:
        _validate_recursive(config, schema, "")
        return True
    except ValueError as e:
        logger.error(f"Configuration validation failed: {e}")
        return False


def _validate_recursive(config: Any, schema: Any, path: str) -> None:
    """Recursively validate configuration."""
    if isinstance(schema, dict):
        if not isinstance(config, dict):
            raise ValueError(f"Expected dict at {path}, got {type(config)}")
        
        # Check required keys
        required_keys = schema.get('_required', [])
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Required key '{key}' missing at {path}")
        
        # Validate nested structures
        for key, value_schema in schema.items():
            if key.startswith('_'):  # Skip meta keys
                continue
            
            if key in config:
                new_path = f"{path}.{key}" if path else key
                _validate_recursive(config[key], value_schema, new_path)
    
    elif isinstance(schema, list):
        if not isinstance(config, list):
            raise ValueError(f"Expected list at {path}, got {type(config)}")
        
        if len(schema) > 0:
            # Validate each item against the first schema element
            for i, item in enumerate(config):
                _validate_recursive(item, schema[0], f"{path}[{i}]")
    
    elif isinstance(schema, type):
        if not isinstance(config, schema):
            raise ValueError(f"Expected {schema.__name__} at {path}, got {type(config).__name__}")


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration for ECoG GAN.
    
    Returns:
        Default configuration dictionary
    """
    return {
        'model': {
            'generator': {
                'latent_dim': 100,
                'out_channels': 100,
                'use_attention': False,
                'attention_config': {
                    'num_heads': 4,
                    'dropout': 0.1
                }
            },
            'critic': {
                'time_window': 0.5,
                'embedding_dim': 64,
                'max_window_nums': 11,
                'use_PE': True,
                'attention_config': {
                    'num_heads': 4,
                    'dropout': 0.1
                }
            }
        },
        'data': {
            'seq_len': 1536,  # 3 seconds at 512 Hz
            'batch_size': 32,
            'shuffle': True,
            'num_workers': 0,
            'preprocessing': {
                'normalization': {
                    'method': 'zscore',
                    'axis': None
                }
            }
        },
        'training': {
            'num_epochs': 100,
            'critic_iterations': 5,
            'checkpoint_frequency': 50
        },
        'optimizer': {
            'generator': {
                'lr': 0.0002,
                'betas': [0.5, 0.999]
            },
            'critic': {
                'lr': 0.0002,
                'betas': [0.5, 0.999]
            }
        },
        'scheduler': {
            'use_scheduler': True,
            'step_size': 100,
            'gamma': 0.9
        },
        'loss': {
            'type': 'wgan_gp',
            'lambda_gp': 10.0,
            'use_feature_matching': False,
            'feature_matching_weight': 1.0,
            'use_spectral_loss': False,
            'spectral_loss_weight': 1.0
        },
        'monitoring': {
            'output_dir': './outputs',
            'save_frequency': 10,
            'log_level': 'INFO'
        },
        'device': {
            'gpu_id': 0,
            'memory_fraction': 0.9
        }
    }


def create_config_from_template(template_name: str, 
                              output_path: Union[str, Path],
                              overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Create configuration from a template.
    
    Args:
        template_name: Name of the template ('default', 'fast_training', 'high_quality')
        output_path: Path to save the configuration
        overrides: Optional overrides to apply
        
    Returns:
        Created configuration
    """
    templates = {
        'default': get_default_config(),
        'fast_training': {
            **get_default_config(),
            'training': {
                'num_epochs': 50,
                'critic_iterations': 3,
                'checkpoint_frequency': 25
            },
            'data': {
                **get_default_config()['data'],
                'batch_size': 64
            }
        },
        'high_quality': {
            **get_default_config(),
            'training': {
                'num_epochs': 500,
                'critic_iterations': 5,
                'checkpoint_frequency': 50
            },
            'optimizer': {
                'generator': {
                    'lr': 0.0001,
                    'betas': [0.5, 0.999]
                },
                'critic': {
                    'lr': 0.0001,
                    'betas': [0.5, 0.999]
                }
            },
            'loss': {
                **get_default_config()['loss'],
                'use_feature_matching': True,
                'use_spectral_loss': True
            }
        }
    }
    
    if template_name not in templates:
        raise ValueError(f"Unknown template: {template_name}")
    
    config = templates[template_name]
    
    if overrides:
        config = merge_configs(config, overrides)
    
    save_config(config, output_path)
    
    return config


def update_config_from_args(config: Dict[str, Any], args: Any) -> Dict[str, Any]:
    """
    Update configuration from command line arguments.
    
    Args:
        config: Base configuration
        args: Parsed command line arguments
        
    Returns:
        Updated configuration
    """
    updates = {}
    
    # Map common command line arguments to config paths
    arg_mapping = {
        'batch_size': 'data.batch_size',
        'learning_rate': 'optimizer.generator.lr',
        'num_epochs': 'training.num_epochs',
        'latent_dim': 'model.generator.latent_dim',
        'output_dir': 'monitoring.output_dir',
        'gpu_id': 'device.gpu_id'
    }
    
    for arg_name, config_path in arg_mapping.items():
        if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
            _set_nested_value(updates, config_path, getattr(args, arg_name))
    
    return merge_configs(config, updates)


def _set_nested_value(dictionary: Dict[str, Any], path: str, value: Any) -> None:
    """Set a nested value in a dictionary using dot notation."""
    keys = path.split('.')
    current = dictionary
    
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    current[keys[-1]] = value


def print_config(config: Dict[str, Any], indent: int = 0) -> None:
    """
    Pretty print configuration.
    
    Args:
        config: Configuration to print
        indent: Current indentation level
    """
    for key, value in config.items():
        if isinstance(value, dict):
            print("  " * indent + f"{key}:")
            print_config(value, indent + 1)
        else:
            print("  " * indent + f"{key}: {value}")


class ConfigManager:
    """Configuration manager for handling multiple configurations."""
    
    def __init__(self, config_dir: Union[str, Path]):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.configs = {}
    
    def load_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """Load all configuration files from the directory."""
        self.configs = {}
        
        for config_file in self.config_dir.glob('*.yaml'):
            config_name = config_file.stem
            self.configs[config_name] = load_config(config_file)
        
        for config_file in self.config_dir.glob('*.yml'):
            config_name = config_file.stem
            self.configs[config_name] = load_config(config_file)
        
        for config_file in self.config_dir.glob('*.json'):
            config_name = config_file.stem
            self.configs[config_name] = load_config(config_file)
        
        logger.info(f"Loaded {len(self.configs)} configurations")
        return self.configs
    
    def get_config(self, name: str) -> Dict[str, Any]:
        """Get configuration by name."""
        if name not in self.configs:
            config_path = self.config_dir / f"{name}.yaml"
            if config_path.exists():
                self.configs[name] = load_config(config_path)
            else:
                raise ValueError(f"Configuration '{name}' not found")
        
        return self.configs[name]
    
    def save_config(self, name: str, config: Dict[str, Any]) -> None:
        """Save configuration with given name."""
        config_path = self.config_dir / f"{name}.yaml"
        save_config(config, config_path)
        self.configs[name] = config
    
    def list_configs(self) -> list:
        """List all available configuration names."""
        return list(self.configs.keys())
