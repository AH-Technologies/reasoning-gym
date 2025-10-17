"""Configuration management utilities."""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional
from copy import deepcopy


class Config:
    """Configuration class with dict-like access and deep merging."""

    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict

    def __getitem__(self, key: str) -> Any:
        return self._config[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._config[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self._config

    def to_dict(self) -> Dict[str, Any]:
        return deepcopy(self._config)

    def __repr__(self) -> str:
        return f"Config({self._config})"


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries, with override taking precedence."""
    result = deepcopy(base)

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)

    return result


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


def resolve_base_configs(config: Dict[str, Any], config_path: Path) -> Dict[str, Any]:
    """Recursively resolve _base_ configs and merge them."""
    if '_base_' not in config:
        return config

    base_paths = config.pop('_base_')
    if isinstance(base_paths, str):
        base_paths = [base_paths]

    merged = {}

    for base_path in base_paths:
        full_path = (config_path.parent / base_path).resolve()

        if not full_path.exists():
            raise FileNotFoundError(f"Base config not found: {full_path}")

        base_config = load_yaml(full_path)
        base_config = resolve_base_configs(base_config, full_path)
        merged = deep_merge(merged, base_config)

    merged = deep_merge(merged, config)

    return merged


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file with base config resolution."""
    path = Path(config_path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    config_dict = load_yaml(path)
    config_dict = resolve_base_configs(config_dict, path)

    return Config(config_dict)


def apply_cli_overrides(config: Config, overrides: Optional[List[str]]) -> Config:
    """Apply command-line overrides to config.

    Supports dot notation for nested keys:
        --override model.name=Qwen2.5-3B training.learning_rate=1e-5
    """
    if not overrides:
        return config

    config_dict = config.to_dict()

    for override in overrides:
        if '=' not in override:
            raise ValueError(f"Invalid override format: {override}. Expected key=value")

        key_path, value = override.split('=', 1)
        keys = key_path.split('.')

        value = _parse_value(value)

        current = config_dict
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    return Config(config_dict)


def _parse_value(value: str) -> Any:
    """Parse string value to appropriate Python type."""
    if value.lower() in ('true', 'false'):
        return value.lower() == 'true'

    if value.lower() in ('none', 'null'):
        return None

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        pass

    return value


def save_config(config: Config, output_path: str) -> None:
    """Save configuration to YAML file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)
