"""
Preset parameter profiles for common exam scenarios.

These presets provide realistic starting points for different types of exams.
"""

from pathlib import Path

from analysis_service.synthetic_data.config import GenerationConfig
from analysis_service.synthetic_data.parameters import load_config

PARAMS_DIR = Path(__file__).parent / "params"


def get_available_presets() -> list[str]:
    return [x.stem for x in PARAMS_DIR.glob("*.yaml")]


def get_preset(name: str) -> GenerationConfig:
    """Get a preset configuration by name."""
    config_path = PARAMS_DIR / f"{name}.yaml"
    if not config_path.exists():
        available_presets = get_available_presets()
        raise ValueError(
            f"Unknown preset: {name}. Available presets: {available_presets}"
        )
    return load_config(config_path)
