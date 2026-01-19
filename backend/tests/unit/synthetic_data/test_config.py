import pytest

from analysis_service.synthetic_data.parameters import load_config
from analysis_service.synthetic_data.presets import PARAMS_DIR, get_preset

########################################################
# Configuration loading
########################################################


def test_configuration_presets_load() -> None:
    """Make sure that all the preset configuration files load."""

    for config_path in PARAMS_DIR.glob("*.yaml"):
        # Skip empty files
        if config_path.read_text().strip() == "":
            continue
        preset = load_config(config_path)
        assert preset is not None


def test_get_preset_baseline_succeeds() -> None:
    preset = get_preset("baseline")
    assert preset is not None


def test_get_preset_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown preset"):
        get_preset("nonexistent_preset")
