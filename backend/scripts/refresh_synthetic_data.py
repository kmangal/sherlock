#!/usr/bin/env python
"""
Refresh synthetic data CSV files from parameter configurations.

Generates CSV files for each non-empty configuration in synthetic_data/params/.
Output files are written to data/synthetic/{preset_name}.csv.
"""

import logging
from pathlib import Path

import typer

from analysis_service.synthetic_data.generators import (
    generate_exam_responses,
    to_csv,
)
from analysis_service.synthetic_data.presets import PARAMS_DIR, get_preset

SYNTHETIC_DATA_DIR = Path(__file__).parent.parent / "data" / "synthetic"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("refresh_synthetic_data")


def get_non_empty_presets() -> list[str]:
    """Return preset names for YAML files that are non-empty."""
    presets = []
    for path in PARAMS_DIR.glob("*.yaml"):
        if path.read_text().strip() != "":
            presets.append(path.stem)
    return sorted(presets)


def main(preset: str | None = None) -> int:
    """Generate CSV files for all non-empty presets."""
    SYNTHETIC_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if preset is None:
        presets = get_non_empty_presets()
        if not presets:
            raise FileNotFoundError("No non-empty preset configurations found")
        logger.info("Found %d non-empty presets: %s", len(presets), presets)
    else:
        logger.info(f"Generating synthetic data for preset {preset}")
        presets = [preset]

    for preset_name in presets:
        output_path = SYNTHETIC_DATA_DIR / f"{preset_name}.csv"
        logger.info("Generating %s -> %s", preset_name, output_path)

        config = get_preset(preset_name)
        data = generate_exam_responses(config)
        to_csv(data, str(output_path))

        logger.info(
            "  Generated %d candidates, %d questions",
            config.n_candidates,
            config.n_questions,
        )

    logger.info(
        "Done. Generated %d CSV files in %s", len(presets), SYNTHETIC_DATA_DIR
    )
    return 0


if __name__ == "__main__":
    typer.run(main)
