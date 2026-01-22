#!/usr/bin/env python
"""
Refresh synthetic data CSV files from parameter configurations.

Generates CSV files for each non-empty configuration in synthetic_data/params/.
Output files are written to data/synthetic/{preset_name}.csv.
"""

import logging
import sys

from analysis_service.synthetic_data.generators import (
    generate_exam_responses,
    to_csv,
)
from analysis_service.synthetic_data.presets import PARAMS_DIR, get_preset
from scripts.paths import SYNTHETIC_DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_non_empty_presets() -> list[str]:
    """Return preset names for YAML files that are non-empty."""
    presets = []
    for path in PARAMS_DIR.glob("*.yaml"):
        if path.read_text().strip() != "":
            presets.append(path.stem)
    return sorted(presets)


def main() -> int:
    """Generate CSV files for all non-empty presets."""
    SYNTHETIC_DATA_DIR.mkdir(parents=True, exist_ok=True)

    presets = get_non_empty_presets()
    if not presets:
        logger.warning("No non-empty preset configurations found")
        return 1

    logger.info("Found %d non-empty presets: %s", len(presets), presets)

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
    sys.exit(main())
