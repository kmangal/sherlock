"""
CSV loading utilities for exam response data.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from analysis_service.core.constants import MISSING_CHAR, MISSING_VALUE
from analysis_service.core.data_models import ResponseMatrix


def _parse_answer_string(answer_string: str) -> list[int]:
    """Parse an answer string into response indices.

    A-Z maps to 0-25, MISSING_CHAR maps to MISSING_VALUE.
    """
    responses: list[int] = []
    for char in answer_string:
        if char == MISSING_CHAR:
            responses.append(MISSING_VALUE)
        elif "A" <= char <= "Z":
            responses.append(ord(char) - ord("A"))
        else:
            raise ValueError(f"Invalid character in answer string: '{char}'")
    return responses


def load_csv_to_response_matrix(
    path: Path,
) -> tuple[list[str], ResponseMatrix]:
    """Load a CSV file with exam responses into a ResponseMatrix.

    Expected CSV columns:
        - candidate_id: unique identifier for each candidate
        - answer_string: string of response letters (e.g., "ABCD*A")

    Returns:
        Tuple of (candidate_ids, ResponseMatrix).

    Raises:
        ValueError: If CSV format is invalid or data is inconsistent.
    """
    df = pd.read_csv(path, dtype=str)

    if "candidate_id" not in df.columns:
        raise ValueError("CSV must have 'candidate_id' column")
    if "answer_string" not in df.columns:
        raise ValueError("CSV must have 'answer_string' column")

    candidate_ids: list[str] = df["candidate_id"].tolist()
    answer_strings: list[str] = df["answer_string"].tolist()

    # Validate all answer strings are the same length
    lengths = {len(s) for s in answer_strings}
    if len(lengths) != 1:
        raise ValueError(
            f"Inconsistent answer string lengths: {sorted(lengths)}"
        )

    # Parse responses
    response_lists = [_parse_answer_string(s) for s in answer_strings]
    responses = np.array(response_lists, dtype=np.int8)

    # Infer n_categories from unique non-missing values
    valid_mask = responses != MISSING_VALUE
    unique_values = set(responses[valid_mask].tolist())
    n_categories = len(unique_values)

    return candidate_ids, ResponseMatrix(
        responses=responses, n_categories=n_categories
    )
