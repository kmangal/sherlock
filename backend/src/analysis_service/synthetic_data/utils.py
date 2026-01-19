"""
String utility functions for synthetic data generation.

Helpers for converting between answer indices and string representations.
"""

import numpy as np
from numpy.typing import NDArray

from analysis_service.core.constants import MISSING_CHAR, MISSING_VALUE


def index_to_letter(index: int) -> str:
    """
    Convert a 0-based index to a letter (0 -> 'A', 1 -> 'B', etc.).

    Args:
        index: 0-based index.

    Returns:
        Corresponding uppercase letter.

    Raises:
        ValueError: If index is out of range [0, 25].
    """
    if not (0 <= index <= 25):
        raise ValueError(f"Index must be in [0, 25], got {index}")
    return chr(ord("A") + index)


def letter_to_index(letter: str) -> int:
    """
    Convert a letter to a 0-based index ('A' -> 0, 'B' -> 1, etc.).

    Args:
        letter: Uppercase letter A-Z.

    Returns:
        Corresponding 0-based index.

    Raises:
        ValueError: If letter is not A-Z.
    """
    if len(letter) != 1 or not ("A" <= letter <= "Z"):
        raise ValueError(f"Letter must be A-Z, got '{letter}'")
    return ord(letter) - ord("A")


def responses_to_string(responses: NDArray[np.int8]) -> str:
    """
    Convert an array of response indices to an answer string.

    Args:
        responses: 1D array of response indices (0-based)

    Returns:
        Answer string (e.g., "ABCD*A").
    """
    chars = []
    for r in responses:
        if r == MISSING_VALUE:
            chars.append(MISSING_CHAR)
        else:
            chars.append(index_to_letter(int(r)))
    return "".join(chars)
