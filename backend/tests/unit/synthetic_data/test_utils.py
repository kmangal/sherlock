import numpy as np
import pytest

from analysis_service.synthetic_data.utils import (
    index_to_letter,
    letter_to_index,
    responses_to_string,
)


def test_index_to_letter() -> None:
    assert index_to_letter(0) == "A"
    assert index_to_letter(1) == "B"
    assert index_to_letter(25) == "Z"


def test_index_to_letter_out_of_range() -> None:
    with pytest.raises(ValueError):
        index_to_letter(-1)
    with pytest.raises(ValueError):
        index_to_letter(26)


def test_letter_to_index() -> None:
    assert letter_to_index("A") == 0
    assert letter_to_index("B") == 1
    assert letter_to_index("Z") == 25


def test_letter_to_index_invalid() -> None:
    with pytest.raises(ValueError):
        letter_to_index("a")
    with pytest.raises(ValueError):
        letter_to_index("AB")


def test_responses_to_string() -> None:
    responses = np.array([0, 1, 2, 3])
    assert responses_to_string(responses) == "ABCD"


def test_responses_to_string_with_missing() -> None:
    responses = np.array([0, -1, 2, -1])
    assert responses_to_string(responses) == "A*C*"
