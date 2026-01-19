from analysis_service.core.utils import get_rng


def test_rng_reproducibility() -> None:
    rng1 = get_rng(42)
    rng2 = get_rng(42)
    assert rng1.random() == rng2.random()
