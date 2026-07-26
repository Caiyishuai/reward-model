"""Tests for training_utils components."""

from training_utils import EarlyStopping


def test_early_stopping_patience() -> None:
    es = EarlyStopping(patience=3, higher_is_better=True)
    assert not es.step(0.5)
    assert not es.step(0.6)
    assert not es.step(0.55)
    assert not es.step(0.54)
    assert es.step(0.53)


def test_early_stopping_improvement_resets() -> None:
    es = EarlyStopping(patience=2, higher_is_better=True)
    assert not es.step(0.5)
    assert not es.step(0.4)
    assert not es.step(0.6)
    assert not es.step(0.5)
    assert es.step(0.4)


def test_early_stopping_lower_is_better() -> None:
    es = EarlyStopping(patience=2, higher_is_better=False)
    assert not es.step(1.0)
    assert not es.step(0.5)
    assert not es.step(0.6)
    assert es.step(0.7)


def test_early_stopping_never_fires_with_improvements() -> None:
    es = EarlyStopping(patience=1, higher_is_better=True)
    for v in [0.1, 0.2, 0.3, 0.4, 0.5]:
        assert not es.step(v)


def test_early_stopping_equal_scores_count_as_no_improvement() -> None:
    es = EarlyStopping(patience=2, higher_is_better=True)
    assert not es.step(0.5)
    assert not es.step(0.5)  # equal = no improvement, counter=1
    assert es.step(0.5)  # counter=2 >= patience=2


def test_early_stopping_patience_1_fires_on_second_no_improvement() -> None:
    es = EarlyStopping(patience=1, higher_is_better=True)
    assert not es.step(0.5)  # first call sets best
    assert es.step(0.4)  # no improvement, counter=1 >= patience=1
