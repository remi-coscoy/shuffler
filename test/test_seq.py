import numpy as np
import pytest

from shuffler.deck_statistics.sequence_stat import sequence_stat


@pytest.fixture
def seq_random_input():
    return np.array([[1, 2, 4], [5, 2, 4]])


@pytest.fixture
def seq_one_input():
    return np.array([[1, 5, 3], [4, 2, 4]])


@pytest.fixture
def seq_zero_point_5_input():
    return np.array([[1, 2, 3], [4, 3, 2]])


@pytest.fixture
def seq_zero_input():
    return np.array([[1, 2, 3], [4, 5, 6]])


def test_sequence_stat(seq_random_input):
    expected_result = 0.666666666
    result = sequence_stat(seq_random_input)
    assert result == pytest.approx(expected_result, rel=1e-9)


def test_sequence_stat_one(seq_one_input):
    expected_result = 1.0
    result = sequence_stat(seq_one_input)
    assert result == pytest.approx(expected_result, rel=1e-9)


def test_sequence_stat_zero_point_five(seq_zero_point_5_input):
    expected_result = 0.5
    result = sequence_stat(seq_zero_point_5_input)
    assert result == pytest.approx(expected_result, rel=1e-9)


def test_sequence_stat_zero(seq_zero_input):
    expected_result = 0.0
    result = sequence_stat(seq_zero_input)
    assert result == pytest.approx(expected_result, rel=1e-9)
