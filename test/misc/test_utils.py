from itertools import permutations

import pytest
import numpy as np 

from src.SpectrogramUtils.misc import utils

def test_get_multi_stft_mono():
    # Simple stft
    result = utils.get_multi_stft(np.random.rand(10000))
    assert isinstance(result, list), "Not a list"
    assert all([isinstance(r, np.ndarray) for r in result]), "All element of the list aren't arrays"
    assert all([len(r.shape) == 2 for r in result]), "the stft aren't 2D"
    assert len(result) == 1, "The number of stft doesn't match the number of audio channels"

def test_get_multi_stft_stereo():
    # Stereo stft
    result = utils.get_multi_stft(np.random.rand(2, 10000))
    assert isinstance(result, list), "Not a list"
    assert all([isinstance(r, np.ndarray) for r in result]), "All element of the list aren't arrays"
    assert all([len(r.shape) == 2 for r in result]), "the stft aren't 2D"
    assert len(result) == 2, "The number of stft doesn't match the number of audio channels"

def test_rpad_rcut_exception():
    with pytest.raises(AssertionError):
        utils.rpad_rcut(np.random.rand(1000), 5000)

def test_rpad_rcut_normal():
    array = np.random.rand(2, 1000)
    result = utils.rpad_rcut(array, 5000)
    assert result.shape == (2, 5000)
    assert np.array_equal(array, result[:, :1000])

def test_lpad_lcut_exception():
    with pytest.raises(AssertionError):
        utils.lpad_lcut(np.random.rand(1000), 5000)

def test_lpad_lcut_normal():
    array = np.random.rand(2, 1000)
    result = utils.lpad_lcut(array, 5000)
    assert result.shape == (2, 5000)
    assert np.array_equal(array, result[:, -1000:])

def test_center_pad_rcut_exception():
    with pytest.raises(AssertionError):
        utils.center_pad_rcut(np.random.rand(1000), 5000)

def test_center_pad_rcut_normal():
    array = np.random.rand(2, 1000)
    result = utils.center_pad_rcut(array, 5000)
    assert result.shape == (2, 5000)
    assert np.array_equal(array, result[:, 2000:3000])

def test_center_pad_rcut_many():
    array = np.random.rand(2, 1000)
    for i in range(200):
        result = utils.center_pad_rcut(array, 5000 + i)
        assert result.shape == (2, 5000 + i)

def test_rpad_rcut_normal_smaller():
    array = np.random.rand(2, 1000)
    result = utils.rpad_rcut(array, 500)
    assert result.shape == (2, 500)
    assert np.array_equal(array[:, :500], result)

def test_lpad_lcut_normal_smaller():
    array = np.random.rand(2, 1000)
    result = utils.lpad_lcut(array, 500)
    assert result.shape == (2, 500)
    assert np.array_equal(array[:, -500:], result)

def test_center_pad_rcut_normal_smaller():
    array = np.random.rand(2, 1000)
    result = utils.center_pad_rcut(array, 500)
    assert result.shape == (2, 500)
    assert np.array_equal(array[:, :500], result)

def test_get_backward_indexer_assert1():
    with pytest.raises(AssertionError, match="forward_indexer should be a 1D array"):
        utils.get_backward_indexer(np.arange(6).reshape(-1, 1))

def test_get_backward_indexer_assert2():
    with pytest.raises(AssertionError, match="forward_indexer should be dtyped int"):
        utils.get_backward_indexer(np.random.rand(2))

def test_get_backward_indexer_assert3():
    with pytest.raises(AssertionError, match="The indexer isn't an arrangement"):
        utils.get_backward_indexer(np.arange(1, 6))

def test_get_backward_indexer_success():
    data = np.random.randint(0, 10, (4, 10))
    for perm in permutations(range(4)):
        forward_indexer = np.array(perm)
        backward_indexer = utils.get_backward_indexer(forward_indexer)
        assert np.array_equal(data, data[forward_indexer][backward_indexer])