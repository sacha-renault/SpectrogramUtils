import numpy as np

import src.SpectrogramUtils.misc.indexers as indexers

def test_ind_1():
    indexer = indexers.get_forward_indexer_amplitude_phase(3, 2)
    assert len(indexer.shape) == 1
    assert indexer.shape == (6, )

    expected_array = np.array([0,2,4,1,3,5])
    assert np.array_equal(expected_array, indexer)

def test_ind_2():
    indexer = indexers.get_forward_indexer_phase_amplitude(3, 2)
    assert len(indexer.shape) == 1
    assert indexer.shape == (6, )

    expected_array = np.array([1,3,5,0,2,4])
    assert np.array_equal(expected_array, indexer)