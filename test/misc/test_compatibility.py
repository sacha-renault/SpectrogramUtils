import warnings

from src.SpectrogramUtils.data.compatibility import are_versions_compatible

def test_compatibility():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Compatibility
        assert are_versions_compatible("0.4.6", "0.3.11")
        assert are_versions_compatible("0.4.6", "0.1.0")
        assert are_versions_compatible("0.4.6", "0.4.5")
        assert are_versions_compatible("0.4.7", "0.5.1")

        # Not compatible
        assert not are_versions_compatible("0.4.6", "0.4.7")
        assert not are_versions_compatible("0.1.0", "4.1.0")
        assert not are_versions_compatible("0.4.1", "1.0.0")
