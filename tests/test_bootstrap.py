"""Bootstrap test to verify pytest is working correctly."""
import pytest


def test_bootstrap_passes():
    """Dummy test to verify pytest infrastructure is set up correctly."""
    assert True, "Bootstrap test should always pass"


def test_basic_arithmetic():
    """Simple sanity check for test environment."""
    assert 1 + 1 == 2


@pytest.mark.unit
def test_imports():
    """Verify key imports work (if dependencies are installed)."""
    try:
        import einops
        import torch

        assert torch is not None
        assert einops is not None
    except ImportError:
        pytest.skip("Core dependencies not installed yet - this is expected for bootstrap")
