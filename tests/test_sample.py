"""Sample tests for the project."""
import pytest


def test_sample():
    """Sample test to verify testing setup."""
    assert True


def test_imports():
    """Test that core modules can be imported."""
    try:
        from src.preprocessing.video_loader import VideoLoader
        from src.models.video_interpreter import VideoInterpreter
        from src.inference.predictor import Predictor
        from src.utils.helpers import ensure_dir, get_project_root
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import modules: {e}")


# TODO: Add more tests as you develop features
