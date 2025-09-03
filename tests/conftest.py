"""
Pytest configuration and shared fixtures for CircleCI Usage Reporter tests.
"""
import os
import tempfile
import pytest


@pytest.fixture
def temp_reports_dir():
    """Create a temporary reports directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        reports_dir = os.path.join(temp_dir, 'reports')
        os.makedirs(reports_dir)
        yield reports_dir


@pytest.fixture
def sample_environment_variables():
    """Sample environment variables for testing."""
    return {
        'ORG_ID': 'test-org-123',
        'CIRCLECI_API_TOKEN': 'test-token-456',
        'START_DATE': '2024-01-01',
        'END_DATE': '2024-01-31'
    }


# Test configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
