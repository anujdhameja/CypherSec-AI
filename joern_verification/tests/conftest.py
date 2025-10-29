"""
Test configuration and shared fixtures for pytest (if used).
"""

import pytest
import tempfile
import os
from pathlib import Path

from joern_verification.tests.test_fixtures import TestEnvironmentManager, TestFixtures


@pytest.fixture(scope="session")
def test_environment():
    """Session-scoped test environment fixture."""
    env_manager = TestEnvironmentManager()
    workspace = env_manager.create_test_workspace()
    
    yield workspace
    
    env_manager.cleanup()


@pytest.fixture(scope="function")
def temp_directory():
    """Function-scoped temporary directory fixture."""
    temp_dir = tempfile.mkdtemp(prefix="joern_test_")
    
    yield temp_dir
    
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="function")
def mock_joern_installation(temp_directory):
    """Function-scoped mock Joern installation fixture."""
    joern_cli_path = TestFixtures.create_mock_joern_installation(temp_directory)
    
    yield joern_cli_path


@pytest.fixture(scope="function")
def sample_test_files(temp_directory):
    """Function-scoped sample test files fixture."""
    test_files = {}
    languages = ['python', 'java', 'c', 'cpp']
    
    for language in languages:
        file_path = TestFixtures.create_test_source_file(temp_directory, language)
        test_files[language] = file_path
    
    yield test_files


# Test configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add unit marker to unit tests
        if "test_discovery" in item.nodeid or "test_generation" in item.nodeid or "test_analysis" in item.nodeid:
            item.add_marker(pytest.mark.unit)
        
        # Add integration marker to integration tests
        if "test_integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Add slow marker to performance tests
        if "performance" in item.name.lower() or "benchmark" in item.name.lower():
            item.add_marker(pytest.mark.slow)