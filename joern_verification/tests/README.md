# Joern Verification Tests

This directory contains comprehensive tests for the Joern multi-language verification system.

## Test Structure

- **Unit Tests**: Test individual components in isolation
  - `test_discovery.py` - Language discovery functionality
  - `test_generation.py` - Test file generation and CPG generation
  - `test_analysis.py` - Result analysis and metrics collection

- **Integration Tests**: Test complete workflows
  - `test_integration.py` - End-to-end verification workflows
  - Performance benchmarks and regression tests

- **Test Utilities**:
  - `test_fixtures.py` - Shared test fixtures and mock objects
  - `test_runner.py` - Custom test runner with reporting
  - `conftest.py` - pytest configuration (if using pytest)

## Running Tests

### Using unittest (built-in)

```bash
# Run all tests
python -m unittest discover joern_verification/tests -v

# Run specific test module
python -m unittest joern_verification.tests.test_discovery -v

# Run specific test class
python -m unittest joern_verification.tests.test_discovery.TestLanguageDiscoveryManager -v

# Run specific test method
python -m unittest joern_verification.tests.test_discovery.TestLanguageDiscoveryManager.test_initialization -v
```

### Using custom test runner

```bash
# Run all tests with summary
python joern_verification/tests/test_runner.py

# Run only unit tests
python joern_verification/tests/test_runner.py --unit

# Run only integration tests
python joern_verification/tests/test_runner.py --integration

# Run specific test
python joern_verification/tests/test_runner.py --test joern_verification.tests.test_discovery.TestLanguageDiscoveryManager.test_initialization

# Verbose output
python joern_verification/tests/test_runner.py --verbose
```

### Using pytest (if available)

```bash
# Run all tests
pytest joern_verification/tests/

# Run only unit tests
pytest joern_verification/tests/ -m unit

# Run only integration tests
pytest joern_verification/tests/ -m integration

# Run with coverage
pytest joern_verification/tests/ --cov=joern_verification
```

## Test Categories

### Unit Tests
- Test individual components in isolation
- Use mocks and fixtures to avoid external dependencies
- Fast execution (< 1 second per test)
- Focus on core functionality and edge cases

### Integration Tests
- Test complete workflows end-to-end
- Use real file system operations (with cleanup)
- May take longer to execute
- Test component interactions and data flow

### Performance Tests
- Benchmark execution time and memory usage
- Test scalability with multiple languages
- Validate resource usage patterns

### Regression Tests
- Ensure consistent behavior across runs
- Test error handling consistency
- Validate that fixes don't break existing functionality

## Test Data

Tests use temporary directories and mock data to avoid affecting the real system:

- Mock Joern installations with fake tool files
- Generated test source files for each language
- Simulated command execution results
- Temporary output directories (automatically cleaned up)

## Mocking Strategy

- **Command Execution**: Mock subprocess calls to avoid running real Joern tools
- **File System**: Use temporary directories that are automatically cleaned up
- **External Dependencies**: Mock network calls and external tool dependencies
- **Time-sensitive Operations**: Mock time.sleep() and timing functions for faster tests

## Test Coverage

The test suite covers:

- ✅ Language discovery and tool scanning
- ✅ Test file generation for all supported languages
- ✅ CPG generation command building and execution
- ✅ Result analysis and categorization
- ✅ Metrics collection and aggregation
- ✅ Report generation and formatting
- ✅ Error handling and edge cases
- ✅ End-to-end verification workflows
- ✅ Performance benchmarking
- ✅ Regression testing

## Adding New Tests

When adding new tests:

1. **Unit Tests**: Add to appropriate `test_*.py` file or create new one
2. **Integration Tests**: Add to `test_integration.py`
3. **Fixtures**: Add reusable test data to `test_fixtures.py`
4. **Mocks**: Use existing mock classes or create new ones in fixtures

### Test Naming Convention

- Test files: `test_<component>.py`
- Test classes: `Test<ComponentName>`
- Test methods: `test_<functionality>_<scenario>`

### Example Test Structure

```python
class TestNewComponent(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        pass
    
    def tearDown(self):
        """Clean up test resources."""
        pass
    
    def test_functionality_success(self):
        """Test successful functionality."""
        pass
    
    def test_functionality_failure(self):
        """Test failure scenarios."""
        pass
    
    def test_functionality_edge_cases(self):
        """Test edge cases and boundary conditions."""
        pass
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're running from the project root directory
2. **Path Issues**: Tests use temporary directories to avoid path conflicts
3. **Mock Failures**: Check that mocks match the actual interface signatures
4. **Cleanup Issues**: Use context managers or tearDown methods for cleanup

### Debug Mode

Run tests with verbose output to see detailed information:

```bash
python -m unittest joern_verification.tests.test_discovery -v
```

### Test Isolation

Each test should be independent and not rely on state from other tests. Use setUp/tearDown methods to ensure clean state.