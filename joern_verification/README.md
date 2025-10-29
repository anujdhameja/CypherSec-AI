# Joern Multi-Language Verification System

A comprehensive system for testing and documenting Joern's CPG (Code Property Graph) generation capabilities across multiple programming languages.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Configuration](#configuration)
- [Supported Languages](#supported-languages)
- [Output and Reports](#output-and-reports)
- [Troubleshooting](#troubleshooting)
- [Examples](#examples)
- [API Reference](#api-reference)
- [Contributing](#contributing)

## Overview

The Joern Multi-Language Verification System systematically tests Joern's ability to generate Code Property Graphs for different programming languages. It provides:

- **Automated Testing**: Tests CPG generation across multiple languages
- **Comprehensive Reporting**: Detailed analysis of success rates, errors, and performance
- **Alternative Recommendations**: Suggests alternative tools for unsupported languages
- **Flexible Configuration**: Customizable settings for different verification scenarios

## Features

### Core Capabilities

- **Language Discovery**: Automatically detects available Joern language frontends
- **Test File Generation**: Creates representative source code samples for each language
- **CPG Generation**: Executes language-specific CPG generation with proper error handling
- **Results Analysis**: Categorizes outcomes and analyzes performance metrics
- **Alternative Tools**: Recommends backup solutions for unsupported languages
- **Multiple Report Formats**: Generates reports in JSON, Markdown, HTML, and CSV formats

### Advanced Features

- **Parallel Processing**: Optional concurrent execution for faster verification
- **Dry Run Mode**: Test configuration without actual CPG generation
- **Custom Timeouts**: Configurable execution limits per language
- **Memory Management**: JVM memory allocation control
- **Progress Tracking**: Real-time status updates during verification
- **Error Recovery**: Continue verification even when some languages fail

## Installation

### Prerequisites

1. **Joern Installation**: Ensure Joern CLI is installed and accessible
   ```bash
   # Download and extract Joern
   wget https://github.com/joernio/joern/releases/latest/download/joern-cli.zip
   unzip joern-cli.zip
   ```

2. **Python Requirements**: Python 3.8 or higher
   ```bash
   pip install -r requirements.txt
   ```

### Setup

1. **Clone or copy the verification system** to your project directory

2. **Configure Joern path** in `joern_verification_config.json`:
   ```json
   {
     "system": {
       "joern_path": "path/to/joern/joern-cli"
     }
   }
   ```

3. **Verify installation**:
   ```bash
   python -m joern_verification.main --validate
   ```

## Quick Start

### Basic Verification

Run verification for all supported languages:
```bash
python -m joern_verification.main
```

### Test Specific Languages

Verify only Python and Java:
```bash
python -m joern_verification.main --languages python java
```

### Validate Setup

Check configuration and Joern installation:
```bash
python -m joern_verification.main --validate
```

### List Supported Languages

See all configured languages:
```bash
python -m joern_verification.main --list-languages
```

## Usage

### Command Line Interface

The system provides a comprehensive CLI with multiple options:

```bash
python -m joern_verification.main [OPTIONS]
```

#### Configuration Options

- `--config PATH`: Use custom configuration file
- `--joern-path PATH`: Override Joern CLI installation path
- `--output-dir PATH`: Override output directory for reports
- `--temp-dir PATH`: Override temporary directory for test files

#### Execution Options

- `--languages LANG [LANG ...]`: Specific languages to verify
- `--timeout SECONDS`: Timeout for CPG generation (default: 300)
- `--memory SIZE`: Memory allocation for JVM (default: 4g)
- `--parallel`: Enable parallel processing (experimental)
- `--dry-run`: Perform dry run without actual CPG generation

#### Report Options

- `--report-format FORMAT [FORMAT ...]`: Output formats (json, markdown, html, csv)
- `--no-report`: Skip report generation
- `--summary-only`: Generate summary report only

#### Information Options

- `--validate`: Validate configuration and setup only
- `--list-languages`: List all supported languages
- `--discover`: Run language discovery only
- `--version`: Show version information

#### Logging Options

- `--verbose, -v`: Increase verbosity (-v, -vv, -vvv)
- `--quiet, -q`: Suppress all output except errors
- `--log-file PATH`: Write logs to file

#### Advanced Options

- `--skip-cleanup`: Skip cleanup of temporary files
- `--continue-on-error`: Continue verification even if some languages fail
- `--export-config PATH`: Export current configuration to file

### Python API

You can also use the system programmatically:

```python
from joern_verification.main import JoernVerificationSystem

# Initialize system
system = JoernVerificationSystem()

# Validate setup
if system.validate_setup():
    # Run verification
    success = system.run_verification(['python', 'java'])
    print(f"Verification {'succeeded' if success else 'failed'}")
```

## Configuration

### Configuration File Structure

The system uses a JSON configuration file (`joern_verification_config.json`):

```json
{
  "system": {
    "joern_path": "joern/joern-cli",
    "output_base_dir": "verification_output",
    "temp_dir": "temp_test_files",
    "max_concurrent_tests": 1,
    "cleanup_temp_files": true,
    "verbose_logging": false
  },
  "languages": {
    "python": {
      "name": "Python",
      "file_extension": ".py",
      "tool_name": "pysrc2cpg.bat",
      "command_template": "{tool_path} {memory_allocation} {input_file} --output {output_dir}",
      "memory_allocation": "-J-Xmx4g",
      "timeout_seconds": 300,
      "alternative_tools": ["tree-sitter", "ast", "libcst"]
    }
  }
}
```

### System Configuration

- `joern_path`: Path to Joern CLI installation directory
- `output_base_dir`: Directory for generated reports and results
- `temp_dir`: Directory for temporary test files
- `max_concurrent_tests`: Maximum number of concurrent language tests
- `cleanup_temp_files`: Whether to clean up temporary files after verification
- `verbose_logging`: Enable detailed logging output

### Language Configuration

Each language entry includes:

- `name`: Display name for the language
- `file_extension`: File extension for test files
- `tool_name`: Joern tool executable name
- `command_template`: Command template with placeholders
- `memory_allocation`: JVM memory settings
- `timeout_seconds`: Maximum execution time
- `alternative_tools`: List of alternative CPG/AST generation tools

### Custom Configuration

Create a custom configuration file:

```bash
# Export current configuration as template
python -m joern_verification.main --export-config my_config.json

# Edit the configuration file
# Use custom configuration
python -m joern_verification.main --config my_config.json
```

## Supported Languages

The system supports the following languages (when corresponding Joern tools are available):

| Language   | Tool              | Extension | Status      |
|------------|-------------------|-----------|-------------|
| C          | c2cpg.bat         | .c        | Supported   |
| C++        | c2cpg.bat         | .cpp      | Supported   |
| C#         | csharpsrc2cpg.bat | .cs       | Supported   |
| Java       | javasrc2cpg.bat   | .java     | Supported   |
| JavaScript | jssrc2cpg.bat     | .js       | Supported   |
| Kotlin     | kotlin2cpg.bat    | .kt       | Supported   |
| PHP        | php2cpg.bat       | .php      | Supported   |
| Python     | pysrc2cpg.bat     | .py       | Supported   |
| Ruby       | rubysrc2cpg.bat   | .rb       | Supported   |
| Swift      | swiftsrc2cpg.bat  | .swift    | Supported   |
| Go         | gosrc2cpg.bat     | .go       | Supported   |

### Language Discovery

The system automatically discovers available languages by scanning the Joern installation:

```bash
# Discover available languages
python -m joern_verification.main --discover
```

## Output and Reports

### Report Formats

The system generates reports in multiple formats:

#### JSON Report
```json
{
  "summary": {
    "total_languages": 5,
    "successful": 3,
    "failed": 2,
    "success_rate": 60.0
  },
  "results": [
    {
      "language": "Python",
      "category": "success",
      "execution_time": 2.34,
      "memory_usage": "256MB"
    }
  ]
}
```

#### Markdown Report
```markdown
# Joern Multi-Language Verification Report

## Summary
- **Total Languages**: 5
- **Successful**: 3
- **Failed**: 2
- **Success Rate**: 60.0%

## Results by Language

### Python ✓
- **Status**: Success
- **Execution Time**: 2.34s
- **Memory Usage**: 256MB
```

### Output Directory Structure

```
verification_output/
├── verification_report.json      # JSON format report
├── verification_report.md        # Markdown format report
├── verification_report.html      # HTML format report
├── verification_report.csv       # CSV format report
└── detailed_results/
    ├── python_analysis.json      # Per-language detailed results
    ├── java_analysis.json
    └── error_logs/
        └── failed_languages.log
```

### Report Contents

Each report includes:

- **Executive Summary**: Overall success rates and key metrics
- **Language Results**: Detailed results for each tested language
- **Performance Analysis**: Execution times and resource usage
- **Error Analysis**: Categorized errors and failure reasons
- **Alternative Recommendations**: Suggested tools for failed languages
- **Next Steps**: Actionable recommendations for improvement

## Troubleshooting

### Common Issues

#### 1. Joern Not Found
```
Error: Joern CLI not found at specified path
```

**Solution**:
- Verify Joern installation path in configuration
- Ensure Joern CLI is properly installed and accessible
- Check file permissions on Joern executables

```bash
# Verify Joern installation
ls -la joern/joern-cli/
# Should show various *2cpg.bat files
```

#### 2. Memory Issues
```
Error: OutOfMemoryError during CPG generation
```

**Solution**:
- Increase memory allocation in configuration or command line
- Reduce concurrent test execution
- Check available system memory

```bash
# Increase memory allocation
python -m joern_verification.main --memory 8g
```

#### 3. Timeout Errors
```
Error: CPG generation timed out after 300 seconds
```

**Solution**:
- Increase timeout value
- Simplify test files
- Check system performance

```bash
# Increase timeout
python -m joern_verification.main --timeout 600
```

#### 4. Permission Errors
```
Error: Permission denied when executing Joern tools
```

**Solution**:
- Ensure execute permissions on Joern tools
- Run with appropriate user privileges
- Check directory write permissions

```bash
# Fix permissions (Linux/Mac)
chmod +x joern/joern-cli/*.bat

# Check write permissions
ls -la verification_output/
```

#### 5. Configuration Errors
```
Error: Invalid configuration file format
```

**Solution**:
- Validate JSON syntax
- Check required configuration fields
- Use configuration validation

```bash
# Validate configuration
python -m joern_verification.main --validate
```

### Debug Mode

Enable verbose logging for troubleshooting:

```bash
# Enable debug logging
python -m joern_verification.main --verbose --log-file debug.log

# Maximum verbosity
python -m joern_verification.main -vvv
```

### Dry Run Testing

Test configuration without actual CPG generation:

```bash
# Dry run mode
python -m joern_verification.main --dry-run
```

### Preserve Debug Files

Keep temporary files for inspection:

```bash
# Skip cleanup for debugging
python -m joern_verification.main --skip-cleanup
```

## Examples

### Example 1: Basic Verification

```bash
# Run basic verification for all languages
python -m joern_verification.main

# Expected output:
# [1/7] Language Discovery
# Found 11 supported languages: C, C++, C#, Java, JavaScript, Kotlin, PHP, Python, Ruby, Swift, Go
# [2/7] Test File Generation
# ✓ Generated test file for Python
# ✓ Generated test file for Java
# ...
```

### Example 2: Custom Configuration

```bash
# Create custom configuration
python -m joern_verification.main --export-config custom.json

# Edit custom.json to modify settings
# {
#   "system": {
#     "joern_path": "/opt/joern/joern-cli",
#     "timeout_seconds": 600
#   }
# }

# Use custom configuration
python -m joern_verification.main --config custom.json
```

### Example 3: Specific Languages with Custom Settings

```bash
# Test Python and Java with increased memory and timeout
python -m joern_verification.main \
  --languages python java \
  --memory 8g \
  --timeout 600 \
  --report-format json markdown
```

### Example 4: Continuous Integration

```bash
# CI-friendly verification with error handling
python -m joern_verification.main \
  --continue-on-error \
  --summary-only \
  --quiet \
  --report-format json

# Check exit code
if [ $? -eq 0 ]; then
  echo "Verification passed"
else
  echo "Verification failed"
fi
```

### Example 5: Development Workflow

```bash
# Quick validation during development
python -m joern_verification.main --validate

# Discover available languages
python -m joern_verification.main --discover

# Test specific language with debug output
python -m joern_verification.main \
  --languages python \
  --verbose \
  --skip-cleanup \
  --log-file python_debug.log
```

## API Reference

### Main Classes

#### JoernVerificationSystem

Main orchestrator class for the verification system.

```python
class JoernVerificationSystem:
    def __init__(self, config_file: Optional[Path] = None)
    def validate_setup(self) -> bool
    def list_supported_languages(self) -> List[str]
    def run_verification(self, languages: Optional[List[str]] = None) -> bool
```

#### ConfigurationManager

Manages system and language configurations.

```python
class ConfigurationManager:
    def __init__(self, config_file: Optional[Path] = None)
    def get_system_config(self) -> SystemConfiguration
    def get_language_config(self, language: str) -> LanguageConfiguration
    def validate_configuration(self) -> List[str]
```

### Core Interfaces

#### LanguageDiscoveryInterface

```python
class LanguageDiscoveryInterface:
    def discover_languages(self) -> DiscoveryResults
    def get_supported_languages(self) -> List[str]
    def validate_tool_availability(self, language: str) -> bool
```

#### TestFileGeneratorInterface

```python
class TestFileGeneratorInterface:
    def generate_test_file(self, language: str) -> Tuple[bool, str]
    def get_test_content(self, language: str) -> str
    def cleanup_test_files(self) -> None
```

#### CPGGenerationInterface

```python
class CPGGenerationInterface:
    def generate_cpg(self, language: str, input_file: Path, 
                    output_dir: Path) -> GenerationResult
    def validate_output(self, output_dir: Path) -> bool
```

### Data Models

#### GenerationResult

```python
@dataclass
class GenerationResult:
    language: str
    input_file: Path
    output_dir: Path
    success: bool
    execution_time: float
    memory_usage: Optional[int]
    stdout: str
    stderr: str
    return_code: int
    output_files: List[str]
    warnings: List[str]
```

#### AnalysisReport

```python
@dataclass
class AnalysisReport:
    language: str
    category: str  # success, success_with_warnings, partial_success, failure
    metrics: Dict[str, Any]
    errors: List[str]
    warnings: List[str]
    recommendations: List[str]
    alternative_tools: List[str]
```

## Contributing

### Development Setup

1. **Clone the repository**
2. **Install development dependencies**:
   ```bash
   pip install -r requirements-dev.txt
   ```
3. **Run tests**:
   ```bash
   python -m pytest joern_verification/tests/
   ```

### Code Style

- Follow PEP 8 guidelines
- Use type hints for all functions
- Add docstrings for all public methods
- Maintain test coverage above 80%

### Adding New Languages

1. **Add language configuration** to `joern_verification_config.json`
2. **Create test template** in `generation/test_templates.py`
3. **Add alternative tools** to `alternatives/database.py`
4. **Update documentation**

### Submitting Changes

1. **Create feature branch**
2. **Add tests for new functionality**
3. **Update documentation**
4. **Submit pull request**

---

For more information, see the [project documentation](https://github.com/your-repo/joern-verification) or [open an issue](https://github.com/your-repo/joern-verification/issues).