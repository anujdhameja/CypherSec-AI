# Installation Guide

This guide provides step-by-step instructions for installing and setting up the Joern Multi-Language Verification System.

## Table of Contents

- [System Requirements](#system-requirements)
- [Prerequisites](#prerequisites)
- [Installation Steps](#installation-steps)
- [Configuration](#configuration)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements

- **Operating System**: Windows 10, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Python**: 3.8 or higher
- **Memory**: 4GB RAM (8GB recommended)
- **Disk Space**: 2GB free space
- **Java**: JRE 8 or higher (for Joern)

### Recommended Requirements

- **Memory**: 16GB RAM
- **Disk Space**: 10GB free space
- **CPU**: Multi-core processor for parallel processing
- **Java**: JRE 11 or higher

## Prerequisites

### 1. Python Installation

Ensure Python 3.8 or higher is installed:

```bash
# Check Python version
python --version
# Should output: Python 3.8.x or higher
```

### 2. Java Installation

Joern requires Java Runtime Environment (JRE) 8 or higher:

```bash
# Check Java version
java -version
# Should output: java version "1.8.x" or higher
```

## Installation Steps

### Step 1: Download Joern CLI

```bash
# Create project directory
mkdir joern-verification-project
cd joern-verification-project

# Download Joern CLI
wget https://github.com/joernio/joern/releases/latest/download/joern-cli.zip
unzip joern-cli.zip

# Verify extraction
ls joern-cli/
```

### Step 2: Set Execute Permissions (Linux/macOS)

```bash
chmod +x joern-cli/*.bat
chmod +x joern-cli/*.sh
```

### Step 3: Install Python Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install required packages
pip install pathlib typing dataclasses json argparse logging datetime
```

### Step 4: Create Configuration File

Create `joern_verification_config.json`:

```json
{
  "system": {
    "joern_path": "joern-cli",
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

## Verification

### Validate Installation

```bash
# Validate setup
python -m joern_verification.main --validate

# Discover available languages
python -m joern_verification.main --discover

# Run quick test
python -m joern_verification.main --languages python --timeout 60
```

## Troubleshooting

### Common Issues

1. **Joern Not Found**: Update `joern_path` in configuration
2. **Java Not Found**: Install Java JRE 8+
3. **Memory Issues**: Reduce `memory_allocation` to `-J-Xmx2g`
4. **Permission Errors**: Run `chmod +x joern-cli/*.bat`

For detailed troubleshooting, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).