# Usage Examples

This document provides comprehensive examples for using the Joern Multi-Language Verification System in various scenarios.

## Table of Contents

- [Basic Usage Examples](#basic-usage-examples)
- [Configuration Examples](#configuration-examples)
- [Advanced Usage Scenarios](#advanced-usage-scenarios)
- [Integration Examples](#integration-examples)
- [Troubleshooting Examples](#troubleshooting-examples)
- [API Usage Examples](#api-usage-examples)
- [Custom Workflow Examples](#custom-workflow-examples)

## Basic Usage Examples

### Example 1: First-Time Setup and Verification

```bash
# Step 1: Validate your setup
python -m joern_verification.main --validate

# Expected output:
# ✓ Joern CLI found at: joern/joern-cli
# ✓ Configuration file is valid
# ✓ Output directory is writable
# ✓ All required dependencies are available
# Setup validation completed successfully

# Step 2: Discover available languages
python -m joern_verification.main --discover

# Expected output:
# Scanning Joern installation...
# Found language tools:
# ✓ c2cpg.bat (C/C++)
# ✓ csharpsrc2cpg.bat (C#)
# ✓ javasrc2cpg.bat (Java)
# ✓ jssrc2cpg.bat (JavaScript)
# ✓ kotlin2cpg.bat (Kotlin)
# ✓ php2cpg.bat (PHP)
# ✓ pysrc2cpg.bat (Python)
# ✓ rubysrc2cpg.bat (Ruby)
# ✓ swiftsrc2cpg.bat (Swift)
# ✓ gosrc2cpg.bat (Go)
# Total: 10 languages discovered

# Step 3: Run basic verification
python -m joern_verification.main

# Expected output:
# [1/7] Language Discovery
# [2/7] Test File Generation
# [3/7] CPG Generation
# [4/7] Results Analysis
# [5/7] Alternative Tool Recommendations
# [6/7] Report Generation
# [7/7] Cleanup
# Verification completed successfully
# Report saved to: verification_output/verification_report.json
```

### Example 2: Testing Specific Languages

```bash
# Test only Python and Java
python -m joern_verification.main --languages python java

# Test with verbose output
python -m joern_verification.main --languages python java --verbose

# Expected verbose output:
# [INFO] Starting verification for languages: python, java
# [INFO] Generating test file for Python...
# [DEBUG] Created test file: temp_test_files/test_sample.py
# [INFO] Executing CPG generation for Python...
# [DEBUG] Command: pysrc2cpg.bat -J-Xmx4g temp_test_files/test_sample.py --output verification_output/python
# [INFO] Python CPG generation completed in 2.34s
# [INFO] Generating test file for Java...
# [DEBUG] Created test file: temp_test_files/test_sample.java
# [INFO] Executing CPG generation for Java...
# [DEBUG] Command: javasrc2cpg.bat -J-Xmx4g temp_test_files/test_sample.java --output verification_output/java
# [INFO] Java CPG generation completed in 3.12s
# [INFO] Analysis complete. 2/2 languages successful.
```

### Example 3: Quick Status Check

```bash
# List all configured languages
python -m joern_verification.main --list-languages

# Expected output:
# Configured Languages:
# 1. C (c2cpg.bat) - Available
# 2. C++ (c2cpg.bat) - Available  
# 3. C# (csharpsrc2cpg.bat) - Available
# 4. Java (javasrc2cpg.bat) - Available
# 5. JavaScript (jssrc2cpg.bat) - Available
# 6. Kotlin (kotlin2cpg.bat) - Available
# 7. PHP (php2cpg.bat) - Available
# 8. Python (pysrc2cpg.bat) - Available
# 9. Ruby (rubysrc2cpg.bat) - Available
# 10. Swift (swiftsrc2cpg.bat) - Available
# 11. Go (gosrc2cpg.bat) - Available
# Total: 11 languages configured, 11 available
```

## Configuration Examples

### Example 4: Custom Configuration File

Create a custom configuration for specific needs:

```bash
# Export current configuration as template
python -m joern_verification.main --export-config my_config.json
```

Edit `my_config.json`:

```json
{
  "system": {
    "joern_path": "/opt/joern/joern-cli",
    "output_base_dir": "custom_verification_output",
    "temp_dir": "custom_temp_files",
    "max_concurrent_tests": 2,
    "cleanup_temp_files": false,
    "verbose_logging": true
  },
  "languages": {
    "python": {
      "name": "Python",
      "file_extension": ".py",
      "tool_name": "pysrc2cpg.bat",
      "command_template": "{tool_path} {memory_allocation} {input_file} --output {output_dir}",
      "memory_allocation": "-J-Xmx8g",
      "timeout_seconds": 600,
      "alternative_tools": ["tree-sitter", "ast", "libcst"]
    },
    "java": {
      "name": "Java",
      "file_extension": ".java",
      "tool_name": "javasrc2cpg.bat",
      "command_template": "{tool_path} {memory_allocation} {input_file} --output {output_dir}",
      "memory_allocation": "-J-Xmx8g",
      "timeout_seconds": 900,
      "alternative_tools": ["eclipse-jdt", "spoon", "javaparser"]
    }
  }
}
```

Use the custom configuration:

```bash
python -m joern_verification.main --config my_config.json
```

### Example 5: Environment-Specific Configurations

**Development Environment** (`dev_config.json`):
```json
{
  "system": {
    "joern_path": "joern/joern-cli",
    "output_base_dir": "dev_verification",
    "cleanup_temp_files": false,
    "verbose_logging": true
  },
  "languages": {
    "python": {
      "timeout_seconds": 60,
      "memory_allocation": "-J-Xmx2g"
    }
  }
}
```

**Production Environment** (`prod_config.json`):
```json
{
  "system": {
    "joern_path": "/opt/joern/joern-cli",
    "output_base_dir": "/var/log/joern_verification",
    "cleanup_temp_files": true,
    "verbose_logging": false,
    "max_concurrent_tests": 4
  },
  "languages": {
    "python": {
      "timeout_seconds": 300,
      "memory_allocation": "-J-Xmx8g"
    }
  }
}
```

**CI/CD Environment** (`ci_config.json`):
```json
{
  "system": {
    "joern_path": "joern/joern-cli",
    "output_base_dir": "ci_verification",
    "cleanup_temp_files": true,
    "verbose_logging": false
  },
  "languages": {
    "python": {
      "timeout_seconds": 180,
      "memory_allocation": "-J-Xmx4g"
    },
    "java": {
      "timeout_seconds": 240,
      "memory_allocation": "-J-Xmx4g"
    }
  }
}
```

## Advanced Usage Scenarios

### Example 6: Performance Optimization

```bash
# High-performance verification with parallel processing
python -m joern_verification.main \
  --parallel \
  --memory 16g \
  --timeout 1200 \
  --languages python java c cpp \
  --report-format json

# Memory-constrained environment
python -m joern_verification.main \
  --memory 2g \
  --timeout 180 \
  --languages python \
  --summary-only

# Quick verification for CI/CD
python -m joern_verification.main \
  --languages python java \
  --timeout 120 \
  --continue-on-error \
  --quiet \
  --report-format json
```

### Example 7: Debugging and Development

```bash
# Debug mode with maximum verbosity
python -m joern_verification.main \
  --languages python \
  --verbose --verbose --verbose \
  --skip-cleanup \
  --log-file debug.log

# Dry run to test configuration
python -m joern_verification.main \
  --dry-run \
  --verbose \
  --languages python java

# Test specific component
python -m joern_verification.main \
  --discover \
  --verbose
```

### Example 8: Custom Report Generation

```bash
# Generate multiple report formats
python -m joern_verification.main \
  --report-format json markdown html csv \
  --output-dir custom_reports

# Summary report only
python -m joern_verification.main \
  --summary-only \
  --report-format markdown

# No report generation (for testing)
python -m joern_verification.main \
  --no-report \
  --verbose
```

## Integration Examples

### Example 9: CI/CD Pipeline Integration

**GitHub Actions** (`.github/workflows/joern-verification.yml`):
```yaml
name: Joern Multi-Language Verification

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  verify-joern:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Download Joern
      run: |
        wget https://github.com/joernio/joern/releases/latest/download/joern-cli.zip
        unzip joern-cli.zip
    
    - name: Run Joern Verification
      run: |
        python -m joern_verification.main \
          --config ci_config.json \
          --continue-on-error \
          --report-format json \
          --quiet
    
    - name: Upload verification report
      uses: actions/upload-artifact@v3
      with:
        name: joern-verification-report
        path: verification_output/verification_report.json
    
    - name: Check verification results
      run: |
        python -c "
        import json
        with open('verification_output/verification_report.json') as f:
            report = json.load(f)
        success_rate = report['summary']['success_rate']
        if success_rate < 80:
            exit(1)
        print(f'Verification passed with {success_rate}% success rate')
        "
```

**Jenkins Pipeline** (`Jenkinsfile`):
```groovy
pipeline {
    agent any
    
    stages {
        stage('Setup') {
            steps {
                sh 'pip install -r requirements.txt'
                sh 'wget -q https://github.com/joernio/joern/releases/latest/download/joern-cli.zip'
                sh 'unzip -q joern-cli.zip'
            }
        }
        
        stage('Verify Joern') {
            steps {
                sh '''
                python -m joern_verification.main \
                  --config ci_config.json \
                  --continue-on-error \
                  --report-format json \
                  --log-file verification.log
                '''
            }
            post {
                always {
                    archiveArtifacts artifacts: 'verification_output/*, verification.log'
                    publishHTML([
                        allowMissing: false,
                        alwaysLinkToLastBuild: true,
                        keepAll: true,
                        reportDir: 'verification_output',
                        reportFiles: 'verification_report.html',
                        reportName: 'Joern Verification Report'
                    ])
                }
            }
        }
    }
}
```

### Example 10: Docker Integration

**Dockerfile**:
```dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    openjdk-11-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Download and setup Joern
RUN wget https://github.com/joernio/joern/releases/latest/download/joern-cli.zip \
    && unzip joern-cli.zip \
    && rm joern-cli.zip

# Copy verification system
COPY joern_verification/ ./joern_verification/
COPY joern_verification_config.json .

# Create output directory
RUN mkdir -p verification_output

# Set entrypoint
ENTRYPOINT ["python", "-m", "joern_verification.main"]
```

**Docker Compose** (`docker-compose.yml`):
```yaml
version: '3.8'

services:
  joern-verification:
    build: .
    volumes:
      - ./verification_output:/app/verification_output
      - ./custom_config.json:/app/joern_verification_config.json
    command: ["--languages", "python", "java", "--report-format", "json", "html"]
    environment:
      - PYTHONUNBUFFERED=1
```

Run with Docker:
```bash
# Build and run
docker-compose up --build

# Run specific languages
docker run --rm -v $(pwd)/verification_output:/app/verification_output \
  joern-verification --languages python java --verbose
```

### Example 11: Makefile Integration

**Makefile**:
```makefile
.PHONY: verify-joern verify-quick verify-all setup clean

# Default verification
verify-joern:
	python -m joern_verification.main --languages python java

# Quick verification for development
verify-quick:
	python -m joern_verification.main \
		--languages python \
		--timeout 60 \
		--summary-only

# Comprehensive verification
verify-all:
	python -m joern_verification.main \
		--report-format json markdown html \
		--verbose

# Setup verification environment
setup:
	pip install -r requirements.txt
	@if [ ! -d "joern-cli" ]; then \
		echo "Downloading Joern..."; \
		wget https://github.com/joernio/joern/releases/latest/download/joern-cli.zip; \
		unzip joern-cli.zip; \
		rm joern-cli.zip; \
	fi
	python -m joern_verification.main --validate

# Clean up verification outputs
clean:
	rm -rf verification_output/
	rm -rf temp_test_files/
	rm -f *.log

# Validate setup
validate:
	python -m joern_verification.main --validate --verbose

# Generate configuration template
config:
	python -m joern_verification.main --export-config template_config.json
	@echo "Configuration template saved to template_config.json"
```

Usage:
```bash
make setup          # Initial setup
make verify-quick   # Quick development check
make verify-joern   # Standard verification
make verify-all     # Comprehensive verification
make clean          # Clean up outputs
```

## Troubleshooting Examples

### Example 12: Common Error Scenarios

**Scenario 1: Memory Issues**
```bash
# Problem: OutOfMemoryError
python -m joern_verification.main --languages java

# Error output:
# Error: java.lang.OutOfMemoryError: Java heap space
# CPG generation failed for Java

# Solution: Increase memory allocation
python -m joern_verification.main \
  --languages java \
  --memory 8g \
  --verbose

# Alternative: Process one language at a time
python -m joern_verification.main --languages java --memory 4g
python -m joern_verification.main --languages python --memory 4g
```

**Scenario 2: Permission Issues**
```bash
# Problem: Permission denied
python -m joern_verification.main

# Error output:
# PermissionError: [Errno 13] Permission denied: 'joern-cli/javasrc2cpg.bat'

# Solution: Fix permissions
chmod +x joern-cli/*.bat
python -m joern_verification.main --validate

# Alternative: Use different Joern path
python -m joern_verification.main --joern-path /opt/joern/joern-cli
```

**Scenario 3: Configuration Issues**
```bash
# Problem: Invalid configuration
python -m joern_verification.main --config broken_config.json

# Error output:
# Error: Invalid configuration file format
# JSONDecodeError: Expecting ',' delimiter: line 5 column 4

# Solution: Validate and fix configuration
python -m json.tool broken_config.json  # Check JSON syntax
python -m joern_verification.main --export-config fixed_config.json
python -m joern_verification.main --config fixed_config.json --validate
```

### Example 13: Debug Workflow

```bash
# Step 1: Basic validation
python -m joern_verification.main --validate --verbose

# Step 2: Test language discovery
python -m joern_verification.main --discover --verbose

# Step 3: Dry run with single language
python -m joern_verification.main \
  --languages python \
  --dry-run \
  --verbose

# Step 4: Actual run with debug output
python -m joern_verification.main \
  --languages python \
  --verbose \
  --skip-cleanup \
  --log-file debug.log

# Step 5: Analyze debug output
grep -i error debug.log
grep -i exception debug.log
tail -50 debug.log
```

## API Usage Examples

### Example 14: Programmatic Usage

**Basic API Usage**:
```python
from joern_verification.main import JoernVerificationSystem
from pathlib import Path

# Initialize system
system = JoernVerificationSystem()

# Validate setup
if not system.validate_setup():
    print("Setup validation failed")
    exit(1)

# Run verification for specific languages
success = system.run_verification(['python', 'java'])
print(f"Verification {'succeeded' if success else 'failed'}")

# Get results
results = system.get_verification_results()
for result in results:
    print(f"{result.language}: {result.category}")
```

**Advanced API Usage**:
```python
from joern_verification.main import JoernVerificationSystem
from joern_verification.config.configuration_manager import ConfigurationManager
from joern_verification.discovery.discovery import LanguageDiscoveryManager
from joern_verification.generation.test_file_generator import TestFileGenerator
from joern_verification.core.cpg_generator import CPGGenerationEngine
from joern_verification.analysis.results_analyzer import ResultsAnalyzer
from joern_verification.reporting.report_generator import ReportGenerator

# Custom configuration
config_manager = ConfigurationManager("custom_config.json")
system_config = config_manager.get_system_config()

# Language discovery
discovery = LanguageDiscoveryManager(system_config.joern_path)
available_languages = discovery.discover_languages()
print(f"Found {len(available_languages.supported_languages)} languages")

# Test file generation
generator = TestFileGenerator(config_manager)
for language in ['python', 'java']:
    success, file_path = generator.generate_test_file(language)
    print(f"Generated test file for {language}: {file_path}")

# CPG generation
cpg_engine = CPGGenerationEngine(config_manager)
for language in ['python', 'java']:
    input_file = Path(f"temp_test_files/test_sample.{config_manager.get_language_config(language).file_extension}")
    output_dir = Path(f"verification_output/{language}")
    
    result = cpg_engine.generate_cpg(language, input_file, output_dir)
    print(f"{language} CPG generation: {'Success' if result.success else 'Failed'}")

# Results analysis
analyzer = ResultsAnalyzer()
analysis_results = []
for result in cpg_results:
    analysis = analyzer.analyze_result(result)
    analysis_results.append(analysis)
    print(f"{analysis.language}: {analysis.category}")

# Report generation
report_generator = ReportGenerator(config_manager)
report_generator.generate_reports(analysis_results, ['json', 'markdown'])
```

**Custom Workflow Example**:
```python
import asyncio
from joern_verification.main import JoernVerificationSystem

class CustomVerificationWorkflow:
    def __init__(self, config_file=None):
        self.system = JoernVerificationSystem(config_file)
        self.results = []
    
    async def run_parallel_verification(self, languages):
        """Run verification for multiple languages in parallel"""
        tasks = []
        for language in languages:
            task = asyncio.create_task(self.verify_language(language))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results
    
    async def verify_language(self, language):
        """Verify a single language asynchronously"""
        # This would need to be implemented with async support
        # For now, this is a conceptual example
        return self.system.run_verification([language])
    
    def generate_custom_report(self):
        """Generate a custom report format"""
        # Custom report logic here
        pass

# Usage
async def main():
    workflow = CustomVerificationWorkflow("custom_config.json")
    results = await workflow.run_parallel_verification(['python', 'java', 'c'])
    workflow.generate_custom_report()

# Run async workflow
asyncio.run(main())
```

## Custom Workflow Examples

### Example 15: Language-Specific Workflows

**Python-Focused Workflow**:
```python
from joern_verification.main import JoernVerificationSystem
from joern_verification.alternatives.tree_sitter_integration import TreeSitterAnalyzer

def python_analysis_workflow():
    """Comprehensive Python analysis workflow"""
    system = JoernVerificationSystem()
    
    # Test Joern Python support
    joern_success = system.run_verification(['python'])
    
    if not joern_success:
        print("Joern Python support failed, trying alternatives...")
        
        # Try tree-sitter as alternative
        tree_sitter = TreeSitterAnalyzer()
        tree_sitter_success = tree_sitter.analyze_python_file("test_sample.py")
        
        if tree_sitter_success:
            print("Tree-sitter analysis successful")
            return True
    
    return joern_success

# Run Python workflow
success = python_analysis_workflow()
print(f"Python analysis workflow: {'Success' if success else 'Failed'}")
```

**Multi-Language Comparison Workflow**:
```python
def language_comparison_workflow():
    """Compare CPG generation across languages"""
    system = JoernVerificationSystem()
    
    languages = ['python', 'java', 'c', 'javascript']
    results = {}
    
    for language in languages:
        print(f"Testing {language}...")
        success = system.run_verification([language])
        
        if success:
            # Analyze CPG quality metrics
            metrics = system.get_language_metrics(language)
            results[language] = {
                'success': True,
                'execution_time': metrics.execution_time,
                'memory_usage': metrics.memory_usage,
                'cpg_nodes': metrics.node_count,
                'cpg_edges': metrics.edge_count
            }
        else:
            results[language] = {'success': False}
    
    # Generate comparison report
    generate_comparison_report(results)
    return results

def generate_comparison_report(results):
    """Generate a comparison report"""
    print("\n=== Language Comparison Report ===")
    for language, data in results.items():
        if data['success']:
            print(f"{language}: ✓ ({data['execution_time']:.2f}s, {data['cpg_nodes']} nodes)")
        else:
            print(f"{language}: ✗ Failed")
```

### Example 16: Continuous Monitoring Workflow

**Scheduled Verification**:
```python
import schedule
import time
from datetime import datetime
from joern_verification.main import JoernVerificationSystem

def scheduled_verification():
    """Run verification on a schedule"""
    system = JoernVerificationSystem()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Running scheduled verification at {timestamp}")
    
    # Run verification
    success = system.run_verification()
    
    # Log results
    with open(f"verification_log_{timestamp}.txt", "w") as f:
        f.write(f"Verification at {timestamp}: {'Success' if success else 'Failed'}\n")
        
        # Add detailed results
        results = system.get_verification_results()
        for result in results:
            f.write(f"{result.language}: {result.category}\n")
    
    return success

# Schedule verification
schedule.every().day.at("09:00").do(scheduled_verification)
schedule.every().week.do(scheduled_verification)

# Run scheduler
while True:
    schedule.run_pending()
    time.sleep(60)
```

**Health Check Workflow**:
```bash
#!/bin/bash
# health_check.sh - System health check script

echo "=== Joern Verification Health Check ==="
echo "Timestamp: $(date)"

# Check Joern installation
echo "Checking Joern installation..."
if python -m joern_verification.main --validate --quiet; then
    echo "✓ Joern installation is healthy"
else
    echo "✗ Joern installation has issues"
    exit 1
fi

# Quick verification test
echo "Running quick verification test..."
if python -m joern_verification.main --languages python --timeout 60 --quiet; then
    echo "✓ Quick verification passed"
else
    echo "✗ Quick verification failed"
    exit 1
fi

# Check disk space
echo "Checking disk space..."
DISK_USAGE=$(df -h . | awk 'NR==2 {print $5}' | sed 's/%//')
if [ "$DISK_USAGE" -gt 90 ]; then
    echo "✗ Disk usage is high: ${DISK_USAGE}%"
    exit 1
else
    echo "✓ Disk usage is acceptable: ${DISK_USAGE}%"
fi

echo "=== Health check completed successfully ==="
```

These examples demonstrate the comprehensive capabilities of the Joern Multi-Language Verification System across various use cases, from basic verification to advanced integration scenarios. Each example includes expected outputs and practical solutions for common challenges.