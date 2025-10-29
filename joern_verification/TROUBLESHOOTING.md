# Troubleshooting Guide

This guide provides solutions for common issues encountered when using the Joern Multi-Language Verification System.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Configuration Problems](#configuration-problems)
- [Execution Errors](#execution-errors)
- [Performance Issues](#performance-issues)
- [Report Generation Problems](#report-generation-problems)
- [Language-Specific Issues](#language-specific-issues)
- [Debug Techniques](#debug-techniques)
- [Getting Help](#getting-help)

## Installation Issues

### Issue: Joern CLI Not Found

**Symptoms**:
```
Error: Joern CLI not found at specified path: joern/joern-cli
FileNotFoundError: [Errno 2] No such file or directory
```

**Causes**:
- Joern CLI not installed
- Incorrect path in configuration
- Missing executable permissions

**Solutions**:

1. **Verify Joern Installation**:
   ```bash
   # Check if Joern directory exists
   ls -la joern/joern-cli/
   
   # Should show files like:
   # c2cpg.bat, javasrc2cpg.bat, pysrc2cpg.bat, etc.
   ```

2. **Download and Install Joern**:
   ```bash
   # Download latest Joern release
   wget https://github.com/joernio/joern/releases/latest/download/joern-cli.zip
   unzip joern-cli.zip
   
   # Verify installation
   ls joern-cli/
   ```

3. **Fix Configuration Path**:
   ```json
   {
     "system": {
       "joern_path": "/absolute/path/to/joern-cli"
     }
   }
   ```

4. **Set Execute Permissions** (Linux/Mac):
   ```bash
   chmod +x joern-cli/*.bat
   chmod +x joern-cli/*.sh
   ```

### Issue: Python Dependencies Missing

**Symptoms**:
```
ModuleNotFoundError: No module named 'some_module'
ImportError: cannot import name 'SomeClass'
```

**Solutions**:

1. **Install Required Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Check Python Version**:
   ```bash
   python --version  # Should be 3.8 or higher
   ```

3. **Use Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   pip install -r requirements.txt
   ```

## Configuration Problems

### Issue: Invalid Configuration File

**Symptoms**:
```
Error: Invalid configuration file format
JSONDecodeError: Expecting ',' delimiter
```

**Solutions**:

1. **Validate JSON Syntax**:
   ```bash
   # Use online JSON validator or:
   python -m json.tool joern_verification_config.json
   ```

2. **Check Required Fields**:
   ```json
   {
     "system": {
       "joern_path": "required",
       "output_base_dir": "required",
       "temp_dir": "required"
     },
     "languages": {
       "python": {
         "name": "required",
         "file_extension": "required",
         "tool_name": "required"
       }
     }
   }
   ```

3. **Use Configuration Validation**:
   ```bash
   python -m joern_verification.main --validate
   ```

4. **Export Default Configuration**:
   ```bash
   python -m joern_verification.main --export-config default.json
   ```

### Issue: Path Resolution Problems

**Symptoms**:
```
Error: Cannot resolve path: ~/joern/joern-cli
Error: Permission denied: /opt/joern
```

**Solutions**:

1. **Use Absolute Paths**:
   ```json
   {
     "system": {
       "joern_path": "/home/user/joern/joern-cli"
     }
   }
   ```

2. **Check Directory Permissions**:
   ```bash
   ls -la /path/to/joern/
   # Ensure read and execute permissions
   ```

3. **Override via Command Line**:
   ```bash
   python -m joern_verification.main --joern-path /correct/path/to/joern-cli
   ```

## Execution Errors

### Issue: Memory Errors

**Symptoms**:
```
java.lang.OutOfMemoryError: Java heap space
Error: Insufficient memory for CPG generation
```

**Solutions**:

1. **Increase Memory Allocation**:
   ```bash
   # Command line
   python -m joern_verification.main --memory 8g
   
   # Configuration file
   "memory_allocation": "-J-Xmx8g"
   ```

2. **Check Available System Memory**:
   ```bash
   # Linux/Mac
   free -h
   
   # Windows
   wmic OS get TotalVisibleMemorySize /value
   ```

3. **Reduce Concurrent Tests**:
   ```json
   {
     "system": {
       "max_concurrent_tests": 1
     }
   }
   ```

4. **Process Languages Individually**:
   ```bash
   python -m joern_verification.main --languages python
   python -m joern_verification.main --languages java
   ```

### Issue: Timeout Errors

**Symptoms**:
```
Error: CPG generation timed out after 300 seconds
TimeoutExpired: Command timed out
```

**Solutions**:

1. **Increase Timeout**:
   ```bash
   python -m joern_verification.main --timeout 600
   ```

2. **Check System Performance**:
   ```bash
   # Monitor CPU and memory usage
   top
   htop
   ```

3. **Simplify Test Files**:
   - Reduce complexity of generated test code
   - Use smaller input files
   - Remove complex language constructs

4. **Use Dry Run for Testing**:
   ```bash
   python -m joern_verification.main --dry-run
   ```

### Issue: Permission Errors

**Symptoms**:
```
PermissionError: [Errno 13] Permission denied
Error: Cannot write to output directory
```

**Solutions**:

1. **Check Directory Permissions**:
   ```bash
   ls -la verification_output/
   mkdir -p verification_output
   chmod 755 verification_output
   ```

2. **Run with Appropriate Privileges**:
   ```bash
   # If necessary (not recommended for regular use)
   sudo python -m joern_verification.main
   ```

3. **Change Output Directory**:
   ```bash
   python -m joern_verification.main --output-dir /tmp/verification
   ```

4. **Fix Joern Tool Permissions**:
   ```bash
   chmod +x joern-cli/*.bat
   ```

## Performance Issues

### Issue: Slow Execution

**Symptoms**:
- Verification takes much longer than expected
- High CPU or memory usage
- System becomes unresponsive

**Solutions**:

1. **Enable Parallel Processing** (if safe):
   ```bash
   python -m joern_verification.main --parallel
   ```

2. **Reduce Language Set**:
   ```bash
   python -m joern_verification.main --languages python java c
   ```

3. **Use Summary Reports Only**:
   ```bash
   python -m joern_verification.main --summary-only
   ```

4. **Monitor Resource Usage**:
   ```bash
   # Monitor during execution
   python -m joern_verification.main --verbose &
   top -p $!
   ```

### Issue: Disk Space Problems

**Symptoms**:
```
OSError: [Errno 28] No space left on device
Error: Cannot create temporary files
```

**Solutions**:

1. **Check Available Disk Space**:
   ```bash
   df -h
   ```

2. **Clean Up Previous Runs**:
   ```bash
   rm -rf verification_output/
   rm -rf temp_test_files/
   ```

3. **Use Different Temporary Directory**:
   ```bash
   python -m joern_verification.main --temp-dir /tmp/joern_temp
   ```

4. **Enable Cleanup**:
   ```json
   {
     "system": {
       "cleanup_temp_files": true
     }
   }
   ```

## Report Generation Problems

### Issue: Report Generation Fails

**Symptoms**:
```
Error: Failed to generate report
FileNotFoundError: Report template not found
```

**Solutions**:

1. **Check Output Directory Permissions**:
   ```bash
   mkdir -p verification_output
   chmod 755 verification_output
   ```

2. **Use Different Report Format**:
   ```bash
   python -m joern_verification.main --report-format json
   ```

3. **Skip Report Generation**:
   ```bash
   python -m joern_verification.main --no-report
   ```

4. **Generate Reports Manually**:
   ```python
   from joern_verification.reporting.report_generator import ReportGenerator
   generator = ReportGenerator()
   # Use generator to create reports from existing data
   ```

### Issue: Corrupted Report Files

**Symptoms**:
- Empty report files
- Malformed JSON/HTML
- Missing report sections

**Solutions**:

1. **Regenerate Reports**:
   ```bash
   rm verification_output/verification_report.*
   python -m joern_verification.main
   ```

2. **Check File Permissions**:
   ```bash
   ls -la verification_output/
   ```

3. **Use Single Format**:
   ```bash
   python -m joern_verification.main --report-format json
   ```

## Language-Specific Issues

### Issue: Python CPG Generation Fails

**Common Errors**:
```
Error: pysrc2cpg.bat not found
SyntaxError in generated Python test file
```

**Solutions**:

1. **Verify Python Tool Availability**:
   ```bash
   ls joern-cli/pysrc2cpg.bat
   ```

2. **Check Python Test File Syntax**:
   ```bash
   python -m py_compile temp_test_files/test_sample.py
   ```

3. **Use Alternative Tools**:
   ```bash
   # Configure tree-sitter as alternative
   pip install tree-sitter tree-sitter-python
   ```

### Issue: Java CPG Generation Fails

**Common Errors**:
```
Error: Could not find or load main class
ClassNotFoundException
```

**Solutions**:

1. **Check Java Installation**:
   ```bash
   java -version
   javac -version
   ```

2. **Verify Classpath**:
   ```bash
   echo $CLASSPATH
   ```

3. **Use Correct Java Version**:
   - Ensure Java 8 or higher is installed
   - Check Joern Java compatibility

### Issue: C/C++ CPG Generation Fails

**Common Errors**:
```
Error: Compiler not found
fatal error: 'stdio.h' file not found
```

**Solutions**:

1. **Install Build Tools**:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install build-essential
   
   # CentOS/RHEL
   sudo yum groupinstall "Development Tools"
   
   # macOS
   xcode-select --install
   ```

2. **Check Compiler Availability**:
   ```bash
   gcc --version
   clang --version
   ```

## Debug Techniques

### Enable Verbose Logging

```bash
# Basic verbose mode
python -m joern_verification.main --verbose

# Maximum verbosity with log file
python -m joern_verification.main -vvv --log-file debug.log
```

### Use Dry Run Mode

```bash
# Test configuration without actual execution
python -m joern_verification.main --dry-run --verbose
```

### Preserve Debug Files

```bash
# Keep temporary files for inspection
python -m joern_verification.main --skip-cleanup --verbose
```

### Manual Component Testing

```python
# Test individual components
from joern_verification.discovery.discovery import LanguageDiscoveryManager

# Test language discovery
discovery = LanguageDiscoveryManager("joern/joern-cli")
results = discovery.discover_languages()
print(results)
```

### Check System State

```bash
# Validate entire system
python -m joern_verification.main --validate --verbose

# Discover available languages
python -m joern_verification.main --discover

# List configured languages
python -m joern_verification.main --list-languages
```

### Analyze Log Files

```bash
# Search for specific errors
grep -i "error" debug.log
grep -i "exception" debug.log
grep -i "failed" debug.log

# Check execution timeline
grep "\\[.*\\]" debug.log
```

## Getting Help

### Self-Diagnosis Checklist

Before seeking help, verify:

1. **✓ Joern Installation**: `ls joern-cli/`
2. **✓ Python Version**: `python --version` (3.8+)
3. **✓ Dependencies**: `pip list | grep -E "(pathlib|typing|dataclasses)"`
4. **✓ Permissions**: `ls -la joern-cli/` (executable files)
5. **✓ Configuration**: `python -m joern_verification.main --validate`
6. **✓ Disk Space**: `df -h`
7. **✓ Memory**: `free -h` (Linux) or check Task Manager (Windows)

### Collect Debug Information

When reporting issues, include:

1. **System Information**:
   ```bash
   python --version
   uname -a  # Linux/Mac
   systeminfo  # Windows
   ```

2. **Configuration**:
   ```bash
   python -m joern_verification.main --export-config debug_config.json
   ```

3. **Error Logs**:
   ```bash
   python -m joern_verification.main --verbose --log-file error.log
   ```

4. **Joern Installation**:
   ```bash
   ls -la joern-cli/
   ```

### Common Support Resources

1. **Joern Documentation**: https://docs.joern.io/
2. **Joern GitHub Issues**: https://github.com/joernio/joern/issues
3. **Python Documentation**: https://docs.python.org/
4. **Stack Overflow**: Search for "joern cpg" or "code property graph"

### Creating Minimal Reproduction

When reporting bugs:

1. **Isolate the Issue**:
   ```bash
   # Test single language
   python -m joern_verification.main --languages python --verbose
   ```

2. **Use Dry Run**:
   ```bash
   python -m joern_verification.main --dry-run --verbose
   ```

3. **Provide Minimal Configuration**:
   ```json
   {
     "system": {
       "joern_path": "joern/joern-cli"
     },
     "languages": {
       "python": {
         "name": "Python",
         "file_extension": ".py",
         "tool_name": "pysrc2cpg.bat"
       }
     }
   }
   ```

### Emergency Recovery

If the system is completely broken:

1. **Reset Configuration**:
   ```bash
   rm joern_verification_config.json
   python -m joern_verification.main --export-config joern_verification_config.json
   ```

2. **Clean All Outputs**:
   ```bash
   rm -rf verification_output/
   rm -rf temp_test_files/
   ```

3. **Reinstall Dependencies**:
   ```bash
   pip uninstall -y -r requirements.txt
   pip install -r requirements.txt
   ```

4. **Verify Basic Functionality**:
   ```bash
   python -c "import joern_verification; print('Import successful')"
   ```

---

If none of these solutions work, please create an issue with:
- Complete error messages
- System information
- Configuration file
- Steps to reproduce
- Debug logs (if available)