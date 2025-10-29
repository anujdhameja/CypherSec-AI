# Example Configurations and Templates

This directory contains example configurations, templates, and sample outputs for the Joern Multi-Language Verification System.

## Directory Structure

```
examples/
├── README.md                           # This file
├── configurations/                     # Example configuration files
│   ├── basic_config.json              # Basic configuration template
│   ├── development_config.json        # Development environment setup
│   ├── production_config.json         # Production environment setup
│   ├── ci_cd_config.json              # CI/CD pipeline configuration
│   ├── minimal_config.json            # Minimal configuration for testing
│   └── comprehensive_config.json      # Full-featured configuration
├── test_files/                        # Sample test files for each language
│   ├── sample_c.c                     # C language test file
│   ├── sample_cpp.cpp                 # C++ language test file
│   ├── sample_csharp.cs               # C# language test file
│   ├── sample_java.java               # Java language test file
│   ├── sample_javascript.js           # JavaScript language test file
│   ├── sample_kotlin.kt               # Kotlin language test file
│   ├── sample_php.php                 # PHP language test file
│   ├── sample_python.py               # Python language test file
│   ├── sample_ruby.rb                 # Ruby language test file
│   ├── sample_swift.swift             # Swift language test file
│   └── sample_go.go                   # Go language test file
├── reports/                           # Sample output reports
│   ├── sample_verification_report.json    # JSON format report
│   ├── sample_verification_report.md      # Markdown format report
│   ├── sample_verification_report.html    # HTML format report
│   └── sample_verification_report.csv     # CSV format report
├── workflows/                         # Integration workflow examples
│   ├── github_actions.yml            # GitHub Actions workflow
│   ├── jenkins_pipeline.groovy       # Jenkins pipeline script
│   ├── docker_compose.yml            # Docker Compose setup
│   └── makefile_integration.mk       # Makefile integration
└── scripts/                          # Utility scripts
    ├── setup_environment.sh          # Environment setup script
    ├── run_verification.sh           # Verification runner script
    └── generate_config.py            # Configuration generator script
```

## Usage

### Configuration Files

Each configuration file in the `configurations/` directory is designed for specific use cases:

- **basic_config.json**: Start here for first-time setup
- **development_config.json**: Optimized for development workflows
- **production_config.json**: Production-ready settings with security considerations
- **ci_cd_config.json**: Streamlined for continuous integration pipelines
- **minimal_config.json**: Bare minimum configuration for testing
- **comprehensive_config.json**: All available options with documentation

### Test Files

The `test_files/` directory contains representative code samples for each supported language. These files demonstrate:

- Common language constructs and patterns
- Syntactically correct code that should parse successfully
- Realistic code complexity for testing CPG generation

### Sample Reports

The `reports/` directory shows example outputs in different formats, helping you understand:

- Report structure and content
- Available metrics and analysis
- How to interpret verification results

### Integration Examples

The `workflows/` directory provides ready-to-use integration examples for:

- **GitHub Actions**: Automated verification in GitHub repositories
- **Jenkins**: Enterprise CI/CD pipeline integration
- **Docker**: Containerized verification environments
- **Make**: Build system integration

### Utility Scripts

The `scripts/` directory contains helpful utilities for:

- **Environment Setup**: Automated installation and configuration
- **Verification Execution**: Simplified command-line interfaces
- **Configuration Generation**: Dynamic configuration creation

## Quick Start

1. **Choose a configuration template** from `configurations/`
2. **Copy and customize** the configuration for your environment
3. **Review test files** in `test_files/` to understand expected inputs
4. **Examine sample reports** in `reports/` to understand outputs
5. **Use integration examples** from `workflows/` for your CI/CD setup

## Customization

All examples are designed to be customized for your specific needs:

- Modify paths and settings in configuration files
- Adapt test files for your specific code patterns
- Customize integration workflows for your environment
- Extend utility scripts for additional functionality

For detailed usage instructions, see the main [README.md](../README.md) and [EXAMPLES.md](../EXAMPLES.md) files.