# Requirements Document

## Introduction

This feature verifies Joern's capability to generate Code Property Graphs (CPGs) for multiple programming languages. The system will test CPG generation across various languages, document supported languages, identify limitations, and provide alternative solutions where needed.

## Glossary

- **Joern**: A platform for analyzing source code using Code Property Graphs
- **CPG**: Code Property Graph - a data structure that represents source code for analysis
- **c2cpg.bat**: Joern's command-line tool for converting source code to CPG format
- **Multi-Language Support**: The ability to process and generate CPGs for different programming languages
- **Language Verification System**: The testing framework that validates CPG generation capabilities
- **Alternative Tool**: Backup solution when Joern doesn't support a specific language

## Requirements

### Requirement 1

**User Story:** As a security researcher, I want to verify which programming languages Joern supports for CPG generation, so that I can plan my code analysis workflow across different codebases.

#### Acceptance Criteria

1. THE Language Verification System SHALL identify all programming languages supported by Joern through documentation analysis
2. THE Language Verification System SHALL extract supported language information from joern/README and documentation files
3. THE Language Verification System SHALL determine which languages c2cpg.bat can process
4. THE Language Verification System SHALL document the complete list of supported languages with their file extensions

### Requirement 2

**User Story:** As a developer, I want to test CPG generation for each supported language, so that I can confirm actual functionality matches documented capabilities.

#### Acceptance Criteria

1. THE Language Verification System SHALL create test source files for each target language (C, C++, C#, Python, Java, PHP)
2. WHEN a test file is created, THE Language Verification System SHALL include representative code constructs for that language
3. THE Language Verification System SHALL attempt CPG generation for each test file using c2cpg.bat
4. THE Language Verification System SHALL capture and log the success or failure status for each language test
5. THE Language Verification System SHALL record any error messages or warnings during CPG generation

### Requirement 3

**User Story:** As a code analysis engineer, I want to identify which languages work immediately and which require additional configuration, so that I can set up my analysis environment correctly.

#### Acceptance Criteria

1. THE Language Verification System SHALL categorize each tested language as "immediately supported", "requires configuration", or "not supported"
2. THE Language Verification System SHALL document any language-specific configuration requirements
3. THE Language Verification System SHALL identify any missing dependencies or tools for each language
4. THE Language Verification System SHALL record performance characteristics and resource usage for successful CPG generations

### Requirement 4

**User Story:** As a multi-language project maintainer, I want to know alternative tools for unsupported languages, so that I can maintain consistent code analysis across my entire codebase.

#### Acceptance Criteria

1. WHEN a language is not supported by Joern, THE Language Verification System SHALL identify alternative CPG generation tools
2. THE Language Verification System SHALL document tree-sitter as an alternative for Python if Joern doesn't support it
3. THE Language Verification System SHALL provide installation and usage instructions for alternative tools
4. THE Language Verification System SHALL compare capabilities between Joern and alternative tools for each language

### Requirement 5

**User Story:** As a project lead, I want a comprehensive report of language support status, so that I can make informed decisions about tooling and workflow design.

#### Acceptance Criteria

1. THE Language Verification System SHALL generate a comprehensive report containing all test results
2. THE Language Verification System SHALL include success rates, error analysis, and performance metrics in the report
3. THE Language Verification System SHALL provide recommendations for each language based on test results
4. THE Language Verification System SHALL document any language-specific issues or limitations discovered during testing
5. THE Language Verification System SHALL include next steps and action items for unsupported or problematic languages