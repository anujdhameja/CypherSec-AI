# Alternative Tool Recommendation System

This module provides intelligent recommendations for alternative CPG/AST generation tools when Joern doesn't support a language or fails to process code.

## Overview

The alternative tool recommendation system consists of:

- **Database**: Comprehensive database of alternative tools with their capabilities, installation instructions, and usage examples
- **Recommender**: Intelligent recommendation engine that suggests tools based on context and requirements
- **Integration**: Utilities to integrate recommendations with the main verification workflow

## Quick Start

```python
from joern_verification.alternatives import (
    AlternativeToolDatabase, 
    AlternativeToolRecommender, 
    RecommendationContext,
    ToolCapability
)

# Initialize the system
database = AlternativeToolDatabase()
recommender = AlternativeToolRecommender(database)

# Create recommendation context
context = RecommendationContext(
    language="python",
    joern_support_status="failed",
    required_capabilities=[ToolCapability.AST_GENERATION],
    integration_complexity_preference="low"
)

# Get recommendations
recommendations = recommender.recommend_alternatives(context)

# Display top recommendation
if recommendations:
    top_rec = recommendations[0]
    print(f"Recommended: {top_rec.tool.name}")
    print(f"Confidence: {top_rec.confidence_score:.2f}")
    print(f"Explanation: {top_rec.explanation}")
```

## Available Alternative Tools

### Tree-sitter
- **Languages**: Python, JavaScript, Java, C, C++, C#, Go, Rust, Ruby, PHP, Swift, Kotlin, TypeScript
- **Capabilities**: AST Generation, Syntax Parsing
- **Complexity**: Medium
- **Best for**: Multi-language projects, fast incremental parsing

### Python AST
- **Languages**: Python only
- **Capabilities**: AST Generation, Syntax Parsing
- **Complexity**: Low
- **Best for**: Python-only projects, no installation required

### ANTLR
- **Languages**: Java, Python, JavaScript, C, C++, C#, Go, PHP, Swift, Kotlin, Ruby
- **Capabilities**: AST Generation, Syntax Parsing
- **Complexity**: High
- **Best for**: Custom grammar requirements, advanced parsing needs

## Integration with Verification Results

```python
from joern_verification.alternatives import AlternativeToolIntegrator
from joern_verification.core.interfaces import GenerationResult

# Process verification results
integrator = AlternativeToolIntegrator()
recommendations_report = integrator.process_verification_results(results)

# Export recommendations
integrator.export_recommendations_to_file(
    recommendations_report, 
    Path("alternative_recommendations.json")
)

# Generate markdown report
markdown_report = integrator.generate_markdown_report(recommendations_report)
```

## Recommendation Context Parameters

- **language**: Target programming language
- **joern_support_status**: "full", "partial", "limited", "none", "failed"
- **required_capabilities**: List of ToolCapability enums
- **integration_complexity_preference**: "low", "medium", "high"
- **project_type**: "research", "production", "educational", "general"
- **time_constraints**: "low", "medium", "high"
- **team_expertise**: "beginner", "medium", "expert"

## Tool Capabilities

- `AST_GENERATION`: Generate Abstract Syntax Trees
- `CPG_GENERATION`: Generate Code Property Graphs
- `SYNTAX_PARSING`: Parse source code syntax
- `SEMANTIC_ANALYSIS`: Perform semantic analysis
- `CONTROL_FLOW`: Analyze control flow
- `DATA_FLOW`: Analyze data flow
- `CALL_GRAPH`: Generate call graphs
- `DEPENDENCY_ANALYSIS`: Analyze dependencies

## Installation Methods

- `PIP`: Python package manager
- `NPM`: Node.js package manager
- `CARGO`: Rust package manager
- `GO_GET`: Go package manager
- `MANUAL`: Manual installation
- `SYSTEM_PACKAGE`: System package manager
- `BINARY_DOWNLOAD`: Direct binary download

## Example Output

```json
{
  "tool_name": "Tree-sitter",
  "confidence_score": 0.95,
  "reason": "joern_failed",
  "explanation": "Tree-sitter is recommended for python because Joern failed to process python code",
  "estimated_setup_time": "30 minutes",
  "integration_guidance": [
    "Install using: pip install tree-sitter",
    "Install python grammar: pip install tree-sitter-python"
  ],
  "pros": [
    "Supports many programming languages",
    "Fast incremental parsing",
    "Active community and development"
  ],
  "cons": [
    "Syntax-only parsing (no semantic analysis)",
    "Different output format than Joern CPG"
  ]
}
```

## Advanced Usage

### Custom Tool Addition

```python
from joern_verification.alternatives.models import AlternativeTool, ToolCapability

# Create custom tool
custom_tool = AlternativeTool(
    name="MyParser",
    description="Custom parser for specific language",
    supported_languages=["mylang"],
    capabilities=[ToolCapability.AST_GENERATION],
    # ... other properties
)

# Add to database
database.add_tool("myparser", custom_tool)
```

### Tool Comparison

```python
# Compare tool with Joern
tree_sitter = database.get_tool("tree-sitter")
comparison = recommender.compare_with_joern(tree_sitter, "python")
print(comparison["feature_comparison"])
```

### Migration Guidance

```python
# Get migration guide
migration_guide = recommender.generate_migration_guide(
    "joern", tree_sitter, "python"
)
print(migration_guide["migration_steps"])
```

## Requirements Addressed

This implementation addresses the following requirements from the specification:

- **4.1**: Identifies alternative CPG generation tools for unsupported languages
- **4.2**: Documents tree-sitter and other alternatives with installation instructions
- **4.3**: Provides installation and usage instructions for each alternative
- **4.4**: Compares capabilities between Joern and alternative tools

## Files Structure

```
joern_verification/alternatives/
├── __init__.py              # Module exports
├── models.py                # Data models and enums
├── database.py              # Alternative tools database
├── recommender.py           # Recommendation engine
├── integration.py           # Integration utilities
├── example_usage.py         # Usage examples
└── README.md               # This documentation
```