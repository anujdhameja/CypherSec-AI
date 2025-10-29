# Joern Verification Reporting System

This module provides comprehensive report generation capabilities for the Joern multi-language verification system. It generates detailed reports with success rates, error analysis, performance metrics, and actionable recommendations.

## Features

- **Multiple Output Formats**: JSON, Markdown, and HTML reports
- **Interactive HTML Reports**: Collapsible sections, sortable tables, and responsive design
- **Comprehensive Analysis**: Performance metrics, error categorization, and trend analysis
- **Actionable Recommendations**: Language-specific suggestions and alternative tools
- **Customizable Configuration**: Flexible report content and formatting options
- **Dashboard Generation**: Summary dashboards with key metrics and visualizations

## Quick Start

```python
from joern_verification.reporting import ReportGenerator, ReportConfiguration
from joern_verification.analysis.metrics_collector import MetricsCollector

# Initialize components
metrics_collector = MetricsCollector()
report_generator = ReportGenerator(metrics_collector)

# Configure report generation
config = ReportConfiguration(
    include_detailed_results=True,
    include_performance_analysis=True,
    include_error_analysis=True,
    output_formats=['json', 'markdown', 'html']
)

# Generate reports
success = report_generator.generate_report(
    analyses=your_analysis_results,
    output_path=Path("verification_report"),
    config=config
)
```

## Report Components

### 1. Executive Summary
- High-level overview of verification results
- Key findings and success rates
- Critical issues requiring attention

### 2. Language Overview
- Sortable table of all tested languages
- Status indicators and performance classifications
- Error and warning counts

### 3. Performance Analysis
- Execution time rankings and statistics
- Resource usage patterns
- Performance trends and consistency metrics

### 4. Error Analysis
- Error categorization and frequency analysis
- Common error patterns
- Resolution suggestions for each error type

### 5. Recommendations
- Immediate actions for critical issues
- Performance improvement suggestions
- Alternative tools for unsupported languages
- Language-specific recommendations

### 6. Detailed Results
- Complete analysis data for each language
- Full error and warning messages
- Metrics and performance data

## Output Formats

### JSON Format
- Machine-readable structured data
- Complete analysis results and metadata
- Suitable for integration with other tools
- Preserves all data types and relationships

### Markdown Format
- Human-readable documentation format
- Tables and formatted sections
- Compatible with documentation systems
- Easy to version control and review

### HTML Format
- Interactive web-based reports
- Responsive design for mobile and desktop
- Collapsible sections and sortable tables
- Professional styling and navigation
- Embedded CSS and JavaScript for standalone viewing

## Configuration Options

```python
config = ReportConfiguration(
    include_detailed_results=True,      # Include full analysis data
    include_performance_analysis=True,  # Include performance metrics
    include_error_analysis=True,        # Include error categorization
    include_recommendations=True,       # Include actionable recommendations
    include_charts=False,              # Include data visualizations
    max_error_details=50,              # Maximum errors to show per language
    max_warning_details=20,            # Maximum warnings to show per language
    group_by_category=True,            # Group results by success category
    sort_by_performance=True,          # Sort languages by performance
    output_formats=['json', 'markdown', 'html']  # Output formats to generate
)
```

## Report Templates

### Default Template
- Complete report with all sections
- Balanced detail level
- Suitable for most use cases

### Executive Template
- High-level summary focused
- Key metrics and recommendations only
- Ideal for management reporting

### Technical Template
- Detailed technical information
- Full error analysis and troubleshooting
- Comprehensive performance data

## Dashboard Features

The reporting system can generate summary dashboards with:

- **Key Metrics Cards**: Success rates, execution times, error counts
- **Interactive Charts**: Success distribution, performance rankings
- **Sortable Tables**: Language summaries with filtering
- **Alert System**: Automatic identification of critical issues

## Error Analysis

The system provides comprehensive error analysis including:

- **Error Categorization**: Memory errors, timeout errors, syntax errors, etc.
- **Pattern Recognition**: Common error patterns across languages
- **Resolution Suggestions**: Specific recommendations for each error type
- **Critical Issue Detection**: Automatic identification of high-priority problems

## Performance Analysis

Performance analysis includes:

- **Execution Time Statistics**: Mean, median, percentiles
- **Resource Usage Tracking**: Memory consumption, output sizes
- **Performance Rankings**: Fastest to slowest languages
- **Trend Analysis**: Performance consistency and patterns
- **Comparative Analysis**: Cross-language performance comparison

## Integration Examples

### With Existing Workflow
```python
# After running verification tests
analyses = run_verification_tests()

# Collect performance metrics
for analysis in analyses:
    metrics_collector.collect_benchmark_data(analysis.generation_result)

# Generate comprehensive report
report_generator.generate_report(analyses, output_path, config)

# Generate summary for quick review
summary = report_generator.format_summary(analyses)
print(summary)
```

### Custom Report Sections
```python
# Add custom sections to reports
custom_section = ReportSection(
    title="Custom Analysis",
    content={"custom_metric": 42},
    section_type="analysis",
    priority=90
)

report.add_section(custom_section)
```

### Dashboard Integration
```python
# Generate dashboard data for web interfaces
dashboard_data = report_generator.generate_summary_dashboard(analyses)

# Use dashboard data in web applications
app.render_dashboard(dashboard_data)
```

## Best Practices

1. **Regular Reporting**: Generate reports after each verification run
2. **Configuration Management**: Use consistent report configurations
3. **Archive Reports**: Keep historical reports for trend analysis
4. **Review Recommendations**: Act on actionable recommendations
5. **Share Results**: Distribute reports to relevant stakeholders

## Troubleshooting

### Common Issues

**Report Generation Fails**
- Check file permissions for output directory
- Verify analysis data is complete and valid
- Review configuration for invalid options

**Missing Data in Reports**
- Ensure metrics collector is properly initialized
- Verify analysis results contain required fields
- Check configuration includes desired sections

**HTML Report Display Issues**
- Ensure modern web browser support
- Check for JavaScript execution restrictions
- Verify CSS rendering in target environment

### Performance Considerations

- Large datasets may require increased memory
- HTML reports with many languages may load slowly
- Consider pagination for very large result sets
- Use appropriate detail levels for report size management

## API Reference

See the individual module documentation for detailed API information:

- `ReportGenerator`: Main report generation class
- `ReportConfiguration`: Configuration options
- `JSONFormatter`, `MarkdownFormatter`, `HTMLFormatter`: Output formatters
- `VerificationReport`: Report data model

## Examples

See `example_usage.py` for complete working examples demonstrating:
- Basic report generation
- Custom configurations
- Multiple output formats
- Dashboard creation
- Error handling