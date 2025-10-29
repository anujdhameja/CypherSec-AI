"""
Example usage of the Joern verification reporting system.

This module demonstrates how to use the reporting components to generate
comprehensive verification reports in multiple formats.
"""

import logging
from pathlib import Path
from datetime import datetime

from ..core.interfaces import AnalysisReport, GenerationResult
from ..analysis.metrics_collector import MetricsCollector
from .report_generator import ReportGenerator
from .models import ReportConfiguration


def create_sample_analysis_reports() -> list[AnalysisReport]:
    """Create sample analysis reports for demonstration."""
    
    # Sample successful analysis
    successful_result = GenerationResult(
        language="java",
        input_file="test_sample.java",
        output_dir="output/java",
        success=True,
        execution_time=12.5,
        memory_usage=256 * 1024 * 1024,  # 256 MB
        stdout="CPG generation completed successfully",
        stderr="",
        return_code=0,
        output_files=[Path("output/java/cpg.bin")],
        warnings=["Deprecated API usage detected"],
        error_message=None
    )
    
    successful_analysis = AnalysisReport(
        language="java",
        category="success_with_warnings",
        metrics={
            'execution_time': 12.5,
            'memory_usage': 256 * 1024 * 1024,
            'output_file_count': 1,
            'output_total_size': 1024 * 1024,  # 1 MB
            'warnings_count': 1,
            'errors_count': 0,
            'success_rate': 1.0
        },
        errors=[],
        warnings=["Deprecated API usage detected"],
        recommendations=[
            "CPG generation successful - ready for analysis",
            "Review warnings for potential issues",
            "Consider testing with larger/more complex files"
        ]
    )
    
    # Sample failed analysis
    failed_result = GenerationResult(
        language="python",
        input_file="test_sample.py",
        output_dir="output/python",
        success=False,
        execution_time=5.2,
        memory_usage=None,
        stdout="",
        stderr="Error: pysrc2cpg.bat not found",
        return_code=1,
        output_files=[],
        warnings=[],
        error_message="Tool not found"
    )
    
    failed_analysis = AnalysisReport(
        language="python",
        category="tool_missing",
        metrics={
            'execution_time': 5.2,
            'memory_usage': None,
            'output_file_count': 0,
            'output_total_size': 0,
            'warnings_count': 0,
            'errors_count': 1,
            'success_rate': 0.0
        },
        errors=["Error: pysrc2cpg.bat not found"],
        warnings=[],
        recommendations=[
            "Install or configure the python frontend tool for Joern",
            "Verify Joern installation and PATH configuration",
            "Consider alternative CPG generation tools for python"
        ]
    )
    
    # Sample partial success analysis
    partial_result = GenerationResult(
        language="cpp",
        input_file="test_sample.cpp",
        output_dir="output/cpp",
        success=True,
        execution_time=45.8,
        memory_usage=512 * 1024 * 1024,  # 512 MB
        stdout="CPG generation completed with warnings",
        stderr="Warning: Some constructs not fully supported",
        return_code=0,
        output_files=[Path("output/cpp/cpg.bin")],
        warnings=["Some constructs not fully supported", "Performance warning"],
        error_message=None
    )
    
    partial_analysis = AnalysisReport(
        language="cpp",
        category="partial_success",
        metrics={
            'execution_time': 45.8,
            'memory_usage': 512 * 1024 * 1024,
            'output_file_count': 1,
            'output_total_size': 2 * 1024 * 1024,  # 2 MB
            'warnings_count': 2,
            'errors_count': 0,
            'success_rate': 0.5
        },
        errors=[],
        warnings=["Some constructs not fully supported", "Performance warning"],
        recommendations=[
            "Review warnings to understand limitations",
            "Validate generated CPG content",
            "Consider if partial results meet requirements"
        ]
    )
    
    return [successful_analysis, failed_analysis, partial_analysis]


def generate_sample_reports():
    """Generate sample reports in all formats."""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Generating sample verification reports...")
    
    # Create sample data
    analyses = create_sample_analysis_reports()
    
    # Initialize metrics collector and add sample data
    metrics_collector = MetricsCollector()
    
    # Simulate collecting metrics for each analysis
    for analysis in analyses:
        # Create a mock GenerationResult for metrics collection
        result = GenerationResult(
            language=analysis.language,
            input_file=f"test_{analysis.language}.ext",
            output_dir=f"output/{analysis.language}",
            success=analysis.category in ['success', 'success_with_warnings', 'partial_success'],
            execution_time=analysis.metrics['execution_time'],
            memory_usage=analysis.metrics.get('memory_usage'),
            stdout="",
            stderr="",
            return_code=0 if analysis.category != 'tool_missing' else 1,
            output_files=[Path(f"output/{analysis.language}/cpg.bin")] if analysis.metrics['output_file_count'] > 0 else [],
            warnings=analysis.warnings,
            error_message=analysis.errors[0] if analysis.errors else None
        )
        
        metrics_collector.collect_benchmark_data(result)
    
    # Initialize report generator
    report_generator = ReportGenerator(metrics_collector)
    
    # Configure report generation
    config = ReportConfiguration(
        include_detailed_results=True,
        include_performance_analysis=True,
        include_error_analysis=True,
        include_recommendations=True,
        include_charts=False,  # Charts require additional libraries
        max_error_details=10,
        max_warning_details=5,
        output_formats=['json', 'markdown', 'html']
    )
    
    # Generate reports
    output_dir = Path("verification_reports")
    output_dir.mkdir(exist_ok=True)
    
    base_path = output_dir / "sample_verification_report"
    
    success = report_generator.generate_report(analyses, base_path, config)
    
    if success:
        logger.info("Sample reports generated successfully!")
        logger.info(f"Reports saved to: {output_dir}")
        logger.info("Generated files:")
        logger.info(f"  - {base_path}.json")
        logger.info(f"  - {base_path}.md")
        logger.info(f"  - {base_path}.html")
        
        # Generate summary
        summary = report_generator.format_summary(analyses)
        print("\n" + "="*60)
        print("VERIFICATION SUMMARY")
        print("="*60)
        print(summary)
        
        # Generate dashboard data
        dashboard = report_generator.generate_summary_dashboard(analyses)
        logger.info(f"Dashboard data: {dashboard['key_metrics']}")
        
    else:
        logger.error("Failed to generate sample reports")
    
    return success


def demonstrate_custom_configuration():
    """Demonstrate custom report configuration options."""
    
    logger = logging.getLogger(__name__)
    logger.info("Demonstrating custom report configurations...")
    
    analyses = create_sample_analysis_reports()
    report_generator = ReportGenerator()
    
    # Executive summary only configuration
    executive_config = ReportConfiguration(
        include_detailed_results=False,
        include_performance_analysis=False,
        include_error_analysis=False,
        include_recommendations=True,
        output_formats=['markdown']
    )
    
    # Technical details configuration
    technical_config = ReportConfiguration(
        include_detailed_results=True,
        include_performance_analysis=True,
        include_error_analysis=True,
        include_recommendations=True,
        max_error_details=50,
        max_warning_details=30,
        output_formats=['json', 'html']
    )
    
    output_dir = Path("custom_reports")
    output_dir.mkdir(exist_ok=True)
    
    # Generate executive report
    executive_path = output_dir / "executive_summary"
    report_generator.generate_report(analyses, executive_path, executive_config)
    logger.info(f"Executive summary generated: {executive_path}.md")
    
    # Generate technical report
    technical_path = output_dir / "technical_details"
    report_generator.generate_report(analyses, technical_path, technical_config)
    logger.info(f"Technical report generated: {technical_path}.json, {technical_path}.html")


if __name__ == "__main__":
    # Generate sample reports
    generate_sample_reports()
    
    # Demonstrate custom configurations
    demonstrate_custom_configuration()
    
    print("\nExample usage completed! Check the generated report files.")