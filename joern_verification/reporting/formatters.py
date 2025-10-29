"""
Output formatters for verification reports.

This module provides formatters for generating reports in multiple formats
including JSON, Markdown, and HTML with comprehensive styling and interactivity.
"""

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from .models import VerificationReport, ReportConfiguration, ChartData, TableData


class BaseFormatter(ABC):
    """Base class for report formatters."""
    
    def __init__(self):
        """Initialize the formatter."""
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def format_report(
        self, 
        report: VerificationReport, 
        output_path: Path,
        config: ReportConfiguration
    ) -> bool:
        """
        Format and save the report.
        
        Args:
            report: VerificationReport to format
            output_path: Path where to save the formatted report
            config: Report configuration
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    def _format_timestamp(self, timestamp: datetime) -> str:
        """Format timestamp for display."""
        return timestamp.strftime("%Y-%m-%d %H:%M:%S")
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 60:
            return f"{seconds:.2f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            remaining_seconds = seconds % 60
            return f"{minutes}m {remaining_seconds:.1f}s"
        else:
            hours = int(seconds // 3600)
            remaining_minutes = int((seconds % 3600) // 60)
            return f"{hours}h {remaining_minutes}m"
    
    def _format_percentage(self, value: float) -> str:
        """Format percentage value."""
        return f"{value:.1f}%"
    
    def _format_file_size(self, bytes_size: int) -> str:
        """Format file size in human-readable format."""
        if bytes_size < 1024:
            return f"{bytes_size} B"
        elif bytes_size < 1024 * 1024:
            return f"{bytes_size / 1024:.1f} KB"
        elif bytes_size < 1024 * 1024 * 1024:
            return f"{bytes_size / (1024 * 1024):.1f} MB"
        else:
            return f"{bytes_size / (1024 * 1024 * 1024):.1f} GB"


class JSONFormatter(BaseFormatter):
    """JSON formatter for verification reports."""
    
    def format_report(
        self, 
        report: VerificationReport, 
        output_path: Path,
        config: ReportConfiguration
    ) -> bool:
        """
        Format report as JSON.
        
        Args:
            report: VerificationReport to format
            output_path: Path where to save the JSON report
            config: Report configuration
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info(f"Generating JSON report at {output_path}")
            
            # Convert report to dictionary
            report_dict = self._report_to_dict(report, config)
            
            # Write JSON file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report_dict, f, indent=2, default=self._json_serializer, ensure_ascii=False)
            
            self.logger.info(f"JSON report generated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to generate JSON report: {e}")
            return False
    
    def _report_to_dict(self, report: VerificationReport, config: ReportConfiguration) -> Dict[str, Any]:
        """Convert VerificationReport to dictionary."""
        report_dict = {
            'metadata': {
                'title': report.metadata.title,
                'generated_at': report.metadata.generated_at.isoformat(),
                'joern_version': report.metadata.joern_version,
                'system_info': report.metadata.system_info,
                'total_languages': report.metadata.total_languages,
                'successful_languages': report.metadata.successful_languages,
                'failed_languages': report.metadata.failed_languages,
                'execution_time': report.metadata.execution_time,
                'report_version': report.metadata.report_version
            },
            'executive_summary': {
                'title': report.executive_summary.title,
                'content': report.executive_summary.content,
                'section_type': report.executive_summary.section_type
            },
            'language_summaries': [
                {
                    'language': summary.language,
                    'status': summary.status,
                    'execution_time': summary.execution_time,
                    'success_rate': summary.success_rate,
                    'error_count': summary.error_count,
                    'warning_count': summary.warning_count,
                    'output_files': summary.output_files,
                    'recommendations': summary.recommendations,
                    'category': summary.category,
                    'performance_class': summary.performance_class
                }
                for summary in report.language_summaries
            ],
            'performance_analysis': {
                'overall_success_rate': report.performance_analysis.overall_success_rate,
                'average_execution_time': report.performance_analysis.average_execution_time,
                'fastest_language': report.performance_analysis.fastest_language,
                'slowest_language': report.performance_analysis.slowest_language,
                'most_reliable_language': report.performance_analysis.most_reliable_language,
                'least_reliable_language': report.performance_analysis.least_reliable_language,
                'performance_rankings': report.performance_analysis.performance_rankings,
                'reliability_rankings': report.performance_analysis.reliability_rankings,
                'resource_usage': report.performance_analysis.resource_usage,
                'trends': report.performance_analysis.trends
            },
            'error_analysis': {
                'total_errors': report.error_analysis.total_errors,
                'error_categories': report.error_analysis.error_categories,
                'common_errors': report.error_analysis.common_errors,
                'languages_with_errors': report.error_analysis.languages_with_errors,
                'resolution_suggestions': report.error_analysis.resolution_suggestions,
                'critical_issues': report.error_analysis.critical_issues
            },
            'recommendations': {
                'immediate_actions': report.recommendations.immediate_actions,
                'performance_improvements': report.recommendations.performance_improvements,
                'reliability_enhancements': report.recommendations.reliability_enhancements,
                'alternative_tools': report.recommendations.alternative_tools,
                'next_steps': report.recommendations.next_steps,
                'language_specific': report.recommendations.language_specific
            }
        }
        
        # Add detailed results if configured
        if config.include_detailed_results and report.detailed_results:
            report_dict['detailed_results'] = [
                {
                    'language': analysis.language,
                    'category': analysis.category,
                    'metrics': analysis.metrics,
                    'errors': analysis.errors,
                    'warnings': analysis.warnings,
                    'recommendations': analysis.recommendations
                }
                for analysis in report.detailed_results
            ]
        
        # Add additional sections
        if report.sections:
            report_dict['additional_sections'] = [
                {
                    'title': section.title,
                    'content': section.content,
                    'section_type': section.section_type,
                    'priority': section.priority,
                    'metadata': section.metadata
                }
                for section in report.sections
            ]
        
        return report_dict
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for complex objects."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, Path):
            return str(obj)
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj)


class MarkdownFormatter(BaseFormatter):
    """Markdown formatter for verification reports."""
    
    def format_report(
        self, 
        report: VerificationReport, 
        output_path: Path,
        config: ReportConfiguration
    ) -> bool:
        """
        Format report as Markdown.
        
        Args:
            report: VerificationReport to format
            output_path: Path where to save the Markdown report
            config: Report configuration
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info(f"Generating Markdown report at {output_path}")
            
            # Generate markdown content
            markdown_content = self._generate_markdown(report, config)
            
            # Write markdown file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            self.logger.info(f"Markdown report generated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to generate Markdown report: {e}")
            return False
    
    def _generate_markdown(self, report: VerificationReport, config: ReportConfiguration) -> str:
        """Generate complete markdown content."""
        sections = []
        
        # Title and metadata
        sections.append(self._generate_header(report))
        
        # Executive summary
        sections.append(self._generate_executive_summary(report))
        
        # Language overview
        sections.append(self._generate_language_overview(report))
        
        # Performance analysis
        if config.include_performance_analysis:
            sections.append(self._generate_performance_section(report))
        
        # Error analysis
        if config.include_error_analysis:
            sections.append(self._generate_error_section(report))
        
        # Recommendations
        sections.append(self._generate_recommendations_section(report))
        
        # Detailed results
        if config.include_detailed_results:
            sections.append(self._generate_detailed_results(report, config))
        
        return '\n\n'.join(sections)
    
    def _generate_header(self, report: VerificationReport) -> str:
        """Generate report header."""
        header = f"""# {report.metadata.title}

**Generated:** {self._format_timestamp(report.metadata.generated_at)}  
**Languages Tested:** {report.metadata.total_languages}  
**Successful:** {report.metadata.successful_languages}  
**Failed:** {report.metadata.failed_languages}  
**Total Execution Time:** {self._format_duration(report.metadata.execution_time)}  
"""
        
        if report.metadata.joern_version:
            header += f"**Joern Version:** {report.metadata.joern_version}  \n"
        
        return header
    
    def _generate_executive_summary(self, report: VerificationReport) -> str:
        """Generate executive summary section."""
        content = report.executive_summary.content
        
        summary = f"""## Executive Summary

**Overview:** {content.get('overview', 'N/A')}

**Success Rate:** {content.get('success_rate', 'N/A')}

### Key Findings

"""
        
        key_findings = content.get('key_findings', [])
        for finding in key_findings:
            summary += f"- {finding}\n"
        
        return summary
    
    def _generate_language_overview(self, report: VerificationReport) -> str:
        """Generate language overview table."""
        overview = """## Language Overview

| Language | Status | Execution Time | Success Rate | Errors | Warnings | Performance |
|----------|--------|----------------|--------------|--------|----------|-------------|
"""
        
        for summary in report.language_summaries:
            status_icon = "✅" if summary.status == "success" else "❌"
            overview += f"| {summary.language} | {status_icon} {summary.category} | {self._format_duration(summary.execution_time)} | {self._format_percentage(summary.success_rate)} | {summary.error_count} | {summary.warning_count} | {summary.performance_class} |\n"
        
        return overview
    
    def _generate_performance_section(self, report: VerificationReport) -> str:
        """Generate performance analysis section."""
        perf = report.performance_analysis
        
        section = f"""## Performance Analysis

**Overall Success Rate:** {self._format_percentage(perf.overall_success_rate)}  
**Average Execution Time:** {self._format_duration(perf.average_execution_time)}  
**Fastest Language:** {perf.fastest_language or 'N/A'}  
**Slowest Language:** {perf.slowest_language or 'N/A'}  

### Performance Rankings

| Rank | Language | Execution Time |
|------|----------|----------------|
"""
        
        for i, (language, time) in enumerate(perf.performance_rankings[:10], 1):
            section += f"| {i} | {language} | {self._format_duration(time)} |\n"
        
        # Resource usage
        if perf.resource_usage:
            section += "\n### Resource Usage\n\n"
            
            if 'memory' in perf.resource_usage:
                memory = perf.resource_usage['memory']
                section += f"**Average Memory Usage:** {memory.get('average_mb', 0):.1f} MB  \n"
                section += f"**Peak Memory Usage:** {memory.get('max_mb', 0):.1f} MB  \n"
            
            if 'output' in perf.resource_usage:
                output = perf.resource_usage['output']
                section += f"**Average Output Size:** {output.get('average_mb', 0):.1f} MB  \n"
                section += f"**Total Output Size:** {output.get('total_mb', 0):.1f} MB  \n"
        
        return section
    
    def _generate_error_section(self, report: VerificationReport) -> str:
        """Generate error analysis section."""
        error_analysis = report.error_analysis
        
        section = f"""## Error Analysis

**Total Errors:** {error_analysis.total_errors}  
**Languages with Errors:** {len(error_analysis.languages_with_errors)}  

### Error Categories

"""
        
        for category, count in error_analysis.error_categories.items():
            section += f"- **{category.replace('_', ' ').title()}:** {count}\n"
        
        # Common errors
        if error_analysis.common_errors:
            section += "\n### Most Common Errors\n\n"
            for i, (error, count) in enumerate(error_analysis.common_errors[:5], 1):
                section += f"{i}. `{error}` ({count} occurrences)\n"
        
        # Critical issues
        if error_analysis.critical_issues:
            section += "\n### Critical Issues\n\n"
            for issue in error_analysis.critical_issues:
                section += f"- ⚠️ {issue}\n"
        
        return section
    
    def _generate_recommendations_section(self, report: VerificationReport) -> str:
        """Generate recommendations section."""
        rec = report.recommendations
        
        section = "## Recommendations\n\n"
        
        # Immediate actions
        if rec.immediate_actions:
            section += "### 🚨 Immediate Actions\n\n"
            for action in rec.immediate_actions:
                section += f"- {action}\n"
            section += "\n"
        
        # Performance improvements
        if rec.performance_improvements:
            section += "### ⚡ Performance Improvements\n\n"
            for improvement in rec.performance_improvements:
                section += f"- {improvement}\n"
            section += "\n"
        
        # Reliability enhancements
        if rec.reliability_enhancements:
            section += "### 🔧 Reliability Enhancements\n\n"
            for enhancement in rec.reliability_enhancements:
                section += f"- {enhancement}\n"
            section += "\n"
        
        # Alternative tools
        if rec.alternative_tools:
            section += "### 🔄 Alternative Tools\n\n"
            for language, tools in rec.alternative_tools.items():
                section += f"**{language}:**\n"
                for tool in tools:
                    section += f"  - {tool}\n"
                section += "\n"
        
        # Next steps
        if rec.next_steps:
            section += "### 📋 Next Steps\n\n"
            for step in rec.next_steps:
                section += f"- {step}\n"
        
        return section
    
    def _generate_detailed_results(self, report: VerificationReport, config: ReportConfiguration) -> str:
        """Generate detailed results section."""
        section = "## Detailed Results\n\n"
        
        for analysis in report.detailed_results:
            section += f"### {analysis.language}\n\n"
            section += f"**Category:** {analysis.category}  \n"
            section += f"**Execution Time:** {self._format_duration(analysis.metrics.get('execution_time', 0))}  \n"
            section += f"**Output Files:** {analysis.metrics.get('output_file_count', 0)}  \n"
            
            if analysis.metrics.get('output_total_size'):
                section += f"**Output Size:** {self._format_file_size(analysis.metrics['output_total_size'])}  \n"
            
            # Errors
            if analysis.errors and len(analysis.errors) <= config.max_error_details:
                section += "\n**Errors:**\n"
                for error in analysis.errors:
                    section += f"- `{error}`\n"
            elif len(analysis.errors) > config.max_error_details:
                section += f"\n**Errors:** {len(analysis.errors)} errors (showing first {config.max_error_details})\n"
                for error in analysis.errors[:config.max_error_details]:
                    section += f"- `{error}`\n"
            
            # Warnings
            if analysis.warnings and len(analysis.warnings) <= config.max_warning_details:
                section += "\n**Warnings:**\n"
                for warning in analysis.warnings:
                    section += f"- `{warning}`\n"
            elif len(analysis.warnings) > config.max_warning_details:
                section += f"\n**Warnings:** {len(analysis.warnings)} warnings (showing first {config.max_warning_details})\n"
                for warning in analysis.warnings[:config.max_warning_details]:
                    section += f"- `{warning}`\n"
            
            # Recommendations
            if analysis.recommendations:
                section += "\n**Recommendations:**\n"
                for recommendation in analysis.recommendations:
                    section += f"- {recommendation}\n"
            
            section += "\n---\n\n"
        
        return section


class HTMLFormatter(BaseFormatter):
    """HTML formatter for verification reports with interactive features."""
    
    def format_report(
        self, 
        report: VerificationReport, 
        output_path: Path,
        config: ReportConfiguration
    ) -> bool:
        """
        Format report as HTML.
        
        Args:
            report: VerificationReport to format
            output_path: Path where to save the HTML report
            config: Report configuration
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info(f"Generating HTML report at {output_path}")
            
            # Generate HTML content
            html_content = self._generate_html(report, config)
            
            # Write HTML file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"HTML report generated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to generate HTML report: {e}")
            return False
    
    def _generate_html(self, report: VerificationReport, config: ReportConfiguration) -> str:
        """Generate complete HTML content."""
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{report.metadata.title}</title>
    {self._generate_css()}
    {self._generate_javascript()}
</head>
<body>
    <div class="container">
        {self._generate_header_html(report)}
        {self._generate_navigation()}
        {self._generate_executive_summary_html(report)}
        {self._generate_dashboard_html(report)}
        {self._generate_language_overview_html(report)}
        {self._generate_performance_html(report, config)}
        {self._generate_error_html(report, config)}
        {self._generate_recommendations_html(report)}
        {self._generate_detailed_results_html(report, config)}
        {self._generate_footer()}
    </div>
</body>
</html>"""
        
        return html
    
    def _generate_css(self) -> str:
        """Generate CSS styles for the HTML report."""
        return """
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .metadata {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        
        .metadata-item {
            background: rgba(255,255,255,0.1);
            padding: 10px;
            border-radius: 5px;
        }
        
        .navigation {
            background: white;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            position: sticky;
            top: 20px;
            z-index: 100;
        }
        
        .nav-links {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
        }
        
        .nav-links a {
            color: #667eea;
            text-decoration: none;
            font-weight: 500;
            padding: 5px 10px;
            border-radius: 5px;
            transition: background-color 0.3s;
        }
        
        .nav-links a:hover {
            background-color: #f0f0f0;
        }
        
        .section {
            background: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .section h2 {
            color: #667eea;
            margin-bottom: 20px;
            font-size: 1.8em;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
        }
        
        .dashboard {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 4px solid #667eea;
        }
        
        .metric-value {
            font-size: 2.5em;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 5px;
        }
        
        .metric-label {
            color: #666;
            font-size: 0.9em;
        }
        
        .status-success {
            color: #28a745;
        }
        
        .status-warning {
            color: #ffc107;
        }
        
        .status-error {
            color: #dc3545;
        }
        
        .language-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        
        .language-table th,
        .language-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }
        
        .language-table th {
            background-color: #f8f9fa;
            font-weight: 600;
            color: #495057;
        }
        
        .language-table tr:hover {
            background-color: #f8f9fa;
        }
        
        .status-badge {
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: 500;
        }
        
        .badge-success {
            background-color: #d4edda;
            color: #155724;
        }
        
        .badge-warning {
            background-color: #fff3cd;
            color: #856404;
        }
        
        .badge-error {
            background-color: #f8d7da;
            color: #721c24;
        }
        
        .recommendations {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }
        
        .recommendation-category {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        
        .recommendation-category h3 {
            margin-bottom: 15px;
            color: #495057;
        }
        
        .recommendation-category ul {
            list-style-type: none;
        }
        
        .recommendation-category li {
            padding: 5px 0;
            padding-left: 20px;
            position: relative;
        }
        
        .recommendation-category li:before {
            content: "→";
            position: absolute;
            left: 0;
            color: #667eea;
            font-weight: bold;
        }
        
        .collapsible {
            cursor: pointer;
            padding: 10px;
            background-color: #f1f1f1;
            border: none;
            text-align: left;
            outline: none;
            font-size: 1.1em;
            font-weight: 500;
            width: 100%;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        
        .collapsible:hover {
            background-color: #ddd;
        }
        
        .collapsible.active {
            background-color: #667eea;
            color: white;
        }
        
        .content {
            padding: 0 18px;
            display: none;
            overflow: hidden;
            background-color: #f9f9f9;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        
        .content.show {
            display: block;
            padding: 18px;
        }
        
        .footer {
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.9em;
        }
        
        .chart-container {
            margin: 20px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 10px;
            }
            
            .header h1 {
                font-size: 2em;
            }
            
            .metadata {
                grid-template-columns: 1fr;
            }
            
            .nav-links {
                flex-direction: column;
                gap: 10px;
            }
            
            .dashboard {
                grid-template-columns: 1fr;
            }
        }
    </style>
        """
    
    def _generate_javascript(self) -> str:
        """Generate JavaScript for interactive features."""
        return """
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            // Collapsible sections
            var coll = document.getElementsByClassName("collapsible");
            for (var i = 0; i < coll.length; i++) {
                coll[i].addEventListener("click", function() {
                    this.classList.toggle("active");
                    var content = this.nextElementSibling;
                    content.classList.toggle("show");
                });
            }
            
            // Smooth scrolling for navigation links
            document.querySelectorAll('a[href^="#"]').forEach(anchor => {
                anchor.addEventListener('click', function (e) {
                    e.preventDefault();
                    const target = document.querySelector(this.getAttribute('href'));
                    if (target) {
                        target.scrollIntoView({
                            behavior: 'smooth',
                            block: 'start'
                        });
                    }
                });
            });
            
            // Table sorting (basic implementation)
            function sortTable(table, column, asc = true) {
                const dirModifier = asc ? 1 : -1;
                const tBody = table.tBodies[0];
                const rows = Array.from(tBody.querySelectorAll("tr"));
                
                const sortedRows = rows.sort((a, b) => {
                    const aColText = a.querySelector(`td:nth-child(${column + 1})`).textContent.trim();
                    const bColText = b.querySelector(`td:nth-child(${column + 1})`).textContent.trim();
                    
                    return aColText > bColText ? (1 * dirModifier) : (-1 * dirModifier);
                });
                
                while (tBody.firstChild) {
                    tBody.removeChild(tBody.firstChild);
                }
                
                tBody.append(...sortedRows);
            }
            
            // Add click handlers to table headers
            document.querySelectorAll(".language-table th").forEach((headerCell, index) => {
                headerCell.addEventListener("click", () => {
                    const table = headerCell.closest("table");
                    const currentIsAscending = headerCell.classList.contains("asc");
                    sortTable(table, index, !currentIsAscending);
                    
                    // Update header classes
                    table.querySelectorAll("th").forEach(th => th.classList.remove("asc", "desc"));
                    headerCell.classList.toggle("asc", !currentIsAscending);
                    headerCell.classList.toggle("desc", currentIsAscending);
                });
            });
        });
    </script>
        """
    
    def _generate_header_html(self, report: VerificationReport) -> str:
        """Generate HTML header section."""
        return f"""
        <div class="header">
            <h1>{report.metadata.title}</h1>
            <div class="metadata">
                <div class="metadata-item">
                    <strong>Generated:</strong><br>
                    {self._format_timestamp(report.metadata.generated_at)}
                </div>
                <div class="metadata-item">
                    <strong>Languages Tested:</strong><br>
                    {report.metadata.total_languages}
                </div>
                <div class="metadata-item">
                    <strong>Successful:</strong><br>
                    {report.metadata.successful_languages}
                </div>
                <div class="metadata-item">
                    <strong>Failed:</strong><br>
                    {report.metadata.failed_languages}
                </div>
                <div class="metadata-item">
                    <strong>Total Time:</strong><br>
                    {self._format_duration(report.metadata.execution_time)}
                </div>
            </div>
        </div>
        """
    
    def _generate_navigation(self) -> str:
        """Generate navigation menu."""
        return """
        <div class="navigation">
            <div class="nav-links">
                <a href="#executive-summary">Executive Summary</a>
                <a href="#dashboard">Dashboard</a>
                <a href="#language-overview">Language Overview</a>
                <a href="#performance">Performance Analysis</a>
                <a href="#errors">Error Analysis</a>
                <a href="#recommendations">Recommendations</a>
                <a href="#detailed-results">Detailed Results</a>
            </div>
        </div>
        """
    
    def _generate_executive_summary_html(self, report: VerificationReport) -> str:
        """Generate executive summary HTML."""
        content = report.executive_summary.content
        
        findings_html = ""
        for finding in content.get('key_findings', []):
            findings_html += f"<li>{finding}</li>"
        
        return f"""
        <div id="executive-summary" class="section">
            <h2>Executive Summary</h2>
            <p><strong>Overview:</strong> {content.get('overview', 'N/A')}</p>
            <p><strong>Success Rate:</strong> <span class="metric-value" style="font-size: 1.2em;">{content.get('success_rate', 'N/A')}</span></p>
            
            <h3>Key Findings</h3>
            <ul>
                {findings_html}
            </ul>
        </div>
        """
    
    def _generate_dashboard_html(self, report: VerificationReport) -> str:
        """Generate dashboard with key metrics."""
        success_rate = report.performance_analysis.overall_success_rate
        avg_time = report.performance_analysis.average_execution_time
        
        return f"""
        <div id="dashboard" class="dashboard">
            <div class="metric-card">
                <div class="metric-value status-{'success' if success_rate >= 80 else 'warning' if success_rate >= 60 else 'error'}">
                    {self._format_percentage(success_rate)}
                </div>
                <div class="metric-label">Success Rate</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-value">{self._format_duration(avg_time)}</div>
                <div class="metric-label">Average Execution Time</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-value status-success">{report.metadata.successful_languages}</div>
                <div class="metric-label">Successful Languages</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-value status-error">{report.metadata.failed_languages}</div>
                <div class="metric-label">Failed Languages</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-value">{report.error_analysis.total_errors}</div>
                <div class="metric-label">Total Errors</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-value">{sum(len(s.warnings) for s in report.language_summaries)}</div>
                <div class="metric-label">Total Warnings</div>
            </div>
        </div>
        """
    
    def _generate_language_overview_html(self, report: VerificationReport) -> str:
        """Generate language overview table."""
        rows_html = ""
        
        for summary in report.language_summaries:
            status_class = "success" if summary.status == "success" else "error"
            status_icon = "✅" if summary.status == "success" else "❌"
            
            rows_html += f"""
            <tr>
                <td>{summary.language}</td>
                <td><span class="status-badge badge-{status_class}">{status_icon} {summary.category}</span></td>
                <td>{self._format_duration(summary.execution_time)}</td>
                <td>{self._format_percentage(summary.success_rate)}</td>
                <td>{summary.error_count}</td>
                <td>{summary.warning_count}</td>
                <td>{summary.output_files}</td>
                <td>{summary.performance_class}</td>
            </tr>
            """
        
        return f"""
        <div id="language-overview" class="section">
            <h2>Language Overview</h2>
            <table class="language-table">
                <thead>
                    <tr>
                        <th>Language</th>
                        <th>Status</th>
                        <th>Execution Time</th>
                        <th>Success Rate</th>
                        <th>Errors</th>
                        <th>Warnings</th>
                        <th>Output Files</th>
                        <th>Performance</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
        """
    
    def _generate_performance_html(self, report: VerificationReport, config: ReportConfiguration) -> str:
        """Generate performance analysis HTML."""
        if not config.include_performance_analysis:
            return ""
        
        perf = report.performance_analysis
        
        rankings_html = ""
        for i, (language, time) in enumerate(perf.performance_rankings[:10], 1):
            rankings_html += f"""
            <tr>
                <td>{i}</td>
                <td>{language}</td>
                <td>{self._format_duration(time)}</td>
            </tr>
            """
        
        return f"""
        <div id="performance" class="section">
            <h2>Performance Analysis</h2>
            
            <div class="dashboard">
                <div class="metric-card">
                    <div class="metric-value">{self._format_duration(perf.average_execution_time)}</div>
                    <div class="metric-label">Average Execution Time</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value status-success">{perf.fastest_language or 'N/A'}</div>
                    <div class="metric-label">Fastest Language</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value status-warning">{perf.slowest_language or 'N/A'}</div>
                    <div class="metric-label">Slowest Language</div>
                </div>
            </div>
            
            <button class="collapsible">Performance Rankings</button>
            <div class="content">
                <table class="language-table">
                    <thead>
                        <tr>
                            <th>Rank</th>
                            <th>Language</th>
                            <th>Execution Time</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rankings_html}
                    </tbody>
                </table>
            </div>
        </div>
        """
    
    def _generate_error_html(self, report: VerificationReport, config: ReportConfiguration) -> str:
        """Generate error analysis HTML."""
        if not config.include_error_analysis:
            return ""
        
        error_analysis = report.error_analysis
        
        categories_html = ""
        for category, count in error_analysis.error_categories.items():
            categories_html += f"<li><strong>{category.replace('_', ' ').title()}:</strong> {count}</li>"
        
        common_errors_html = ""
        for i, (error, count) in enumerate(error_analysis.common_errors[:5], 1):
            common_errors_html += f"<li><code>{error}</code> ({count} occurrences)</li>"
        
        critical_issues_html = ""
        for issue in error_analysis.critical_issues:
            critical_issues_html += f"<li>⚠️ {issue}</li>"
        
        return f"""
        <div id="errors" class="section">
            <h2>Error Analysis</h2>
            
            <div class="dashboard">
                <div class="metric-card">
                    <div class="metric-value status-error">{error_analysis.total_errors}</div>
                    <div class="metric-label">Total Errors</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{len(error_analysis.languages_with_errors)}</div>
                    <div class="metric-label">Languages with Errors</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value status-warning">{len(error_analysis.critical_issues)}</div>
                    <div class="metric-label">Critical Issues</div>
                </div>
            </div>
            
            <button class="collapsible">Error Categories</button>
            <div class="content">
                <ul>{categories_html}</ul>
            </div>
            
            <button class="collapsible">Most Common Errors</button>
            <div class="content">
                <ol>{common_errors_html}</ol>
            </div>
            
            <button class="collapsible">Critical Issues</button>
            <div class="content">
                <ul>{critical_issues_html}</ul>
            </div>
        </div>
        """
    
    def _generate_recommendations_html(self, report: VerificationReport) -> str:
        """Generate recommendations HTML."""
        rec = report.recommendations
        
        return f"""
        <div id="recommendations" class="section">
            <h2>Recommendations</h2>
            
            <div class="recommendations">
                {self._generate_recommendation_category("🚨 Immediate Actions", rec.immediate_actions)}
                {self._generate_recommendation_category("⚡ Performance Improvements", rec.performance_improvements)}
                {self._generate_recommendation_category("🔧 Reliability Enhancements", rec.reliability_enhancements)}
                {self._generate_recommendation_category("📋 Next Steps", rec.next_steps)}
            </div>
            
            {self._generate_alternative_tools_html(rec.alternative_tools)}
            {self._generate_language_specific_html(rec.language_specific)}
        </div>
        """
    
    def _generate_recommendation_category(self, title: str, recommendations: List[str]) -> str:
        """Generate HTML for a recommendation category."""
        if not recommendations:
            return ""
        
        items_html = ""
        for rec in recommendations:
            items_html += f"<li>{rec}</li>"
        
        return f"""
        <div class="recommendation-category">
            <h3>{title}</h3>
            <ul>{items_html}</ul>
        </div>
        """
    
    def _generate_alternative_tools_html(self, alternative_tools: Dict[str, List[str]]) -> str:
        """Generate alternative tools HTML."""
        if not alternative_tools:
            return ""
        
        tools_html = ""
        for language, tools in alternative_tools.items():
            tool_items = ""
            for tool in tools:
                tool_items += f"<li>{tool}</li>"
            
            tools_html += f"""
            <button class="collapsible">{language} - Alternative Tools</button>
            <div class="content">
                <ul>{tool_items}</ul>
            </div>
            """
        
        return f"""
        <h3>🔄 Alternative Tools</h3>
        {tools_html}
        """
    
    def _generate_language_specific_html(self, language_specific: Dict[str, List[str]]) -> str:
        """Generate language-specific recommendations HTML."""
        if not language_specific:
            return ""
        
        lang_html = ""
        for language, recommendations in language_specific.items():
            rec_items = ""
            for rec in recommendations:
                rec_items += f"<li>{rec}</li>"
            
            lang_html += f"""
            <button class="collapsible">{language} - Specific Recommendations</button>
            <div class="content">
                <ul>{rec_items}</ul>
            </div>
            """
        
        return f"""
        <h3>🎯 Language-Specific Recommendations</h3>
        {lang_html}
        """
    
    def _generate_detailed_results_html(self, report: VerificationReport, config: ReportConfiguration) -> str:
        """Generate detailed results HTML."""
        if not config.include_detailed_results or not report.detailed_results:
            return ""
        
        results_html = ""
        for analysis in report.detailed_results:
            errors_html = ""
            if analysis.errors:
                for error in analysis.errors[:config.max_error_details]:
                    errors_html += f"<li><code>{error}</code></li>"
                if len(analysis.errors) > config.max_error_details:
                    errors_html += f"<li><em>... and {len(analysis.errors) - config.max_error_details} more errors</em></li>"
            
            warnings_html = ""
            if analysis.warnings:
                for warning in analysis.warnings[:config.max_warning_details]:
                    warnings_html += f"<li><code>{warning}</code></li>"
                if len(analysis.warnings) > config.max_warning_details:
                    warnings_html += f"<li><em>... and {len(analysis.warnings) - config.max_warning_details} more warnings</em></li>"
            
            recommendations_html = ""
            for rec in analysis.recommendations:
                recommendations_html += f"<li>{rec}</li>"
            
            results_html += f"""
            <button class="collapsible">{analysis.language} - Detailed Results</button>
            <div class="content">
                <p><strong>Category:</strong> {analysis.category}</p>
                <p><strong>Execution Time:</strong> {self._format_duration(analysis.metrics.get('execution_time', 0))}</p>
                <p><strong>Output Files:</strong> {analysis.metrics.get('output_file_count', 0)}</p>
                
                {f'<h4>Errors ({len(analysis.errors)})</h4><ul>{errors_html}</ul>' if errors_html else ''}
                {f'<h4>Warnings ({len(analysis.warnings)})</h4><ul>{warnings_html}</ul>' if warnings_html else ''}
                {f'<h4>Recommendations</h4><ul>{recommendations_html}</ul>' if recommendations_html else ''}
            </div>
            """
        
        return f"""
        <div id="detailed-results" class="section">
            <h2>Detailed Results</h2>
            {results_html}
        </div>
        """
    
    def _generate_footer(self) -> str:
        """Generate footer HTML."""
        return f"""
        <div class="footer">
            <p>Generated by Joern Multi-Language Verification System on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        """