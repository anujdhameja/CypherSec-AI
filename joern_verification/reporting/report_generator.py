"""
Main report generation system for Joern multi-language verification.

This module provides the core ReportGenerator class that orchestrates the creation
of comprehensive verification reports with performance analysis and recommendations.
"""

import logging
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from ..core.interfaces import AnalysisReport, ReportGeneratorInterface
from ..analysis.metrics_collector import MetricsCollector, ErrorAnalysis, WarningAnalysis
from .models import (
    VerificationReport, ReportMetadata, LanguageSummary, ReportSection,
    PerformanceAnalysis, ErrorAnalysisSummary, RecommendationSet,
    ReportConfiguration, ChartData, TableData
)


class ReportGenerator(ReportGeneratorInterface):
    """
    Comprehensive report generator for Joern verification results.
    
    Generates detailed reports with success rates, error analysis, performance metrics,
    and actionable recommendations for each programming language tested.
    """
    
    def __init__(self, metrics_collector: Optional[MetricsCollector] = None):
        """
        Initialize the report generator.
        
        Args:
            metrics_collector: Optional MetricsCollector for performance data
        """
        self.logger = logging.getLogger(__name__)
        self.metrics_collector = metrics_collector or MetricsCollector()
        
    def generate_report(
        self, 
        analyses: List[AnalysisReport], 
        output_path: Path,
        config: Optional[ReportConfiguration] = None
    ) -> bool:
        """
        Generate a comprehensive verification report.
        
        Args:
            analyses: List of AnalysisReport objects to include
            output_path: Path where the report should be saved
            config: Optional configuration for report generation
            
        Returns:
            True if report was generated successfully, False otherwise
        """
        try:
            self.logger.info(f"Generating verification report for {len(analyses)} languages")
            
            # Use default configuration if none provided
            if config is None:
                config = ReportConfiguration()
            
            # Validate configuration
            config_issues = config.validate()
            if config_issues:
                self.logger.error(f"Configuration validation failed: {config_issues}")
                return False
            
            # Build comprehensive report
            report = self._build_verification_report(analyses, config)
            
            # Generate reports in requested formats
            success = True
            for format_type in config.output_formats:
                format_path = self._get_format_path(output_path, format_type)
                if not self._generate_format_report(report, format_path, format_type, config):
                    success = False
                    self.logger.error(f"Failed to generate {format_type} report")
            
            if success:
                self.logger.info(f"Successfully generated reports at {output_path}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to generate report: {e}")
            return False
    
    def format_summary(self, analyses: List[AnalysisReport]) -> str:
        """
        Format a summary of all analysis results.
        
        Args:
            analyses: List of AnalysisReport objects to summarize
            
        Returns:
            Formatted summary string
        """
        if not analyses:
            return "No analysis results available."
        
        # Calculate summary statistics
        total_languages = len(analyses)
        successful = sum(1 for a in analyses if a.category in ['success', 'success_with_warnings'])
        failed = total_languages - successful
        
        # Calculate average execution time
        execution_times = [a.metrics.get('execution_time', 0) for a in analyses]
        avg_time = statistics.mean(execution_times) if execution_times else 0
        
        # Count total errors and warnings
        total_errors = sum(len(a.errors) for a in analyses)
        total_warnings = sum(len(a.warnings) for a in analyses)
        
        # Format summary
        summary = f"""
Joern Multi-Language Verification Summary
========================================

Total Languages Tested: {total_languages}
Successful: {successful} ({successful/total_languages*100:.1f}%)
Failed: {failed} ({failed/total_languages*100:.1f}%)

Average Execution Time: {avg_time:.2f} seconds
Total Errors: {total_errors}
Total Warnings: {total_warnings}

Language Results:
"""
        
        # Add individual language results
        for analysis in sorted(analyses, key=lambda a: a.language):
            status_icon = "✓" if analysis.category in ['success', 'success_with_warnings'] else "✗"
            exec_time = analysis.metrics.get('execution_time', 0)
            summary += f"  {status_icon} {analysis.language:<12} - {analysis.category:<20} ({exec_time:.2f}s)\n"
        
        return summary
    
    def _build_verification_report(
        self, 
        analyses: List[AnalysisReport], 
        config: ReportConfiguration
    ) -> VerificationReport:
        """
        Build a comprehensive verification report from analysis results.
        
        Args:
            analyses: List of analysis results
            config: Report configuration
            
        Returns:
            Complete VerificationReport object
        """
        # Generate metadata
        metadata = self._generate_metadata(analyses)
        
        # Generate executive summary
        executive_summary = self._generate_executive_summary(analyses)
        
        # Generate language summaries
        language_summaries = self._generate_language_summaries(analyses)
        
        # Generate performance analysis
        performance_analysis = self._generate_performance_analysis(analyses)
        
        # Generate error analysis
        error_analysis = self._generate_error_analysis(analyses)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(analyses, performance_analysis, error_analysis)
        
        # Create the report
        report = VerificationReport(
            metadata=metadata,
            executive_summary=executive_summary,
            language_summaries=language_summaries,
            performance_analysis=performance_analysis,
            error_analysis=error_analysis,
            recommendations=recommendations,
            detailed_results=analyses if config.include_detailed_results else []
        )
        
        # Add additional sections based on configuration
        if config.include_performance_analysis:
            report.add_section(self._create_performance_section(performance_analysis))
        
        if config.include_error_analysis:
            report.add_section(self._create_error_section(error_analysis))
        
        return report
    
    def _generate_metadata(self, analyses: List[AnalysisReport]) -> ReportMetadata:
        """Generate report metadata."""
        successful = sum(1 for a in analyses if a.category in ['success', 'success_with_warnings'])
        failed = len(analyses) - successful
        
        # Calculate total execution time
        total_time = sum(a.metrics.get('execution_time', 0) for a in analyses)
        
        return ReportMetadata(
            title="Joern Multi-Language Verification Report",
            generated_at=datetime.now(),
            total_languages=len(analyses),
            successful_languages=successful,
            failed_languages=failed,
            execution_time=total_time
        )
    
    def _generate_executive_summary(self, analyses: List[AnalysisReport]) -> ReportSection:
        """Generate executive summary section."""
        total_languages = len(analyses)
        successful = sum(1 for a in analyses if a.category in ['success', 'success_with_warnings'])
        success_rate = (successful / total_languages * 100) if total_languages > 0 else 0
        
        # Identify key findings
        key_findings = []
        
        if success_rate >= 80:
            key_findings.append(f"Excellent overall success rate of {success_rate:.1f}%")
        elif success_rate >= 60:
            key_findings.append(f"Good success rate of {success_rate:.1f}% with room for improvement")
        else:
            key_findings.append(f"Low success rate of {success_rate:.1f}% requires attention")
        
        # Find fastest and slowest languages
        execution_times = [(a.language, a.metrics.get('execution_time', 0)) for a in analyses]
        if execution_times:
            fastest = min(execution_times, key=lambda x: x[1])
            slowest = max(execution_times, key=lambda x: x[1])
            key_findings.append(f"Performance range: {fastest[0]} ({fastest[1]:.2f}s) to {slowest[0]} ({slowest[1]:.2f}s)")
        
        # Count critical issues
        critical_errors = sum(1 for a in analyses if len(a.errors) > 0)
        if critical_errors > 0:
            key_findings.append(f"{critical_errors} languages have critical errors requiring attention")
        
        summary_content = {
            'overview': f"Tested {total_languages} programming languages with Joern CPG generation",
            'success_rate': f"{success_rate:.1f}%",
            'successful_languages': successful,
            'failed_languages': total_languages - successful,
            'key_findings': key_findings,
            'recommendation_count': len(self._get_top_recommendations(analyses))
        }
        
        return ReportSection(
            title="Executive Summary",
            content=summary_content,
            section_type="summary",
            priority=100
        )
    
    def _generate_language_summaries(self, analyses: List[AnalysisReport]) -> List[LanguageSummary]:
        """Generate summary for each language."""
        summaries = []
        
        for analysis in analyses:
            # Determine status
            status = "success" if analysis.category in ['success', 'success_with_warnings'] else "failure"
            
            # Get performance classification
            perf_class = self._classify_performance(analysis.metrics.get('execution_time', 0))
            
            # Calculate success rate (100% for success, 0% for failure, 50% for partial)
            success_rate = 100.0 if status == "success" else 0.0
            if analysis.category == "partial_success":
                success_rate = 50.0
            
            summary = LanguageSummary(
                language=analysis.language,
                status=status,
                execution_time=analysis.metrics.get('execution_time', 0),
                success_rate=success_rate,
                error_count=len(analysis.errors),
                warning_count=len(analysis.warnings),
                output_files=analysis.metrics.get('output_file_count', 0),
                recommendations=analysis.recommendations[:3],  # Top 3 recommendations
                category=analysis.category,
                performance_class=perf_class
            )
            
            summaries.append(summary)
        
        # Sort by language name
        summaries.sort(key=lambda s: s.language)
        
        return summaries
    
    def _generate_performance_analysis(self, analyses: List[AnalysisReport]) -> PerformanceAnalysis:
        """Generate comprehensive performance analysis."""
        if not analyses:
            return PerformanceAnalysis(
                overall_success_rate=0.0,
                average_execution_time=0.0,
                fastest_language=None,
                slowest_language=None,
                most_reliable_language=None,
                least_reliable_language=None,
                performance_rankings=[],
                reliability_rankings=[],
                resource_usage={},
                trends={}
            )
        
        # Calculate overall metrics
        successful = sum(1 for a in analyses if a.category in ['success', 'success_with_warnings'])
        overall_success_rate = (successful / len(analyses)) * 100
        
        execution_times = [a.metrics.get('execution_time', 0) for a in analyses]
        average_execution_time = statistics.mean(execution_times) if execution_times else 0
        
        # Find fastest and slowest languages
        time_rankings = [(a.language, a.metrics.get('execution_time', 0)) for a in analyses]
        time_rankings.sort(key=lambda x: x[1])
        
        fastest_language = time_rankings[0][0] if time_rankings else None
        slowest_language = time_rankings[-1][0] if time_rankings else None
        
        # Find most and least reliable languages
        reliability_rankings = []
        for analysis in analyses:
            success_score = 100 if analysis.category in ['success', 'success_with_warnings'] else 0
            if analysis.category == 'partial_success':
                success_score = 50
            reliability_rankings.append((analysis.language, success_score))
        
        reliability_rankings.sort(key=lambda x: x[1], reverse=True)
        
        most_reliable_language = reliability_rankings[0][0] if reliability_rankings else None
        least_reliable_language = reliability_rankings[-1][0] if reliability_rankings else None
        
        # Analyze resource usage
        resource_usage = self._analyze_resource_usage(analyses)
        
        # Analyze trends (simplified)
        trends = self._analyze_performance_trends(analyses)
        
        return PerformanceAnalysis(
            overall_success_rate=overall_success_rate,
            average_execution_time=average_execution_time,
            fastest_language=fastest_language,
            slowest_language=slowest_language,
            most_reliable_language=most_reliable_language,
            least_reliable_language=least_reliable_language,
            performance_rankings=time_rankings,
            reliability_rankings=reliability_rankings,
            resource_usage=resource_usage,
            trends=trends
        )
    
    def _generate_error_analysis(self, analyses: List[AnalysisReport]) -> ErrorAnalysisSummary:
        """Generate comprehensive error analysis."""
        # Use metrics collector for detailed error analysis
        error_analysis = self.metrics_collector.analyze_error_patterns(analyses)
        
        # Count total errors
        total_errors = sum(len(a.errors) for a in analyses)
        
        # Find languages with errors
        languages_with_errors = [a.language for a in analyses if a.errors]
        
        # Identify critical issues (languages with multiple errors)
        critical_issues = []
        for analysis in analyses:
            if len(analysis.errors) > 2:
                critical_issues.append(f"{analysis.language}: {len(analysis.errors)} errors")
        
        return ErrorAnalysisSummary(
            total_errors=total_errors,
            error_categories=error_analysis.error_categories,
            common_errors=error_analysis.common_errors,
            languages_with_errors=languages_with_errors,
            resolution_suggestions=error_analysis.resolution_suggestions,
            critical_issues=critical_issues
        )
    
    def _generate_recommendations(
        self, 
        analyses: List[AnalysisReport],
        performance_analysis: PerformanceAnalysis,
        error_analysis: ErrorAnalysisSummary
    ) -> RecommendationSet:
        """Generate comprehensive recommendations."""
        immediate_actions = []
        performance_improvements = []
        reliability_enhancements = []
        alternative_tools = {}
        next_steps = []
        language_specific = {}
        
        # Immediate actions based on critical issues
        if error_analysis.critical_issues:
            immediate_actions.append("Address critical errors in languages with multiple failures")
        
        if performance_analysis.overall_success_rate < 70:
            immediate_actions.append("Investigate common failure patterns to improve overall success rate")
        
        # Performance improvements
        if performance_analysis.average_execution_time > 30:
            performance_improvements.append("Optimize CPG generation for better performance")
        
        if performance_analysis.slowest_language:
            performance_improvements.append(f"Focus optimization on {performance_analysis.slowest_language}")
        
        # Reliability enhancements
        failed_languages = [a.language for a in analyses if a.category in ['failure', 'tool_missing']]
        if failed_languages:
            reliability_enhancements.append(f"Investigate setup issues for: {', '.join(failed_languages)}")
        
        # Alternative tools for failed languages
        for analysis in analyses:
            if analysis.category in ['failure', 'tool_missing']:
                alternative_tools[analysis.language] = [
                    "Consider tree-sitter parser",
                    "Explore language-specific AST tools",
                    "Check for updated Joern language frontends"
                ]
        
        # Next steps
        next_steps.extend([
            "Test with larger, more complex source files",
            "Benchmark performance with production codebases",
            "Set up automated verification pipeline",
            "Document language-specific configuration requirements"
        ])
        
        # Language-specific recommendations
        for analysis in analyses:
            lang_recommendations = []
            
            if analysis.category == 'success_with_warnings':
                lang_recommendations.append("Review warnings for potential quality issues")
            elif analysis.category == 'partial_success':
                lang_recommendations.append("Investigate partial generation causes")
            elif analysis.category == 'failure':
                lang_recommendations.extend(analysis.recommendations[:2])
            
            if lang_recommendations:
                language_specific[analysis.language] = lang_recommendations
        
        return RecommendationSet(
            immediate_actions=immediate_actions,
            performance_improvements=performance_improvements,
            reliability_enhancements=reliability_enhancements,
            alternative_tools=alternative_tools,
            next_steps=next_steps,
            language_specific=language_specific
        )
    
    def _get_top_recommendations(self, analyses: List[AnalysisReport]) -> List[str]:
        """Get top recommendations across all analyses."""
        all_recommendations = []
        for analysis in analyses:
            all_recommendations.extend(analysis.recommendations)
        
        # Count frequency of recommendations
        from collections import Counter
        recommendation_counts = Counter(all_recommendations)
        
        # Return top 5 most common recommendations
        return [rec for rec, count in recommendation_counts.most_common(5)]
    
    def _classify_performance(self, execution_time: float) -> str:
        """Classify performance based on execution time."""
        if execution_time < 5.0:
            return "excellent"
        elif execution_time < 15.0:
            return "good"
        elif execution_time < 60.0:
            return "acceptable"
        elif execution_time < 300.0:
            return "slow"
        else:
            return "very_slow"
    
    def _analyze_resource_usage(self, analyses: List[AnalysisReport]) -> Dict[str, Any]:
        """Analyze resource usage patterns."""
        memory_usage = []
        output_sizes = []
        
        for analysis in analyses:
            if analysis.metrics.get('memory_usage'):
                memory_usage.append(analysis.metrics['memory_usage'])
            
            if analysis.metrics.get('output_total_size'):
                output_sizes.append(analysis.metrics['output_total_size'])
        
        resource_analysis = {}
        
        if memory_usage:
            resource_analysis['memory'] = {
                'average_mb': statistics.mean(memory_usage) / (1024 * 1024),
                'max_mb': max(memory_usage) / (1024 * 1024),
                'languages_tracked': len(memory_usage)
            }
        
        if output_sizes:
            resource_analysis['output'] = {
                'average_mb': statistics.mean(output_sizes) / (1024 * 1024),
                'total_mb': sum(output_sizes) / (1024 * 1024),
                'largest_mb': max(output_sizes) / (1024 * 1024)
            }
        
        return resource_analysis
    
    def _analyze_performance_trends(self, analyses: List[AnalysisReport]) -> Dict[str, Any]:
        """Analyze performance trends (simplified analysis)."""
        execution_times = [a.metrics.get('execution_time', 0) for a in analyses]
        
        if len(execution_times) < 2:
            return {'trend': 'insufficient_data'}
        
        # Simple trend analysis
        avg_time = statistics.mean(execution_times)
        std_dev = statistics.stdev(execution_times) if len(execution_times) > 1 else 0
        
        # Classify consistency
        cv = std_dev / avg_time if avg_time > 0 else 0
        
        if cv < 0.2:
            consistency = "consistent"
        elif cv < 0.5:
            consistency = "moderate"
        else:
            consistency = "inconsistent"
        
        return {
            'average_time': avg_time,
            'consistency': consistency,
            'coefficient_of_variation': cv,
            'performance_spread': max(execution_times) - min(execution_times) if execution_times else 0
        }
    
    def _create_performance_section(self, performance_analysis: PerformanceAnalysis) -> ReportSection:
        """Create performance analysis section."""
        content = {
            'overall_success_rate': f"{performance_analysis.overall_success_rate:.1f}%",
            'average_execution_time': f"{performance_analysis.average_execution_time:.2f} seconds",
            'fastest_language': performance_analysis.fastest_language,
            'slowest_language': performance_analysis.slowest_language,
            'performance_rankings': performance_analysis.performance_rankings[:10],  # Top 10
            'resource_usage': performance_analysis.resource_usage
        }
        
        return ReportSection(
            title="Performance Analysis",
            content=content,
            section_type="analysis",
            priority=80
        )
    
    def _create_error_section(self, error_analysis: ErrorAnalysisSummary) -> ReportSection:
        """Create error analysis section."""
        content = {
            'total_errors': error_analysis.total_errors,
            'error_categories': error_analysis.error_categories,
            'common_errors': error_analysis.common_errors[:10],  # Top 10
            'languages_with_errors': error_analysis.languages_with_errors,
            'critical_issues': error_analysis.critical_issues,
            'resolution_suggestions': error_analysis.resolution_suggestions
        }
        
        return ReportSection(
            title="Error Analysis",
            content=content,
            section_type="analysis",
            priority=70
        )
    
    def _generate_format_report(
        self, 
        report: VerificationReport, 
        output_path: Path, 
        format_type: str,
        config: ReportConfiguration
    ) -> bool:
        """Generate report in specific format."""
        try:
            if format_type == 'json':
                from .formatters import JSONFormatter
                formatter = JSONFormatter()
            elif format_type == 'markdown':
                from .formatters import MarkdownFormatter
                formatter = MarkdownFormatter()
            elif format_type == 'html':
                from .formatters import HTMLFormatter
                formatter = HTMLFormatter()
            else:
                self.logger.error(f"Unsupported format: {format_type}")
                return False
            
            return formatter.format_report(report, output_path, config)
            
        except Exception as e:
            self.logger.error(f"Failed to generate {format_type} report: {e}")
            return False
    
    def _get_format_path(self, base_path: Path, format_type: str) -> Path:
        """Get output path for specific format."""
        if format_type == 'json':
            return base_path.with_suffix('.json')
        elif format_type == 'markdown':
            return base_path.with_suffix('.md')
        elif format_type == 'html':
            return base_path.with_suffix('.html')
        elif format_type == 'csv':
            return base_path.with_suffix('.csv')
        else:
            return base_path
    
    def generate_summary_dashboard(self, analyses: List[AnalysisReport]) -> Dict[str, Any]:
        """
        Generate a summary dashboard with key metrics and visualizations.
        
        Args:
            analyses: List of analysis results
            
        Returns:
            Dictionary containing dashboard data
        """
        dashboard = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_languages': len(analyses)
            },
            'key_metrics': self._generate_key_metrics(analyses),
            'charts': self._generate_dashboard_charts(analyses),
            'tables': self._generate_dashboard_tables(analyses),
            'alerts': self._generate_dashboard_alerts(analyses)
        }
        
        return dashboard
    
    def _generate_key_metrics(self, analyses: List[AnalysisReport]) -> Dict[str, Any]:
        """Generate key metrics for dashboard."""
        total = len(analyses)
        successful = sum(1 for a in analyses if a.category in ['success', 'success_with_warnings'])
        
        execution_times = [a.metrics.get('execution_time', 0) for a in analyses]
        avg_time = statistics.mean(execution_times) if execution_times else 0
        
        return {
            'total_languages': total,
            'successful_languages': successful,
            'success_rate': (successful / total * 100) if total > 0 else 0,
            'average_execution_time': avg_time,
            'total_errors': sum(len(a.errors) for a in analyses),
            'total_warnings': sum(len(a.warnings) for a in analyses)
        }
    
    def _generate_dashboard_charts(self, analyses: List[AnalysisReport]) -> List[ChartData]:
        """Generate charts for dashboard."""
        charts = []
        
        # Success rate pie chart
        categories = {}
        for analysis in analyses:
            categories[analysis.category] = categories.get(analysis.category, 0) + 1
        
        charts.append(ChartData(
            chart_type='pie',
            title='Language Success Distribution',
            data=categories,
            labels=list(categories.keys()),
            values=list(categories.values()),
            description='Distribution of success categories across all tested languages'
        ))
        
        # Execution time bar chart
        time_data = [(a.language, a.metrics.get('execution_time', 0)) for a in analyses]
        time_data.sort(key=lambda x: x[1])
        
        charts.append(ChartData(
            chart_type='bar',
            title='Execution Time by Language',
            data={lang: time for lang, time in time_data},
            labels=[lang for lang, _ in time_data],
            values=[time for _, time in time_data],
            description='CPG generation execution time for each language'
        ))
        
        return charts
    
    def _generate_dashboard_tables(self, analyses: List[AnalysisReport]) -> List[TableData]:
        """Generate tables for dashboard."""
        tables = []
        
        # Language summary table
        headers = ['Language', 'Status', 'Execution Time (s)', 'Errors', 'Warnings', 'Output Files']
        rows = []
        
        for analysis in sorted(analyses, key=lambda a: a.language):
            rows.append([
                analysis.language,
                analysis.category,
                f"{analysis.metrics.get('execution_time', 0):.2f}",
                len(analysis.errors),
                len(analysis.warnings),
                analysis.metrics.get('output_file_count', 0)
            ])
        
        tables.append(TableData(
            title='Language Summary',
            headers=headers,
            rows=rows,
            column_types=['text', 'text', 'number', 'number', 'number', 'number'],
            sortable=True
        ))
        
        return tables
    
    def _generate_dashboard_alerts(self, analyses: List[AnalysisReport]) -> List[Dict[str, str]]:
        """Generate alerts for dashboard."""
        alerts = []
        
        # Check for critical issues
        failed_languages = [a.language for a in analyses if a.category == 'failure']
        if failed_languages:
            alerts.append({
                'type': 'error',
                'title': 'Failed Languages',
                'message': f"{len(failed_languages)} languages failed CPG generation: {', '.join(failed_languages)}"
            })
        
        # Check for performance issues
        slow_languages = [a.language for a in analyses if a.metrics.get('execution_time', 0) > 60]
        if slow_languages:
            alerts.append({
                'type': 'warning',
                'title': 'Performance Issues',
                'message': f"Slow execution detected for: {', '.join(slow_languages)}"
            })
        
        # Check success rate
        total = len(analyses)
        successful = sum(1 for a in analyses if a.category in ['success', 'success_with_warnings'])
        success_rate = (successful / total * 100) if total > 0 else 0
        
        if success_rate < 70:
            alerts.append({
                'type': 'warning',
                'title': 'Low Success Rate',
                'message': f"Overall success rate is {success_rate:.1f}% - consider investigating common issues"
            })
        elif success_rate >= 90:
            alerts.append({
                'type': 'success',
                'title': 'Excellent Results',
                'message': f"High success rate of {success_rate:.1f}% achieved"
            })
        
        return alerts