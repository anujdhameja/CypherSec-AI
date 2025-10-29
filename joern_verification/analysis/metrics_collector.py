"""
Metrics collection and analysis system for CPG generation performance.

This module provides comprehensive performance benchmarking, error analysis,
and warning detection for multi-language CPG generation verification.
"""

import time
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from pathlib import Path
from collections import defaultdict, Counter
import statistics
import json

from ..core.interfaces import GenerationResult, AnalysisReport


@dataclass
class LanguageBenchmark:
    """Performance benchmark data for a specific language."""
    language: str
    execution_times: List[float] = field(default_factory=list)
    memory_usage: List[int] = field(default_factory=list)
    success_count: int = 0
    failure_count: int = 0
    warning_count: int = 0
    error_count: int = 0
    output_sizes: List[int] = field(default_factory=list)
    
    @property
    def total_attempts(self) -> int:
        """Total number of CPG generation attempts."""
        return self.success_count + self.failure_count
    
    @property
    def success_rate(self) -> float:
        """Success rate as a percentage."""
        if self.total_attempts == 0:
            return 0.0
        return (self.success_count / self.total_attempts) * 100.0
    
    @property
    def average_execution_time(self) -> float:
        """Average execution time in seconds."""
        if not self.execution_times:
            return 0.0
        return statistics.mean(self.execution_times)
    
    @property
    def median_execution_time(self) -> float:
        """Median execution time in seconds."""
        if not self.execution_times:
            return 0.0
        return statistics.median(self.execution_times)


@dataclass
class ErrorAnalysis:
    """Analysis of error patterns and frequencies."""
    error_categories: Dict[str, int] = field(default_factory=dict)
    common_errors: List[Tuple[str, int]] = field(default_factory=list)
    error_by_language: Dict[str, List[str]] = field(default_factory=dict)
    resolution_suggestions: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class WarningAnalysis:
    """Analysis of warning patterns and classifications."""
    warning_categories: Dict[str, int] = field(default_factory=dict)
    warning_by_language: Dict[str, List[str]] = field(default_factory=dict)
    severity_distribution: Dict[str, int] = field(default_factory=dict)


class MetricsCollector:
    """Comprehensive metrics collection and analysis system."""
    
    def __init__(self):
        """Initialize the metrics collector."""
        self.logger = logging.getLogger(__name__)
        self.benchmarks: Dict[str, LanguageBenchmark] = {}
        self.system_metrics: List[Dict[str, Any]] = []
        self.error_patterns = self._initialize_error_patterns()
        self.warning_patterns = self._initialize_warning_patterns()
        self.performance_thresholds = self._initialize_performance_thresholds()
        
    def collect_benchmark_data(self, result: GenerationResult) -> None:
        """
        Collect benchmark data from a generation result.
        
        Args:
            result: GenerationResult to collect metrics from
        """
        language = result.language.lower()
        
        # Initialize benchmark if not exists
        if language not in self.benchmarks:
            self.benchmarks[language] = LanguageBenchmark(language=language)
        
        benchmark = self.benchmarks[language]
        
        # Update execution times
        benchmark.execution_times.append(result.execution_time)
        
        # Update memory usage if available
        if result.memory_usage is not None:
            benchmark.memory_usage.append(result.memory_usage)
        
        # Update success/failure counts
        if result.success:
            benchmark.success_count += 1
        else:
            benchmark.failure_count += 1
        
        # Update warning and error counts
        benchmark.warning_count += len(result.warnings or [])
        
        # Count errors in stderr
        if result.stderr:
            error_lines = [line for line in result.stderr.split('\n') 
                          if any(keyword in line.lower() for keyword in ['error', 'exception', 'failed'])]
            benchmark.error_count += len(error_lines)
        
        # Update output sizes
        total_output_size = 0
        for output_file in result.output_files:
            try:
                if output_file.exists():
                    total_output_size += output_file.stat().st_size
            except (OSError, AttributeError):
                pass
        
        if total_output_size > 0:
            benchmark.output_sizes.append(total_output_size)
        
        self.logger.debug(f"Collected benchmark data for {language}")
    
    def _initialize_error_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Initialize comprehensive error pattern recognition."""
        return {
            'memory_error': [
                re.compile(r'OutOfMemoryError|out of memory|heap space|memory.*exhausted', re.IGNORECASE),
                re.compile(r'java\.lang\.OutOfMemoryError|GC overhead limit exceeded', re.IGNORECASE),
                re.compile(r'Cannot allocate memory|insufficient memory', re.IGNORECASE)
            ],
            'timeout_error': [
                re.compile(r'timeout|timed out|time limit exceeded|execution.*timeout', re.IGNORECASE),
                re.compile(r'process.*killed|terminated.*timeout', re.IGNORECASE)
            ],
            'syntax_error': [
                re.compile(r'syntax error|parse error|parsing failed|compilation.*failed', re.IGNORECASE),
                re.compile(r'unexpected token|invalid syntax|malformed', re.IGNORECASE),
                re.compile(r'SyntaxError|ParseException|CompilationException', re.IGNORECASE)
            ],
            'dependency_error': [
                re.compile(r'command not found|file not found|no such file|module.*not.*found', re.IGNORECASE),
                re.compile(r'ClassNotFoundException|NoClassDefFoundError|ImportError|FileNotFoundException', re.IGNORECASE),
                re.compile(r'dependency.*missing|required.*not.*found|not found', re.IGNORECASE)
            ],
            'permission_error': [
                re.compile(r'permission denied|access denied|unauthorized|forbidden', re.IGNORECASE),
                re.compile(r'SecurityException|AccessControlException', re.IGNORECASE)
            ],
            'network_error': [
                re.compile(r'network.*error|connection.*failed|host.*unreachable|NetworkException', re.IGNORECASE),
                re.compile(r'ConnectException|SocketException|UnknownHostException', re.IGNORECASE)
            ],
            'configuration_error': [
                re.compile(r'configuration.*error|invalid.*config|misconfigured', re.IGNORECASE),
                re.compile(r'ConfigurationException|IllegalArgumentException', re.IGNORECASE)
            ]
        }
    
    def _initialize_warning_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Initialize comprehensive warning pattern recognition."""
        return {
            'deprecation_warning': [
                re.compile(r'deprecated|obsolete|legacy.*usage', re.IGNORECASE),
                re.compile(r'DeprecationWarning|@deprecated', re.IGNORECASE)
            ],
            'performance_warning': [
                re.compile(r'performance.*warning|slow.*operation|inefficient', re.IGNORECASE),
                re.compile(r'optimization.*recommended|consider.*optimizing', re.IGNORECASE)
            ],
            'compatibility_warning': [
                re.compile(r'compatibility.*warning|version.*mismatch|incompatible', re.IGNORECASE),
                re.compile(r'unsupported.*version|requires.*newer', re.IGNORECASE)
            ],
            'security_warning': [
                re.compile(r'security.*warning|unsafe.*operation|potential.*vulnerability', re.IGNORECASE),
                re.compile(r'SecurityWarning|insecure.*usage', re.IGNORECASE)
            ],
            'syntax_warning': [
                re.compile(r'syntax.*warning|style.*warning|formatting', re.IGNORECASE),
                re.compile(r'SyntaxWarning|StyleWarning', re.IGNORECASE)
            ]
        }
    
    def _initialize_performance_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Initialize performance classification thresholds."""
        return {
            'execution_time': {
                'excellent': 5.0,    # < 5 seconds
                'good': 15.0,        # < 15 seconds
                'acceptable': 60.0,  # < 1 minute
                'slow': 300.0,       # < 5 minutes
                # > 5 minutes = very_slow
            },
            'memory_usage': {
                'low': 100 * 1024 * 1024,      # < 100 MB
                'moderate': 500 * 1024 * 1024,  # < 500 MB
                'high': 1024 * 1024 * 1024,     # < 1 GB
                # > 1 GB = very_high
            },
            'output_size': {
                'small': 1 * 1024 * 1024,      # < 1 MB
                'medium': 10 * 1024 * 1024,     # < 10 MB
                'large': 100 * 1024 * 1024,     # < 100 MB
                # > 100 MB = very_large
            }
        }
    
    def analyze_performance_patterns(self) -> Dict[str, Any]:
        """
        Analyze performance patterns across all languages.
        
        Returns:
            Dictionary containing performance analysis results
        """
        if not self.benchmarks:
            return {}
        
        analysis = {
            'language_count': len(self.benchmarks),
            'total_attempts': sum(b.total_attempts for b in self.benchmarks.values()),
            'overall_success_rate': self._calculate_overall_success_rate(),
            'performance_rankings': self._rank_languages_by_performance(),
            'reliability_rankings': self._rank_languages_by_reliability(),
            'execution_time_stats': self._analyze_execution_times(),
            'output_size_stats': self._analyze_output_sizes(),
            'resource_usage_patterns': self._analyze_resource_usage()
        }
        
        return analysis
    
    def analyze_error_patterns(self, analyses: List[AnalysisReport]) -> ErrorAnalysis:
        """
        Analyze error patterns across multiple analysis reports.
        
        Args:
            analyses: List of AnalysisReport objects to analyze
            
        Returns:
            ErrorAnalysis with categorized error information
        """
        error_analysis = ErrorAnalysis()
        
        # Collect all errors by category and language
        for analysis in analyses:
            language = analysis.language
            
            # Initialize language error list
            if language not in error_analysis.error_by_language:
                error_analysis.error_by_language[language] = []
            
            # Process errors
            for error in analysis.errors:
                error_analysis.error_by_language[language].append(error)
                
                # Categorize error
                category = self._categorize_error_message(error)
                error_analysis.error_categories[category] = error_analysis.error_categories.get(category, 0) + 1
        
        # Find most common errors
        all_errors = []
        for errors in error_analysis.error_by_language.values():
            all_errors.extend(errors)
        
        error_counter = Counter(all_errors)
        error_analysis.common_errors = error_counter.most_common(10)
        
        # Generate resolution suggestions
        error_analysis.resolution_suggestions = self._generate_error_resolutions(error_analysis.error_categories)
        
        return error_analysis
    
    def analyze_warning_patterns(self, analyses: List[AnalysisReport]) -> WarningAnalysis:
        """
        Analyze warning patterns and classify by severity.
        
        Args:
            analyses: List of AnalysisReport objects to analyze
            
        Returns:
            WarningAnalysis with categorized warning information
        """
        warning_analysis = WarningAnalysis()
        
        # Collect all warnings by language
        for analysis in analyses:
            language = analysis.language
            
            # Initialize language warning list
            if language not in warning_analysis.warning_by_language:
                warning_analysis.warning_by_language[language] = []
            
            # Process warnings
            for warning in analysis.warnings:
                warning_analysis.warning_by_language[language].append(warning)
                
                # Categorize warning
                category = self._categorize_warning_message(warning)
                warning_analysis.warning_categories[category] = warning_analysis.warning_categories.get(category, 0) + 1
                
                # Classify severity
                severity = self._classify_warning_severity(warning)
                warning_analysis.severity_distribution[severity] = warning_analysis.severity_distribution.get(severity, 0) + 1
        
        return warning_analysis
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.
        
        Returns:
            Dictionary containing detailed performance analysis
        """
        report = {
            'summary': self._generate_performance_summary(),
            'language_details': self._generate_language_details(),
            'comparative_analysis': self._generate_comparative_analysis(),
            'recommendations': self._generate_performance_recommendations()
        }
        
        return report
    
    def _calculate_overall_success_rate(self) -> float:
        """Calculate overall success rate across all languages."""
        total_success = sum(b.success_count for b in self.benchmarks.values())
        total_attempts = sum(b.total_attempts for b in self.benchmarks.values())
        
        if total_attempts == 0:
            return 0.0
        
        return (total_success / total_attempts) * 100.0
    
    def _rank_languages_by_performance(self) -> List[Tuple[str, float]]:
        """
        Rank languages by average execution time (faster is better).
        
        Returns:
            List of (language, avg_time) tuples sorted by performance
        """
        rankings = []
        
        for language, benchmark in self.benchmarks.items():
            if benchmark.execution_times:
                avg_time = benchmark.average_execution_time
                rankings.append((language, avg_time))
        
        # Sort by execution time (ascending - faster is better)
        rankings.sort(key=lambda x: x[1])
        
        return rankings
    
    def _rank_languages_by_reliability(self) -> List[Tuple[str, float]]:
        """
        Rank languages by success rate (higher is better).
        
        Returns:
            List of (language, success_rate) tuples sorted by reliability
        """
        rankings = []
        
        for language, benchmark in self.benchmarks.items():
            if benchmark.total_attempts > 0:
                rankings.append((language, benchmark.success_rate))
        
        # Sort by success rate (descending - higher is better)
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        return rankings
    
    def _analyze_execution_times(self) -> Dict[str, Any]:
        """Analyze execution time statistics across all languages."""
        all_times = []
        for benchmark in self.benchmarks.values():
            all_times.extend(benchmark.execution_times)
        
        if not all_times:
            return {}
        
        return {
            'min': min(all_times),
            'max': max(all_times),
            'mean': statistics.mean(all_times),
            'median': statistics.median(all_times),
            'std_dev': statistics.stdev(all_times) if len(all_times) > 1 else 0,
            'percentile_95': self._percentile(all_times, 95),
            'total_samples': len(all_times)
        }
    
    def _analyze_output_sizes(self) -> Dict[str, Any]:
        """Analyze output file size statistics."""
        all_sizes = []
        for benchmark in self.benchmarks.values():
            all_sizes.extend(benchmark.output_sizes)
        
        if not all_sizes:
            return {}
        
        return {
            'min_bytes': min(all_sizes),
            'max_bytes': max(all_sizes),
            'mean_bytes': statistics.mean(all_sizes),
            'median_bytes': statistics.median(all_sizes),
            'total_output_mb': sum(all_sizes) / (1024 * 1024),
            'average_output_mb': statistics.mean(all_sizes) / (1024 * 1024)
        }
    
    def _analyze_resource_usage(self) -> Dict[str, Any]:
        """Analyze resource usage patterns."""
        memory_usage = []
        for benchmark in self.benchmarks.values():
            memory_usage.extend(benchmark.memory_usage)
        
        resource_analysis = {
            'memory_samples': len(memory_usage),
            'languages_with_memory_data': sum(1 for b in self.benchmarks.values() if b.memory_usage)
        }
        
        if memory_usage:
            resource_analysis.update({
                'min_memory_mb': min(memory_usage) / (1024 * 1024),
                'max_memory_mb': max(memory_usage) / (1024 * 1024),
                'avg_memory_mb': statistics.mean(memory_usage) / (1024 * 1024)
            })
        
        return resource_analysis
    
    def _categorize_error_message(self, error: str) -> str:
        """Categorize an error message using comprehensive pattern matching."""
        # Check each error category pattern
        for category, patterns in self.error_patterns.items():
            for pattern in patterns:
                if pattern.search(error):
                    return category
        
        # Default to general error if no pattern matches
        return 'general_error'
    
    def _categorize_warning_message(self, warning: str) -> str:
        """Categorize a warning message using comprehensive pattern matching."""
        # Check each warning category pattern
        for category, patterns in self.warning_patterns.items():
            for pattern in patterns:
                if pattern.search(warning):
                    return category
        
        # Default to general warning if no pattern matches
        return 'general_warning'
    
    def _classify_warning_severity(self, warning: str) -> str:
        """Classify warning severity level."""
        warning_lower = warning.lower()
        
        if any(keyword in warning_lower for keyword in ['critical', 'severe', 'security']):
            return 'high'
        elif any(keyword in warning_lower for keyword in ['important', 'significant']):
            return 'medium'
        else:
            return 'low'
    
    def _generate_error_resolutions(self, error_categories: Dict[str, int]) -> Dict[str, List[str]]:
        """Generate resolution suggestions for error categories."""
        resolutions = {
            'memory_error': [
                'Increase JVM heap size with -J-Xmx parameter (e.g., -J-Xmx8g)',
                'Use smaller input files for testing',
                'Process files in smaller batches',
                'Monitor system memory usage and close other applications',
                'Consider using 64-bit JVM if not already'
            ],
            'timeout_error': [
                'Increase command timeout duration in configuration',
                'Use simpler test files to verify basic functionality',
                'Check system performance and CPU load',
                'Consider parallel processing limitations',
                'Verify system is not under heavy load'
            ],
            'syntax_error': [
                'Verify source code syntax is correct for the target language',
                'Check language version compatibility with Joern',
                'Use different test file samples with known good syntax',
                'Review language-specific requirements and standards',
                'Validate test files with language-specific compilers'
            ],
            'dependency_error': [
                'Install required language tools and runtimes',
                'Verify PATH configuration includes all necessary tools',
                'Check Joern installation completeness',
                'Install language-specific dependencies and SDKs',
                'Verify tool versions are compatible'
            ],
            'permission_error': [
                'Check file and directory permissions',
                'Run with appropriate user privileges',
                'Verify tool execution permissions',
                'Check antivirus software interference',
                'Ensure write permissions for output directories'
            ],
            'network_error': [
                'Check internet connectivity',
                'Verify proxy settings if applicable',
                'Check firewall configuration',
                'Retry operation after network issues resolve',
                'Use offline mode if available'
            ],
            'configuration_error': [
                'Review Joern configuration files',
                'Verify tool paths and settings',
                'Check environment variables',
                'Reset to default configuration if needed',
                'Validate configuration file syntax'
            ],
            'general_error': [
                'Review complete error messages for specific details',
                'Check system logs for additional information',
                'Verify input file integrity and format',
                'Test with minimal examples first',
                'Consult Joern documentation for troubleshooting'
            ]
        }
        
        return {category: resolutions.get(category, ['Review error details and documentation']) 
                for category in error_categories.keys()}
    
    def _generate_performance_summary(self) -> Dict[str, Any]:
        """Generate high-level performance summary."""
        if not self.benchmarks:
            return {}
        
        total_languages = len(self.benchmarks)
        successful_languages = sum(1 for b in self.benchmarks.values() if b.success_count > 0)
        
        return {
            'total_languages_tested': total_languages,
            'successful_languages': successful_languages,
            'overall_success_rate': self._calculate_overall_success_rate(),
            'fastest_language': self._get_fastest_language(),
            'most_reliable_language': self._get_most_reliable_language(),
            'total_execution_time': sum(sum(b.execution_times) for b in self.benchmarks.values()),
            'average_time_per_language': self._get_average_time_per_language()
        }
    
    def _generate_language_details(self) -> Dict[str, Dict[str, Any]]:
        """Generate detailed metrics for each language."""
        details = {}
        
        for language, benchmark in self.benchmarks.items():
            details[language] = {
                'attempts': benchmark.total_attempts,
                'successes': benchmark.success_count,
                'failures': benchmark.failure_count,
                'success_rate': benchmark.success_rate,
                'avg_execution_time': benchmark.average_execution_time,
                'median_execution_time': benchmark.median_execution_time,
                'min_execution_time': min(benchmark.execution_times) if benchmark.execution_times else 0,
                'max_execution_time': max(benchmark.execution_times) if benchmark.execution_times else 0,
                'warning_count': benchmark.warning_count,
                'error_count': benchmark.error_count,
                'avg_output_size': statistics.mean(benchmark.output_sizes) if benchmark.output_sizes else 0
            }
        
        return details
    
    def _generate_comparative_analysis(self) -> Dict[str, Any]:
        """Generate comparative analysis between languages."""
        performance_rankings = self._rank_languages_by_performance()
        reliability_rankings = self._rank_languages_by_reliability()
        
        return {
            'performance_rankings': performance_rankings,
            'reliability_rankings': reliability_rankings,
            'performance_vs_reliability': self._correlate_performance_reliability(),
            'outliers': self._identify_performance_outliers()
        }
    
    def _generate_performance_recommendations(self) -> List[str]:
        """Generate performance improvement recommendations."""
        recommendations = []
        
        # Analyze overall patterns
        avg_time = self._get_average_time_per_language()
        if avg_time > 30:
            recommendations.append("Consider optimizing for better overall performance")
        
        # Check for high failure rates
        overall_success = self._calculate_overall_success_rate()
        if overall_success < 70:
            recommendations.append("Investigate common failure causes to improve reliability")
        
        # Memory usage recommendations
        if any(b.memory_usage for b in self.benchmarks.values()):
            recommendations.append("Monitor memory usage patterns for optimization opportunities")
        
        # Language-specific recommendations
        slow_languages = [lang for lang, time in self._rank_languages_by_performance()[-3:]]
        if slow_languages:
            recommendations.append(f"Focus optimization efforts on slower languages: {', '.join(slow_languages)}")
        
        return recommendations
    
    def benchmark_language_performance(self, language: str) -> Dict[str, Any]:
        """
        Generate detailed performance benchmark for a specific language.
        
        Args:
            language: Programming language to benchmark
            
        Returns:
            Dictionary containing comprehensive performance metrics
        """
        if language not in self.benchmarks:
            return {}
        
        benchmark = self.benchmarks[language]
        
        # Calculate detailed statistics
        performance_metrics = {
            'language': language,
            'sample_size': len(benchmark.execution_times),
            'success_metrics': {
                'total_attempts': benchmark.total_attempts,
                'successful_attempts': benchmark.success_count,
                'failed_attempts': benchmark.failure_count,
                'success_rate_percent': benchmark.success_rate
            },
            'timing_metrics': self._calculate_timing_metrics(benchmark.execution_times),
            'resource_metrics': self._calculate_resource_metrics(benchmark),
            'quality_metrics': self._calculate_quality_metrics(benchmark),
            'performance_classification': self._classify_language_performance(benchmark),
            'trend_analysis': self._analyze_performance_trends(benchmark)
        }
        
        return performance_metrics
    
    def _calculate_timing_metrics(self, execution_times: List[float]) -> Dict[str, Any]:
        """Calculate comprehensive timing statistics."""
        if not execution_times:
            return {}
        
        sorted_times = sorted(execution_times)
        
        return {
            'count': len(execution_times),
            'min_seconds': min(execution_times),
            'max_seconds': max(execution_times),
            'mean_seconds': statistics.mean(execution_times),
            'median_seconds': statistics.median(execution_times),
            'std_dev_seconds': statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
            'percentiles': {
                'p25': self._percentile(execution_times, 25),
                'p75': self._percentile(execution_times, 75),
                'p90': self._percentile(execution_times, 90),
                'p95': self._percentile(execution_times, 95),
                'p99': self._percentile(execution_times, 99)
            },
            'variance': statistics.variance(execution_times) if len(execution_times) > 1 else 0,
            'coefficient_of_variation': (statistics.stdev(execution_times) / statistics.mean(execution_times)) if len(execution_times) > 1 and statistics.mean(execution_times) > 0 else 0
        }
    
    def _calculate_resource_metrics(self, benchmark: LanguageBenchmark) -> Dict[str, Any]:
        """Calculate resource usage metrics."""
        metrics = {
            'memory_usage': {},
            'output_metrics': {}
        }
        
        # Memory usage metrics
        if benchmark.memory_usage:
            metrics['memory_usage'] = {
                'samples': len(benchmark.memory_usage),
                'min_bytes': min(benchmark.memory_usage),
                'max_bytes': max(benchmark.memory_usage),
                'mean_bytes': statistics.mean(benchmark.memory_usage),
                'median_bytes': statistics.median(benchmark.memory_usage),
                'min_mb': min(benchmark.memory_usage) / (1024 * 1024),
                'max_mb': max(benchmark.memory_usage) / (1024 * 1024),
                'mean_mb': statistics.mean(benchmark.memory_usage) / (1024 * 1024),
                'classification': self._classify_memory_usage(statistics.mean(benchmark.memory_usage))
            }
        
        # Output size metrics
        if benchmark.output_sizes:
            total_output = sum(benchmark.output_sizes)
            metrics['output_metrics'] = {
                'file_count': len(benchmark.output_sizes),
                'total_bytes': total_output,
                'total_mb': total_output / (1024 * 1024),
                'average_file_size_bytes': statistics.mean(benchmark.output_sizes),
                'average_file_size_mb': statistics.mean(benchmark.output_sizes) / (1024 * 1024),
                'largest_file_bytes': max(benchmark.output_sizes),
                'smallest_file_bytes': min(benchmark.output_sizes),
                'size_classification': self._classify_output_size(statistics.mean(benchmark.output_sizes))
            }
        
        return metrics
    
    def _calculate_quality_metrics(self, benchmark: LanguageBenchmark) -> Dict[str, Any]:
        """Calculate code quality and reliability metrics."""
        return {
            'reliability': {
                'success_rate': benchmark.success_rate,
                'failure_rate': (benchmark.failure_count / benchmark.total_attempts * 100) if benchmark.total_attempts > 0 else 0,
                'reliability_classification': self._classify_reliability(benchmark.success_rate)
            },
            'issues': {
                'total_warnings': benchmark.warning_count,
                'total_errors': benchmark.error_count,
                'warnings_per_attempt': benchmark.warning_count / benchmark.total_attempts if benchmark.total_attempts > 0 else 0,
                'errors_per_attempt': benchmark.error_count / benchmark.total_attempts if benchmark.total_attempts > 0 else 0,
                'issue_severity': self._classify_issue_severity(benchmark.warning_count, benchmark.error_count)
            }
        }
    
    def _classify_language_performance(self, benchmark: LanguageBenchmark) -> Dict[str, str]:
        """Classify overall language performance across multiple dimensions."""
        avg_time = benchmark.average_execution_time
        success_rate = benchmark.success_rate
        
        # Time classification
        time_class = 'very_slow'
        for threshold_name, threshold_value in self.performance_thresholds['execution_time'].items():
            if avg_time < threshold_value:
                time_class = threshold_name
                break
        
        # Reliability classification
        reliability_class = self._classify_reliability(success_rate)
        
        # Overall classification
        if time_class in ['excellent', 'good'] and reliability_class == 'excellent':
            overall_class = 'excellent'
        elif time_class in ['excellent', 'good', 'acceptable'] and reliability_class in ['excellent', 'good']:
            overall_class = 'good'
        elif reliability_class in ['excellent', 'good']:
            overall_class = 'acceptable'
        else:
            overall_class = 'poor'
        
        return {
            'execution_time': time_class,
            'reliability': reliability_class,
            'overall': overall_class
        }
    
    def _classify_memory_usage(self, avg_memory_bytes: float) -> str:
        """Classify memory usage level."""
        thresholds = self.performance_thresholds['memory_usage']
        
        if avg_memory_bytes < thresholds['low']:
            return 'low'
        elif avg_memory_bytes < thresholds['moderate']:
            return 'moderate'
        elif avg_memory_bytes < thresholds['high']:
            return 'high'
        else:
            return 'very_high'
    
    def _classify_output_size(self, avg_size_bytes: float) -> str:
        """Classify output file size."""
        thresholds = self.performance_thresholds['output_size']
        
        if avg_size_bytes < thresholds['small']:
            return 'small'
        elif avg_size_bytes < thresholds['medium']:
            return 'medium'
        elif avg_size_bytes < thresholds['large']:
            return 'large'
        else:
            return 'very_large'
    
    def _classify_reliability(self, success_rate: float) -> str:
        """Classify reliability based on success rate."""
        if success_rate >= 95:
            return 'excellent'
        elif success_rate >= 80:
            return 'good'
        elif success_rate >= 60:
            return 'acceptable'
        elif success_rate >= 30:
            return 'poor'
        else:
            return 'very_poor'
    
    def _classify_issue_severity(self, warning_count: int, error_count: int) -> str:
        """Classify overall issue severity."""
        if error_count == 0 and warning_count <= 2:
            return 'low'
        elif error_count <= 1 and warning_count <= 5:
            return 'moderate'
        elif error_count <= 3 and warning_count <= 10:
            return 'high'
        else:
            return 'critical'
    
    def _analyze_performance_trends(self, benchmark: LanguageBenchmark) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        if len(benchmark.execution_times) < 3:
            return {'trend': 'insufficient_data'}
        
        # Simple trend analysis using linear regression concept
        times = benchmark.execution_times
        n = len(times)
        x_values = list(range(n))
        
        # Calculate trend direction
        x_mean = statistics.mean(x_values)
        y_mean = statistics.mean(times)
        
        numerator = sum((x_values[i] - x_mean) * (times[i] - y_mean) for i in range(n))
        denominator = sum((x_values[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            slope = 0
        else:
            slope = numerator / denominator
        
        # Classify trend
        if abs(slope) < 0.1:
            trend = 'stable'
        elif slope > 0:
            trend = 'degrading'
        else:
            trend = 'improving'
        
        return {
            'trend': trend,
            'slope': slope,
            'consistency': self._calculate_consistency(times),
            'volatility': statistics.stdev(times) / statistics.mean(times) if statistics.mean(times) > 0 else 0
        }
    
    def _calculate_consistency(self, values: List[float]) -> str:
        """Calculate consistency of performance values."""
        if len(values) < 2:
            return 'unknown'
        
        cv = statistics.stdev(values) / statistics.mean(values) if statistics.mean(values) > 0 else float('inf')
        
        if cv < 0.1:
            return 'very_consistent'
        elif cv < 0.2:
            return 'consistent'
        elif cv < 0.5:
            return 'moderate'
        else:
            return 'inconsistent'
    
    def generate_comparative_benchmark_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive comparative benchmark report across all languages.
        
        Returns:
            Dictionary containing detailed comparative analysis
        """
        if not self.benchmarks:
            return {}
        
        # Generate individual language benchmarks
        language_benchmarks = {}
        for language in self.benchmarks.keys():
            language_benchmarks[language] = self.benchmark_language_performance(language)
        
        # Comparative analysis
        comparative_analysis = {
            'summary': self._generate_benchmark_summary(language_benchmarks),
            'rankings': self._generate_comprehensive_rankings(language_benchmarks),
            'correlations': self._analyze_performance_correlations(language_benchmarks),
            'recommendations': self._generate_benchmark_recommendations(language_benchmarks),
            'language_details': language_benchmarks
        }
        
        return comparative_analysis
    
    def _generate_benchmark_summary(self, benchmarks: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate high-level benchmark summary."""
        total_languages = len(benchmarks)
        successful_languages = sum(1 for b in benchmarks.values() 
                                 if b.get('success_metrics', {}).get('success_rate_percent', 0) > 0)
        
        all_times = []
        all_success_rates = []
        
        for benchmark in benchmarks.values():
            timing = benchmark.get('timing_metrics', {})
            success = benchmark.get('success_metrics', {})
            
            if timing.get('mean_seconds'):
                all_times.append(timing['mean_seconds'])
            if success.get('success_rate_percent') is not None:
                all_success_rates.append(success['success_rate_percent'])
        
        return {
            'total_languages_tested': total_languages,
            'languages_with_success': successful_languages,
            'overall_success_rate': statistics.mean(all_success_rates) if all_success_rates else 0,
            'average_execution_time': statistics.mean(all_times) if all_times else 0,
            'fastest_average_time': min(all_times) if all_times else 0,
            'slowest_average_time': max(all_times) if all_times else 0,
            'performance_variance': statistics.stdev(all_times) if len(all_times) > 1 else 0
        }
    
    def _generate_comprehensive_rankings(self, benchmarks: Dict[str, Dict[str, Any]]) -> Dict[str, List[Tuple[str, Any]]]:
        """Generate comprehensive rankings across multiple metrics."""
        rankings = {}
        
        # Performance rankings (execution time - lower is better)
        performance_data = []
        for lang, benchmark in benchmarks.items():
            timing = benchmark.get('timing_metrics', {})
            if timing.get('mean_seconds') is not None:
                performance_data.append((lang, timing['mean_seconds']))
        
        rankings['execution_time'] = sorted(performance_data, key=lambda x: x[1])
        
        # Reliability rankings (success rate - higher is better)
        reliability_data = []
        for lang, benchmark in benchmarks.items():
            success = benchmark.get('success_metrics', {})
            if success.get('success_rate_percent') is not None:
                reliability_data.append((lang, success['success_rate_percent']))
        
        rankings['reliability'] = sorted(reliability_data, key=lambda x: x[1], reverse=True)
        
        # Memory efficiency rankings (lower is better)
        memory_data = []
        for lang, benchmark in benchmarks.items():
            resource = benchmark.get('resource_metrics', {}).get('memory_usage', {})
            if resource.get('mean_mb') is not None:
                memory_data.append((lang, resource['mean_mb']))
        
        rankings['memory_efficiency'] = sorted(memory_data, key=lambda x: x[1])
        
        # Output size rankings
        output_data = []
        for lang, benchmark in benchmarks.items():
            resource = benchmark.get('resource_metrics', {}).get('output_metrics', {})
            if resource.get('average_file_size_mb') is not None:
                output_data.append((lang, resource['average_file_size_mb']))
        
        rankings['output_size'] = sorted(output_data, key=lambda x: x[1])
        
        return rankings
    
    def _analyze_performance_correlations(self, benchmarks: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze correlations between different performance metrics."""
        # Extract data for correlation analysis
        execution_times = []
        success_rates = []
        memory_usage = []
        output_sizes = []
        languages = []
        
        for lang, benchmark in benchmarks.items():
            timing = benchmark.get('timing_metrics', {})
            success = benchmark.get('success_metrics', {})
            resource = benchmark.get('resource_metrics', {})
            
            if (timing.get('mean_seconds') is not None and 
                success.get('success_rate_percent') is not None):
                
                languages.append(lang)
                execution_times.append(timing['mean_seconds'])
                success_rates.append(success['success_rate_percent'])
                
                memory = resource.get('memory_usage', {}).get('mean_mb', 0)
                memory_usage.append(memory)
                
                output = resource.get('output_metrics', {}).get('average_file_size_mb', 0)
                output_sizes.append(output)
        
        correlations = {}
        
        if len(execution_times) > 2:
            # Simple correlation analysis
            correlations['time_vs_reliability'] = self._calculate_correlation(execution_times, success_rates)
            
            if any(m > 0 for m in memory_usage):
                correlations['time_vs_memory'] = self._calculate_correlation(execution_times, memory_usage)
            
            if any(o > 0 for o in output_sizes):
                correlations['time_vs_output_size'] = self._calculate_correlation(execution_times, output_sizes)
        
        return correlations
    
    def _calculate_correlation(self, x_values: List[float], y_values: List[float]) -> Dict[str, Any]:
        """Calculate correlation coefficient between two datasets."""
        if len(x_values) != len(y_values) or len(x_values) < 2:
            return {'correlation': 0, 'strength': 'insufficient_data'}
        
        n = len(x_values)
        x_mean = statistics.mean(x_values)
        y_mean = statistics.mean(y_values)
        
        numerator = sum((x_values[i] - x_mean) * (y_values[i] - y_mean) for i in range(n))
        x_variance = sum((x_values[i] - x_mean) ** 2 for i in range(n))
        y_variance = sum((y_values[i] - y_mean) ** 2 for i in range(n))
        
        denominator = (x_variance * y_variance) ** 0.5
        
        if denominator == 0:
            correlation = 0
        else:
            correlation = numerator / denominator
        
        # Classify correlation strength
        abs_corr = abs(correlation)
        if abs_corr < 0.1:
            strength = 'negligible'
        elif abs_corr < 0.3:
            strength = 'weak'
        elif abs_corr < 0.5:
            strength = 'moderate'
        elif abs_corr < 0.7:
            strength = 'strong'
        else:
            strength = 'very_strong'
        
        return {
            'correlation': correlation,
            'strength': strength,
            'direction': 'positive' if correlation > 0 else 'negative' if correlation < 0 else 'none'
        }
    
    def _generate_benchmark_recommendations(self, benchmarks: Dict[str, Dict[str, Any]]) -> List[str]:
        """Generate actionable recommendations based on benchmark analysis."""
        recommendations = []
        
        # Analyze overall patterns
        slow_languages = []
        unreliable_languages = []
        memory_intensive_languages = []
        
        for lang, benchmark in benchmarks.items():
            timing = benchmark.get('timing_metrics', {})
            success = benchmark.get('success_metrics', {})
            resource = benchmark.get('resource_metrics', {})
            classification = benchmark.get('performance_classification', {})
            
            # Identify problematic languages
            if classification.get('execution_time') in ['slow', 'very_slow']:
                slow_languages.append(lang)
            
            if classification.get('reliability') in ['poor', 'very_poor']:
                unreliable_languages.append(lang)
            
            memory_class = resource.get('memory_usage', {}).get('classification')
            if memory_class in ['high', 'very_high']:
                memory_intensive_languages.append(lang)
        
        # Generate specific recommendations
        if slow_languages:
            recommendations.append(f"Optimize performance for slow languages: {', '.join(slow_languages)}")
        
        if unreliable_languages:
            recommendations.append(f"Investigate reliability issues for: {', '.join(unreliable_languages)}")
        
        if memory_intensive_languages:
            recommendations.append(f"Consider memory optimization for: {', '.join(memory_intensive_languages)}")
        
        # General recommendations
        total_languages = len(benchmarks)
        successful_count = sum(1 for b in benchmarks.values() 
                             if b.get('success_metrics', {}).get('success_rate_percent', 0) > 80)
        
        if successful_count / total_languages < 0.7:
            recommendations.append("Overall success rate is below 70% - review common failure patterns")
        
        if not recommendations:
            recommendations.append("Performance benchmarks look good - consider testing with larger datasets")
        
        return recommendations
    
    def export_metrics_data(self, output_path: Path, format_type: str = 'json') -> bool:
        """
        Export collected metrics data to file.
        
        Args:
            output_path: Path where to save the metrics data
            format_type: Export format ('json', 'csv')
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            if format_type.lower() == 'json':
                # Generate comprehensive report
                report_data = {
                    'metadata': {
                        'export_timestamp': time.time(),
                        'total_languages': len(self.benchmarks),
                        'format_version': '1.0'
                    },
                    'benchmarks': {lang: {
                        'language': benchmark.language,
                        'execution_times': benchmark.execution_times,
                        'memory_usage': benchmark.memory_usage,
                        'success_count': benchmark.success_count,
                        'failure_count': benchmark.failure_count,
                        'warning_count': benchmark.warning_count,
                        'error_count': benchmark.error_count,
                        'output_sizes': benchmark.output_sizes,
                        'success_rate': benchmark.success_rate,
                        'average_execution_time': benchmark.average_execution_time
                    } for lang, benchmark in self.benchmarks.items()},
                    'performance_analysis': self.analyze_performance_patterns(),
                    'comparative_report': self.generate_comparative_benchmark_report()
                }
                
                with open(output_path, 'w') as f:
                    json.dump(report_data, f, indent=2, default=str)
                
                self.logger.info(f"Metrics data exported to {output_path}")
                return True
                
            else:
                self.logger.error(f"Unsupported export format: {format_type}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to export metrics data: {e}")
            return False
    
    def _get_fastest_language(self) -> Optional[str]:
        """Get the language with fastest average execution time."""
        rankings = self._rank_languages_by_performance()
        return rankings[0][0] if rankings else None
    
    def _get_most_reliable_language(self) -> Optional[str]:
        """Get the language with highest success rate."""
        rankings = self._rank_languages_by_reliability()
        return rankings[0][0] if rankings else None
    
    def _get_average_time_per_language(self) -> float:
        """Get average execution time across all languages."""
        all_times = []
        for benchmark in self.benchmarks.values():
            all_times.extend(benchmark.execution_times)
        
        return statistics.mean(all_times) if all_times else 0.0
    
    def _correlate_performance_reliability(self) -> Dict[str, Any]:
        """Analyze correlation between performance and reliability."""
        # This is a simplified correlation analysis
        fast_and_reliable = []
        slow_but_reliable = []
        fast_but_unreliable = []
        
        for language, benchmark in self.benchmarks.items():
            if not benchmark.execution_times or benchmark.total_attempts == 0:
                continue
                
            avg_time = benchmark.average_execution_time
            success_rate = benchmark.success_rate
            
            if avg_time < 10 and success_rate > 80:
                fast_and_reliable.append(language)
            elif avg_time > 30 and success_rate > 80:
                slow_but_reliable.append(language)
            elif avg_time < 10 and success_rate < 50:
                fast_but_unreliable.append(language)
        
        return {
            'fast_and_reliable': fast_and_reliable,
            'slow_but_reliable': slow_but_reliable,
            'fast_but_unreliable': fast_but_unreliable
        }
    
    def _identify_performance_outliers(self) -> Dict[str, List[str]]:
        """Identify performance outliers."""
        all_times = []
        for benchmark in self.benchmarks.values():
            all_times.extend(benchmark.execution_times)
        
        if len(all_times) < 2:
            return {}
        
        mean_time = statistics.mean(all_times)
        std_dev = statistics.stdev(all_times)
        
        outliers = {
            'unusually_fast': [],
            'unusually_slow': []
        }
        
        for language, benchmark in self.benchmarks.items():
            if benchmark.execution_times:
                avg_time = benchmark.average_execution_time
                
                if avg_time < mean_time - 2 * std_dev:
                    outliers['unusually_fast'].append(language)
                elif avg_time > mean_time + 2 * std_dev:
                    outliers['unusually_slow'].append(language)
        
        return outliers
    
    @staticmethod
    def _percentile(data: List[float], percentile: float) -> float:
        """Calculate percentile of a dataset."""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))