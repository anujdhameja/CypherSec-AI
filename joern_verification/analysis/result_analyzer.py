"""
Results analysis system for CPG generation outcomes.

This module provides comprehensive analysis of CPG generation results,
including categorization, error parsing, and performance metrics collection.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Pattern
from pathlib import Path

from ..core.interfaces import GenerationResult, AnalysisReport, ResultsAnalyzerInterface


@dataclass
class ErrorPattern:
    """Pattern for matching and categorizing error messages."""
    pattern: Pattern[str]
    category: str
    severity: str
    description: str


@dataclass
class PerformanceMetrics:
    """Performance metrics for CPG generation."""
    execution_time: float
    memory_usage: Optional[int] = None
    output_file_count: int = 0
    output_total_size: int = 0
    warnings_count: int = 0
    errors_count: int = 0


class ResultAnalyzer(ResultsAnalyzerInterface):
    """Analyzer for CPG generation results with comprehensive categorization."""
    
    # Result categories
    CATEGORY_SUCCESS = "success"
    CATEGORY_SUCCESS_WITH_WARNINGS = "success_with_warnings"
    CATEGORY_PARTIAL_SUCCESS = "partial_success"
    CATEGORY_FAILURE = "failure"
    CATEGORY_TOOL_MISSING = "tool_missing"
    CATEGORY_TIMEOUT = "timeout"
    CATEGORY_MEMORY_ERROR = "memory_error"
    CATEGORY_SYNTAX_ERROR = "syntax_error"
    
    def __init__(self):
        """Initialize the result analyzer with error patterns."""
        self.logger = logging.getLogger(__name__)
        self.error_patterns = self._initialize_error_patterns()
        
    def _initialize_error_patterns(self) -> List[ErrorPattern]:
        """
        Initialize error patterns for categorizing different types of failures.
        
        Returns:
            List of ErrorPattern objects for matching errors
        """
        patterns = [
            # Memory-related errors
            ErrorPattern(
                pattern=re.compile(r'OutOfMemoryError|out of memory|heap space', re.IGNORECASE),
                category="memory_error",
                severity="high",
                description="Insufficient memory for CPG generation"
            ),
            ErrorPattern(
                pattern=re.compile(r'java\.lang\.OutOfMemoryError', re.IGNORECASE),
                category="memory_error", 
                severity="high",
                description="JVM out of memory error"
            ),
            
            # Timeout-related errors
            ErrorPattern(
                pattern=re.compile(r'timeout|timed out|time limit exceeded', re.IGNORECASE),
                category="timeout",
                severity="medium",
                description="Command execution timeout"
            ),
            
            # Syntax and parsing errors
            ErrorPattern(
                pattern=re.compile(r'syntax error|parse error|parsing failed', re.IGNORECASE),
                category="syntax_error",
                severity="medium",
                description="Source code syntax or parsing error"
            ),
            ErrorPattern(
                pattern=re.compile(r'compilation failed|compile error', re.IGNORECASE),
                category="syntax_error",
                severity="medium", 
                description="Source code compilation error"
            ),
            
            # Tool availability errors
            ErrorPattern(
                pattern=re.compile(r'command not found|file not found|no such file', re.IGNORECASE),
                category="tool_missing",
                severity="high",
                description="Required tool or file not found"
            ),
            ErrorPattern(
                pattern=re.compile(r'permission denied|access denied', re.IGNORECASE),
                category="tool_missing",
                severity="high",
                description="Permission denied accessing tool or files"
            ),
            
            # Language-specific errors
            ErrorPattern(
                pattern=re.compile(r'unsupported language|language not supported', re.IGNORECASE),
                category="tool_missing",
                severity="high",
                description="Programming language not supported"
            ),
            
            # General failure patterns
            ErrorPattern(
                pattern=re.compile(r'failed to|error:|exception:', re.IGNORECASE),
                category="failure",
                severity="medium",
                description="General execution failure"
            ),
        ]
        
        return patterns
    
    def analyze_result(self, result: GenerationResult) -> AnalysisReport:
        """
        Analyze a CPG generation result and create comprehensive analysis report.
        
        Args:
            result: GenerationResult to analyze
            
        Returns:
            AnalysisReport with categorized results and metrics
        """
        self.logger.info(f"Analyzing result for {result.language}")
        
        # Categorize the outcome
        category = self.categorize_outcome(result)
        
        # Extract performance metrics
        metrics = self.extract_metrics(result)
        
        # Parse errors and warnings
        errors = self._parse_errors(result.stderr, result.error_message)
        warnings = self._parse_warnings(result.stderr, result.warnings or [])
        
        # Generate recommendations
        recommendations = self._generate_recommendations(result, category, errors, warnings)
        
        # Create analysis report
        report = AnalysisReport(
            language=result.language,
            category=category,
            metrics=metrics,
            errors=errors,
            warnings=warnings,
            recommendations=recommendations
        )
        
        self.logger.info(f"Analysis complete for {result.language}: {category}")
        return report
    
    def categorize_outcome(self, result: GenerationResult) -> str:
        """
        Categorize the outcome of a CPG generation attempt.
        
        Args:
            result: GenerationResult to categorize
            
        Returns:
            Category string indicating the type of outcome
        """
        # Check for timeout
        if hasattr(result, 'timeout_occurred') and result.timeout_occurred:
            return self.CATEGORY_TIMEOUT
            
        # Check for tool missing (command not found, etc.)
        if result.return_code == -1 or "not found" in result.stderr.lower():
            return self.CATEGORY_TOOL_MISSING
            
        # Check for specific error patterns
        error_text = f"{result.stderr} {result.error_message or ''}"
        for pattern in self.error_patterns:
            if pattern.pattern.search(error_text):
                return pattern.category
                
        # Check basic success/failure
        if not result.success:
            return self.CATEGORY_FAILURE
            
        # Check for output files
        if not result.output_files:
            return self.CATEGORY_FAILURE
            
        # Check for warnings
        warnings_count = len(result.warnings or [])
        if warnings_count > 0 or "warn" in result.stderr.lower():
            return self.CATEGORY_SUCCESS_WITH_WARNINGS
            
        # Check for partial success (some output but with issues)
        if result.output_files and result.stderr:
            # If we have output but also errors in stderr, it might be partial
            if any(keyword in result.stderr.lower() for keyword in ['error', 'failed', 'exception']):
                return self.CATEGORY_PARTIAL_SUCCESS
                
        # Full success
        return self.CATEGORY_SUCCESS
    
    def extract_metrics(self, result: GenerationResult) -> Dict[str, Any]:
        """
        Extract performance and quality metrics from a generation result.
        
        Args:
            result: GenerationResult to extract metrics from
            
        Returns:
            Dictionary of metrics and measurements
        """
        # Calculate output file metrics
        output_file_count = len(result.output_files)
        output_total_size = 0
        
        for file_path in result.output_files:
            try:
                if file_path.exists():
                    output_total_size += file_path.stat().st_size
            except (OSError, AttributeError):
                # Handle cases where file_path might not be a Path object
                pass
                
        # Count warnings and errors
        warnings_count = len(result.warnings or [])
        errors_count = self._count_errors_in_stderr(result.stderr)
        
        # Create performance metrics
        performance = PerformanceMetrics(
            execution_time=result.execution_time,
            memory_usage=result.memory_usage,
            output_file_count=output_file_count,
            output_total_size=output_total_size,
            warnings_count=warnings_count,
            errors_count=errors_count
        )
        
        # Convert to dictionary with additional computed metrics
        metrics = {
            'execution_time': performance.execution_time,
            'memory_usage': performance.memory_usage,
            'output_file_count': performance.output_file_count,
            'output_total_size': performance.output_total_size,
            'output_size_mb': round(performance.output_total_size / (1024 * 1024), 2),
            'warnings_count': performance.warnings_count,
            'errors_count': performance.errors_count,
            'success_rate': 1.0 if result.success else 0.0,
            'has_output': output_file_count > 0,
            'return_code': result.return_code,
            'stdout_length': len(result.stdout),
            'stderr_length': len(result.stderr)
        }
        
        # Add performance classification
        metrics['performance_class'] = self._classify_performance(performance.execution_time)
        
        return metrics
    
    def _parse_errors(self, stderr: str, error_message: Optional[str]) -> List[str]:
        """
        Parse and extract error messages from stderr and error_message.
        
        Args:
            stderr: Standard error output
            error_message: Additional error message
            
        Returns:
            List of parsed error messages
        """
        errors = []
        
        # Parse stderr for error lines
        if stderr:
            lines = stderr.split('\n')
            for line in lines:
                line = line.strip()
                if line and any(keyword in line.lower() for keyword in ['error', 'exception', 'failed']):
                    # Skip warning lines that contain 'error' in context
                    if not any(warn_keyword in line.lower() for warn_keyword in ['warn', 'warning']):
                        errors.append(line)
        
        # Add explicit error message
        if error_message and error_message not in errors:
            errors.append(error_message)
            
        return errors
    
    def _parse_warnings(self, stderr: str, existing_warnings: List[str]) -> List[str]:
        """
        Parse and extract warning messages from stderr.
        
        Args:
            stderr: Standard error output
            existing_warnings: Already extracted warnings
            
        Returns:
            List of parsed warning messages
        """
        warnings = list(existing_warnings)  # Copy existing warnings
        
        if stderr:
            lines = stderr.split('\n')
            for line in lines:
                line = line.strip()
                if line and any(keyword in line.lower() for keyword in ['warn', 'warning']):
                    if line not in warnings:
                        warnings.append(line)
                        
        return warnings
    
    def _count_errors_in_stderr(self, stderr: str) -> int:
        """
        Count the number of error messages in stderr.
        
        Args:
            stderr: Standard error output
            
        Returns:
            Number of error messages found
        """
        if not stderr:
            return 0
            
        error_count = 0
        lines = stderr.split('\n')
        
        for line in lines:
            line = line.strip()
            if line and any(keyword in line.lower() for keyword in ['error', 'exception', 'failed']):
                # Skip warning lines that contain 'error' in context
                if not any(warn_keyword in line.lower() for warn_keyword in ['warn', 'warning']):
                    error_count += 1
                    
        return error_count
    
    def _classify_performance(self, execution_time: float) -> str:
        """
        Classify performance based on execution time.
        
        Args:
            execution_time: Execution time in seconds
            
        Returns:
            Performance classification string
        """
        if execution_time < 5.0:
            return "fast"
        elif execution_time < 30.0:
            return "moderate"
        elif execution_time < 120.0:
            return "slow"
        else:
            return "very_slow"
    
    def _generate_recommendations(
        self,
        result: GenerationResult,
        category: str,
        errors: List[str],
        warnings: List[str]
    ) -> List[str]:
        """
        Generate actionable recommendations based on analysis results.
        
        Args:
            result: Original generation result
            category: Categorized outcome
            errors: Parsed error messages
            warnings: Parsed warning messages
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Category-specific recommendations
        if category == self.CATEGORY_TOOL_MISSING:
            recommendations.append(f"Install or configure the {result.language} frontend tool for Joern")
            recommendations.append("Verify Joern installation and PATH configuration")
            recommendations.append(f"Consider alternative CPG generation tools for {result.language}")
            
        elif category == self.CATEGORY_MEMORY_ERROR:
            recommendations.append("Increase JVM heap size (e.g., -J-Xmx8g)")
            recommendations.append("Use smaller input files for testing")
            recommendations.append("Consider processing files in batches")
            
        elif category == self.CATEGORY_TIMEOUT:
            recommendations.append("Increase timeout duration for complex files")
            recommendations.append("Use simpler test files to verify basic functionality")
            recommendations.append("Check system resources and performance")
            
        elif category == self.CATEGORY_SYNTAX_ERROR:
            recommendations.append("Verify test file syntax is correct for the language")
            recommendations.append("Try with different source code samples")
            recommendations.append("Check language version compatibility")
            
        elif category == self.CATEGORY_FAILURE:
            recommendations.append("Review error messages for specific issues")
            recommendations.append("Verify input file format and content")
            recommendations.append("Check Joern tool configuration")
            
        elif category == self.CATEGORY_PARTIAL_SUCCESS:
            recommendations.append("Review warnings to understand limitations")
            recommendations.append("Validate generated CPG content")
            recommendations.append("Consider if partial results meet requirements")
            
        elif category == self.CATEGORY_SUCCESS_WITH_WARNINGS:
            recommendations.append("Review warnings for potential issues")
            recommendations.append("Validate CPG quality and completeness")
            
        elif category == self.CATEGORY_SUCCESS:
            recommendations.append("CPG generation successful - ready for analysis")
            recommendations.append("Consider testing with larger/more complex files")
        
        # Performance-based recommendations
        if result.execution_time > 60.0:
            recommendations.append("Consider optimizing for better performance")
            recommendations.append("Monitor system resources during execution")
            
        # Warning-based recommendations
        if len(warnings) > 5:
            recommendations.append("High number of warnings - review for quality issues")
            
        return recommendations
    
    def get_category_summary(self, analyses: List[AnalysisReport]) -> Dict[str, int]:
        """
        Get summary of categories across multiple analysis reports.
        
        Args:
            analyses: List of AnalysisReport objects
            
        Returns:
            Dictionary mapping categories to counts
        """
        category_counts = {}
        
        for analysis in analyses:
            category = analysis.category
            category_counts[category] = category_counts.get(category, 0) + 1
            
        return category_counts
    
    def get_performance_summary(self, analyses: List[AnalysisReport]) -> Dict[str, Any]:
        """
        Get performance summary across multiple analysis reports.
        
        Args:
            analyses: List of AnalysisReport objects
            
        Returns:
            Dictionary with performance statistics
        """
        if not analyses:
            return {}
            
        execution_times = []
        success_count = 0
        total_warnings = 0
        total_errors = 0
        
        for analysis in analyses:
            metrics = analysis.metrics
            execution_times.append(metrics.get('execution_time', 0))
            
            if analysis.category in [self.CATEGORY_SUCCESS, self.CATEGORY_SUCCESS_WITH_WARNINGS]:
                success_count += 1
                
            total_warnings += metrics.get('warnings_count', 0)
            total_errors += metrics.get('errors_count', 0)
        
        return {
            'total_languages': len(analyses),
            'successful_languages': success_count,
            'success_rate': success_count / len(analyses) if analyses else 0,
            'average_execution_time': sum(execution_times) / len(execution_times) if execution_times else 0,
            'min_execution_time': min(execution_times) if execution_times else 0,
            'max_execution_time': max(execution_times) if execution_times else 0,
            'total_warnings': total_warnings,
            'total_errors': total_errors,
            'average_warnings_per_language': total_warnings / len(analyses) if analyses else 0
        }