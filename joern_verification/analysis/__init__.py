"""
Analysis module for Joern multi-language verification system.

This module provides comprehensive analysis capabilities including result categorization,
performance metrics collection, error pattern analysis, and warning detection.
"""

from .result_analyzer import ResultAnalyzer, ErrorPattern, PerformanceMetrics
from .metrics_collector import (
    MetricsCollector, 
    LanguageBenchmark, 
    ErrorAnalysis, 
    WarningAnalysis
)

__all__ = [
    'ResultAnalyzer',
    'ErrorPattern', 
    'PerformanceMetrics',
    'MetricsCollector',
    'LanguageBenchmark',
    'ErrorAnalysis',
    'WarningAnalysis'
]