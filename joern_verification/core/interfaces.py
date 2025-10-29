"""
Core interfaces for the Joern multi-language verification system.

This module defines the base interfaces that all components must implement
to ensure consistent behavior across the verification workflow.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from pathlib import Path


@dataclass
class GenerationResult:
    """Result of CPG generation for a specific language."""
    language: str
    input_file: str
    output_dir: str
    success: bool
    execution_time: float
    memory_usage: Optional[int]
    stdout: str
    stderr: str
    return_code: int
    output_files: List[Path]
    warnings: Optional[List[str]] = None
    error_message: Optional[str] = None
    timeout_occurred: bool = False


@dataclass
class AnalysisReport:
    """Analysis report for a language verification test."""
    language: str
    category: str  # success, success_with_warnings, partial_success, failure, tool_missing
    metrics: Dict[str, Any]
    errors: List[str]
    warnings: List[str]
    recommendations: List[str]


@dataclass
class LanguageSupport:
    """Information about language support in Joern."""
    language: str
    tool_available: bool
    tool_path: str
    supported: bool
    alternative_tools: List[str]
    notes: str


class LanguageDiscoveryInterface(ABC):
    """Interface for discovering available language frontends in Joern."""
    
    @abstractmethod
    def discover_languages(self) -> List[LanguageSupport]:
        """
        Discover all available language frontends in the Joern installation.
        
        Returns:
            List of LanguageSupport objects describing available languages
        """
        pass
    
    @abstractmethod
    def get_tool_path(self, language: str) -> Optional[str]:
        """
        Get the path to the tool for a specific language.
        
        Args:
            language: Programming language name
            
        Returns:
            Path to the language tool or None if not available
        """
        pass
    
    @abstractmethod
    def validate_tool_availability(self, language: str) -> bool:
        """
        Validate that a language tool is available and executable.
        
        Args:
            language: Programming language name
            
        Returns:
            True if tool is available and executable, False otherwise
        """
        pass


class TestFileGeneratorInterface(ABC):
    """Interface for generating test source files for different languages."""
    
    @abstractmethod
    def generate_test_file(self, language: str, output_path: Path) -> bool:
        """
        Generate a test source file for the specified language.
        
        Args:
            language: Programming language name
            output_path: Path where the test file should be created
            
        Returns:
            True if file was generated successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def get_test_content(self, language: str) -> str:
        """
        Get the test content template for a specific language.
        
        Args:
            language: Programming language name
            
        Returns:
            Test source code content for the language
        """
        pass
    
    @abstractmethod
    def validate_syntax(self, language: str, content: str) -> bool:
        """
        Validate that the generated content has correct syntax.
        
        Args:
            language: Programming language name
            content: Source code content to validate
            
        Returns:
            True if syntax is valid, False otherwise
        """
        pass


class CPGGenerationInterface(ABC):
    """Interface for executing CPG generation using Joern tools."""
    
    @abstractmethod
    def generate_cpg(self, language: str, input_file: Path, output_dir: Path) -> GenerationResult:
        """
        Generate CPG for a source file using the appropriate Joern tool.
        
        Args:
            language: Programming language name
            input_file: Path to the source file
            output_dir: Directory where CPG should be generated
            
        Returns:
            GenerationResult containing execution details and outcomes
        """
        pass
    
    @abstractmethod
    def get_command_template(self, language: str) -> str:
        """
        Get the command template for CPG generation for a specific language.
        
        Args:
            language: Programming language name
            
        Returns:
            Command template string with placeholders
        """
        pass
    
    @abstractmethod
    def execute_command(self, command: str, timeout: int = 300) -> GenerationResult:
        """
        Execute a CPG generation command with proper resource management.
        
        Args:
            command: Command to execute
            timeout: Maximum execution time in seconds
            
        Returns:
            GenerationResult containing execution details
        """
        pass


class ResultsAnalyzerInterface(ABC):
    """Interface for analyzing CPG generation results."""
    
    @abstractmethod
    def analyze_result(self, result: GenerationResult) -> AnalysisReport:
        """
        Analyze a CPG generation result and categorize the outcome.
        
        Args:
            result: GenerationResult to analyze
            
        Returns:
            AnalysisReport with categorized results and metrics
        """
        pass
    
    @abstractmethod
    def categorize_outcome(self, result: GenerationResult) -> str:
        """
        Categorize the outcome of a CPG generation attempt.
        
        Args:
            result: GenerationResult to categorize
            
        Returns:
            Category string (success, failure, etc.)
        """
        pass
    
    @abstractmethod
    def extract_metrics(self, result: GenerationResult) -> Dict[str, Any]:
        """
        Extract performance and quality metrics from a generation result.
        
        Args:
            result: GenerationResult to extract metrics from
            
        Returns:
            Dictionary of metrics and measurements
        """
        pass


class ReportGeneratorInterface(ABC):
    """Interface for generating verification reports."""
    
    @abstractmethod
    def generate_report(self, analyses: List[AnalysisReport], output_path: Path) -> bool:
        """
        Generate a comprehensive verification report.
        
        Args:
            analyses: List of AnalysisReport objects to include
            output_path: Path where the report should be saved
            
        Returns:
            True if report was generated successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def format_summary(self, analyses: List[AnalysisReport]) -> str:
        """
        Format a summary of all analysis results.
        
        Args:
            analyses: List of AnalysisReport objects to summarize
            
        Returns:
            Formatted summary string
        """
        pass