"""
Data models for report generation.

This module defines the data structures used for organizing and presenting
verification results in various report formats.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path

from ..core.interfaces import AnalysisReport


@dataclass
class ReportMetadata:
    """Metadata for verification reports."""
    title: str
    generated_at: datetime
    joern_version: Optional[str] = None
    system_info: Optional[Dict[str, str]] = None
    total_languages: int = 0
    successful_languages: int = 0
    failed_languages: int = 0
    execution_time: float = 0.0
    report_version: str = "1.0"


@dataclass
class LanguageSummary:
    """Summary information for a specific language."""
    language: str
    status: str  # success, failure, partial, etc.
    execution_time: float
    success_rate: float
    error_count: int
    warning_count: int
    output_files: int
    recommendations: List[str]
    category: str
    performance_class: str


@dataclass
class ReportSection:
    """A section within a verification report."""
    title: str
    content: Union[str, Dict[str, Any], List[Any]]
    section_type: str  # summary, details, analysis, recommendations
    priority: int = 0  # Higher priority sections appear first
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class PerformanceAnalysis:
    """Performance analysis data for reports."""
    overall_success_rate: float
    average_execution_time: float
    fastest_language: Optional[str]
    slowest_language: Optional[str]
    most_reliable_language: Optional[str]
    least_reliable_language: Optional[str]
    performance_rankings: List[tuple]
    reliability_rankings: List[tuple]
    resource_usage: Dict[str, Any]
    trends: Dict[str, Any]


@dataclass
class ErrorAnalysisSummary:
    """Error analysis summary for reports."""
    total_errors: int
    error_categories: Dict[str, int]
    common_errors: List[tuple]
    languages_with_errors: List[str]
    resolution_suggestions: Dict[str, List[str]]
    critical_issues: List[str]


@dataclass
class RecommendationSet:
    """Set of recommendations organized by category."""
    immediate_actions: List[str]
    performance_improvements: List[str]
    reliability_enhancements: List[str]
    alternative_tools: Dict[str, List[str]]
    next_steps: List[str]
    language_specific: Dict[str, List[str]]


@dataclass
class VerificationReport:
    """Complete verification report structure."""
    metadata: ReportMetadata
    executive_summary: ReportSection
    language_summaries: List[LanguageSummary]
    performance_analysis: PerformanceAnalysis
    error_analysis: ErrorAnalysisSummary
    recommendations: RecommendationSet
    detailed_results: List[AnalysisReport]
    sections: List[ReportSection] = field(default_factory=list)
    
    def add_section(self, section: ReportSection) -> None:
        """Add a section to the report."""
        self.sections.append(section)
        # Sort sections by priority (higher priority first)
        self.sections.sort(key=lambda s: s.priority, reverse=True)
    
    def get_section(self, title: str) -> Optional[ReportSection]:
        """Get a section by title."""
        for section in self.sections:
            if section.title == title:
                return section
        return None
    
    def get_sections_by_type(self, section_type: str) -> List[ReportSection]:
        """Get all sections of a specific type."""
        return [s for s in self.sections if s.section_type == section_type]


@dataclass
class ReportConfiguration:
    """Configuration for report generation."""
    include_detailed_results: bool = True
    include_performance_analysis: bool = True
    include_error_analysis: bool = True
    include_recommendations: bool = True
    include_charts: bool = False
    max_error_details: int = 50
    max_warning_details: int = 20
    group_by_category: bool = True
    sort_by_performance: bool = True
    output_formats: List[str] = field(default_factory=lambda: ['json', 'markdown'])
    custom_sections: List[str] = field(default_factory=list)
    
    def validate(self) -> List[str]:
        """Validate configuration and return any issues."""
        issues = []
        
        if not self.output_formats:
            issues.append("At least one output format must be specified")
        
        valid_formats = ['json', 'markdown', 'html', 'csv']
        for fmt in self.output_formats:
            if fmt not in valid_formats:
                issues.append(f"Invalid output format: {fmt}")
        
        if self.max_error_details < 0:
            issues.append("max_error_details must be non-negative")
        
        if self.max_warning_details < 0:
            issues.append("max_warning_details must be non-negative")
        
        return issues


@dataclass
class ChartData:
    """Data structure for charts and visualizations."""
    chart_type: str  # bar, pie, line, scatter
    title: str
    data: Dict[str, Any]
    labels: List[str]
    values: List[Union[int, float]]
    colors: Optional[List[str]] = None
    description: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chart data to dictionary for serialization."""
        return {
            'type': self.chart_type,
            'title': self.title,
            'data': self.data,
            'labels': self.labels,
            'values': self.values,
            'colors': self.colors,
            'description': self.description
        }


@dataclass
class TableData:
    """Data structure for tables in reports."""
    title: str
    headers: List[str]
    rows: List[List[Any]]
    column_types: Optional[List[str]] = None  # text, number, percentage, time
    sortable: bool = True
    filterable: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert table data to dictionary for serialization."""
        return {
            'title': self.title,
            'headers': self.headers,
            'rows': self.rows,
            'column_types': self.column_types,
            'sortable': self.sortable,
            'filterable': self.filterable
        }
    
    def add_row(self, row: List[Any]) -> None:
        """Add a row to the table."""
        if len(row) != len(self.headers):
            raise ValueError(f"Row length {len(row)} doesn't match headers length {len(self.headers)}")
        self.rows.append(row)
    
    def sort_by_column(self, column_index: int, reverse: bool = False) -> None:
        """Sort table by specified column."""
        if 0 <= column_index < len(self.headers):
            self.rows.sort(key=lambda row: row[column_index], reverse=reverse)


@dataclass
class ReportTemplate:
    """Template for customizing report structure and appearance."""
    name: str
    description: str
    sections: List[str]  # Ordered list of section names to include
    formatting_options: Dict[str, Any]
    custom_css: Optional[str] = None
    custom_js: Optional[str] = None
    
    @classmethod
    def get_default_template(cls) -> 'ReportTemplate':
        """Get the default report template."""
        return cls(
            name="default",
            description="Standard verification report template",
            sections=[
                "executive_summary",
                "language_overview", 
                "performance_analysis",
                "error_analysis",
                "recommendations",
                "detailed_results"
            ],
            formatting_options={
                "show_charts": True,
                "show_tables": True,
                "compact_mode": False,
                "color_scheme": "default"
            }
        )
    
    @classmethod
    def get_executive_template(cls) -> 'ReportTemplate':
        """Get a template focused on executive summary."""
        return cls(
            name="executive",
            description="Executive summary focused report",
            sections=[
                "executive_summary",
                "key_metrics",
                "recommendations"
            ],
            formatting_options={
                "show_charts": True,
                "show_tables": False,
                "compact_mode": True,
                "color_scheme": "professional"
            }
        )
    
    @classmethod
    def get_technical_template(cls) -> 'ReportTemplate':
        """Get a template focused on technical details."""
        return cls(
            name="technical",
            description="Technical details focused report",
            sections=[
                "language_overview",
                "performance_analysis", 
                "error_analysis",
                "detailed_results",
                "troubleshooting_guide"
            ],
            formatting_options={
                "show_charts": True,
                "show_tables": True,
                "compact_mode": False,
                "color_scheme": "technical"
            }
        )