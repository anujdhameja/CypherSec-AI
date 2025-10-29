"""
Data models for alternative tool recommendation system.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any


class InstallationMethod(Enum):
    """Installation methods for alternative tools."""
    PIP = "pip"
    NPM = "npm"
    CARGO = "cargo"
    GO_GET = "go_get"
    MANUAL = "manual"
    SYSTEM_PACKAGE = "system_package"
    BINARY_DOWNLOAD = "binary_download"


class ToolCapability(Enum):
    """Capabilities that alternative tools can provide."""
    AST_GENERATION = "ast_generation"
    CPG_GENERATION = "cpg_generation"
    SYNTAX_PARSING = "syntax_parsing"
    SEMANTIC_ANALYSIS = "semantic_analysis"
    CONTROL_FLOW = "control_flow"
    DATA_FLOW = "data_flow"
    CALL_GRAPH = "call_graph"
    DEPENDENCY_ANALYSIS = "dependency_analysis"


@dataclass
class InstallationInstruction:
    """Installation instructions for an alternative tool."""
    method: InstallationMethod
    command: str
    description: str
    prerequisites: List[str] = field(default_factory=list)
    post_install_steps: List[str] = field(default_factory=list)
    verification_command: Optional[str] = None


@dataclass
class UsageExample:
    """Usage example for an alternative tool."""
    description: str
    command: str
    input_example: str
    expected_output_format: str
    notes: Optional[str] = None


@dataclass
class ToolComparison:
    """Comparison between Joern and an alternative tool."""
    feature: str
    joern_support: str  # "full", "partial", "none"
    alternative_support: str  # "full", "partial", "none"
    notes: Optional[str] = None


@dataclass
class AlternativeTool:
    """Represents an alternative tool for CPG/AST generation."""
    name: str
    description: str
    supported_languages: List[str]
    capabilities: List[ToolCapability]
    installation: List[InstallationInstruction]
    usage_examples: List[UsageExample]
    comparison_with_joern: List[ToolComparison]
    official_website: Optional[str] = None
    documentation_url: Optional[str] = None
    github_url: Optional[str] = None
    license: Optional[str] = None
    maturity_level: str = "stable"  # "experimental", "beta", "stable", "mature"
    last_updated: Optional[str] = None
    pros: List[str] = field(default_factory=list)
    cons: List[str] = field(default_factory=list)
    integration_complexity: str = "medium"  # "low", "medium", "high"


@dataclass
class LanguageAlternatives:
    """Alternative tools available for a specific language."""
    language: str
    primary_alternatives: List[AlternativeTool]
    secondary_alternatives: List[AlternativeTool]
    recommended_tool: Optional[str] = None
    joern_support_status: str = "unknown"  # "full", "partial", "limited", "none", "unknown"
    notes: Optional[str] = None