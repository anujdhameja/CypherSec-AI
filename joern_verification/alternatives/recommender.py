"""
Recommendation engine for alternative CPG/AST generation tools.

This module provides intelligent recommendations for alternative tools
based on language support, capabilities, and integration requirements.
"""

from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from .database import AlternativeToolDatabase
from .models import AlternativeTool, LanguageAlternatives, ToolCapability


class RecommendationReason(Enum):
    """Reasons for recommending an alternative tool."""
    JOERN_NOT_SUPPORTED = "joern_not_supported"
    JOERN_LIMITED_SUPPORT = "joern_limited_support"
    JOERN_FAILED = "joern_failed"
    BETTER_LANGUAGE_SUPPORT = "better_language_support"
    ADDITIONAL_CAPABILITIES = "additional_capabilities"
    EASIER_INTEGRATION = "easier_integration"
    PERFORMANCE_BENEFITS = "performance_benefits"


@dataclass
class ToolRecommendation:
    """Represents a tool recommendation with reasoning."""
    tool: AlternativeTool
    confidence_score: float  # 0.0 to 1.0
    reason: RecommendationReason
    explanation: str
    integration_guidance: List[str]
    pros_for_use_case: List[str]
    cons_for_use_case: List[str]
    estimated_setup_time: str  # "minutes", "hours", "days"


@dataclass
class RecommendationContext:
    """Context information for generating recommendations."""
    language: str
    joern_support_status: str  # "full", "partial", "limited", "none", "failed"
    required_capabilities: List[ToolCapability]
    integration_complexity_preference: str  # "low", "medium", "high"
    project_type: str = "general"  # "research", "production", "educational", "general"
    time_constraints: str = "medium"  # "low", "medium", "high"
    team_expertise: str = "medium"  # "beginner", "medium", "expert"


class AlternativeToolRecommender:
    """Recommendation engine for alternative CPG/AST generation tools."""
    
    def __init__(self, database: Optional[AlternativeToolDatabase] = None):
        """
        Initialize the recommendation engine.
        
        Args:
            database: Alternative tool database (creates new if None)
        """
        self.database = database or AlternativeToolDatabase()
    
    def recommend_alternatives(
        self, 
        context: RecommendationContext
    ) -> List[ToolRecommendation]:
        """
        Generate tool recommendations based on context.
        
        Args:
            context: Recommendation context with requirements and constraints
            
        Returns:
            List of tool recommendations sorted by confidence score
        """
        recommendations = []
        
        # Get language-specific alternatives
        language_alternatives = self.database.get_alternatives_for_language(context.language)
        
        if not language_alternatives:
            # No specific alternatives defined, search by language support
            available_tools = self.database.search_tools_by_language(context.language)
            if available_tools:
                for tool in available_tools:
                    recommendation = self._create_generic_recommendation(tool, context)
                    recommendations.append(recommendation)
        else:
            # Use predefined language alternatives
            recommendations.extend(
                self._evaluate_primary_alternatives(language_alternatives, context)
            )
            recommendations.extend(
                self._evaluate_secondary_alternatives(language_alternatives, context)
            )
        
        # Sort by confidence score (descending)
        recommendations.sort(key=lambda x: x.confidence_score, reverse=True)
        
        return recommendations
    
    def _evaluate_primary_alternatives(
        self, 
        language_alternatives: LanguageAlternatives, 
        context: RecommendationContext
    ) -> List[ToolRecommendation]:
        """Evaluate primary alternative tools for a language."""
        recommendations = []
        
        for tool in language_alternatives.primary_alternatives:
            confidence_score = self._calculate_confidence_score(tool, context)
            reason = self._determine_recommendation_reason(context)
            
            recommendation = ToolRecommendation(
                tool=tool,
                confidence_score=confidence_score,
                reason=reason,
                explanation=self._generate_explanation(tool, context, reason),
                integration_guidance=self._generate_integration_guidance(tool, context),
                pros_for_use_case=self._identify_pros_for_use_case(tool, context),
                cons_for_use_case=self._identify_cons_for_use_case(tool, context),
                estimated_setup_time=self._estimate_setup_time(tool, context)
            )
            
            recommendations.append(recommendation)
        
        return recommendations
    
    def _evaluate_secondary_alternatives(
        self, 
        language_alternatives: LanguageAlternatives, 
        context: RecommendationContext
    ) -> List[ToolRecommendation]:
        """Evaluate secondary alternative tools for a language."""
        recommendations = []
        
        for tool in language_alternatives.secondary_alternatives:
            # Secondary alternatives get lower base confidence
            confidence_score = self._calculate_confidence_score(tool, context) * 0.7
            reason = self._determine_recommendation_reason(context)
            
            recommendation = ToolRecommendation(
                tool=tool,
                confidence_score=confidence_score,
                reason=reason,
                explanation=self._generate_explanation(tool, context, reason, is_secondary=True),
                integration_guidance=self._generate_integration_guidance(tool, context),
                pros_for_use_case=self._identify_pros_for_use_case(tool, context),
                cons_for_use_case=self._identify_cons_for_use_case(tool, context),
                estimated_setup_time=self._estimate_setup_time(tool, context)
            )
            
            recommendations.append(recommendation)
        
        return recommendations
    
    def _create_generic_recommendation(
        self, 
        tool: AlternativeTool, 
        context: RecommendationContext
    ) -> ToolRecommendation:
        """Create a generic recommendation for tools not in language-specific lists."""
        confidence_score = self._calculate_confidence_score(tool, context) * 0.5  # Lower confidence for generic
        reason = RecommendationReason.BETTER_LANGUAGE_SUPPORT
        
        return ToolRecommendation(
            tool=tool,
            confidence_score=confidence_score,
            reason=reason,
            explanation=f"{tool.name} supports {context.language} but may require additional configuration",
            integration_guidance=self._generate_integration_guidance(tool, context),
            pros_for_use_case=self._identify_pros_for_use_case(tool, context),
            cons_for_use_case=self._identify_cons_for_use_case(tool, context),
            estimated_setup_time=self._estimate_setup_time(tool, context)
        )
    
    def _calculate_confidence_score(
        self, 
        tool: AlternativeTool, 
        context: RecommendationContext
    ) -> float:
        """Calculate confidence score for a tool recommendation."""
        score = 0.5  # Base score
        
        # Language support bonus
        if context.language.lower() in [lang.lower() for lang in tool.supported_languages]:
            score += 0.2
        
        # Capability matching
        capability_match_ratio = len(
            set(context.required_capabilities) & set(tool.capabilities)
        ) / max(len(context.required_capabilities), 1)
        score += capability_match_ratio * 0.2
        
        # Integration complexity preference
        complexity_scores = {"low": 0.8, "medium": 0.6, "high": 0.4}
        if tool.integration_complexity in complexity_scores:
            if context.integration_complexity_preference == "low":
                score += (1.0 - complexity_scores[tool.integration_complexity]) * 0.1
            elif context.integration_complexity_preference == "high":
                score += complexity_scores[tool.integration_complexity] * 0.1
        
        # Maturity level bonus
        maturity_scores = {"experimental": 0.0, "beta": 0.05, "stable": 0.1, "mature": 0.15}
        score += maturity_scores.get(tool.maturity_level, 0.0)
        
        # Team expertise adjustment
        if context.team_expertise == "beginner" and tool.integration_complexity == "high":
            score -= 0.1
        elif context.team_expertise == "expert" and tool.integration_complexity == "low":
            score += 0.05
        
        return min(1.0, max(0.0, score))
    
    def _determine_recommendation_reason(self, context: RecommendationContext) -> RecommendationReason:
        """Determine the primary reason for recommending alternatives."""
        if context.joern_support_status == "none":
            return RecommendationReason.JOERN_NOT_SUPPORTED
        elif context.joern_support_status == "failed":
            return RecommendationReason.JOERN_FAILED
        elif context.joern_support_status in ["partial", "limited"]:
            return RecommendationReason.JOERN_LIMITED_SUPPORT
        else:
            return RecommendationReason.ADDITIONAL_CAPABILITIES
    
    def _generate_explanation(
        self, 
        tool: AlternativeTool, 
        context: RecommendationContext, 
        reason: RecommendationReason,
        is_secondary: bool = False
    ) -> str:
        """Generate explanation for why this tool is recommended."""
        base_explanation = f"{tool.name} is recommended for {context.language}"
        
        if is_secondary:
            base_explanation += " as a secondary alternative"
        
        reason_explanations = {
            RecommendationReason.JOERN_NOT_SUPPORTED: 
                f" because Joern does not support {context.language}",
            RecommendationReason.JOERN_FAILED: 
                f" because Joern failed to process {context.language} code",
            RecommendationReason.JOERN_LIMITED_SUPPORT: 
                f" because Joern has limited support for {context.language}",
            RecommendationReason.BETTER_LANGUAGE_SUPPORT: 
                f" due to its comprehensive {context.language} support",
            RecommendationReason.ADDITIONAL_CAPABILITIES: 
                " for additional parsing capabilities",
            RecommendationReason.EASIER_INTEGRATION: 
                " due to its easier integration process",
            RecommendationReason.PERFORMANCE_BENEFITS: 
                " for better performance characteristics"
        }
        
        explanation = base_explanation + reason_explanations.get(reason, "")
        
        # Add tool-specific benefits
        if tool.name == "Tree-sitter":
            explanation += ". Tree-sitter provides fast, incremental parsing with excellent language coverage."
        elif tool.name == "Python AST":
            explanation += ". Python AST is built into Python and requires no additional setup."
        elif tool.name == "ANTLR":
            explanation += ". ANTLR offers flexible grammar-based parsing for custom language support."
        
        return explanation
    
    def _generate_integration_guidance(
        self, 
        tool: AlternativeTool, 
        context: RecommendationContext
    ) -> List[str]:
        """Generate integration guidance for the tool."""
        guidance = []
        
        # Basic installation guidance
        if tool.installation:
            primary_install = tool.installation[0]
            guidance.append(f"Install using: {primary_install.command}")
            
            if primary_install.prerequisites:
                guidance.append(f"Prerequisites: {', '.join(primary_install.prerequisites)}")
        
        # Integration complexity guidance
        if tool.integration_complexity == "high":
            guidance.append("Consider allocating additional time for setup and configuration")
            guidance.append("Review documentation thoroughly before implementation")
        elif tool.integration_complexity == "low":
            guidance.append("Quick setup - can be integrated in minutes")
        
        # Language-specific guidance
        if context.language == "python" and tool.name == "Python AST":
            guidance.append("Use ast.parse() to generate syntax trees")
            guidance.append("Consider ast.NodeVisitor for tree traversal")
        elif tool.name == "Tree-sitter":
            guidance.append(f"Install {context.language} grammar: pip install tree-sitter-{context.language}")
            guidance.append("Use Language.build_library() to compile grammars")
        
        # Project type guidance
        if context.project_type == "production":
            guidance.append("Ensure tool stability and long-term support")
            guidance.append("Consider performance implications for large codebases")
        elif context.project_type == "research":
            guidance.append("Focus on parsing accuracy and feature completeness")
        
        return guidance
    
    def _identify_pros_for_use_case(
        self, 
        tool: AlternativeTool, 
        context: RecommendationContext
    ) -> List[str]:
        """Identify pros specific to the use case."""
        pros = tool.pros.copy()
        
        # Add context-specific pros
        if context.integration_complexity_preference == "low" and tool.integration_complexity == "low":
            pros.append("Matches preference for simple integration")
        
        if context.time_constraints == "high" and tool.name == "Python AST":
            pros.append("No installation time required")
        
        if context.project_type == "educational" and tool.name == "Tree-sitter":
            pros.append("Good learning tool for understanding parsing")
        
        return pros
    
    def _identify_cons_for_use_case(
        self, 
        tool: AlternativeTool, 
        context: RecommendationContext
    ) -> List[str]:
        """Identify cons specific to the use case."""
        cons = tool.cons.copy()
        
        # Add context-specific cons
        if context.integration_complexity_preference == "low" and tool.integration_complexity == "high":
            cons.append("May be too complex for current requirements")
        
        if context.team_expertise == "beginner" and tool.integration_complexity == "high":
            cons.append("Requires advanced technical expertise")
        
        if context.time_constraints == "high" and tool.integration_complexity == "high":
            cons.append("Setup time may exceed project constraints")
        
        return cons
    
    def _estimate_setup_time(
        self, 
        tool: AlternativeTool, 
        context: RecommendationContext
    ) -> str:
        """Estimate setup time based on tool complexity and team expertise."""
        base_times = {
            "low": {"beginner": "30 minutes", "medium": "15 minutes", "expert": "10 minutes"},
            "medium": {"beginner": "2 hours", "medium": "1 hour", "expert": "30 minutes"},
            "high": {"beginner": "1 day", "medium": "4 hours", "expert": "2 hours"}
        }
        
        complexity = tool.integration_complexity
        expertise = context.team_expertise
        
        return base_times.get(complexity, {}).get(expertise, "1 hour")
    
    def compare_with_joern(
        self, 
        tool: AlternativeTool, 
        language: str
    ) -> Dict[str, Any]:
        """
        Generate detailed comparison between a tool and Joern for a specific language.
        
        Args:
            tool: Alternative tool to compare
            language: Programming language for comparison
            
        Returns:
            Dictionary containing comparison details
        """
        comparison = {
            "tool_name": tool.name,
            "language": language,
            "feature_comparison": tool.comparison_with_joern,
            "capability_analysis": {
                "joern_capabilities": [
                    "CPG Generation", "Semantic Analysis", "Control Flow Analysis",
                    "Data Flow Analysis", "Call Graph Generation"
                ],
                "tool_capabilities": [cap.value for cap in tool.capabilities],
                "unique_to_joern": [],
                "unique_to_tool": [],
                "shared_capabilities": []
            },
            "integration_comparison": {
                "joern_complexity": "medium",
                "tool_complexity": tool.integration_complexity,
                "joern_setup_time": "1-2 hours",
                "tool_setup_time": self._estimate_setup_time(
                    tool, 
                    RecommendationContext(
                        language=language,
                        joern_support_status="full",
                        required_capabilities=[],
                        integration_complexity_preference="medium"
                    )
                )
            },
            "use_case_recommendations": {
                "prefer_joern_when": [
                    "Need comprehensive semantic analysis",
                    "Require CPG format specifically",
                    "Working with multiple languages supported by Joern",
                    "Need advanced security analysis features"
                ],
                "prefer_alternative_when": [
                    f"Joern doesn't support {language}",
                    "Need faster parsing performance",
                    "Prefer simpler integration",
                    "Working in constrained environments"
                ]
            }
        }
        
        # Analyze capability overlaps
        joern_caps = {"cpg_generation", "semantic_analysis", "control_flow", "data_flow", "call_graph"}
        tool_caps = {cap.value for cap in tool.capabilities}
        
        comparison["capability_analysis"]["shared_capabilities"] = list(joern_caps & tool_caps)
        comparison["capability_analysis"]["unique_to_joern"] = list(joern_caps - tool_caps)
        comparison["capability_analysis"]["unique_to_tool"] = list(tool_caps - joern_caps)
        
        return comparison
    
    def generate_migration_guide(
        self, 
        from_tool: str, 
        to_tool: AlternativeTool, 
        language: str
    ) -> Dict[str, Any]:
        """
        Generate migration guide from one tool to another.
        
        Args:
            from_tool: Source tool name ("joern" or alternative tool name)
            to_tool: Target alternative tool
            language: Programming language context
            
        Returns:
            Dictionary containing migration guidance
        """
        migration_guide = {
            "from_tool": from_tool,
            "to_tool": to_tool.name,
            "language": language,
            "migration_steps": [],
            "output_format_changes": {},
            "workflow_adjustments": [],
            "potential_challenges": [],
            "mitigation_strategies": []
        }
        
        if from_tool.lower() == "joern":
            migration_guide["migration_steps"] = [
                f"Install {to_tool.name} using recommended method",
                "Adapt input file processing to new tool format",
                "Update output parsing to handle new format",
                "Test with sample files to verify functionality",
                "Update documentation and team processes"
            ]
            
            migration_guide["output_format_changes"] = {
                "joern_output": "CPG format with nodes and edges",
                "new_output": "AST format specific to chosen tool",
                "conversion_needed": True
            }
            
            migration_guide["workflow_adjustments"] = [
                "Update build scripts and automation",
                "Modify analysis pipelines",
                "Retrain team on new tool usage"
            ]
            
            migration_guide["potential_challenges"] = [
                "Different output format requires parser updates",
                "May lose some semantic analysis capabilities",
                "Team learning curve for new tool"
            ]
            
            migration_guide["mitigation_strategies"] = [
                "Create output format conversion utilities",
                "Implement gradual migration with parallel testing",
                "Provide team training and documentation"
            ]
        
        return migration_guide