"""
Integration utilities for alternative tool recommendations.

This module provides utilities to integrate the alternative tool recommendation
system with the main Joern verification workflow.
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
import json

from .database import AlternativeToolDatabase
from .recommender import AlternativeToolRecommender, RecommendationContext, ToolRecommendation
from .models import ToolCapability
from ..core.interfaces import GenerationResult, AnalysisReport


class AlternativeToolIntegrator:
    """Integrates alternative tool recommendations with verification results."""
    
    def __init__(self):
        """Initialize the integrator with database and recommender."""
        self.database = AlternativeToolDatabase()
        self.recommender = AlternativeToolRecommender(self.database)
    
    def process_verification_results(
        self, 
        results: Dict[str, GenerationResult]
    ) -> Dict[str, Any]:
        """
        Process verification results and generate alternative recommendations.
        
        Args:
            results: Dictionary mapping language to GenerationResult
            
        Returns:
            Dictionary containing recommendations for each language
        """
        recommendations_report = {
            "summary": {
                "total_languages_tested": len(results),
                "successful_languages": 0,
                "failed_languages": 0,
                "languages_with_alternatives": 0
            },
            "language_recommendations": {},
            "alternative_tools_overview": self._generate_tools_overview()
        }
        
        for language, result in results.items():
            # Determine Joern support status
            joern_status = self._determine_joern_status(result)
            
            # Create recommendation context
            context = RecommendationContext(
                language=language,
                joern_support_status=joern_status,
                required_capabilities=[
                    ToolCapability.AST_GENERATION,
                    ToolCapability.SYNTAX_PARSING
                ],
                integration_complexity_preference="medium",
                project_type="general",
                team_expertise="medium"
            )
            
            # Generate recommendations
            recommendations = self.recommender.recommend_alternatives(context)
            
            # Update summary statistics
            if result.success:
                recommendations_report["summary"]["successful_languages"] += 1
            else:
                recommendations_report["summary"]["failed_languages"] += 1
            
            if recommendations:
                recommendations_report["summary"]["languages_with_alternatives"] += 1
            
            # Store language-specific recommendations
            recommendations_report["language_recommendations"][language] = {
                "joern_status": joern_status,
                "joern_result": {
                    "success": result.success,
                    "execution_time": result.execution_time,
                    "errors": result.stderr if result.stderr else None
                },
                "recommendations": [self._serialize_recommendation(rec) for rec in recommendations],
                "top_recommendation": self._serialize_recommendation(recommendations[0]) if recommendations else None
            }
        
        return recommendations_report
    
    def _determine_joern_status(self, result: GenerationResult) -> str:
        """Determine Joern support status from generation result."""
        if result.success:
            if result.stderr and "warning" in result.stderr.lower():
                return "partial"
            return "full"
        else:
            if "not found" in result.stderr.lower() or "command not found" in result.stderr.lower():
                return "none"
            return "failed"
    
    def _serialize_recommendation(self, recommendation: ToolRecommendation) -> Dict[str, Any]:
        """Serialize a tool recommendation for JSON output."""
        return {
            "tool_name": recommendation.tool.name,
            "confidence_score": recommendation.confidence_score,
            "reason": recommendation.reason.value,
            "explanation": recommendation.explanation,
            "integration_guidance": recommendation.integration_guidance,
            "pros": recommendation.pros_for_use_case,
            "cons": recommendation.cons_for_use_case,
            "estimated_setup_time": recommendation.estimated_setup_time,
            "tool_details": {
                "description": recommendation.tool.description,
                "supported_languages": recommendation.tool.supported_languages,
                "capabilities": [cap.value for cap in recommendation.tool.capabilities],
                "maturity_level": recommendation.tool.maturity_level,
                "integration_complexity": recommendation.tool.integration_complexity,
                "official_website": recommendation.tool.official_website,
                "documentation_url": recommendation.tool.documentation_url,
                "github_url": recommendation.tool.github_url,
                "license": recommendation.tool.license
            },
            "installation_instructions": [
                {
                    "method": inst.method.value,
                    "command": inst.command,
                    "description": inst.description,
                    "prerequisites": inst.prerequisites,
                    "verification_command": inst.verification_command
                }
                for inst in recommendation.tool.installation
            ],
            "usage_examples": [
                {
                    "description": ex.description,
                    "command": ex.command,
                    "input_example": ex.input_example,
                    "expected_output_format": ex.expected_output_format,
                    "notes": ex.notes
                }
                for ex in recommendation.tool.usage_examples
            ]
        }
    
    def _generate_tools_overview(self) -> Dict[str, Any]:
        """Generate overview of all available alternative tools."""
        all_tools = self.database.get_all_tools()
        
        overview = {
            "total_tools": len(all_tools),
            "tools_by_category": {
                "universal_parsers": [],
                "language_specific": [],
                "grammar_generators": []
            },
            "tools_by_complexity": {
                "low": [],
                "medium": [],
                "high": []
            },
            "tools_by_maturity": {
                "experimental": [],
                "beta": [],
                "stable": [],
                "mature": []
            }
        }
        
        for tool_name, tool in all_tools.items():
            # Categorize by type
            if len(tool.supported_languages) > 5:
                overview["tools_by_category"]["universal_parsers"].append(tool_name)
            elif len(tool.supported_languages) == 1:
                overview["tools_by_category"]["language_specific"].append(tool_name)
            else:
                overview["tools_by_category"]["grammar_generators"].append(tool_name)
            
            # Categorize by complexity
            overview["tools_by_complexity"][tool.integration_complexity].append(tool_name)
            
            # Categorize by maturity
            overview["tools_by_maturity"][tool.maturity_level].append(tool_name)
        
        return overview
    
    def generate_detailed_comparison_report(
        self, 
        language: str, 
        joern_result: GenerationResult
    ) -> Dict[str, Any]:
        """
        Generate detailed comparison report for a specific language.
        
        Args:
            language: Programming language
            joern_result: Joern generation result for comparison
            
        Returns:
            Detailed comparison report
        """
        alternatives = self.database.get_alternatives_for_language(language)
        if not alternatives:
            return {"error": f"No alternatives found for {language}"}
        
        report = {
            "language": language,
            "joern_performance": {
                "success": joern_result.success,
                "execution_time": joern_result.execution_time,
                "memory_usage": getattr(joern_result, 'memory_usage', 'unknown'),
                "output_size": len(joern_result.output_files) if joern_result.output_files else 0
            },
            "alternative_comparisons": []
        }
        
        for tool in alternatives.primary_alternatives + alternatives.secondary_alternatives:
            comparison = self.recommender.compare_with_joern(tool, language)
            report["alternative_comparisons"].append(comparison)
        
        return report
    
    def export_recommendations_to_file(
        self, 
        recommendations: Dict[str, Any], 
        output_path: Path
    ):
        """
        Export recommendations to a JSON file.
        
        Args:
            recommendations: Recommendations dictionary
            output_path: Path to output file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(recommendations, f, indent=2, ensure_ascii=False)
    
    def generate_markdown_report(
        self, 
        recommendations: Dict[str, Any]
    ) -> str:
        """
        Generate a markdown report from recommendations.
        
        Args:
            recommendations: Recommendations dictionary
            
        Returns:
            Markdown formatted report
        """
        md_lines = [
            "# Alternative Tool Recommendations Report",
            "",
            "## Summary",
            "",
            f"- **Total Languages Tested**: {recommendations['summary']['total_languages_tested']}",
            f"- **Successful with Joern**: {recommendations['summary']['successful_languages']}",
            f"- **Failed with Joern**: {recommendations['summary']['failed_languages']}",
            f"- **Languages with Alternatives**: {recommendations['summary']['languages_with_alternatives']}",
            "",
            "## Language-Specific Recommendations",
            ""
        ]
        
        for language, lang_data in recommendations["language_recommendations"].items():
            md_lines.extend([
                f"### {language.title()}",
                "",
                f"**Joern Status**: {lang_data['joern_status']}",
                ""
            ])
            
            if lang_data['joern_result']['success']:
                md_lines.append(f"✅ Joern successfully processed {language} in {lang_data['joern_result']['execution_time']:.2f}s")
            else:
                md_lines.append(f"❌ Joern failed to process {language}")
                if lang_data['joern_result']['errors']:
                    md_lines.append(f"Error: `{lang_data['joern_result']['errors'][:100]}...`")
            
            md_lines.append("")
            
            if lang_data['recommendations']:
                md_lines.extend([
                    "#### Recommended Alternatives",
                    ""
                ])
                
                for i, rec in enumerate(lang_data['recommendations'][:3], 1):  # Top 3 recommendations
                    md_lines.extend([
                        f"**{i}. {rec['tool_name']}** (Confidence: {rec['confidence_score']:.2f})",
                        f"- {rec['explanation']}",
                        f"- Setup time: {rec['estimated_setup_time']}",
                        f"- Integration complexity: {rec['tool_details']['integration_complexity']}",
                        ""
                    ])
            else:
                md_lines.extend([
                    "No specific alternatives recommended.",
                    ""
                ])
            
            md_lines.append("---")
            md_lines.append("")
        
        # Add tools overview
        md_lines.extend([
            "## Available Alternative Tools Overview",
            "",
            f"**Total Tools**: {recommendations['alternative_tools_overview']['total_tools']}",
            ""
        ])
        
        tools_overview = recommendations['alternative_tools_overview']
        
        md_lines.extend([
            "### By Category",
            f"- **Universal Parsers**: {', '.join(tools_overview['tools_by_category']['universal_parsers'])}",
            f"- **Language Specific**: {', '.join(tools_overview['tools_by_category']['language_specific'])}",
            f"- **Grammar Generators**: {', '.join(tools_overview['tools_by_category']['grammar_generators'])}",
            "",
            "### By Integration Complexity",
            f"- **Low**: {', '.join(tools_overview['tools_by_complexity']['low'])}",
            f"- **Medium**: {', '.join(tools_overview['tools_by_complexity']['medium'])}",
            f"- **High**: {', '.join(tools_overview['tools_by_complexity']['high'])}",
            ""
        ])
        
        return "\n".join(md_lines)