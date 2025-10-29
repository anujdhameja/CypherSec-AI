"""
Example usage of the alternative tool recommendation system.

This script demonstrates how to use the alternative tool database and
recommendation engine to get suggestions for CPG/AST generation tools.
"""

from pathlib import Path
from .database import AlternativeToolDatabase
from .recommender import AlternativeToolRecommender, RecommendationContext
from .integration import AlternativeToolIntegrator
from .models import ToolCapability
from ..core.interfaces import GenerationResult


def example_basic_usage():
    """Example of basic usage of the recommendation system."""
    print("=== Alternative Tool Recommendation System Example ===\n")
    
    # Initialize the database and recommender
    database = AlternativeToolDatabase()
    recommender = AlternativeToolRecommender(database)
    
    # Example 1: Get alternatives for Python
    print("1. Getting alternatives for Python:")
    python_alternatives = database.get_alternatives_for_language("python")
    if python_alternatives:
        print(f"   Recommended tool: {python_alternatives.recommended_tool}")
        print(f"   Primary alternatives: {[tool.name for tool in python_alternatives.primary_alternatives]}")
        print(f"   Secondary alternatives: {[tool.name for tool in python_alternatives.secondary_alternatives]}")
    print()
    
    # Example 2: Get tool information
    print("2. Getting information about Tree-sitter:")
    tree_sitter = database.get_tool("tree-sitter")
    if tree_sitter:
        print(f"   Description: {tree_sitter.description}")
        print(f"   Supported languages: {', '.join(tree_sitter.supported_languages[:5])}...")
        print(f"   Maturity level: {tree_sitter.maturity_level}")
        print(f"   Integration complexity: {tree_sitter.integration_complexity}")
    print()
    
    # Example 3: Generate recommendations based on context
    print("3. Generating recommendations for failed Python processing:")
    context = RecommendationContext(
        language="python",
        joern_support_status="failed",
        required_capabilities=[ToolCapability.AST_GENERATION, ToolCapability.SYNTAX_PARSING],
        integration_complexity_preference="low",
        project_type="research",
        team_expertise="medium"
    )
    
    recommendations = recommender.recommend_alternatives(context)
    for i, rec in enumerate(recommendations[:2], 1):
        print(f"   {i}. {rec.tool.name} (Confidence: {rec.confidence_score:.2f})")
        print(f"      Reason: {rec.explanation}")
        print(f"      Setup time: {rec.estimated_setup_time}")
    print()


def example_integration_with_results():
    """Example of integrating recommendations with verification results."""
    print("=== Integration with Verification Results ===\n")
    
    # Create mock verification results
    mock_results = {
        "python": GenerationResult(
            language="python",
            input_file="test_sample.py",
            output_dir="output/python",
            success=False,
            execution_time=2.5,
            memory_usage=None,
            stdout="",
            stderr="Error: pysrc2cpg.bat not found",
            return_code=1,
            output_files=[]
        ),
        "javascript": GenerationResult(
            language="javascript",
            input_file="test_sample.js",
            output_dir="output/javascript",
            success=True,
            execution_time=1.2,
            memory_usage=None,
            stdout="CPG generated successfully",
            stderr="",
            return_code=0,
            output_files=[Path("output/javascript/cpg.bin")]
        )
    }
    
    # Process results and generate recommendations
    integrator = AlternativeToolIntegrator()
    recommendations_report = integrator.process_verification_results(mock_results)
    
    print("Processing Results Summary:")
    print(f"- Total languages tested: {recommendations_report['summary']['total_languages_tested']}")
    print(f"- Successful with Joern: {recommendations_report['summary']['successful_languages']}")
    print(f"- Failed with Joern: {recommendations_report['summary']['failed_languages']}")
    print(f"- Languages with alternatives: {recommendations_report['summary']['languages_with_alternatives']}")
    print()
    
    # Show recommendations for failed language
    python_rec = recommendations_report['language_recommendations']['python']
    print("Python Recommendations:")
    print(f"- Joern status: {python_rec['joern_status']}")
    if python_rec['top_recommendation']:
        top_rec = python_rec['top_recommendation']
        print(f"- Top recommendation: {top_rec['tool_name']}")
        print(f"- Confidence: {top_rec['confidence_score']:.2f}")
        print(f"- Explanation: {top_rec['explanation']}")
    print()


def example_comparison_and_migration():
    """Example of tool comparison and migration guidance."""
    print("=== Tool Comparison and Migration ===\n")
    
    database = AlternativeToolDatabase()
    recommender = AlternativeToolRecommender(database)
    
    # Compare Tree-sitter with Joern for Python
    tree_sitter = database.get_tool("tree-sitter")
    if tree_sitter:
        comparison = recommender.compare_with_joern(tree_sitter, "python")
        
        print("Tree-sitter vs Joern for Python:")
        print("Feature Comparison:")
        for comp in comparison["feature_comparison"]:
            print(f"- {comp.feature}: Joern={comp.joern_support}, Tree-sitter={comp.alternative_support}")
        
        print("\nWhen to prefer each tool:")
        print("Prefer Joern when:")
        for reason in comparison["use_case_recommendations"]["prefer_joern_when"]:
            print(f"  • {reason}")
        
        print("Prefer Tree-sitter when:")
        for reason in comparison["use_case_recommendations"]["prefer_alternative_when"]:
            print(f"  • {reason}")
        print()
    
    # Generate migration guide
    if tree_sitter:
        migration_guide = recommender.generate_migration_guide("joern", tree_sitter, "python")
        
        print("Migration Guide from Joern to Tree-sitter:")
        print("Steps:")
        for step in migration_guide["migration_steps"]:
            print(f"  1. {step}")
        
        print("\nPotential Challenges:")
        for challenge in migration_guide["potential_challenges"]:
            print(f"  • {challenge}")
        print()


def example_export_recommendations():
    """Example of exporting recommendations to files."""
    print("=== Exporting Recommendations ===\n")
    
    # Create mock results and generate recommendations
    mock_results = {
        "python": GenerationResult(
            language="python",
            input_file="test_sample.py",
            output_dir="output/python",
            success=False,
            execution_time=2.5,
            memory_usage=None,
            stdout="",
            stderr="Tool not found",
            return_code=1,
            output_files=[]
        )
    }
    
    integrator = AlternativeToolIntegrator()
    recommendations = integrator.process_verification_results(mock_results)
    
    # Export to JSON
    output_dir = Path("verification_output")
    json_path = output_dir / "alternative_recommendations.json"
    integrator.export_recommendations_to_file(recommendations, json_path)
    print(f"Recommendations exported to: {json_path}")
    
    # Generate markdown report
    markdown_report = integrator.generate_markdown_report(recommendations)
    markdown_path = output_dir / "alternative_recommendations.md"
    
    output_dir.mkdir(exist_ok=True)
    with open(markdown_path, 'w', encoding='utf-8') as f:
        f.write(markdown_report)
    
    print(f"Markdown report generated: {markdown_path}")
    print()


if __name__ == "__main__":
    """Run all examples."""
    try:
        example_basic_usage()
        example_integration_with_results()
        example_comparison_and_migration()
        example_export_recommendations()
        
        print("✅ All examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()