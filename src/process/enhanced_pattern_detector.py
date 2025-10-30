#!/usr/bin/env python3
"""
Enhanced Pattern Detector with Context-Aware Analysis

This module replaces the basic VulnerabilityPatternDetector with a sophisticated
context-aware system that can distinguish between safe and unsafe usage of
dangerous functions across multiple programming languages.

Key Features:
- Context-aware analysis (looks at surrounding code)
- Multi-language support (C, C++, Java, Python, JavaScript, Go)
- Safe pattern recognition (reduces false positives)
- Comprehensive vulnerability database (500+ functions)
- Smart risk scoring based on actual usage patterns
"""

import re
import sys
import os
from typing import Dict, List, Tuple, Optional

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.process.comprehensive_vulnerability_database import (
    EnhancedVulnerabilityPatternDetector,
    ComprehensiveVulnerabilityDatabase,
    Language,
    RiskLevel
)


class ContextAwareVulnerabilityDetector:
    """
    Context-aware vulnerability detector that solves the false positive problem.
    
    This is the MAIN SOLUTION that replaces our old pattern detector.
    """
    
    def __init__(self):
        self.enhanced_detector = EnhancedVulnerabilityPatternDetector()
        self.db = self.enhanced_detector.db
        
        print(f"🚀 Context-Aware Vulnerability Detector initialized")
        print(f"📊 Supporting {len(Language)} languages with {len(self.db.vulnerability_db)} dangerous functions")
    
    def annotate_vulnerable_lines(self, source_code: str, context_window: int = 5) -> Dict[int, float]:
        """
        Enhanced version of annotate_vulnerable_lines that provides context-aware analysis.
        
        This method maintains compatibility with existing code while providing
        much more accurate results.
        
        Args:
            source_code: Source code to analyze
            context_window: Number of lines to consider for context
            
        Returns:
            Dict mapping line_number -> vulnerability_score (0.0-1.0)
            Only returns lines that are ACTUALLY vulnerable (not safe usage)
        """
        
        # Use enhanced analysis
        analysis_results = self.enhanced_detector.analyze_source_code(source_code, context_window)
        
        # Convert to old format for compatibility
        vulnerable_lines = {}
        
        # Only include lines that are actually vulnerable (not safe usage)
        for line_num, info in analysis_results['vulnerable_lines'].items():
            vulnerable_lines[line_num] = info['score']
        
        # Log the improvement
        total_dangerous_functions = len(analysis_results['vulnerable_lines']) + len(analysis_results['safe_lines'])
        actual_vulnerabilities = len(analysis_results['vulnerable_lines'])
        
        if total_dangerous_functions > 0:
            accuracy_improvement = (total_dangerous_functions - actual_vulnerabilities) / total_dangerous_functions
            print(f"🎯 Context Analysis Results:")
            print(f"   Total dangerous functions found: {total_dangerous_functions}")
            print(f"   Actually vulnerable: {actual_vulnerabilities}")
            print(f"   Safe usage detected: {len(analysis_results['safe_lines'])}")
            print(f"   False positive reduction: {accuracy_improvement:.1%}")
        
        return vulnerable_lines
    
    def get_detailed_analysis(self, source_code: str) -> Dict:
        """
        Get detailed analysis with explanations and recommendations.
        
        This provides much more information than the basic annotate_vulnerable_lines.
        """
        return self.enhanced_detector.analyze_source_code(source_code)
    
    def explain_vulnerability_pattern(self, line_content: str, attention_score: float, 
                                    source_code: str = "") -> str:
        """
        Enhanced explanation that considers context and provides detailed reasoning.
        
        This replaces the basic explain_vulnerability_pattern method.
        """
        
        if source_code:
            # Get full analysis for better explanation
            analysis = self.get_detailed_analysis(source_code)
            
            # Find the line in analysis results
            for line_num, info in analysis['vulnerable_lines'].items():
                if line_content.strip() in source_code.split('\n')[line_num]:
                    return f"{info['explanation']} (Risk: {info['score']:.2f})"
            
            for line_num, info in analysis['safe_lines'].items():
                if line_content.strip() in source_code.split('\n')[line_num]:
                    return f"SAFE USAGE: {info['explanation']} (Risk: {info['score']:.2f})"
        
        # Fallback to basic analysis
        language = self.db.detect_language(line_content)
        dangerous_functions = self.db.get_all_dangerous_functions(language)
        
        for func_name in dangerous_functions:
            if func_name in line_content:
                analysis = self.db.analyze_function_in_context(
                    func_name, line_content, [], language
                )
                return analysis['explanation']
        
        # Default explanation
        if attention_score > 0.15:
            return "Model detected potential security issue based on code patterns"
        else:
            return "Line flagged by attention mechanism - review recommended"
    
    def create_attention_supervision_mask(self, graph_data, source_code: str, 
                                        node_to_line_mapping: Optional[Dict[int, int]] = None) -> 'torch.Tensor':
        """
        Create enhanced attention supervision mask using context-aware analysis.
        
        This creates much more accurate supervision signals for training.
        """
        import torch
        
        num_nodes = graph_data.x.size(0)
        
        # Get detailed analysis
        analysis = self.get_detailed_analysis(source_code)
        
        # Create supervision mask
        supervision_mask = torch.zeros(num_nodes, dtype=torch.float)
        
        # Create default mapping if not provided
        if node_to_line_mapping is None:
            node_to_line_mapping = {i: i for i in range(num_nodes)}
        
        # Set supervision signals
        for node_idx, line_num in node_to_line_mapping.items():
            if node_idx < num_nodes:
                if line_num in analysis['vulnerable_lines']:
                    # High supervision for actually vulnerable lines
                    supervision_mask[node_idx] = analysis['vulnerable_lines'][line_num]['score']
                elif line_num in analysis['safe_lines']:
                    # Low supervision for safe usage (teach model to ignore)
                    supervision_mask[node_idx] = 0.1
                else:
                    # No supervision for neutral lines
                    supervision_mask[node_idx] = 0.0
        
        return supervision_mask
    
    def get_language_specific_functions(self, language_name: str) -> List[str]:
        """Get dangerous functions for a specific language"""
        
        try:
            language = Language(language_name.lower())
            return self.db.get_all_dangerous_functions(language)
        except ValueError:
            print(f"⚠️ Unknown language: {language_name}")
            return []
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages"""
        return [lang.value for lang in Language]
    
    def get_database_summary(self) -> Dict:
        """Get summary of the vulnerability database"""
        return self.db.get_vulnerability_summary()


def test_context_aware_detector():
    """Test the context-aware detector with various scenarios"""
    
    print("🧪 Testing Context-Aware Vulnerability Detector")
    print("="*70)
    
    detector = ContextAwareVulnerabilityDetector()
    
    # Test Case 1: Unsafe strcpy
    print("\n📋 Test Case 1: Unsafe strcpy")
    unsafe_code = '''void unsafe_function(char* input) {
    char buffer[64];
    strcpy(buffer, input);  // UNSAFE - no bounds check
    printf("Result: %s\\n", buffer);
}'''
    
    unsafe_results = detector.annotate_vulnerable_lines(unsafe_code)
    print(f"Vulnerable lines detected: {list(unsafe_results.keys())}")
    
    # Test Case 2: Safe strncpy with bounds checking
    print("\n📋 Test Case 2: Safe strncpy with bounds checking")
    safe_code = '''void safe_function(char* input) {
    char buffer[64];
    if (strlen(input) < sizeof(buffer) - 1) {
        strncpy(buffer, input, sizeof(buffer) - 1);
        buffer[sizeof(buffer) - 1] = '\\0';
    }
    printf("Result: %s\\n", buffer);
}'''
    
    safe_results = detector.annotate_vulnerable_lines(safe_code)
    print(f"Vulnerable lines detected: {list(safe_results.keys())}")
    
    # Test Case 3: Mixed safe and unsafe
    print("\n📋 Test Case 3: Mixed safe and unsafe")
    mixed_code = '''void mixed_function(char* input) {
    char safe_buffer[100];
    char unsafe_buffer[50];
    
    // Safe usage
    if (strlen(input) < 99) {
        strncpy(safe_buffer, input, 99);
        safe_buffer[99] = '\\0';
    }
    
    // Unsafe usage
    strcpy(unsafe_buffer, input);
    printf(unsafe_buffer);  // Format string vuln
}'''
    
    mixed_results = detector.annotate_vulnerable_lines(mixed_code)
    print(f"Vulnerable lines detected: {list(mixed_results.keys())}")
    
    # Get detailed analysis
    detailed = detector.get_detailed_analysis(mixed_code)
    print(f"\n📊 Detailed Analysis:")
    print(f"   Language: {detailed['language'].value}")
    print(f"   Vulnerable lines: {len(detailed['vulnerable_lines'])}")
    print(f"   Safe lines: {len(detailed['safe_lines'])}")
    print(f"   Overall risk: {detailed['overall_risk_score']:.2f}")
    
    # Show explanations
    print(f"\n💡 Explanations:")
    for line_num, info in detailed['vulnerable_lines'].items():
        print(f"   Line {line_num}: {info['explanation']}")
    
    for line_num, info in detailed['safe_lines'].items():
        print(f"   Line {line_num}: {info['explanation']}")
    
    print(f"\n🎯 Context-aware detection test completed!")


if __name__ == "__main__":
    test_context_aware_detector()