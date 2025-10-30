#!/usr/bin/env python3
"""
Test Pattern-Enhanced Vulnerability Detection

This script tests our enhanced system with pattern detection integration
to see if combining attention mechanism with pattern matching improves
vulnerability detection accuracy.

Usage:
    python test_pattern_enhanced_detection.py
"""

import torch
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append('src')

from src.inference.enhanced_explainable_predictor import EnhancedExplainablePredictor
from src.process.vulnerability_pattern_detector import VulnerabilityPatternDetector
from torch_geometric.data import Data
import numpy as np


def create_realistic_vulnerable_code_with_source():
    """Create vulnerable code with actual source code string"""
    
    source_code = '''void process_user_input(char* user_data) {
    char buffer[64];
    char temp_buffer[32];
    char output[128];
    
    if (user_data != NULL) {
        strcpy(buffer, user_data);
        strcat(buffer, "_processed");
        sprintf(temp_buffer, "%s", buffer);
        printf(temp_buffer);
        strcpy(output, buffer);
        return;
    }
    printf("No input provided\\n");
}'''
    
    lines = source_code.split('\n')
    num_nodes = len(lines)
    
    # Create node features with emphasis on vulnerable patterns
    x = torch.randn(num_nodes, 100) * 0.8
    
    # EMPHASIZE VULNERABLE LINES with distinct patterns
    vulnerable_lines = [5, 6, 7, 8, 9]  # strcpy, strcat, sprintf, printf, strcpy
    for line_idx in vulnerable_lines:
        if line_idx < num_nodes:
            # Add strong vulnerability signal to these nodes
            x[line_idx] += torch.randn(100) * 1.5  # Stronger signal
            # Add specific patterns for different vulnerability types
            if 'strcpy' in lines[line_idx]:
                x[line_idx, :20] += 2.5  # Buffer overflow pattern
            elif 'strcat' in lines[line_idx]:
                x[line_idx, 20:40] += 2.2  # String concatenation pattern
            elif 'sprintf' in lines[line_idx]:
                x[line_idx, 40:60] += 2.0  # Format string pattern
            elif 'printf' in lines[line_idx] and '%s' not in lines[line_idx]:
                x[line_idx, 60:80] += 1.8  # Format string vulnerability pattern
    
    # Create realistic control and data flow edges
    edges = []
    
    # Sequential control flow
    for i in range(num_nodes - 1):
        edges.extend([[i, i+1], [i+1, i]])
    
    # Data dependencies (realistic flow)
    data_flow_edges = [
        # user_data flows to strcpy
        [0, 5], [5, 0],
        # buffer flows through the vulnerable chain
        [1, 5], [5, 1],    # buffer declared, used in strcpy
        [5, 6], [6, 5],    # strcpy result used in strcat
        [6, 7], [7, 6],    # strcat result used in sprintf
        [1, 7], [7, 1],    # buffer used in sprintf
        [2, 7], [7, 2],    # temp_buffer used in sprintf
        [7, 8], [8, 8],    # sprintf result used in printf
        [2, 8], [8, 2],    # temp_buffer used in printf
        [1, 9], [9, 1],    # buffer used in final strcpy
        [3, 9], [9, 3],    # output used in strcpy
        [5, 9], [9, 5],    # buffer flows to output
        # Control flow dependencies
        [4, 5], [5, 4],    # if condition controls strcpy
        [4, 6], [6, 4],    # if condition controls strcat
        [4, 7], [7, 4],    # if condition controls sprintf
        [4, 8], [8, 4],    # if condition controls printf
        [4, 9], [9, 4],    # if condition controls final strcpy
    ]
    
    edges.extend(data_flow_edges)
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    # Create node to line mapping
    node_to_line_mapping = {i: i for i in range(num_nodes)}
    
    return Data(x=x, edge_index=edge_index), lines, node_to_line_mapping, source_code


def create_safe_code_with_source():
    """Create safe code with actual source code string"""
    
    source_code = '''void safe_process_input(char* user_data) {
    char buffer[64];
    char temp_buffer[32];
    char output[128];
    size_t input_len;
    
    if (user_data != NULL) {
        input_len = strlen(user_data);
        if (input_len < 50) {
            strncpy(buffer, user_data, 50);
            buffer[50] = '\\0';
            strncat(buffer, "_ok", 12);
            snprintf(temp_buffer, 31, "%s", buffer);
            printf("%s\\n", temp_buffer);
            strncpy(output, buffer, 127);
            output[127] = '\\0';
        } else {
            printf("Input too long\\n");
        }
    } else {
        printf("No input provided\\n");
    }
}'''
    
    lines = source_code.split('\n')
    num_nodes = len(lines)
    
    # Create more uniform node features (safer patterns)
    x = torch.randn(num_nodes, 100) * 0.6
    
    # Slightly emphasize safety checks
    safety_lines = [6, 7, 8, 9, 10, 11, 12, 13, 14]  # Length checks and safe functions
    for line_idx in safety_lines:
        if line_idx < num_nodes:
            x[line_idx] += torch.randn(100) * 0.4  # Weaker signal for safe patterns
    
    # Create edges for safe control flow
    edges = []
    
    # Sequential control flow
    for i in range(num_nodes - 1):
        edges.extend([[i, i+1], [i+1, i]])
    
    # Safe data flow patterns
    safe_data_flow = [
        [0, 6], [6, 0],    # user_data to strlen
        [6, 7], [7, 6],    # length check to condition
        [1, 8], [8, 1],    # buffer to strncpy
        [8, 10], [10, 8],  # strncpy to strncat
        [10, 11], [11, 10], # strncat to snprintf
        [2, 11], [11, 2],  # temp_buffer to snprintf
        [11, 12], [12, 11], # snprintf to printf
        [1, 13], [13, 1],  # buffer to final strncpy
        [3, 13], [13, 3],  # output to strncpy
    ]
    
    edges.extend(safe_data_flow)
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    # Create node to line mapping
    node_to_line_mapping = {i: i for i in range(num_nodes)}
    
    return Data(x=x, edge_index=edge_index), lines, node_to_line_mapping, source_code


def main():
    print("="*90)
    print("🔍 PATTERN-ENHANCED VULNERABILITY DETECTION TEST")
    print("="*90)
    
    try:
        # Test pattern detector first
        print("\n🧪 Testing Vulnerability Pattern Detector...")
        detector = VulnerabilityPatternDetector()
        
        # Test vulnerable code
        print("\n" + "="*80)
        print("📋 TEST 1: PATTERN DETECTION ON VULNERABLE CODE")
        print("="*80)
        
        vuln_graph, vuln_lines, vuln_mapping, vuln_source = create_realistic_vulnerable_code_with_source()
        
        print("📄 Vulnerable C function:")
        for i, line in enumerate(vuln_lines):
            marker = " ← VULNERABLE!" if i in [5, 6, 7, 8, 9] else ""
            print(f"   {i:2d}: {line}{marker}")
        
        # Test pattern detection
        pattern_annotations = detector.annotate_vulnerable_lines(vuln_source)
        
        print(f"\n🎯 Pattern Detection Results:")
        for line_num, score in sorted(pattern_annotations.items()):
            if line_num < len(vuln_lines):
                explanation = detector.explain_vulnerability_pattern(vuln_lines[line_num], score)
                print(f"   Line {line_num:2d}: {score:.2f} - {explanation}")
        
        # Initialize enhanced predictor
        print("\n🔧 Initializing enhanced explainable predictor...")
        predictor = EnhancedExplainablePredictor('models/enhanced_attention_model.pth')
        
        # Test with pattern integration
        print(f"\n🔮 Running pattern-enhanced vulnerability analysis...")
        vuln_result = predictor.predict_with_line_level_analysis(
            vuln_graph,
            node_labels=vuln_lines,
            node_to_line_mapping=vuln_mapping,
            source_code=vuln_source,  # NEW: Pass source code for pattern analysis
            top_k=15
        )
        
        # Display enhanced analysis
        predictor.print_enhanced_explanation(vuln_result, "Pattern-Enhanced Vulnerable Code Analysis")
        predictor.visualize_line_level_attention(vuln_result)
        
        # Test 2: Safe code
        print("\n" + "="*80)
        print("📋 TEST 2: PATTERN DETECTION ON SAFE CODE")
        print("="*80)
        
        safe_graph, safe_lines, safe_mapping, safe_source = create_safe_code_with_source()
        
        print("📄 Safe C function:")
        for i, line in enumerate(safe_lines):
            marker = " ← SAFE" if i in [6, 7, 8, 9, 10, 11, 12, 13, 14] else ""
            print(f"   {i:2d}: {line}{marker}")
        
        # Test pattern detection on safe code
        safe_pattern_annotations = detector.annotate_vulnerable_lines(safe_source)
        
        print(f"\n🎯 Pattern Detection Results:")
        if safe_pattern_annotations:
            for line_num, score in sorted(safe_pattern_annotations.items()):
                if line_num < len(safe_lines):
                    explanation = detector.explain_vulnerability_pattern(safe_lines[line_num], score)
                    print(f"   Line {line_num:2d}: {score:.2f} - {explanation}")
        else:
            print("   ✅ No vulnerable patterns detected")
        
        # Test with pattern integration
        print(f"\n🔮 Running pattern-enhanced analysis on safe code...")
        safe_result = predictor.predict_with_line_level_analysis(
            safe_graph,
            node_labels=safe_lines,
            node_to_line_mapping=safe_mapping,
            source_code=safe_source,  # NEW: Pass source code for pattern analysis
            top_k=15
        )
        
        # Display enhanced analysis
        predictor.print_enhanced_explanation(safe_result, "Pattern-Enhanced Safe Code Analysis")
        predictor.visualize_line_level_attention(safe_result)
        
        # Enhanced Comparison Analysis
        print("\n" + "="*80)
        print("📊 PATTERN-ENHANCED COMPARISON ANALYSIS")
        print("="*80)
        
        print(f"\n🎯 Pattern Detection Comparison:")
        print(f"   Vulnerable Code:")
        print(f"     Pattern-detected lines: {len(pattern_annotations)}")
        print(f"     Attention-detected lines: {len(vuln_result['vulnerable_lines'])}")
        if 'pattern_analysis' in vuln_result:
            pattern_analysis = vuln_result.get('pattern_analysis')
            if pattern_analysis:
                print(f"     Agreement rate: {pattern_analysis.get('agreement_rate', 0):.1%}")
                print(f"     Both methods: {len(pattern_analysis.get('overlap_lines', []))}")
                print(f"     Pattern only: {len(pattern_analysis.get('pattern_only_lines', []))}")
                print(f"     Attention only: {len(pattern_analysis.get('attention_only_lines', []))}")
        
        print(f"\n   Safe Code:")
        print(f"     Pattern-detected lines: {len(safe_pattern_annotations)}")
        print(f"     Attention-detected lines: {len(safe_result['vulnerable_lines'])}")
        
        # Accuracy Analysis
        print(f"\n💡 Enhanced Detection Accuracy:")
        
        # Check pattern detection accuracy
        actual_vulnerable_lines = {5, 6, 7, 8, 9}  # strcpy, strcat, sprintf, printf, strcpy
        pattern_detected_lines = set(pattern_annotations.keys())
        attention_detected_lines = {line['line_number'] for line in vuln_result['vulnerable_lines'] 
                                  if line['risk_level'] in ['HIGH', 'MEDIUM']}
        
        pattern_overlap = actual_vulnerable_lines & pattern_detected_lines
        attention_overlap = actual_vulnerable_lines & attention_detected_lines
        combined_overlap = actual_vulnerable_lines & (pattern_detected_lines | attention_detected_lines)
        
        pattern_accuracy = len(pattern_overlap) / len(actual_vulnerable_lines)
        attention_accuracy = len(attention_overlap) / len(actual_vulnerable_lines)
        combined_accuracy = len(combined_overlap) / len(actual_vulnerable_lines)
        
        print(f"   Actual vulnerable lines: {sorted(actual_vulnerable_lines)}")
        print(f"   Pattern detection accuracy: {pattern_accuracy:.1%} ({sorted(pattern_overlap)})")
        print(f"   Attention detection accuracy: {attention_accuracy:.1%} ({sorted(attention_overlap)})")
        print(f"   Combined detection accuracy: {combined_accuracy:.1%} ({sorted(combined_overlap)})")
        
        # Overall Assessment
        print(f"\n🎉 Pattern-Enhanced Detection Results:")
        
        if combined_accuracy >= 0.8:
            print(f"   ✅ EXCELLENT: Combined approach detected {combined_accuracy:.1%} of vulnerabilities!")
        elif combined_accuracy >= 0.6:
            print(f"   ✅ GOOD: Combined approach detected {combined_accuracy:.1%} of vulnerabilities")
        elif combined_accuracy >= 0.4:
            print(f"   ⚠️  FAIR: Combined approach detected {combined_accuracy:.1%} of vulnerabilities")
        else:
            print(f"   ❌ POOR: Combined approach only detected {combined_accuracy:.1%} of vulnerabilities")
        
        # Improvement analysis
        improvement = combined_accuracy - attention_accuracy
        if improvement > 0:
            print(f"   📈 IMPROVEMENT: Pattern integration improved detection by {improvement:.1%}")
        elif improvement == 0:
            print(f"   📊 NO CHANGE: Pattern integration maintained same accuracy")
        else:
            print(f"   📉 REGRESSION: Pattern integration decreased accuracy by {abs(improvement):.1%}")
        
        print(f"\n🚀 Pattern-enhanced vulnerability detection test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()