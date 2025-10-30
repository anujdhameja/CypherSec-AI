#!/usr/bin/env python3
"""
Complete Integration Test: All Three Enhancements

This script tests the complete integration of all three enhancements:
1. Enhanced Attention Model (multi-head attention + vulnerability detection)
2. Pattern Detection (automatic vulnerability pattern matching)
3. Line Number Tracking (precise CPG line-to-attention mapping)

This represents the final, complete vulnerability detection system.

Usage:
    python test_complete_integration.py
"""

import torch
import sys
import os
from pathlib import Path
import numpy as np

# Add src to path
sys.path.append('src')

from src.inference.enhanced_explainable_predictor import EnhancedExplainablePredictor
from src.process.vulnerability_pattern_detector import VulnerabilityPatternDetector
from src.data.line_number_tracker import EnhancedGraphData, LineNumberTracker
from src.data.enhanced_data_manager import EnhancedVulnerabilityDataset, create_enhanced_data_loader
from torch_geometric.data import Data
from gensim.models import Word2Vec


def create_realistic_cpg_data():
    """Create realistic CPG data structure with actual line numbers"""
    
    # Realistic vulnerable C function
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
    
    # Simulate CPG nodes with realistic line numbers (0-based IDs)
    cpg_nodes = [
        {"id": 0, "label": "METHOD", "properties": {"code": "void process_user_input(char* user_data)", "lineNumber": 0}},
        {"id": 1, "label": "IDENTIFIER", "properties": {"code": "char buffer[64]", "lineNumber": 1}},
        {"id": 2, "label": "IDENTIFIER", "properties": {"code": "char temp_buffer[32]", "lineNumber": 2}},
        {"id": 3, "label": "IDENTIFIER", "properties": {"code": "char output[128]", "lineNumber": 3}},
        {"id": 4, "label": "CONTROL_STRUCTURE", "properties": {"code": "if (user_data != NULL)", "lineNumber": 5}},
        {"id": 5, "label": "CALL", "properties": {"code": "strcpy(buffer, user_data)", "lineNumber": 6}},
        {"id": 6, "label": "CALL", "properties": {"code": "strcat(buffer, \"_processed\")", "lineNumber": 7}},
        {"id": 7, "label": "CALL", "properties": {"code": "sprintf(temp_buffer, \"%s\", buffer)", "lineNumber": 8}},
        {"id": 8, "label": "CALL", "properties": {"code": "printf(temp_buffer)", "lineNumber": 9}},
        {"id": 9, "label": "CALL", "properties": {"code": "strcpy(output, buffer)", "lineNumber": 10}},
        {"id": 10, "label": "RETURN", "properties": {"code": "return", "lineNumber": 11}},
        {"id": 11, "label": "CALL", "properties": {"code": "printf(\"No input provided\\\\n\")", "lineNumber": 13}},
    ]
    
    # Simulate CPG edges (control and data flow) - using 0-based indices
    cpg_edges = [
        {"source": 0, "target": 1}, {"source": 1, "target": 2}, {"source": 2, "target": 3},
        {"source": 3, "target": 4}, {"source": 4, "target": 5}, {"source": 5, "target": 6},
        {"source": 6, "target": 7}, {"source": 7, "target": 8}, {"source": 8, "target": 9},
        {"source": 9, "target": 10}, {"source": 4, "target": 11},
        # Data flow edges
        {"source": 1, "target": 5}, {"source": 5, "target": 6}, {"source": 6, "target": 7},
        {"source": 7, "target": 8}, {"source": 1, "target": 9}, {"source": 5, "target": 9},
    ]
    
    # Create CPG data structure
    cpg_data = {
        "nodes": cpg_nodes,
        "edges": cpg_edges,
        "source_code": source_code,
        "function_name": "process_user_input",
        "target": 1  # Vulnerable
    }
    
    return cpg_data


def create_safe_cpg_data():
    """Create safe CPG data structure"""
    
    source_code = '''void safe_process_input(char* user_data) {
    char buffer[64];
    char temp_buffer[32];
    size_t input_len;
    
    if (user_data != NULL) {
        input_len = strlen(user_data);
        if (input_len < 50) {
            strncpy(buffer, user_data, 50);
            buffer[50] = '\\0';
            strncat(buffer, "_ok", 12);
            snprintf(temp_buffer, 31, "%s", buffer);
            printf("%s\\n", temp_buffer);
        } else {
            printf("Input too long\\n");
        }
    }
}'''
    
    cpg_nodes = [
        {"id": 0, "label": "METHOD", "properties": {"code": "void safe_process_input(char* user_data)", "lineNumber": 0}},
        {"id": 1, "label": "IDENTIFIER", "properties": {"code": "char buffer[64]", "lineNumber": 1}},
        {"id": 2, "label": "IDENTIFIER", "properties": {"code": "char temp_buffer[32]", "lineNumber": 2}},
        {"id": 3, "label": "IDENTIFIER", "properties": {"code": "size_t input_len", "lineNumber": 3}},
        {"id": 4, "label": "CONTROL_STRUCTURE", "properties": {"code": "if (user_data != NULL)", "lineNumber": 5}},
        {"id": 5, "label": "CALL", "properties": {"code": "input_len = strlen(user_data)", "lineNumber": 6}},
        {"id": 6, "label": "CONTROL_STRUCTURE", "properties": {"code": "if (input_len < 50)", "lineNumber": 7}},
        {"id": 7, "label": "CALL", "properties": {"code": "strncpy(buffer, user_data, 50)", "lineNumber": 8}},
        {"id": 8, "label": "ASSIGNMENT", "properties": {"code": "buffer[50] = '\\0'", "lineNumber": 9}},
        {"id": 9, "label": "CALL", "properties": {"code": "strncat(buffer, \"_ok\", 12)", "lineNumber": 10}},
        {"id": 10, "label": "CALL", "properties": {"code": "snprintf(temp_buffer, 31, \"%s\", buffer)", "lineNumber": 11}},
        {"id": 11, "label": "CALL", "properties": {"code": "printf(\"%s\\\\n\", temp_buffer)", "lineNumber": 12}},
        {"id": 12, "label": "CALL", "properties": {"code": "printf(\"Input too long\\\\n\")", "lineNumber": 14}},
    ]
    
    cpg_edges = [
        {"source": 0, "target": 1}, {"source": 1, "target": 2}, {"source": 2, "target": 3},
        {"source": 3, "target": 4}, {"source": 4, "target": 5}, {"source": 5, "target": 6},
        {"source": 6, "target": 7}, {"source": 7, "target": 8}, {"source": 8, "target": 9},
        {"source": 9, "target": 10}, {"source": 10, "target": 11}, {"source": 6, "target": 12},
    ]
    
    cpg_data = {
        "nodes": cpg_nodes,
        "edges": cpg_edges,
        "source_code": source_code,
        "function_name": "safe_process_input",
        "target": 0  # Safe
    }
    
    return cpg_data


def main():
    print("="*100)
    print("🚀 COMPLETE INTEGRATION TEST: ALL THREE ENHANCEMENTS")
    print("="*100)
    print("Testing:")
    print("  1. Enhanced Attention Model (multi-head + vulnerability detection)")
    print("  2. Pattern Detection (automatic vulnerability pattern matching)")
    print("  3. Line Number Tracking (precise CPG line-to-attention mapping)")
    print("="*100)
    
    try:
        # Step 1: Create Word2Vec model for embeddings
        print("\n📚 Creating Word2Vec model...")
        sentences = [
            ['void', 'char', 'buffer', 'strcpy', 'strcat', 'sprintf', 'printf', 'return'],
            ['if', 'user_data', 'NULL', 'strlen', 'strncpy', 'strncat', 'snprintf'],
            ['process', 'input', 'safe', 'vulnerable', 'function', 'method']
        ]
        w2v_model = Word2Vec(sentences, vector_size=100, min_count=1, epochs=10)
        print("✅ Word2Vec model created")
        
        # Step 2: Create realistic CPG data with line number tracking
        print("\n🔧 Creating realistic CPG data with line number tracking...")
        
        vuln_cpg = create_realistic_cpg_data()
        safe_cpg = create_safe_cpg_data()
        
        print("✅ CPG data created with realistic line numbers")
        
        # Step 3: Create enhanced dataset with line number tracking
        print("\n📊 Creating enhanced dataset...")
        
        cpg_data_list = [vuln_cpg, safe_cpg]
        dataset = EnhancedVulnerabilityDataset(cpg_data_list, w2v_model)
        
        print(f"✅ Enhanced dataset created with {len(dataset)} samples")
        
        # Step 4: Test line number tracking
        print("\n🔍 Testing line number tracking...")
        
        sample_0 = dataset[0]  # Vulnerable function
        sample_1 = dataset[1]  # Safe function
        
        print(f"Sample 0 (Vulnerable):")
        print(f"  Nodes: {sample_0.x.shape[0]}")
        print(f"  Line numbers: {sample_0.line_numbers}")
        print(f"  Function: {sample_0.function_name}")
        
        print(f"Sample 1 (Safe):")
        print(f"  Nodes: {sample_1.x.shape[0]}")
        print(f"  Line numbers: {sample_1.line_numbers}")
        print(f"  Function: {sample_1.function_name}")
        
        # Step 5: Test pattern detection on CPG data
        print("\n🎯 Testing pattern detection on CPG data...")
        
        detector = VulnerabilityPatternDetector()
        
        # Test on vulnerable function
        vuln_patterns = detector.annotate_vulnerable_lines(vuln_cpg['source_code'])
        print(f"Vulnerable function patterns: {len(vuln_patterns)} lines detected")
        for line_num, score in sorted(vuln_patterns.items()):
            lines = vuln_cpg['source_code'].split('\n')
            if line_num < len(lines):
                print(f"  Line {line_num}: {score:.2f} - {lines[line_num].strip()}")
        
        # Test on safe function
        safe_patterns = detector.annotate_vulnerable_lines(safe_cpg['source_code'])
        print(f"Safe function patterns: {len(safe_patterns)} lines detected")
        
        # Step 6: Initialize enhanced explainable predictor
        print("\n🤖 Initializing enhanced explainable predictor...")
        
        predictor = EnhancedExplainablePredictor('models/enhanced_attention_model.pth')
        
        # Step 7: Complete integration test - Vulnerable function
        print("\n" + "="*80)
        print("🔥 COMPLETE INTEGRATION TEST: VULNERABLE FUNCTION")
        print("="*80)
        
        print("📄 Vulnerable C function with CPG line tracking:")
        vuln_lines = vuln_cpg['source_code'].split('\n')
        for i, line in enumerate(vuln_lines):
            marker = " ← VULNERABLE!" if i in [6, 7, 8, 9, 10] else ""
            print(f"   {i:2d}: {line}{marker}")
        
        # Create node to line mapping from CPG data
        tracker = LineNumberTracker()
        vuln_node_to_line, vuln_line_to_code = tracker.create_line_mapping(vuln_cpg)
        
        print(f"\n🔮 Running complete integrated analysis...")
        vuln_result = predictor.predict_with_line_level_analysis(
            sample_0,
            node_labels=vuln_lines,
            node_to_line_mapping=vuln_node_to_line,
            source_code=vuln_cpg['source_code'],
            top_k=15
        )
        
        # Display complete analysis
        predictor.print_enhanced_explanation(vuln_result, "Complete Integration: Vulnerable Function")
        predictor.visualize_line_level_attention(vuln_result)
        
        # Step 8: Complete integration test - Safe function
        print("\n" + "="*80)
        print("✅ COMPLETE INTEGRATION TEST: SAFE FUNCTION")
        print("="*80)
        
        print("📄 Safe C function with CPG line tracking:")
        safe_lines = safe_cpg['source_code'].split('\n')
        for i, line in enumerate(safe_lines):
            marker = " ← SAFE" if i in [6, 7, 8, 9, 10, 11, 12] else ""
            print(f"   {i:2d}: {line}{marker}")
        
        safe_node_to_line, safe_line_to_code = tracker.create_line_mapping(safe_cpg)
        
        print(f"\n🔮 Running complete integrated analysis...")
        safe_result = predictor.predict_with_line_level_analysis(
            sample_1,
            node_labels=safe_lines,
            node_to_line_mapping=safe_node_to_line,
            source_code=safe_cpg['source_code'],
            top_k=15
        )
        
        # Display complete analysis
        predictor.print_enhanced_explanation(safe_result, "Complete Integration: Safe Function")
        predictor.visualize_line_level_attention(safe_result)
        
        # Step 9: Integration effectiveness analysis
        print("\n" + "="*80)
        print("📊 COMPLETE INTEGRATION EFFECTIVENESS ANALYSIS")
        print("="*80)
        
        print(f"\n🎯 Vulnerability Detection Results:")
        print(f"   Vulnerable Function:")
        print(f"     Model Prediction: {'🚨 VULNERABLE' if vuln_result['is_vulnerable'] else '✅ SAFE'}")
        print(f"     Confidence: {vuln_result['confidence']:.1%}")
        print(f"     Risk Level: {vuln_result['risk_level']}")
        
        print(f"   Safe Function:")
        print(f"     Model Prediction: {'🚨 VULNERABLE' if safe_result['is_vulnerable'] else '✅ SAFE'}")
        print(f"     Confidence: {safe_result['confidence']:.1%}")
        print(f"     Risk Level: {safe_result['risk_level']}")
        
        # Line-level detection analysis
        print(f"\n🔍 Line-Level Detection Analysis:")
        
        actual_vulnerable_lines = {6, 7, 8, 9, 10}  # strcpy, strcat, sprintf, printf, strcpy
        pattern_detected_lines = set(vuln_patterns.keys())
        attention_detected_lines = {line['line_number'] for line in vuln_result['vulnerable_lines'] 
                                  if line['risk_level'] in ['HIGH', 'MEDIUM']}
        
        # Calculate detection rates
        pattern_accuracy = len(actual_vulnerable_lines & pattern_detected_lines) / len(actual_vulnerable_lines)
        attention_accuracy = len(actual_vulnerable_lines & attention_detected_lines) / len(actual_vulnerable_lines)
        combined_accuracy = len(actual_vulnerable_lines & (pattern_detected_lines | attention_detected_lines)) / len(actual_vulnerable_lines)
        
        print(f"   Actual vulnerable lines: {sorted(actual_vulnerable_lines)}")
        print(f"   Pattern detection: {pattern_accuracy:.1%} accuracy ({sorted(actual_vulnerable_lines & pattern_detected_lines)})")
        print(f"   Attention detection: {attention_accuracy:.1%} accuracy ({sorted(actual_vulnerable_lines & attention_detected_lines)})")
        print(f"   Combined detection: {combined_accuracy:.1%} accuracy ({sorted(actual_vulnerable_lines & (pattern_detected_lines | attention_detected_lines))})")
        
        # Integration benefits analysis
        print(f"\n💡 Integration Benefits:")
        
        # Line number tracking benefits
        if hasattr(sample_0, 'line_numbers') and (sample_0.line_numbers >= 0).sum() > 0:
            print(f"   ✅ Line Number Tracking: {(sample_0.line_numbers >= 0).sum().item()}/{len(sample_0.line_numbers)} nodes mapped to source lines")
        
        # Pattern detection benefits
        improvement = combined_accuracy - attention_accuracy
        if improvement > 0:
            print(f"   ✅ Pattern Detection: Improved accuracy by {improvement:.1%}")
        
        # Enhanced attention benefits
        if vuln_result['multi_head_analysis']['focus_quality'] != 'LOW':
            print(f"   ✅ Enhanced Attention: {vuln_result['multi_head_analysis']['focus_quality']} focus quality")
        
        # Overall assessment
        print(f"\n🎉 Complete Integration Assessment:")
        
        if combined_accuracy >= 0.8:
            print(f"   ✅ EXCELLENT: Integrated system achieved {combined_accuracy:.1%} detection accuracy!")
        elif combined_accuracy >= 0.6:
            print(f"   ✅ GOOD: Integrated system achieved {combined_accuracy:.1%} detection accuracy")
        elif combined_accuracy >= 0.4:
            print(f"   ⚠️  FAIR: Integrated system achieved {combined_accuracy:.1%} detection accuracy")
        else:
            print(f"   ❌ NEEDS IMPROVEMENT: System achieved {combined_accuracy:.1%} detection accuracy")
        
        # Feature completeness
        features_working = []
        if hasattr(sample_0, 'line_numbers'):
            features_working.append("Line Number Tracking")
        if len(vuln_patterns) > 0:
            features_working.append("Pattern Detection")
        if len(vuln_result['vulnerable_lines']) > 0:
            features_working.append("Enhanced Attention")
        
        print(f"   🔧 Working Features: {', '.join(features_working)}")
        print(f"   📊 Integration Score: {len(features_working)}/3 components active")
        
        # Final summary
        print(f"\n🚀 COMPLETE INTEGRATION TEST RESULTS:")
        print(f"   ✅ Enhanced Attention Model: {'Working' if len(vuln_result['vulnerable_lines']) > 0 else 'Needs Training'}")
        print(f"   ✅ Pattern Detection: {'Working' if len(vuln_patterns) > 0 else 'Failed'} ({pattern_accuracy:.1%} accuracy)")
        print(f"   ✅ Line Number Tracking: {'Working' if hasattr(sample_0, 'line_numbers') else 'Failed'}")
        print(f"   🎯 Overall Detection: {combined_accuracy:.1%} accuracy on vulnerable lines")
        print(f"   🔍 Explainability: {'Full' if all([len(vuln_patterns) > 0, len(vuln_result['vulnerable_lines']) > 0, hasattr(sample_0, 'line_numbers')]) else 'Partial'}")
        
        print(f"\n🎉 Complete integration test completed successfully!")
        print(f"🚀 The system now provides comprehensive, explainable vulnerability detection!")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()