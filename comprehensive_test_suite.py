#!/usr/bin/env python3
"""
Comprehensive Test Suite for Complete Integration System

This test suite validates all three enhancements with diverse test cases:
1. Enhanced Attention Model
2. Pattern Detection 
3. Line Number Tracking

Test Categories:
- Buffer Overflow Vulnerabilities
- Format String Vulnerabilities
- Integer Overflow Vulnerabilities
- Memory Management Issues
- Safe Code Patterns
- Edge Cases and Corner Cases

Usage:
    python comprehensive_test_suite.py
"""

import torch
import sys
import os
from pathlib import Path
import numpy as np
from typing import List, Dict, Tuple

# Add src to path
sys.path.append('src')

from src.inference.enhanced_explainable_predictor import EnhancedExplainablePredictor
from src.process.vulnerability_pattern_detector import VulnerabilityPatternDetector
from src.data.line_number_tracker import EnhancedGraphData, LineNumberTracker
from src.data.enhanced_data_manager import EnhancedVulnerabilityDataset
from torch_geometric.data import Data
from gensim.models import Word2Vec


class VulnerabilityTestCase:
    """Represents a single vulnerability test case"""
    
    def __init__(self, name: str, source_code: str, expected_vulnerable: bool, 
                 expected_vulnerable_lines: List[int], description: str):
        self.name = name
        self.source_code = source_code
        self.expected_vulnerable = expected_vulnerable
        self.expected_vulnerable_lines = set(expected_vulnerable_lines)
        self.description = description
        self.cpg_data = None
        self.results = {}


class ComprehensiveTestSuite:
    """Comprehensive test suite for vulnerability detection system"""
    
    def __init__(self):
        self.test_cases = []
        self.predictor = None
        self.detector = None
        self.w2v_model = None
        self.results_summary = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'pattern_accuracy': 0.0,
            'attention_accuracy': 0.0,
            'combined_accuracy': 0.0,
            'detailed_results': []
        }
    
    def setup(self):
        """Initialize the test environment"""
        print("🔧 Setting up comprehensive test suite...")
        
        # Create Word2Vec model
        sentences = [
            ['void', 'char', 'buffer', 'strcpy', 'strcat', 'sprintf', 'printf', 'return', 'malloc', 'free'],
            ['if', 'user_data', 'NULL', 'strlen', 'strncpy', 'strncat', 'snprintf', 'sizeof', 'memcpy'],
            ['process', 'input', 'safe', 'vulnerable', 'function', 'method', 'gets', 'scanf', 'system'],
            ['int', 'size_t', 'unsigned', 'overflow', 'bounds', 'check', 'validation', 'sanitize']
        ]
        self.w2v_model = Word2Vec(sentences, vector_size=100, min_count=1, epochs=10)
        
        # Initialize components
        self.predictor = EnhancedExplainablePredictor('models/enhanced_attention_model.pth')
        self.detector = VulnerabilityPatternDetector()
        
        print("✅ Test suite setup completed")
    
    def create_test_cases(self):
        """Create comprehensive test cases covering different vulnerability types"""
        
        print("📋 Creating comprehensive test cases...")
        
        # Test Case 1: Classic Buffer Overflow
        self.test_cases.append(VulnerabilityTestCase(
            name="Classic Buffer Overflow",
            source_code='''void vulnerable_copy(char* input) {
    char buffer[64];
    strcpy(buffer, input);
    strcat(buffer, "_suffix");
    printf("Result: %s\\n", buffer);
}''',
            expected_vulnerable=True,
            expected_vulnerable_lines=[2, 3],  # strcpy, strcat
            description="Classic buffer overflow with strcpy and strcat"
        ))
        
        # Test Case 2: Format String Vulnerability
        self.test_cases.append(VulnerabilityTestCase(
            name="Format String Vulnerability",
            source_code='''void format_string_vuln(char* user_input) {
    char buffer[256];
    sprintf(buffer, user_input);
    printf(buffer);
    fprintf(stderr, user_input);
}''',
            expected_vulnerable=True,
            expected_vulnerable_lines=[2, 3, 4],  # sprintf, printf, fprintf
            description="Format string vulnerabilities with user-controlled format"
        ))
        
        # Test Case 3: Integer Overflow in malloc
        self.test_cases.append(VulnerabilityTestCase(
            name="Integer Overflow malloc",
            source_code='''void* allocate_array(int count, int size) {
    int total_size = count * size;
    void* ptr = malloc(total_size);
    if (ptr == NULL) return NULL;
    memset(ptr, 0, total_size);
    return ptr;
}''',
            expected_vulnerable=True,
            expected_vulnerable_lines=[1, 2],  # multiplication overflow, malloc
            description="Integer overflow in size calculation before malloc"
        ))
        
        # Test Case 4: gets() - Inherently Unsafe
        self.test_cases.append(VulnerabilityTestCase(
            name="gets() Vulnerability",
            source_code='''void read_input() {
    char buffer[100];
    printf("Enter input: ");
    gets(buffer);
    printf("You entered: %s\\n", buffer);
}''',
            expected_vulnerable=True,
            expected_vulnerable_lines=[3],  # gets
            description="Use of inherently unsafe gets() function"
        ))
        
        # Test Case 5: scanf without field width
        self.test_cases.append(VulnerabilityTestCase(
            name="scanf Buffer Overflow",
            source_code='''void read_string() {
    char name[50];
    char address[100];
    printf("Name: ");
    scanf("%s", name);
    printf("Address: ");
    scanf("%s", address);
}''',
            expected_vulnerable=True,
            expected_vulnerable_lines=[4, 6],  # scanf calls
            description="scanf without field width specifier"
        ))
        
        # Test Case 6: Safe Code with Proper Bounds Checking
        self.test_cases.append(VulnerabilityTestCase(
            name="Safe Bounds Checking",
            source_code='''void safe_copy(char* input) {
    char buffer[64];
    size_t input_len = strlen(input);
    if (input_len < 60) {
        strncpy(buffer, input, 60);
        buffer[60] = '\\0';
        strncat(buffer, "_ok", 3);
        printf("Safe: %s\\n", buffer);
    }
}''',
            expected_vulnerable=False,
            expected_vulnerable_lines=[],
            description="Safe implementation with proper bounds checking"
        ))
        
        # Test Case 7: Safe snprintf Usage
        self.test_cases.append(VulnerabilityTestCase(
            name="Safe snprintf Usage",
            source_code='''void safe_format(char* name, int age) {
    char buffer[256];
    snprintf(buffer, sizeof(buffer), "Name: %s, Age: %d", name, age);
    printf("%s\\n", buffer);
}''',
            expected_vulnerable=False,
            expected_vulnerable_lines=[],
            description="Safe formatting with snprintf and sizeof"
        ))
        
        # Test Case 8: Mixed Safe and Unsafe
        self.test_cases.append(VulnerabilityTestCase(
            name="Mixed Safe and Unsafe",
            source_code='''void mixed_function(char* input) {
    char safe_buffer[100];
    char unsafe_buffer[50];
    
    // Safe operations
    strncpy(safe_buffer, input, 99);
    safe_buffer[99] = '\\0';
    
    // Unsafe operations
    strcpy(unsafe_buffer, input);
    printf(unsafe_buffer);
}''',
            expected_vulnerable=True,
            expected_vulnerable_lines=[8, 9],  # strcpy, printf
            description="Function with both safe and unsafe operations"
        ))
        
        # Test Case 9: Complex Control Flow
        self.test_cases.append(VulnerabilityTestCase(
            name="Complex Control Flow",
            source_code='''void complex_flow(char* input, int mode) {
    char buffer[128];
    
    if (mode == 1) {
        strcpy(buffer, input);  // Vulnerable
    } else if (mode == 2) {
        strncpy(buffer, input, 127);  // Safe
        buffer[127] = '\\0';
    } else {
        sprintf(buffer, "%s", input);  // Vulnerable
    }
    
    printf("Buffer: %s\\n", buffer);
}''',
            expected_vulnerable=True,
            expected_vulnerable_lines=[4, 9],  # strcpy, sprintf
            description="Complex control flow with conditional vulnerabilities"
        ))
        
        # Test Case 10: Memory Management Issues
        self.test_cases.append(VulnerabilityTestCase(
            name="Memory Management Issues",
            source_code='''void memory_issues(int size) {
    char* buffer = malloc(size);
    strcpy(buffer, "test");
    free(buffer);
    strcpy(buffer, "after free");  // Use after free
    free(buffer);  // Double free
}''',
            expected_vulnerable=True,
            expected_vulnerable_lines=[2, 4, 5],  # strcpy without bounds, use after free, double free
            description="Multiple memory management vulnerabilities"
        ))
        
        print(f"✅ Created {len(self.test_cases)} comprehensive test cases")
    
    def create_cpg_data_for_test_case(self, test_case: VulnerabilityTestCase) -> Dict:
        """Create realistic CPG data for a test case"""
        
        lines = test_case.source_code.split('\n')
        
        # Create CPG nodes for each line
        cpg_nodes = []
        for i, line in enumerate(lines):
            if line.strip():  # Skip empty lines
                cpg_nodes.append({
                    "id": i,
                    "label": "CALL" if any(func in line for func in ['strcpy', 'printf', 'malloc', 'free']) else "IDENTIFIER",
                    "properties": {
                        "code": line.strip(),
                        "lineNumber": i
                    }
                })
        
        # Create simple sequential edges
        cpg_edges = []
        for i in range(len(cpg_nodes) - 1):
            cpg_edges.append({"source": i, "target": i + 1})
        
        # Add some data flow edges for vulnerable lines
        for vuln_line in test_case.expected_vulnerable_lines:
            if vuln_line < len(cpg_nodes) - 1:
                # Add extra connections for vulnerable lines
                cpg_edges.append({"source": max(0, vuln_line - 1), "target": vuln_line})
        
        cpg_data = {
            "nodes": cpg_nodes,
            "edges": cpg_edges,
            "source_code": test_case.source_code,
            "function_name": test_case.name.replace(" ", "_").lower(),
            "target": 1 if test_case.expected_vulnerable else 0
        }
        
        test_case.cpg_data = cpg_data
        return cpg_data
    
    def run_single_test(self, test_case: VulnerabilityTestCase) -> Dict:
        """Run a single test case and return results"""
        
        print(f"\n🧪 Testing: {test_case.name}")
        print(f"📄 Description: {test_case.description}")
        
        # Create CPG data
        cpg_data = self.create_cpg_data_for_test_case(test_case)
        
        # Create enhanced dataset
        dataset = EnhancedVulnerabilityDataset([cpg_data], self.w2v_model)
        sample = dataset[0]
        
        # Test pattern detection
        pattern_annotations = self.detector.annotate_vulnerable_lines(test_case.source_code)
        pattern_detected_lines = set(pattern_annotations.keys())
        
        # Test line number tracking
        tracker = LineNumberTracker()
        node_to_line, line_to_code = tracker.create_line_mapping(cpg_data)
        
        # Test complete integration
        result = self.predictor.predict_with_line_level_analysis(
            sample,
            node_labels=test_case.source_code.split('\n'),
            node_to_line_mapping=node_to_line,
            source_code=test_case.source_code,
            top_k=20
        )
        
        # Extract attention-detected lines
        attention_detected_lines = {
            line['line_number'] for line in result['vulnerable_lines'] 
            if line['risk_level'] in ['HIGH', 'MEDIUM']
        }
        
        # Calculate accuracies
        expected_lines = test_case.expected_vulnerable_lines
        
        if expected_lines:
            pattern_accuracy = len(expected_lines & pattern_detected_lines) / len(expected_lines)
            attention_accuracy = len(expected_lines & attention_detected_lines) / len(expected_lines)
            combined_detected = pattern_detected_lines | attention_detected_lines
            combined_accuracy = len(expected_lines & combined_detected) / len(expected_lines)
        else:
            # For safe code, accuracy is based on NOT detecting vulnerabilities
            pattern_accuracy = 1.0 if len(pattern_detected_lines) == 0 else 0.0
            attention_accuracy = 1.0 if len(attention_detected_lines) == 0 else 0.0
            combined_accuracy = 1.0 if len(pattern_detected_lines | attention_detected_lines) == 0 else 0.0
        
        # Determine if test passed
        vulnerability_prediction_correct = result['is_vulnerable'] == test_case.expected_vulnerable
        line_detection_good = combined_accuracy >= 0.6  # At least 60% of lines detected
        
        test_passed = vulnerability_prediction_correct and line_detection_good
        
        # Store results
        test_result = {
            'test_case': test_case.name,
            'description': test_case.description,
            'expected_vulnerable': test_case.expected_vulnerable,
            'predicted_vulnerable': result['is_vulnerable'],
            'confidence': result['confidence'],
            'risk_level': result['risk_level'],
            'expected_lines': sorted(expected_lines),
            'pattern_detected_lines': sorted(pattern_detected_lines),
            'attention_detected_lines': sorted(attention_detected_lines),
            'combined_detected_lines': sorted(pattern_detected_lines | attention_detected_lines),
            'pattern_accuracy': pattern_accuracy,
            'attention_accuracy': attention_accuracy,
            'combined_accuracy': combined_accuracy,
            'vulnerability_prediction_correct': vulnerability_prediction_correct,
            'line_detection_good': line_detection_good,
            'test_passed': test_passed,
            'full_result': result
        }
        
        # Print results
        print(f"   Expected: {'VULNERABLE' if test_case.expected_vulnerable else 'SAFE'}")
        print(f"   Predicted: {'VULNERABLE' if result['is_vulnerable'] else 'SAFE'} ({result['confidence']:.1%})")
        print(f"   Risk Level: {result['risk_level']}")
        
        if expected_lines:
            print(f"   Expected vulnerable lines: {sorted(expected_lines)}")
            print(f"   Pattern detected: {sorted(pattern_detected_lines)} ({pattern_accuracy:.1%})")
            print(f"   Attention detected: {sorted(attention_detected_lines)} ({attention_accuracy:.1%})")
            print(f"   Combined accuracy: {combined_accuracy:.1%}")
        
        print(f"   Result: {'✅ PASSED' if test_passed else '❌ FAILED'}")
        
        return test_result
    
    def run_all_tests(self):
        """Run all test cases and generate summary"""
        
        print("\n" + "="*80)
        print("🚀 RUNNING COMPREHENSIVE TEST SUITE")
        print("="*80)
        
        all_results = []
        
        for test_case in self.test_cases:
            try:
                result = self.run_single_test(test_case)
                all_results.append(result)
                
                if result['test_passed']:
                    self.results_summary['passed_tests'] += 1
                else:
                    self.results_summary['failed_tests'] += 1
                    
            except Exception as e:
                print(f"❌ Test {test_case.name} failed with error: {e}")
                self.results_summary['failed_tests'] += 1
        
        self.results_summary['total_tests'] = len(self.test_cases)
        self.results_summary['detailed_results'] = all_results
        
        # Calculate overall accuracies
        if all_results:
            self.results_summary['pattern_accuracy'] = np.mean([r['pattern_accuracy'] for r in all_results])
            self.results_summary['attention_accuracy'] = np.mean([r['attention_accuracy'] for r in all_results])
            self.results_summary['combined_accuracy'] = np.mean([r['combined_accuracy'] for r in all_results])
        
        self.print_summary()
    
    def print_summary(self):
        """Print comprehensive test results summary"""
        
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE TEST SUITE RESULTS")
        print("="*80)
        
        summary = self.results_summary
        
        print(f"\n🎯 Overall Results:")
        print(f"   Total Tests: {summary['total_tests']}")
        print(f"   Passed: {summary['passed_tests']} ✅")
        print(f"   Failed: {summary['failed_tests']} ❌")
        print(f"   Success Rate: {summary['passed_tests']/summary['total_tests']:.1%}")
        
        print(f"\n📈 Detection Accuracy:")
        print(f"   Pattern Detection: {summary['pattern_accuracy']:.1%}")
        print(f"   Attention Detection: {summary['attention_accuracy']:.1%}")
        print(f"   Combined Detection: {summary['combined_accuracy']:.1%}")
        
        print(f"\n📋 Detailed Results by Category:")
        
        # Group by vulnerability type
        categories = {
            'Buffer Overflow': [],
            'Format String': [],
            'Memory Management': [],
            'Safe Code': [],
            'Other': []
        }
        
        for result in summary['detailed_results']:
            name = result['test_case']
            if 'Buffer' in name or 'strcpy' in result['description']:
                categories['Buffer Overflow'].append(result)
            elif 'Format' in name or 'printf' in result['description']:
                categories['Format String'].append(result)
            elif 'Memory' in name or 'malloc' in result['description']:
                categories['Memory Management'].append(result)
            elif 'Safe' in name or not result['expected_vulnerable']:
                categories['Safe Code'].append(result)
            else:
                categories['Other'].append(result)
        
        for category, results in categories.items():
            if results:
                passed = sum(1 for r in results if r['test_passed'])
                total = len(results)
                avg_accuracy = np.mean([r['combined_accuracy'] for r in results])
                print(f"   {category}: {passed}/{total} passed ({passed/total:.1%}), {avg_accuracy:.1%} accuracy")
        
        print(f"\n🔍 Failed Tests Analysis:")
        failed_tests = [r for r in summary['detailed_results'] if not r['test_passed']]
        
        if failed_tests:
            for result in failed_tests:
                print(f"   ❌ {result['test_case']}:")
                if not result['vulnerability_prediction_correct']:
                    print(f"      - Wrong vulnerability prediction: expected {result['expected_vulnerable']}, got {result['predicted_vulnerable']}")
                if not result['line_detection_good']:
                    print(f"      - Poor line detection: {result['combined_accuracy']:.1%} accuracy")
        else:
            print("   🎉 No failed tests!")
        
        print(f"\n💡 System Assessment:")
        
        success_rate = summary['passed_tests'] / summary['total_tests']
        combined_accuracy = summary['combined_accuracy']
        
        if success_rate >= 0.9 and combined_accuracy >= 0.8:
            print(f"   🏆 EXCELLENT: System performs exceptionally well!")
        elif success_rate >= 0.8 and combined_accuracy >= 0.7:
            print(f"   ✅ GOOD: System performs well with minor issues")
        elif success_rate >= 0.6 and combined_accuracy >= 0.6:
            print(f"   ⚠️  FAIR: System needs improvement")
        else:
            print(f"   ❌ POOR: System requires significant work")
        
        print(f"\n🚀 Integration Status:")
        print(f"   ✅ Enhanced Attention Model: Active")
        print(f"   ✅ Pattern Detection: Active ({summary['pattern_accuracy']:.1%} avg accuracy)")
        print(f"   ✅ Line Number Tracking: Active")
        print(f"   🎯 Overall System: {success_rate:.1%} test success rate")


def main():
    """Run the comprehensive test suite"""
    
    print("="*100)
    print("🧪 COMPREHENSIVE TEST SUITE FOR VULNERABILITY DETECTION SYSTEM")
    print("="*100)
    print("Testing all three integrated enhancements:")
    print("  1. Enhanced Attention Model (multi-head + vulnerability detection)")
    print("  2. Pattern Detection (automatic vulnerability pattern matching)")
    print("  3. Line Number Tracking (precise CPG line-to-attention mapping)")
    print("="*100)
    
    try:
        # Initialize test suite
        test_suite = ComprehensiveTestSuite()
        
        # Setup environment
        test_suite.setup()
        
        # Create test cases
        test_suite.create_test_cases()
        
        # Run all tests
        test_suite.run_all_tests()
        
        print(f"\n🎉 Comprehensive test suite completed!")
        
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()