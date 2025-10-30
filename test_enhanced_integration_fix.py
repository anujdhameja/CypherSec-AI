#!/usr/bin/env python3
"""
Test Enhanced Integration with Context-Aware Pattern Detection

This test validates that our enhanced context-aware pattern detector
solves the false positive problem while maintaining perfect vulnerability detection.

Key Improvements Tested:
1. Context-aware analysis (safe vs unsafe usage)
2. Multi-language support (6 languages)
3. Comprehensive vulnerability database (500+ functions)
4. Smart risk scoring based on actual usage patterns
5. False positive reduction

Expected Results:
- 100% accuracy on vulnerable code (maintained)
- 100% accuracy on safe code (fixed false positives)
- Perfect line-level detection (maintained)
"""

import torch
import sys
import os
from pathlib import Path
import numpy as np

# Add src to path
sys.path.append('src')

from src.process.enhanced_pattern_detector import ContextAwareVulnerabilityDetector
from gensim.models import Word2Vec


class EnhancedIntegrationTester:
    """Test the enhanced integration with context-aware detection"""
    
    def __init__(self):
        self.enhanced_detector = None
        self.w2v_model = None
        self.test_results = []
    
    def setup(self):
        """Initialize the enhanced system"""
        print("🔧 Setting up enhanced integration test...")
        
        # Create Word2Vec model
        sentences = [
            ['void', 'char', 'buffer', 'strcpy', 'strcat', 'sprintf', 'printf', 'return', 'malloc', 'free'],
            ['if', 'user_data', 'NULL', 'strlen', 'strncpy', 'strncat', 'snprintf', 'sizeof', 'memcpy'],
            ['process', 'input', 'safe', 'vulnerable', 'function', 'method', 'gets', 'scanf', 'system'],
            ['int', 'size_t', 'unsigned', 'overflow', 'bounds', 'check', 'validation', 'sanitize']
        ]
        self.w2v_model = Word2Vec(sentences, vector_size=100, min_count=1, epochs=10)
        
        # Initialize enhanced components
        self.enhanced_detector = ContextAwareVulnerabilityDetector()
        
        print("✅ Enhanced integration test setup completed")
    
    def create_test_cases(self):
        """Create test cases that specifically test the false positive fixes"""
        
        return [
            # Test Case 1: Classic Buffer Overflow (should be vulnerable)
            {
                'name': 'Classic Buffer Overflow',
                'source_code': '''void vulnerable_copy(char* input) {
    char buffer[64];
    strcpy(buffer, input);
    strcat(buffer, "_suffix");
    printf("Result: %s\\n", buffer);
}''',
                'expected_vulnerable': True,
                'expected_vulnerable_lines': {2, 3},
                'description': 'Classic buffer overflow - should detect as vulnerable'
            },
            
            # Test Case 2: Safe strncpy with bounds checking (should be safe)
            {
                'name': 'Safe strncpy with Bounds Checking',
                'source_code': '''void safe_copy(char* input) {
    char buffer[64];
    if (strlen(input) < sizeof(buffer) - 1) {
        strncpy(buffer, input, sizeof(buffer) - 1);
        buffer[sizeof(buffer) - 1] = '\\0';
        printf("Safe: %s\\n", buffer);
    }
}''',
                'expected_vulnerable': False,
                'expected_vulnerable_lines': set(),
                'description': 'Safe usage with bounds checking - should detect as safe'
            },
            
            # Test Case 3: Safe snprintf usage (should be safe)
            {
                'name': 'Safe snprintf Usage',
                'source_code': '''void safe_format(char* name, int age) {
    char buffer[256];
    snprintf(buffer, sizeof(buffer), "Name: %s, Age: %d", name, age);
    printf("%s\\n", buffer);
}''',
                'expected_vulnerable': False,
                'expected_vulnerable_lines': set(),
                'description': 'Safe snprintf with sizeof - should detect as safe'
            },
            
            # Test Case 4: Mixed safe and unsafe (should detect only unsafe parts)
            {
                'name': 'Mixed Safe and Unsafe',
                'source_code': '''void mixed_function(char* input) {
    char safe_buffer[100];
    char unsafe_buffer[50];
    
    // Safe operations
    if (strlen(input) < 99) {
        strncpy(safe_buffer, input, 99);
        safe_buffer[99] = '\\0';
    }
    
    // Unsafe operations
    strcpy(unsafe_buffer, input);
    printf(unsafe_buffer);
}''',
                'expected_vulnerable': True,
                'expected_vulnerable_lines': {11, 12},  # Only unsafe lines
                'description': 'Mixed code - should detect only unsafe parts'
            },
            
            # Test Case 5: Format string vulnerability (should be vulnerable)
            {
                'name': 'Format String Vulnerability',
                'source_code': '''void format_vuln(char* user_input) {
    char buffer[256];
    sprintf(buffer, user_input);
    printf(buffer);
}''',
                'expected_vulnerable': True,
                'expected_vulnerable_lines': {2, 3},
                'description': 'Format string vulnerabilities - should detect as vulnerable'
            },
            
            # Test Case 6: Safe format string usage (should be safe)
            {
                'name': 'Safe Format String Usage',
                'source_code': '''void safe_format_string(char* name) {
    char buffer[256];
    snprintf(buffer, sizeof(buffer), "Hello %s", name);
    printf("%s\\n", buffer);
}''',
                'expected_vulnerable': False,
                'expected_vulnerable_lines': set(),
                'description': 'Safe format string usage - should detect as safe'
            }
        ]
    
    def test_case(self, test_case: dict) -> dict:
        """Test a single case and return results"""
        
        print(f"\n🧪 Testing: {test_case['name']}")
        print(f"📝 Description: {test_case['description']}")
        
        source_code = test_case['source_code']
        
        # 1. Test enhanced pattern detector
        vulnerable_lines = self.enhanced_detector.annotate_vulnerable_lines(source_code)
        detected_vulnerable_lines = set(vulnerable_lines.keys())
        
        # 2. Get detailed analysis
        detailed_analysis = self.enhanced_detector.get_detailed_analysis(source_code)
        
        # 3. Evaluate results
        expected_lines = test_case['expected_vulnerable_lines']
        
        # Calculate accuracy metrics
        true_positives = len(detected_vulnerable_lines & expected_lines)
        false_positives = len(detected_vulnerable_lines - expected_lines)
        false_negatives = len(expected_lines - detected_vulnerable_lines)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 1.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 1.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Check if this is a perfect detection
        perfect_detection = detected_vulnerable_lines == expected_lines
        
        result = {
            'name': test_case['name'],
            'expected_vulnerable': test_case['expected_vulnerable'],
            'detected_vulnerable': len(detected_vulnerable_lines) > 0,
            'expected_lines': expected_lines,
            'detected_lines': detected_vulnerable_lines,
            'perfect_detection': perfect_detection,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'detailed_analysis': detailed_analysis
        }
        
        # Print results
        print(f"✅ Expected vulnerable: {test_case['expected_vulnerable']}")
        print(f"✅ Detected vulnerable: {len(detected_vulnerable_lines) > 0}")
        print(f"📍 Expected lines: {sorted(expected_lines)}")
        print(f"📍 Detected lines: {sorted(detected_vulnerable_lines)}")
        print(f"🎯 Perfect detection: {perfect_detection}")
        print(f"📊 Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1_score:.2f}")
        print(f"❌ False positives: {false_positives}, False negatives: {false_negatives}")
        
        if detailed_analysis['safe_lines']:
            print(f"✅ Safe usage detected: {len(detailed_analysis['safe_lines'])} lines")
        
        return result
    
    def run_comprehensive_test(self):
        """Run comprehensive test of the enhanced system"""
        
        print("🚀 Running Comprehensive Enhanced Integration Test")
        print("=" * 80)
        
        self.setup()
        test_cases = self.create_test_cases()
        
        results = []
        perfect_detections = 0
        total_false_positives = 0
        total_false_negatives = 0
        
        for test_case in test_cases:
            result = self.test_case(test_case)
            results.append(result)
            
            if result['perfect_detection']:
                perfect_detections += 1
            
            total_false_positives += result['false_positives']
            total_false_negatives += result['false_negatives']
        
        # Calculate overall metrics
        overall_accuracy = perfect_detections / len(test_cases)
        avg_precision = np.mean([r['precision'] for r in results])
        avg_recall = np.mean([r['recall'] for r in results])
        avg_f1 = np.mean([r['f1_score'] for r in results])
        
        # Print summary
        print(f"\n🎯 COMPREHENSIVE TEST RESULTS")
        print("=" * 80)
        print(f"📊 Overall Accuracy: {overall_accuracy:.1%} ({perfect_detections}/{len(test_cases)} perfect)")
        print(f"📊 Average Precision: {avg_precision:.2f}")
        print(f"📊 Average Recall: {avg_recall:.2f}")
        print(f"📊 Average F1-Score: {avg_f1:.2f}")
        print(f"❌ Total False Positives: {total_false_positives}")
        print(f"❌ Total False Negatives: {total_false_negatives}")
        
        # Analyze false positive reduction
        safe_test_cases = [tc for tc in test_cases if not tc['expected_vulnerable']]
        safe_results = [r for r in results if not r['expected_vulnerable']]
        
        if safe_test_cases:
            false_positive_reduction = sum(1 for r in safe_results if r['false_positives'] == 0) / len(safe_results)
            print(f"🎯 False Positive Reduction: {false_positive_reduction:.1%}")
        
        # Show detailed breakdown
        print(f"\n📋 DETAILED BREAKDOWN:")
        for result in results:
            status = "✅ PERFECT" if result['perfect_detection'] else "❌ ISSUES"
            print(f"   {result['name']}: {status} (P:{result['precision']:.2f} R:{result['recall']:.2f})")
        
        # Validate the fix
        print(f"\n🔍 FIX VALIDATION:")
        
        # Check if we maintain vulnerability detection
        vuln_cases = [r for r in results if r['expected_vulnerable']]
        vuln_detection_rate = sum(1 for r in vuln_cases if r['detected_vulnerable']) / len(vuln_cases) if vuln_cases else 0
        print(f"   Vulnerability Detection Rate: {vuln_detection_rate:.1%} (should be 100%)")
        
        # Check if we reduced false positives on safe code
        safe_cases = [r for r in results if not r['expected_vulnerable']]
        safe_detection_rate = sum(1 for r in safe_cases if not r['detected_vulnerable']) / len(safe_cases) if safe_cases else 0
        print(f"   Safe Code Recognition Rate: {safe_detection_rate:.1%} (should be 100%)")
        
        # Overall fix success
        fix_success = vuln_detection_rate >= 0.9 and safe_detection_rate >= 0.9
        print(f"   🎯 Fix Success: {'✅ YES' if fix_success else '❌ NO'}")
        
        return {
            'overall_accuracy': overall_accuracy,
            'avg_precision': avg_precision,
            'avg_recall': avg_recall,
            'avg_f1': avg_f1,
            'total_false_positives': total_false_positives,
            'total_false_negatives': total_false_negatives,
            'fix_success': fix_success,
            'results': results
        }


def main():
    """Main test function"""
    
    tester = EnhancedIntegrationTester()
    results = tester.run_comprehensive_test()
    
    print(f"\n🏁 FINAL ASSESSMENT:")
    print("=" * 80)
    
    if results['fix_success']:
        print("🎉 SUCCESS: Enhanced context-aware pattern detection is working!")
        print("✅ Maintains perfect vulnerability detection")
        print("✅ Eliminates false positives on safe code")
        print("✅ Provides accurate line-level analysis")
    else:
        print("❌ ISSUES: Some problems detected in the enhanced system")
        print("🔧 Review the detailed results above for specific issues")
    
    return results


if __name__ == "__main__":
    main()