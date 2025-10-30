#!/usr/bin/env python3
"""
Comprehensive Test of Enhanced Context-Aware Vulnerability Detection Pipeline

This script tests the complete enhanced pipeline with real source code examples
to demonstrate the improvements in accuracy and false positive reduction.

Test Cases:
1. Classic buffer overflow vulnerabilities
2. Safe usage patterns with bounds checking
3. Format string vulnerabilities
4. Mixed safe and unsafe code
5. Multi-language examples
"""

import sys
import os
from pathlib import Path
import torch
from torch_geometric.data import Data

# Add src to path
sys.path.append('src')

try:
    from src.inference.enhanced_vulnerability_predictor import EnhancedVulnerabilityPredictor
    from src.process.vulnerability_pattern_detector import VulnerabilityPatternDetector
    ENHANCED_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Enhanced predictor not available: {e}")
    print("   Falling back to pattern detection only")
    from src.process.vulnerability_pattern_detector import VulnerabilityPatternDetector
    ENHANCED_AVAILABLE = False


class EnhancedPipelineTester:
    """Comprehensive tester for the enhanced vulnerability detection pipeline"""
    
    def __init__(self):
        self.test_results = []
        
        if ENHANCED_AVAILABLE:
            print("🚀 Initializing Enhanced Vulnerability Predictor...")
            try:
                self.predictor = EnhancedVulnerabilityPredictor()
                self.enhanced_mode = True
            except Exception as e:
                print(f"⚠️ Could not initialize enhanced predictor: {e}")
                print("   Using pattern detection only")
                self.pattern_detector = VulnerabilityPatternDetector(use_enhanced=True)
                self.enhanced_mode = False
        else:
            print("📋 Using pattern detection only")
            self.pattern_detector = VulnerabilityPatternDetector(use_enhanced=True)
            self.enhanced_mode = False
    
    def run_comprehensive_test(self):
        """Run comprehensive test suite"""
        
        print("🧪 COMPREHENSIVE ENHANCED PIPELINE TEST")
        print("="*80)
        
        test_cases = self.create_test_cases()
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n📋 TEST CASE {i}: {test_case['name']}")
            print("-" * 60)
            
            result = self.test_single_case(test_case)
            self.test_results.append(result)
            
            self.print_test_result(result)
        
        # Print summary
        self.print_test_summary()
    
    def create_test_cases(self):
        """Create comprehensive test cases"""
        
        return [
            {
                'name': 'Classic Buffer Overflow (C)',
                'language': 'C',
                'source_code': '''
void vulnerable_function(char* user_input) {
    char buffer[64];
    char temp[32];
    
    // Classic buffer overflow vulnerabilities
    strcpy(buffer, user_input);           // No bounds checking
    strcat(buffer, "_suffix");            // No bounds checking
    sprintf(temp, "%s_temp", buffer);     // No bounds checking
    gets(buffer);                         // Inherently unsafe
    
    printf("Processed: %s\\n", temp);
}
''',
                'expected_vulnerable': True,
                'expected_issues': ['strcpy', 'strcat', 'sprintf', 'gets'],
                'description': 'Multiple buffer overflow vulnerabilities without bounds checking'
            },
            
            {
                'name': 'Safe Buffer Operations (C)',
                'language': 'C',
                'source_code': '''
void safe_function(char* user_input) {
    char buffer[128];
    char temp[64];
    
    // Safe operations with proper bounds checking
    if (user_input != NULL && strlen(user_input) < sizeof(buffer) - 20) {
        strncpy(buffer, user_input, sizeof(buffer) - 20);
        buffer[sizeof(buffer) - 1] = '\\0';
        
        strncat(buffer, "_suffix", 10);
        
        snprintf(temp, sizeof(temp), "Result: %s", buffer);
        printf("%s\\n", temp);
    }
}
''',
                'expected_vulnerable': False,
                'expected_issues': [],
                'description': 'Safe buffer operations with proper bounds checking and validation'
            },
            
            {
                'name': 'Format String Vulnerabilities (C)',
                'language': 'C',
                'source_code': '''
void format_string_vuln(char* user_data) {
    char buffer[256];
    
    // Format string vulnerabilities
    printf(user_data);                    // Direct user input as format
    sprintf(buffer, user_data);           // User input as format string
    fprintf(stderr, user_data);           // User input as format string
    
    // This would be safe
    printf("User data: %s\\n", user_data);
}
''',
                'expected_vulnerable': True,
                'expected_issues': ['printf', 'sprintf', 'fprintf'],
                'description': 'Format string vulnerabilities with user-controlled format strings'
            },
            
            {
                'name': 'Mixed Safe and Unsafe (C)',
                'language': 'C',
                'source_code': '''
void mixed_function(char* input) {
    char safe_buffer[200];
    char unsafe_buffer[50];
    
    // Safe operations
    if (input != NULL && strlen(input) < 180) {
        strncpy(safe_buffer, input, 180);
        safe_buffer[180] = '\\0';
        printf("Safe: %s\\n", safe_buffer);
    }
    
    // Unsafe operations
    strcpy(unsafe_buffer, input);         // No bounds check
    printf(unsafe_buffer);                // Format string vuln
    
    // More safe operations
    snprintf(safe_buffer, sizeof(safe_buffer), "Processed: %s", input);
}
''',
                'expected_vulnerable': True,
                'expected_issues': ['strcpy', 'printf'],
                'description': 'Mixed code with both safe and unsafe patterns'
            },
            
            {
                'name': 'Python Code Injection (Python)',
                'language': 'Python',
                'source_code': '''
def process_user_code(user_input):
    # Dangerous code execution
    result = eval(user_input)             # Code injection risk
    exec(user_input)                      # Code execution risk
    
    # Safe alternative would be:
    # import json
    # result = json.loads(user_input)
    
    return result

def safe_processing(user_data):
    import json
    try:
        # Safe JSON parsing instead of eval
        data = json.loads(user_data)
        return data
    except json.JSONDecodeError:
        return None
''',
                'expected_vulnerable': True,
                'expected_issues': ['eval', 'exec'],
                'description': 'Python code injection vulnerabilities vs safe alternatives'
            },
            
            {
                'name': 'JavaScript XSS Vulnerabilities (JavaScript)',
                'language': 'JavaScript',
                'source_code': '''
function processUserInput(userInput) {
    // XSS vulnerabilities
    document.getElementById("output").innerHTML = userInput;  // XSS risk
    document.write(userInput);                               // XSS risk
    eval("var result = " + userInput);                       // Code injection
    
    // Safe alternatives
    document.getElementById("safe").textContent = userInput; // Safe
    
    setTimeout(function() {
        console.log("Safe timer");
    }, 1000);
}
''',
                'expected_vulnerable': True,
                'expected_issues': ['innerHTML', 'document.write', 'eval'],
                'description': 'JavaScript XSS and code injection vulnerabilities'
            }
        ]
    
    def test_single_case(self, test_case):
        """Test a single case"""
        
        source_code = test_case['source_code']
        
        if self.enhanced_mode:
            # Use enhanced predictor with GNN + patterns
            graph_data = self.create_dummy_graph(len(source_code.split('\n')))
            result = self.predictor.predict_with_source_code(graph_data, source_code)
            
            return {
                'test_case': test_case,
                'enhanced_result': result,
                'vulnerable_detected': result['combined_assessment']['is_vulnerable'],
                'confidence': result['combined_assessment']['confidence'],
                'vulnerable_lines': list(result['source_analysis']['vulnerability_scores'].keys()),
                'explanations': result['explanations']['line_explanations'],
                'recommendations': result['explanations']['recommendations']
            }
        else:
            # Use pattern detection only
            vulnerable_lines = self.pattern_detector.annotate_vulnerable_lines(source_code)
            detailed_analysis = self.pattern_detector.enhanced_detector.get_detailed_analysis(source_code)
            
            return {
                'test_case': test_case,
                'pattern_result': detailed_analysis,
                'vulnerable_detected': len(vulnerable_lines) > 0,
                'confidence': detailed_analysis['overall_risk_score'],
                'vulnerable_lines': list(vulnerable_lines.keys()),
                'explanations': {
                    line_num: {
                        'explanation': self.pattern_detector.explain_vulnerability_pattern(
                            source_code.split('\n')[line_num], score, source_code
                        ),
                        'vulnerability_score': score
                    }
                    for line_num, score in vulnerable_lines.items()
                },
                'recommendations': detailed_analysis['recommendations']
            }
    
    def create_dummy_graph(self, num_lines):
        """Create a dummy graph for testing"""
        num_nodes = max(8, num_lines // 2)
        x = torch.randn(num_nodes, 100)
        
        # Create edges
        edges = []
        for i in range(num_nodes - 1):
            edges.extend([[i, i+1], [i+1, i]])
        
        # Add some complexity
        if num_nodes > 4:
            edges.extend([[0, num_nodes//2], [num_nodes//2, 0]])
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return Data(x=x, edge_index=edge_index)
    
    def print_test_result(self, result):
        """Print result for a single test case"""
        
        test_case = result['test_case']
        
        print(f"📝 Description: {test_case['description']}")
        print(f"🔤 Language: {test_case['language']}")
        
        # Prediction results
        vulnerable = result['vulnerable_detected']
        expected = test_case['expected_vulnerable']
        correct = "✅" if vulnerable == expected else "❌"
        
        print(f"\n🎯 PREDICTION: {correct}")
        print(f"   Expected: {'Vulnerable' if expected else 'Safe'}")
        print(f"   Detected: {'Vulnerable' if vulnerable else 'Safe'}")
        print(f"   Confidence: {result['confidence']:.2%}")
        
        # Vulnerable lines
        if result['vulnerable_lines']:
            print(f"\n📍 VULNERABLE LINES: {result['vulnerable_lines']}")
            
            # Show explanations
            for line_num, details in result['explanations'].items():
                if isinstance(details, dict) and 'explanation' in details:
                    print(f"   Line {line_num}: {details['explanation']}")
        else:
            print(f"\n✅ NO VULNERABLE LINES DETECTED")
        
        # Recommendations
        if result['recommendations']:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in result['recommendations']:
                print(f"   • {rec}")
        
        print()
    
    def print_test_summary(self):
        """Print overall test summary"""
        
        print("\n🎯 TEST SUMMARY")
        print("="*80)
        
        total_tests = len(self.test_results)
        correct_predictions = sum(
            1 for r in self.test_results 
            if r['vulnerable_detected'] == r['test_case']['expected_vulnerable']
        )
        
        accuracy = correct_predictions / total_tests if total_tests > 0 else 0
        
        print(f"📊 Overall Results:")
        print(f"   Total tests: {total_tests}")
        print(f"   Correct predictions: {correct_predictions}")
        print(f"   Accuracy: {accuracy:.1%}")
        
        # Breakdown by language
        languages = {}
        for result in self.test_results:
            lang = result['test_case']['language']
            if lang not in languages:
                languages[lang] = {'total': 0, 'correct': 0}
            
            languages[lang]['total'] += 1
            if result['vulnerable_detected'] == result['test_case']['expected_vulnerable']:
                languages[lang]['correct'] += 1
        
        print(f"\n📋 Results by Language:")
        for lang, stats in languages.items():
            lang_accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            print(f"   {lang}: {stats['correct']}/{stats['total']} ({lang_accuracy:.1%})")
        
        # Show false positives and negatives
        false_positives = sum(
            1 for r in self.test_results 
            if r['vulnerable_detected'] and not r['test_case']['expected_vulnerable']
        )
        false_negatives = sum(
            1 for r in self.test_results 
            if not r['vulnerable_detected'] and r['test_case']['expected_vulnerable']
        )
        
        print(f"\n🔍 Error Analysis:")
        print(f"   False positives: {false_positives}")
        print(f"   False negatives: {false_negatives}")
        
        if accuracy >= 0.8:
            print(f"\n🎉 EXCELLENT: High accuracy achieved!")
        elif accuracy >= 0.6:
            print(f"\n✅ GOOD: Reasonable accuracy achieved")
        else:
            print(f"\n⚠️ NEEDS IMPROVEMENT: Low accuracy detected")
        
        print("="*80)


def main():
    """Main test function"""
    
    print("🚀 ENHANCED VULNERABILITY DETECTION PIPELINE TEST")
    print("="*80)
    print("Testing the complete enhanced pipeline with real source code examples")
    print("This demonstrates the context-aware improvements and false positive reduction")
    print("="*80)
    
    tester = EnhancedPipelineTester()
    tester.run_comprehensive_test()
    
    print("\n✅ Comprehensive pipeline testing completed!")


if __name__ == "__main__":
    main()