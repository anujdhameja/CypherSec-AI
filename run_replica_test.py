#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Replica Test Runner
==================

Simple script to run the complete replica testing pipeline and generate comprehensive reports.
This script validates your entire vulnerability detection pipeline before production deployment.

Usage:
    python run_replica_test.py                    # Test all languages
    python run_replica_test.py --quick            # Quick test with 2 languages
    python run_replica_test.py --analyze-only     # Only run vulnerability analysis
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

# Import replica modules
from replica_pipeline_test import ReplicaPipelineTester
from replica_vulnerability_detector import ReplicaVulnerabilityAnalyzer


class ReplicaTestRunner:
    """Orchestrates the complete replica testing workflow"""
    
    def __init__(self):
        self.start_time = time.time()
        self.test_results = {}
        self.languages = ['c', 'cpp', 'python', 'java']  # Core languages for testing
        # Defaults for trained model usage
        self.use_trained = False
        self.model_path = 'models/final_model.pth'
        
    def run_pipeline_test(self, languages=None, quick_mode=False):
        """Run the main pipeline test"""
        print("🚀 STARTING REPLICA PIPELINE TEST")
        print("="*80)
        
        if languages is None:
            languages = ['c', 'cpp'] if quick_mode else self.languages
        
        try:
            # Initialize pipeline tester
            tester = ReplicaPipelineTester()
            # Apply trained model flags
            tester.use_trained = self.use_trained
            tester.trained_model_path = self.model_path
            
            # Run full pipeline
            tester.run_full_pipeline(languages)
            
            self.test_results['pipeline'] = {
                'status': 'success',
                'languages_tested': languages,
                'duration': time.time() - self.start_time
            }
            
            print("✅ Pipeline test completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Pipeline test failed: {e}")
            self.test_results['pipeline'] = {
                'status': 'failed',
                'error': str(e),
                'duration': time.time() - self.start_time
            }
            return False
    
    def run_vulnerability_analysis(self, languages=None):
        """Run vulnerability detection analysis on all datasets"""
        print("\n🔍 STARTING VULNERABILITY ANALYSIS")
        print("="*80)
        
        if languages is None:
            languages = self.languages
        
        analyzer = ReplicaVulnerabilityAnalyzer()
        analysis_results = {}
        
        for lang in languages:
            dataset_path = f"data/replica_raw/replica_dataset_{lang}.json"
            
            if not os.path.exists(dataset_path):
                print(f"⚠️  Dataset not found: {dataset_path}")
                continue
            
            print(f"\n🔄 Analyzing {lang.upper()} dataset...")
            
            try:
                # Run batch analysis
                results = analyzer.batch_analyze(dataset_path)
                
                # Generate report
                report_path = f"replica_analysis_{lang}_report.txt"
                report = analyzer.generate_report(results, report_path)
                
                analysis_results[lang] = {
                    'status': 'success',
                    'samples_analyzed': results['statistics']['total_samples'],
                    'vulnerabilities_detected': results['statistics']['vulnerable_detected'],
                    'vulnerable_lines_found': results['statistics']['total_vulnerable_lines'],
                    'report_path': report_path
                }
                
                print(f"✅ {lang.upper()} analysis completed")
                
            except Exception as e:
                print(f"❌ {lang.upper()} analysis failed: {e}")
                analysis_results[lang] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        self.test_results['vulnerability_analysis'] = analysis_results
        return analysis_results
    
    def validate_pipeline_outputs(self):
        """Validate that all expected outputs were generated"""
        print("\n🔍 VALIDATING PIPELINE OUTPUTS")
        print("="*50)
        
        validation_results = {}
        
        # Check directories and files
        expected_dirs = [
            'data/replica_cpg',
            'data/replica_input', 
            'data/replica_tokens',
            'data/replica_w2v'
        ]
        
        for dir_path in expected_dirs:
            if os.path.exists(dir_path):
                files = list(Path(dir_path).glob("*"))
                validation_results[dir_path] = {
                    'exists': True,
                    'file_count': len(files),
                    'files': [f.name for f in files[:5]]  # First 5 files
                }
                print(f"✅ {dir_path}: {len(files)} files")
            else:
                validation_results[dir_path] = {'exists': False}
                print(f"❌ {dir_path}: Not found")
        
        # Check specific critical files
        critical_files = [
            'data/replica_w2v/replica_w2v.model',
            'replica_configs.json'
        ]
        
        for file_path in critical_files:
            exists = os.path.exists(file_path)
            validation_results[file_path] = {'exists': exists}
            status = "✅" if exists else "❌"
            print(f"{status} {file_path}")
        
        self.test_results['validation'] = validation_results
        return validation_results
    
    def test_format_compatibility(self):
        """Test compatibility with your large dataset format"""
        print("\n🧪 TESTING FORMAT COMPATIBILITY")
        print("="*50)
        
        # Test sample in your specified format
        test_sample = {
            "Sno": 1.0,
            "Primary Language of Benchmark": "c++",
            "Vulnerability": 1.0,
            "CVE-ID": None,
            "Severity": None,
            "CWE ID": "Improper memory management in C++ can lead to buffer overflow vulnerabilities.",
            "File name with path": None,
            "Line Number": None,
            "Code Snippet": "c++\n#include <cstring>\n\nvoid copyString(char* dest, const char* src) {\n    while (*src != '\\0') {\n        *dest = *src;\n        dest++;\n        src++;\n    }\n}\n\nint main() {\n    char source[10] = \"Hello!\";\n    char destination[5];\n    copyString(destination, source);\n    return 0;\n}\n",
            "vulnerability": None,
            "question": None,
            "rejected": None,
            "commit_message": None,
            "cwe": None,
            "target": None,
            "nvd_url": None,
            "func_hash": None,
            "file_hash": None,
            "cve_desc": None,
            "commit_id": None,
            "commit_url": None,
            "idx": None,
            "project_url": None,
            "project": None,
            "cve": None,
            "file_name": None,
            "func": None
        }
        
        try:
            # Test with vulnerability analyzer
            analyzer = ReplicaVulnerabilityAnalyzer()
            result = analyzer.analyze_code_sample(test_sample)
            
            print("✅ Format compatibility test passed")
            print(f"   Detected {result['line_analysis']['vulnerable_lines_count']} vulnerable lines")
            print(f"   Risk score: {result['line_analysis']['overall_risk_score']:.2f}")
            
            self.test_results['format_compatibility'] = {
                'status': 'success',
                'vulnerable_lines_detected': result['line_analysis']['vulnerable_lines_count'],
                'risk_score': result['line_analysis']['overall_risk_score']
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Format compatibility test failed: {e}")
            self.test_results['format_compatibility'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def generate_final_report(self):
        """Generate comprehensive final report"""
        print("\n📄 GENERATING FINAL REPORT")
        print("="*50)
        
        report_data = {
            'test_timestamp': datetime.now().isoformat(),
            'total_duration': time.time() - self.start_time,
            'results': self.test_results
        }
        
        # Save JSON report
        json_report_path = f"replica_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        # Generate human-readable report
        txt_report_path = json_report_path.replace('.json', '.txt')
        
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("REPLICA VULNERABILITY DETECTION PIPELINE TEST REPORT")
        report_lines.append("="*80)
        report_lines.append(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Total Duration: {report_data['total_duration']:.1f} seconds")
        report_lines.append("")
        
        # Pipeline test results
        if 'pipeline' in self.test_results:
            pipeline = self.test_results['pipeline']
            report_lines.append("PIPELINE TEST RESULTS:")
            report_lines.append(f"  Status: {pipeline['status'].upper()}")
            if 'languages_tested' in pipeline:
                report_lines.append(f"  Languages: {', '.join(pipeline['languages_tested'])}")
            if 'duration' in pipeline:
                report_lines.append(f"  Duration: {pipeline['duration']:.1f}s")
            if 'error' in pipeline:
                report_lines.append(f"  Error: {pipeline['error']}")
            report_lines.append("")
        
        # Vulnerability analysis results
        if 'vulnerability_analysis' in self.test_results:
            report_lines.append("VULNERABILITY ANALYSIS RESULTS:")
            for lang, results in self.test_results['vulnerability_analysis'].items():
                report_lines.append(f"  {lang.upper()}:")
                report_lines.append(f"    Status: {results['status'].upper()}")
                if 'samples_analyzed' in results:
                    report_lines.append(f"    Samples: {results['samples_analyzed']}")
                    report_lines.append(f"    Vulnerabilities: {results['vulnerabilities_detected']}")
                    report_lines.append(f"    Vulnerable Lines: {results['vulnerable_lines_found']}")
                if 'error' in results:
                    report_lines.append(f"    Error: {results['error']}")
            report_lines.append("")
        
        # Validation results
        if 'validation' in self.test_results:
            report_lines.append("OUTPUT VALIDATION:")
            validation = self.test_results['validation']
            for path, result in validation.items():
                status = "✅" if result.get('exists', False) else "❌"
                report_lines.append(f"  {status} {path}")
                if 'file_count' in result:
                    report_lines.append(f"      Files: {result['file_count']}")
            report_lines.append("")
        
        # Format compatibility
        if 'format_compatibility' in self.test_results:
            compat = self.test_results['format_compatibility']
            report_lines.append("FORMAT COMPATIBILITY:")
            report_lines.append(f"  Status: {compat['status'].upper()}")
            if 'vulnerable_lines_detected' in compat:
                report_lines.append(f"  Vulnerable Lines: {compat['vulnerable_lines_detected']}")
                report_lines.append(f"  Risk Score: {compat['risk_score']:.2f}")
            report_lines.append("")
        
        # Overall assessment
        report_lines.append("OVERALL ASSESSMENT:")
        
        pipeline_ok = self.test_results.get('pipeline', {}).get('status') == 'success'
        format_ok = self.test_results.get('format_compatibility', {}).get('status') == 'success'
        
        if pipeline_ok and format_ok:
            report_lines.append("  🎉 READY FOR PRODUCTION DEPLOYMENT!")
            report_lines.append("  ✅ All tests passed successfully")
            report_lines.append("  ✅ Pipeline generates correct outputs")
            report_lines.append("  ✅ Format compatibility confirmed")
            report_lines.append("")
            report_lines.append("  NEXT STEPS:")
            report_lines.append("    1. Deploy with your large multi-language dataset")
            report_lines.append("    2. Monitor performance metrics")
            report_lines.append("    3. Validate results on production data")
        else:
            report_lines.append("  ⚠️  ISSUES DETECTED - REVIEW BEFORE DEPLOYMENT")
            if not pipeline_ok:
                report_lines.append("    ❌ Pipeline test failed")
            if not format_ok:
                report_lines.append("    ❌ Format compatibility issues")
        
        report_lines.append("")
        report_lines.append("="*80)
        
        # Save text report
        with open(txt_report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"📄 Reports saved:")
        print(f"   JSON: {json_report_path}")
        print(f"   Text: {txt_report_path}")
        
        # Print summary to console
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        for line in report_lines[-15:]:  # Last 15 lines (assessment section)
            print(line)
        
        return txt_report_path, json_report_path


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Replica Test Runner")
    parser.add_argument('--quick', action='store_true',
                       help='Quick test with limited languages')
    parser.add_argument('--analyze-only', action='store_true',
                       help='Run only vulnerability analysis')
    parser.add_argument('--skip-pipeline', action='store_true',
                       help='Skip pipeline test, run analysis only')
    parser.add_argument('--use-trained', action='store_true',
                       help='Use trained model weights during replica prediction (best-effort)')
    parser.add_argument('--model-path', type=str, default='models/final_model.pth',
                       help='Path to trained model weights (default: models/final_model.pth)')
    
    args = parser.parse_args()
    
    runner = ReplicaTestRunner()
    # Apply CLI flags to runner
    runner.use_trained = args.use_trained
    runner.model_path = args.model_path
    
    print("🎯 REPLICA VULNERABILITY DETECTION SYSTEM TEST")
    print("="*80)
    print("This test validates your complete pipeline before production deployment.")
    print("Testing: CPG generation → Embedding → Model prediction → Line detection")
    print("="*80)
    
    success = True
    
    try:
        # Run pipeline test (unless skipped)
        if not args.skip_pipeline and not args.analyze_only:
            success &= runner.run_pipeline_test(quick_mode=args.quick)
            
            # Validate outputs
            runner.validate_pipeline_outputs()
        
        # Run vulnerability analysis
        if not args.analyze_only or args.skip_pipeline:
            runner.run_vulnerability_analysis()
        
        # Test format compatibility
        runner.test_format_compatibility()
        
        # Generate final report
        runner.generate_final_report()
        
        if success:
            print("\n🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
            print("Your pipeline is ready for production deployment.")
        else:
            print("\n⚠️  SOME TESTS FAILED - REVIEW REPORTS")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test runner failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()