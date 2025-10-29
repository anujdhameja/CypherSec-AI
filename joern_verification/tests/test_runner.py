"""
Test runner for Joern verification system tests.
"""

import unittest
import sys
import os
import time
from pathlib import Path
from typing import List, Optional

# Add the parent directory to the path so we can import joern_verification modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class JoernTestRunner:
    """Custom test runner for Joern verification tests."""
    
    def __init__(self, verbosity: int = 2):
        """
        Initialize test runner.
        
        Args:
            verbosity: Test output verbosity level
        """
        self.verbosity = verbosity
        self.test_results = {}
    
    def discover_tests(self, test_dir: Optional[str] = None) -> unittest.TestSuite:
        """
        Discover all tests in the test directory.
        
        Args:
            test_dir: Directory to search for tests (uses current dir if None)
            
        Returns:
            Test suite containing all discovered tests
        """
        if test_dir is None:
            test_dir = str(Path(__file__).parent)
        
        loader = unittest.TestLoader()
        suite = loader.discover(test_dir, pattern='test_*.py')
        return suite
    
    def run_unit_tests(self) -> unittest.TestResult:
        """
        Run unit tests only.
        
        Returns:
            Test results
        """
        print("Running Unit Tests...")
        print("=" * 50)
        
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        # Add specific unit test modules
        unit_test_modules = [
            'test_discovery',
            'test_generation', 
            'test_analysis'
        ]
        
        for module_name in unit_test_modules:
            try:
                module = __import__(f'joern_verification.tests.{module_name}', fromlist=[module_name])
                suite.addTests(loader.loadTestsFromModule(module))
            except ImportError as e:
                print(f"Warning: Could not import {module_name}: {e}")
        
        runner = unittest.TextTestRunner(verbosity=self.verbosity)
        result = runner.run(suite)
        
        self.test_results['unit_tests'] = {
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / max(result.testsRun, 1)
        }
        
        return result
    
    def run_integration_tests(self) -> unittest.TestResult:
        """
        Run integration tests only.
        
        Returns:
            Test results
        """
        print("\nRunning Integration Tests...")
        print("=" * 50)
        
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        # Add integration test module
        try:
            from joern_verification.tests import test_integration
            suite.addTests(loader.loadTestsFromModule(test_integration))
        except ImportError as e:
            print(f"Warning: Could not import integration tests: {e}")
            return unittest.TestResult()
        
        runner = unittest.TextTestRunner(verbosity=self.verbosity)
        result = runner.run(suite)
        
        self.test_results['integration_tests'] = {
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / max(result.testsRun, 1)
        }
        
        return result
    
    def run_all_tests(self) -> dict:
        """
        Run all tests (unit and integration).
        
        Returns:
            Dictionary with test results summary
        """
        print("Joern Verification System - Test Suite")
        print("=" * 60)
        
        start_time = time.time()
        
        # Run unit tests
        unit_result = self.run_unit_tests()
        
        # Run integration tests
        integration_result = self.run_integration_tests()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Generate summary
        summary = self._generate_summary(total_time)
        self._print_summary(summary)
        
        return summary
    
    def run_specific_test(self, test_name: str) -> unittest.TestResult:
        """
        Run a specific test by name.
        
        Args:
            test_name: Name of the test to run
            
        Returns:
            Test results
        """
        print(f"Running specific test: {test_name}")
        print("=" * 50)
        
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromName(test_name)
        
        runner = unittest.TextTestRunner(verbosity=self.verbosity)
        result = runner.run(suite)
        
        return result
    
    def _generate_summary(self, total_time: float) -> dict:
        """Generate test results summary."""
        unit_stats = self.test_results.get('unit_tests', {})
        integration_stats = self.test_results.get('integration_tests', {})
        
        total_tests = unit_stats.get('tests_run', 0) + integration_stats.get('tests_run', 0)
        total_failures = unit_stats.get('failures', 0) + integration_stats.get('failures', 0)
        total_errors = unit_stats.get('errors', 0) + integration_stats.get('errors', 0)
        
        overall_success_rate = (total_tests - total_failures - total_errors) / max(total_tests, 1)
        
        return {
            'total_time': total_time,
            'total_tests': total_tests,
            'total_failures': total_failures,
            'total_errors': total_errors,
            'overall_success_rate': overall_success_rate,
            'unit_tests': unit_stats,
            'integration_tests': integration_stats
        }
    
    def _print_summary(self, summary: dict):
        """Print test results summary."""
        print("\n" + "=" * 60)
        print("TEST RESULTS SUMMARY")
        print("=" * 60)
        
        print(f"Total execution time: {summary['total_time']:.2f} seconds")
        print(f"Total tests run: {summary['total_tests']}")
        print(f"Total failures: {summary['total_failures']}")
        print(f"Total errors: {summary['total_errors']}")
        print(f"Overall success rate: {summary['overall_success_rate']:.1%}")
        
        if 'unit_tests' in summary:
            unit = summary['unit_tests']
            print(f"\nUnit Tests: {unit.get('tests_run', 0)} tests, "
                  f"{unit.get('success_rate', 0):.1%} success rate")
        
        if 'integration_tests' in summary:
            integration = summary['integration_tests']
            print(f"Integration Tests: {integration.get('tests_run', 0)} tests, "
                  f"{integration.get('success_rate', 0):.1%} success rate")
        
        # Overall status
        if summary['overall_success_rate'] >= 0.9:
            print("\n✅ TEST SUITE PASSED")
        elif summary['overall_success_rate'] >= 0.7:
            print("\n⚠️  TEST SUITE PASSED WITH WARNINGS")
        else:
            print("\n❌ TEST SUITE FAILED")
        
        print("=" * 60)


def main():
    """Main entry point for test runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Joern Verification Test Runner')
    parser.add_argument('--unit', action='store_true', help='Run unit tests only')
    parser.add_argument('--integration', action='store_true', help='Run integration tests only')
    parser.add_argument('--test', type=str, help='Run specific test by name')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    verbosity = 2 if args.verbose else 1
    runner = JoernTestRunner(verbosity=verbosity)
    
    try:
        if args.test:
            # Run specific test
            result = runner.run_specific_test(args.test)
            success = len(result.failures) == 0 and len(result.errors) == 0
        elif args.unit:
            # Run unit tests only
            result = runner.run_unit_tests()
            success = len(result.failures) == 0 and len(result.errors) == 0
        elif args.integration:
            # Run integration tests only
            result = runner.run_integration_tests()
            success = len(result.failures) == 0 and len(result.errors) == 0
        else:
            # Run all tests
            summary = runner.run_all_tests()
            success = summary['overall_success_rate'] >= 0.9
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\nTest execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError running tests: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()