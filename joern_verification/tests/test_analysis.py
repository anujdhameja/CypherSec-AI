"""
Unit tests for analysis and reporting functionality.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json
from pathlib import Path

from joern_verification.analysis.result_analyzer import ResultAnalyzer
from joern_verification.analysis.metrics_collector import MetricsCollector
from joern_verification.core.interfaces import GenerationResult, AnalysisReport


class TestResultAnalyzer(unittest.TestCase):
    """Test cases for ResultAnalyzer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = ResultAnalyzer()
    
    def test_initialization(self):
        """Test ResultAnalyzer initialization."""
        self.assertIsInstance(self.analyzer, ResultAnalyzer)
    
    def test_analyze_successful_result(self):
        """Test analyzing successful CPG generation result."""
        result = GenerationResult(
            language="python",
            input_file="test.py",
            output_dir="output",
            success=True,
            execution_time=2.5,
            memory_usage=512,
            stdout="CPG generated successfully",
            stderr="",
            return_code=0,
            output_files=[Path("output/test.cpg")],
            warnings=[],
            error_message=None
        )
        
        analysis = self.analyzer.analyze_result(result)
        
        self.assertIsInstance(analysis, AnalysisReport)
        self.assertEqual(analysis.language, "python")
        self.assertEqual(analysis.category, "success")
        self.assertEqual(len(analysis.errors), 0)
        self.assertEqual(len(analysis.warnings), 0)
        self.assertGreater(len(analysis.recommendations), 0)
    
    def test_analyze_successful_result_with_warnings(self):
        """Test analyzing successful result with warnings."""
        result = GenerationResult(
            language="java",
            input_file="Test.java",
            output_dir="output",
            success=True,
            execution_time=3.0,
            memory_usage=1024,
            stdout="CPG generated with warnings",
            stderr="Warning: Deprecated API usage",
            return_code=0,
            output_files=[Path("output/Test.cpg")],
            warnings=["Warning: Deprecated API usage"],
            error_message=None
        )
        
        analysis = self.analyzer.analyze_result(result)
        
        self.assertEqual(analysis.category, "success_with_warnings")
        self.assertEqual(len(analysis.warnings), 1)
        self.assertIn("Deprecated API usage", analysis.warnings[0])
    
    def test_analyze_failed_result(self):
        """Test analyzing failed CPG generation result."""
        result = GenerationResult(
            language="cpp",
            input_file="test.cpp",
            output_dir="output",
            success=False,
            execution_time=1.0,
            memory_usage=None,
            stdout="",
            stderr="Error: Compilation failed",
            return_code=1,
            output_files=[],
            warnings=[],
            error_message="Compilation failed"
        )
        
        analysis = self.analyzer.analyze_result(result)
        
        self.assertEqual(analysis.category, "failure")
        self.assertGreater(len(analysis.errors), 0)
        self.assertIn("Compilation failed", analysis.errors[0])
    
    def test_analyze_partial_success_result(self):
        """Test analyzing partial success result."""
        result = GenerationResult(
            language="javascript",
            input_file="test.js",
            output_dir="output",
            success=True,
            execution_time=2.0,
            memory_usage=256,
            stdout="Partial CPG generated",
            stderr="Warning: Some constructs not supported",
            return_code=0,
            output_files=[Path("output/partial.cpg")],
            warnings=["Some constructs not supported"],
            error_message=None
        )
        
        # Mock partial success detection
        with patch.object(self.analyzer, '_detect_partial_success', return_value=True):
            analysis = self.analyzer.analyze_result(result)
        
        self.assertEqual(analysis.category, "partial_success")
        self.assertGreater(len(analysis.warnings), 0)
    
    def test_categorize_outcome_success(self):
        """Test outcome categorization for success."""
        result = GenerationResult(
            language="python",
            input_file="test.py",
            output_dir="output",
            success=True,
            execution_time=1.0,
            memory_usage=None,
            stdout="Success",
            stderr="",
            return_code=0,
            output_files=[Path("test.cpg")],
            warnings=[]
        )
        
        category = self.analyzer.categorize_outcome(result)
        self.assertEqual(category, "success")
    
    def test_categorize_outcome_failure(self):
        """Test outcome categorization for failure."""
        result = GenerationResult(
            language="python",
            input_file="test.py",
            output_dir="output",
            success=False,
            execution_time=0.5,
            memory_usage=None,
            stdout="",
            stderr="Error occurred",
            return_code=1,
            output_files=[],
            warnings=[]
        )
        
        category = self.analyzer.categorize_outcome(result)
        self.assertEqual(category, "failure")
    
    def test_extract_metrics(self):
        """Test metrics extraction from result."""
        result = GenerationResult(
            language="java",
            input_file="Test.java",
            output_dir="output",
            success=True,
            execution_time=2.5,
            memory_usage=1024,
            stdout="Success",
            stderr="",
            return_code=0,
            output_files=[Path("Test.cpg"), Path("metadata.json")],
            warnings=["Minor warning"]
        )
        
        metrics = self.analyzer.extract_metrics(result)
        
        self.assertIsInstance(metrics, dict)
        self.assertEqual(metrics['execution_time'], 2.5)
        self.assertEqual(metrics['memory_usage'], 1024)
        self.assertEqual(metrics['output_file_count'], 2)
        self.assertEqual(metrics['warning_count'], 1)
        self.assertEqual(metrics['success_rate'], 1.0)
    
    def test_extract_error_patterns(self):
        """Test error pattern extraction."""
        stderr = """
        Error: Compilation failed at line 15
        Warning: Deprecated function usage
        Error: Missing dependency: libfoo
        """
        
        patterns = self.analyzer._extract_error_patterns(stderr)
        
        self.assertIsInstance(patterns, list)
        self.assertGreater(len(patterns), 0)
        # Should contain error-related patterns
        error_patterns = [p for p in patterns if 'error' in p.lower()]
        self.assertGreater(len(error_patterns), 0)


class TestMetricsCollector(unittest.TestCase):
    """Test cases for MetricsCollector."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.collector = MetricsCollector()
    
    def test_initialization(self):
        """Test MetricsCollector initialization."""
        self.assertIsInstance(self.collector, MetricsCollector)
        self.assertIsInstance(self.collector.metrics, dict)
    
    def test_collect_performance_metrics(self):
        """Test collecting performance metrics."""
        result = GenerationResult(
            language="python",
            input_file="test.py",
            output_dir="output",
            success=True,
            execution_time=1.5,
            memory_usage=512,
            stdout="Success",
            stderr="",
            return_code=0,
            output_files=[Path("test.cpg")]
        )
        
        metrics = self.collector.collect_performance_metrics(result)
        
        self.assertIsInstance(metrics, dict)
        self.assertEqual(metrics['execution_time'], 1.5)
        self.assertEqual(metrics['memory_usage'], 512)
        self.assertIn('throughput', metrics)
    
    def test_collect_quality_metrics(self):
        """Test collecting quality metrics."""
        result = GenerationResult(
            language="java",
            input_file="Test.java",
            output_dir="output",
            success=True,
            execution_time=2.0,
            memory_usage=1024,
            stdout="Success",
            stderr="Warning: Minor issue",
            return_code=0,
            output_files=[Path("Test.cpg"), Path("metadata.json")],
            warnings=["Minor issue"]
        )
        
        metrics = self.collector.collect_quality_metrics(result)
        
        self.assertIsInstance(metrics, dict)
        self.assertEqual(metrics['success'], True)
        self.assertEqual(metrics['warning_count'], 1)
        self.assertEqual(metrics['output_file_count'], 2)
        self.assertIn('completeness_score', metrics)
    
    def test_collect_error_metrics(self):
        """Test collecting error metrics."""
        result = GenerationResult(
            language="cpp",
            input_file="test.cpp",
            output_dir="output",
            success=False,
            execution_time=0.5,
            memory_usage=None,
            stdout="",
            stderr="Error: Compilation failed\nError: Missing header",
            return_code=1,
            output_files=[],
            error_message="Multiple errors occurred"
        )
        
        metrics = self.collector.collect_error_metrics(result)
        
        self.assertIsInstance(metrics, dict)
        self.assertEqual(metrics['success'], False)
        self.assertEqual(metrics['return_code'], 1)
        self.assertGreater(metrics['error_count'], 0)
        self.assertIn('error_types', metrics)
    
    def test_aggregate_metrics(self):
        """Test aggregating metrics from multiple results."""
        results = [
            GenerationResult(
                language="python",
                input_file="test1.py",
                output_dir="output1",
                success=True,
                execution_time=1.0,
                memory_usage=256,
                stdout="Success",
                stderr="",
                return_code=0,
                output_files=[Path("test1.cpg")]
            ),
            GenerationResult(
                language="java",
                input_file="Test2.java",
                output_dir="output2",
                success=True,
                execution_time=2.0,
                memory_usage=512,
                stdout="Success",
                stderr="Warning: Minor",
                return_code=0,
                output_files=[Path("Test2.cpg")],
                warnings=["Minor"]
            ),
            GenerationResult(
                language="cpp",
                input_file="test3.cpp",
                output_dir="output3",
                success=False,
                execution_time=0.5,
                memory_usage=None,
                stdout="",
                stderr="Error: Failed",
                return_code=1,
                output_files=[],
                error_message="Failed"
            )
        ]
        
        aggregated = self.collector.aggregate_metrics(results)
        
        self.assertIsInstance(aggregated, dict)
        self.assertEqual(aggregated['total_tests'], 3)
        self.assertEqual(aggregated['successful_tests'], 2)
        self.assertEqual(aggregated['failed_tests'], 1)
        self.assertAlmostEqual(aggregated['success_rate'], 2/3, places=2)
        self.assertAlmostEqual(aggregated['average_execution_time'], 1.17, places=1)
    
    def test_get_language_statistics(self):
        """Test getting statistics by language."""
        results = [
            GenerationResult(language="python", input_file="test1.py", output_dir="out1", 
                           success=True, execution_time=1.0, memory_usage=None, stdout="", 
                           stderr="", return_code=0, output_files=[]),
            GenerationResult(language="python", input_file="test2.py", output_dir="out2", 
                           success=False, execution_time=0.5, memory_usage=None, stdout="", 
                           stderr="Error", return_code=1, output_files=[]),
            GenerationResult(language="java", input_file="Test.java", output_dir="out3", 
                           success=True, execution_time=2.0, memory_usage=None, stdout="", 
                           stderr="", return_code=0, output_files=[])
        ]
        
        stats = self.collector.get_language_statistics(results)
        
        self.assertIsInstance(stats, dict)
        self.assertIn('python', stats)
        self.assertIn('java', stats)
        
        python_stats = stats['python']
        self.assertEqual(python_stats['total_tests'], 2)
        self.assertEqual(python_stats['successful_tests'], 1)
        self.assertEqual(python_stats['success_rate'], 0.5)
        
        java_stats = stats['java']
        self.assertEqual(java_stats['total_tests'], 1)
        self.assertEqual(java_stats['successful_tests'], 1)
        self.assertEqual(java_stats['success_rate'], 1.0)
    
    def test_export_metrics(self):
        """Test exporting metrics to file."""
        metrics = {
            'total_tests': 5,
            'successful_tests': 3,
            'success_rate': 0.6,
            'languages': ['python', 'java', 'cpp']
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_path = f.name
        
        try:
            success = self.collector.export_metrics(metrics, output_path)
            
            self.assertTrue(success)
            self.assertTrue(Path(output_path).exists())
            
            # Verify exported content
            with open(output_path, 'r') as f:
                exported = json.load(f)
            
            self.assertEqual(exported['total_tests'], 5)
            self.assertEqual(exported['successful_tests'], 3)
            self.assertEqual(exported['success_rate'], 0.6)
            
        finally:
            # Cleanup
            if Path(output_path).exists():
                Path(output_path).unlink()


if __name__ == '__main__':
    unittest.main()