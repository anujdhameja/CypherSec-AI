"""
Integration tests for the complete Joern verification workflow.
"""

import unittest
from unittest.mock import patch, Mock
import tempfile
import os
import json
from pathlib import Path

from joern_verification.main import JoernVerificationOrchestrator
from joern_verification.tests.test_fixtures import TestEnvironmentManager, TestFixtures, MockCommandExecutor


class TestJoernVerificationWorkflow(unittest.TestCase):
    """Integration tests for the complete verification workflow."""
    
    def setUp(self):
        """Set up test environment."""
        self.env_manager = TestEnvironmentManager()
        self.workspace = self.env_manager.create_test_workspace()
        
        # Create orchestrator with test workspace
        self.orchestrator = JoernVerificationOrchestrator(
            joern_cli_path=self.workspace['joern_cli_path'],
            output_dir=self.workspace['output_dir']
        )
    
    def tearDown(self):
        """Clean up test environment."""
        self.env_manager.cleanup()
    
    @patch('joern_verification.generation.command_executor.CommandExecutor')
    def test_complete_verification_workflow_success(self, mock_executor_class):
        """Test complete verification workflow with successful results."""
        # Mock command executor to simulate successful CPG generation
        mock_executor = MockCommandExecutor(simulate_success=True)
        mock_executor_class.return_value = mock_executor
        
        # Run complete verification
        results = self.orchestrator.run_complete_verification()
        
        # Verify results structure
        self.assertIsInstance(results, dict)
        self.assertIn('discovery_results', results)
        self.assertIn('generation_results', results)
        self.assertIn('analysis_results', results)
        self.assertIn('report_path', results)
        
        # Verify discovery results
        discovery = results['discovery_results']
        self.assertEqual(discovery['discovery_status'], 'completed')
        self.assertIn('discovered_tools', discovery)
        
        # Verify generation results
        generation = results['generation_results']
        self.assertIsInstance(generation, list)
        self.assertGreater(len(generation), 0)
        
        # Verify analysis results
        analysis = results['analysis_results']
        self.assertIsInstance(analysis, list)
        self.assertEqual(len(analysis), len(generation))
        
        # Verify report was generated
        report_path = results['report_path']
        self.assertTrue(os.path.exists(report_path))
    
    @patch('joern_verification.generation.command_executor.CommandExecutor')
    def test_complete_verification_workflow_mixed_results(self, mock_executor_class):
        """Test verification workflow with mixed success/failure results."""
        # Mock command executor to simulate mixed results
        mock_executor = Mock()
        mock_executor_class.return_value = mock_executor
        
        # Simulate different outcomes for different languages
        def mock_execute_command(command, working_dir=None, timeout=None):
            if 'pysrc2cpg.bat' in command:
                return TestFixtures.create_mock_execution_result(success=True)
            elif 'javasrc2cpg.bat' in command:
                return TestFixtures.create_mock_execution_result(
                    success=True,
                    stderr="Warning: Deprecated API usage"
                )
            else:
                return TestFixtures.create_mock_execution_result(
                    success=False,
                    return_code=1,
                    stderr="Error: Tool not found",
                    error_message="Tool execution failed"
                )
        
        mock_executor.execute_command.side_effect = mock_execute_command
        mock_executor.validate_tool_availability.return_value = True
        mock_executor.get_memory_args.return_value = ["-J-Xmx4g"]
        
        # Run verification
        results = self.orchestrator.run_complete_verification()
        
        # Verify mixed results are handled correctly
        generation_results = results['generation_results']
        analysis_results = results['analysis_results']
        
        # Should have both successful and failed results
        successful = [r for r in generation_results if r.success]
        failed = [r for r in generation_results if not r.success]
        
        self.assertGreater(len(successful), 0)
        self.assertGreater(len(failed), 0)
        
        # Analysis should categorize results correctly
        success_analyses = [a for a in analysis_results if a.category == 'success']
        failure_analyses = [a for a in analysis_results if a.category == 'failure']
        
        self.assertEqual(len(success_analyses), len(successful))
        self.assertEqual(len(failure_analyses), len(failed))
    
    def test_language_discovery_integration(self):
        """Test language discovery integration with real file system."""
        # Run discovery
        discovery_results = self.orchestrator.discovery_manager.discover_languages()
        
        # Verify discovery completed
        self.assertEqual(discovery_results['discovery_status'], 'completed')
        self.assertTrue(self.orchestrator.discovery_manager.discovery_completed)
        
        # Verify tools were discovered
        discovered_tools = discovery_results['discovered_tools']
        self.assertIsInstance(discovered_tools, list)
        
        # Should find the mock tools we created
        tool_names = [tool['name'] for tool in discovered_tools]
        expected_tools = ['c2cpg.bat', 'javasrc2cpg.bat', 'pysrc2cpg.bat']
        
        for expected_tool in expected_tools:
            self.assertIn(expected_tool, tool_names)
    
    def test_test_file_generation_integration(self):
        """Test test file generation integration."""
        # Generate test files for specific languages
        languages = ['python', 'java', 'c']
        
        with self.orchestrator.test_file_manager.create_test_environment(languages) as generator:
            created_files = generator.get_created_files()
            
            # Verify files were created
            self.assertEqual(len(created_files), len(languages))
            
            # Verify file contents
            for file_path in created_files:
                self.assertTrue(os.path.exists(file_path))
                
                with open(file_path, 'r') as f:
                    content = f.read()
                self.assertGreater(len(content), 0)
                
                # Verify syntax validation
                file_ext = Path(file_path).suffix
                if file_ext == '.py':
                    self.assertTrue(generator.validate_syntax('python', content))
                elif file_ext == '.java':
                    self.assertTrue(generator.validate_syntax('java', content))
                elif file_ext == '.c':
                    self.assertTrue(generator.validate_syntax('c', content))
    
    @patch('joern_verification.generation.command_executor.CommandExecutor')
    def test_cpg_generation_integration(self, mock_executor_class):
        """Test CPG generation integration."""
        # Setup mock executor
        mock_executor = MockCommandExecutor(simulate_success=True)
        mock_executor_class.return_value = mock_executor
        
        # Create test file
        test_file = TestFixtures.create_test_source_file(
            self.workspace['test_files_dir'], 
            'python'
        )
        
        # Generate CPG
        result = self.orchestrator.cpg_generator.generate_cpg(
            language='python',
            input_file=Path(test_file),
            output_dir=Path(self.workspace['output_dir'])
        )
        
        # Verify result
        self.assertTrue(result.success)
        self.assertEqual(result.language, 'python')
        self.assertEqual(result.input_file, Path(test_file))
        self.assertGreater(result.execution_time, 0)
    
    def test_analysis_and_reporting_integration(self):
        """Test analysis and reporting integration."""
        # Create mock generation results
        generation_results = [
            TestFixtures.create_mock_generation_result('python', success=True),
            TestFixtures.create_mock_generation_result('java', success=True, warnings=['Minor warning']),
            TestFixtures.create_mock_generation_result('cpp', success=False, error_message='Compilation failed')
        ]
        
        # Analyze results
        analysis_results = []
        for gen_result in generation_results:
            analysis = self.orchestrator.result_analyzer.analyze_result(gen_result)
            analysis_results.append(analysis)
        
        # Verify analyses
        self.assertEqual(len(analysis_results), 3)
        
        categories = [a.category for a in analysis_results]
        self.assertIn('success', categories)
        self.assertIn('success_with_warnings', categories)
        self.assertIn('failure', categories)
        
        # Generate report
        report_path = os.path.join(self.workspace['output_dir'], 'verification_report.json')
        success = self.orchestrator.report_generator.generate_report(
            analysis_results, 
            Path(report_path)
        )
        
        # Verify report
        self.assertTrue(success)
        self.assertTrue(os.path.exists(report_path))
        
        # Verify report content
        with open(report_path, 'r') as f:
            report_data = json.load(f)
        
        self.assertIn('summary', report_data)
        self.assertIn('language_results', report_data)
        self.assertIn('recommendations', report_data)


class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmarking tests for the verification system."""
    
    def setUp(self):
        """Set up performance test environment."""
        self.env_manager = TestEnvironmentManager()
        self.workspace = self.env_manager.create_test_workspace()
    
    def tearDown(self):
        """Clean up test environment."""
        self.env_manager.cleanup()
    
    @patch('joern_verification.generation.command_executor.CommandExecutor')
    def test_performance_with_multiple_languages(self, mock_executor_class):
        """Test performance with multiple languages."""
        import time
        
        # Setup mock executor with realistic timing
        mock_executor = Mock()
        mock_executor_class.return_value = mock_executor
        
        def mock_execute_with_delay(command, working_dir=None, timeout=None):
            time.sleep(0.1)  # Simulate processing time
            return TestFixtures.create_mock_execution_result(
                success=True,
                execution_time=0.1
            )
        
        mock_executor.execute_command.side_effect = mock_execute_with_delay
        mock_executor.validate_tool_availability.return_value = True
        mock_executor.get_memory_args.return_value = ["-J-Xmx4g"]
        
        # Create orchestrator
        orchestrator = JoernVerificationOrchestrator(
            joern_cli_path=self.workspace['joern_cli_path'],
            output_dir=self.workspace['output_dir']
        )
        
        # Measure performance
        start_time = time.time()
        results = orchestrator.run_complete_verification()
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Verify performance metrics
        self.assertLess(execution_time, 10.0)  # Should complete within 10 seconds
        
        # Verify results quality
        generation_results = results['generation_results']
        self.assertGreater(len(generation_results), 0)
        
        # Calculate throughput
        throughput = len(generation_results) / execution_time
        self.assertGreater(throughput, 0.1)  # At least 0.1 languages per second
    
    def test_memory_usage_tracking(self):
        """Test memory usage tracking during verification."""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create large test environment
        large_workspace = self.env_manager.create_test_workspace(
            languages=['python', 'java', 'c', 'cpp', 'csharp', 'javascript']
        )
        
        # Create orchestrator
        orchestrator = JoernVerificationOrchestrator(
            joern_cli_path=large_workspace['joern_cli_path'],
            output_dir=large_workspace['output_dir']
        )
        
        # Run discovery (lightweight operation)
        orchestrator.discovery_manager.discover_languages()
        
        # Get peak memory usage
        peak_memory = process.memory_info().rss
        memory_increase = peak_memory - initial_memory
        
        # Verify memory usage is reasonable (less than 100MB increase)
        self.assertLess(memory_increase, 100 * 1024 * 1024)
    
    @patch('joern_verification.generation.command_executor.CommandExecutor')
    def test_concurrent_processing_simulation(self, mock_executor_class):
        """Test simulation of concurrent processing capabilities."""
        # Setup mock executor
        mock_executor = MockCommandExecutor(simulate_success=True)
        mock_executor_class.return_value = mock_executor
        
        # Create orchestrator
        orchestrator = JoernVerificationOrchestrator(
            joern_cli_path=self.workspace['joern_cli_path'],
            output_dir=self.workspace['output_dir']
        )
        
        # Simulate processing multiple files
        test_files = []
        languages = ['python', 'java', 'c', 'cpp']
        
        for language in languages:
            file_path = TestFixtures.create_test_source_file(
                self.workspace['test_files_dir'],
                language
            )
            test_files.append((language, file_path))
        
        # Process all files
        results = []
        for language, file_path in test_files:
            result = orchestrator.cpg_generator.generate_cpg(
                language=language,
                input_file=Path(file_path),
                output_dir=Path(self.workspace['output_dir'])
            )
            results.append(result)
        
        # Verify all processed successfully
        self.assertEqual(len(results), len(test_files))
        successful_results = [r for r in results if r.success]
        self.assertEqual(len(successful_results), len(test_files))


class TestRegressionTests(unittest.TestCase):
    """Regression tests to ensure consistent behavior."""
    
    def setUp(self):
        """Set up regression test environment."""
        self.env_manager = TestEnvironmentManager()
        self.workspace = self.env_manager.create_test_workspace()
    
    def tearDown(self):
        """Clean up test environment."""
        self.env_manager.cleanup()
    
    def test_consistent_discovery_results(self):
        """Test that discovery results are consistent across multiple runs."""
        orchestrator = JoernVerificationOrchestrator(
            joern_cli_path=self.workspace['joern_cli_path'],
            output_dir=self.workspace['output_dir']
        )
        
        # Run discovery multiple times
        results1 = orchestrator.discovery_manager.discover_languages()
        
        # Reset and run again
        orchestrator.discovery_manager.discovery_completed = False
        results2 = orchestrator.discovery_manager.discover_languages()
        
        # Compare results
        self.assertEqual(results1['discovery_status'], results2['discovery_status'])
        self.assertEqual(
            len(results1['discovered_tools']), 
            len(results2['discovered_tools'])
        )
        
        # Tool names should be identical
        tools1 = {tool['name'] for tool in results1['discovered_tools']}
        tools2 = {tool['name'] for tool in results2['discovered_tools']}
        self.assertEqual(tools1, tools2)
    
    def test_consistent_test_file_generation(self):
        """Test that test file generation is consistent."""
        from joern_verification.generation.test_file_generator import TestFileGenerator
        
        # Generate files multiple times
        generator1 = TestFileGenerator(self.workspace['test_files_dir'])
        generator2 = TestFileGenerator(self.workspace['test_files_dir'])
        
        # Generate content for same language
        content1 = generator1.get_test_content('python')
        content2 = generator2.get_test_content('python')
        
        # Content should be identical
        self.assertEqual(content1, content2)
        
        # Cleanup
        generator1.cleanup_test_files()
        generator2.cleanup_test_files()
    
    @patch('joern_verification.generation.command_executor.CommandExecutor')
    def test_error_handling_consistency(self, mock_executor_class):
        """Test that error handling is consistent."""
        # Setup mock executor to simulate errors
        mock_executor = Mock()
        mock_executor_class.return_value = mock_executor
        mock_executor.validate_tool_availability.return_value = True
        mock_executor.get_memory_args.return_value = ["-J-Xmx4g"]
        
        # Simulate consistent error
        mock_executor.execute_command.return_value = TestFixtures.create_mock_execution_result(
            success=False,
            return_code=1,
            stderr="Error: Consistent test error",
            error_message="Test error message"
        )
        
        orchestrator = JoernVerificationOrchestrator(
            joern_cli_path=self.workspace['joern_cli_path'],
            output_dir=self.workspace['output_dir']
        )
        
        # Create test file
        test_file = TestFixtures.create_test_source_file(
            self.workspace['test_files_dir'],
            'python'
        )
        
        # Generate CPG multiple times
        result1 = orchestrator.cpg_generator.generate_cpg(
            language='python',
            input_file=Path(test_file),
            output_dir=Path(self.workspace['output_dir'])
        )
        
        result2 = orchestrator.cpg_generator.generate_cpg(
            language='python',
            input_file=Path(test_file),
            output_dir=Path(self.workspace['output_dir'])
        )
        
        # Results should be consistent
        self.assertEqual(result1.success, result2.success)
        self.assertEqual(result1.return_code, result2.return_code)
        self.assertEqual(result1.error_message, result2.error_message)


if __name__ == '__main__':
    # Run integration tests
    unittest.main(verbosity=2)