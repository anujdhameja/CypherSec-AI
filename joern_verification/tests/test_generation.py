"""
Unit tests for test file generation and CPG generation functionality.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from pathlib import Path

from joern_verification.generation.test_file_generator import TestFileGenerator, TestFileManager
from joern_verification.generation.cpg_generator import CPGGenerator, GenerationResult
from joern_verification.generation.command_executor import CommandExecutor, ExecutionResult


class TestTestFileGenerator(unittest.TestCase):
    """Test cases for TestFileGenerator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.generator = TestFileGenerator(self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.generator.cleanup_test_files()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test TestFileGenerator initialization."""
        self.assertEqual(self.generator.base_output_dir, self.temp_dir)
        self.assertIsInstance(self.generator.created_files, list)
        self.assertIsInstance(self.generator.created_dirs, list)
        self.assertTrue(os.path.exists(self.temp_dir))
    
    def test_get_test_content(self):
        """Test getting test content for languages."""
        # Test with supported language
        content = self.generator.get_test_content("python")
        self.assertIsInstance(content, str)
        self.assertGreater(len(content), 0)
        
        # Test with unsupported language
        content = self.generator.get_test_content("nonexistent")
        self.assertIsNone(content)
    
    def test_generate_test_file_success(self):
        """Test successful test file generation."""
        success, path = self.generator.generate_test_file("python")
        
        self.assertTrue(success)
        self.assertTrue(os.path.exists(path))
        self.assertIn(path, self.generator.created_files)
        
        # Verify file content
        with open(path, 'r') as f:
            content = f.read()
        self.assertGreater(len(content), 0)
    
    def test_generate_test_file_unsupported_language(self):
        """Test test file generation for unsupported language."""
        success, error = self.generator.generate_test_file("nonexistent")
        
        self.assertFalse(success)
        self.assertIn("No template available", error)
    
    def test_generate_test_file_custom_path(self):
        """Test test file generation with custom path."""
        custom_path = os.path.join(self.temp_dir, "custom", "test.py")
        success, path = self.generator.generate_test_file("python", custom_path)
        
        self.assertTrue(success)
        self.assertEqual(path, custom_path)
        self.assertTrue(os.path.exists(custom_path))
    
    def test_validate_syntax_python(self):
        """Test Python syntax validation."""
        valid_code = "def hello():\n    print('Hello, World!')\n"
        invalid_code = "def hello(\n    print('Hello, World!')\n"
        
        self.assertTrue(self.generator.validate_syntax("python", valid_code))
        self.assertFalse(self.generator.validate_syntax("python", invalid_code))
    
    def test_validate_syntax_java(self):
        """Test Java syntax validation."""
        valid_code = "public class Test { public static void main(String[] args) {} }"
        invalid_code = "invalid java code"
        
        self.assertTrue(self.generator.validate_syntax("java", valid_code))
        self.assertFalse(self.generator.validate_syntax("java", invalid_code))
    
    def test_generate_all_test_files(self):
        """Test generating test files for all supported languages."""
        results = self.generator.generate_all_test_files()
        
        self.assertIsInstance(results, dict)
        self.assertGreater(len(results), 0)
        
        # Check that all results have the expected format
        for language, (success, path_or_error) in results.items():
            self.assertIsInstance(success, bool)
            self.assertIsInstance(path_or_error, str)
            if success:
                self.assertTrue(os.path.exists(path_or_error))
    
    def test_cleanup_test_files(self):
        """Test cleanup of generated test files."""
        # Generate some test files
        self.generator.generate_test_file("python")
        self.generator.generate_test_file("java")
        
        created_files = self.generator.created_files.copy()
        self.assertGreater(len(created_files), 0)
        
        # Cleanup
        self.generator.cleanup_test_files()
        
        # Verify files are removed
        for file_path in created_files:
            self.assertFalse(os.path.exists(file_path))
        
        self.assertEqual(len(self.generator.created_files), 0)
    
    def test_context_manager(self):
        """Test TestFileGenerator as context manager."""
        with TestFileGenerator(self.temp_dir) as gen:
            success, path = gen.generate_test_file("python")
            self.assertTrue(success)
            self.assertTrue(os.path.exists(path))
        
        # File should be cleaned up after context exit
        self.assertFalse(os.path.exists(path))


class TestCPGGenerator(unittest.TestCase):
    """Test cases for CPGGenerator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.joern_cli_path = Path(self.temp_dir) / "joern-cli"
        self.joern_cli_path.mkdir(parents=True, exist_ok=True)
        
        # Create mock executor
        self.mock_executor = Mock(spec=CommandExecutor)
        self.generator = CPGGenerator(self.joern_cli_path, self.mock_executor)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test CPGGenerator initialization."""
        self.assertEqual(self.generator.joern_cli_path, self.joern_cli_path)
        self.assertEqual(self.generator.executor, self.mock_executor)
    
    def test_initialization_invalid_path(self):
        """Test initialization with invalid Joern path."""
        with self.assertRaises(ValueError):
            CPGGenerator("/invalid/path")
    
    def test_get_tool_path_supported_language(self):
        """Test getting tool path for supported language."""
        # Mock tool availability
        self.mock_executor.validate_tool_availability.return_value = True
        
        tool_path = self.generator.get_tool_path("python")
        expected_path = self.joern_cli_path / "pysrc2cpg.bat"
        
        self.assertEqual(tool_path, expected_path)
        self.mock_executor.validate_tool_availability.assert_called_with(expected_path)
    
    def test_get_tool_path_unsupported_language(self):
        """Test getting tool path for unsupported language."""
        tool_path = self.generator.get_tool_path("nonexistent")
        self.assertIsNone(tool_path)
    
    def test_get_tool_path_unavailable_tool(self):
        """Test getting tool path when tool is not available."""
        # Mock tool as unavailable
        self.mock_executor.validate_tool_availability.return_value = False
        
        tool_path = self.generator.get_tool_path("python")
        self.assertIsNone(tool_path)
    
    def test_build_command_supported_language(self):
        """Test building command for supported language."""
        # Mock tool availability
        self.mock_executor.validate_tool_availability.return_value = True
        self.mock_executor.get_memory_args.return_value = ["-J-Xmx4g"]
        
        input_file = Path("test.py")
        output_dir = Path("output")
        
        command = self.generator.build_command("python", input_file, output_dir)
        
        self.assertIsInstance(command, list)
        self.assertIn(str(self.joern_cli_path / "pysrc2cpg.bat"), command)
        self.assertIn(str(input_file), command)
        self.assertIn(str(output_dir), command)
    
    def test_build_command_unsupported_language(self):
        """Test building command for unsupported language."""
        input_file = Path("test.xyz")
        output_dir = Path("output")
        
        command = self.generator.build_command("nonexistent", input_file, output_dir)
        self.assertIsNone(command)
    
    def test_generate_cpg_nonexistent_input(self):
        """Test CPG generation with non-existent input file."""
        input_file = Path("nonexistent.py")
        output_dir = Path("output")
        
        result = self.generator.generate_cpg("python", input_file, output_dir)
        
        self.assertIsInstance(result, GenerationResult)
        self.assertFalse(result.success)
        self.assertIn("Input file does not exist", result.error_message)
    
    def test_generate_cpg_success(self):
        """Test successful CPG generation."""
        # Create test input file
        input_file = Path(self.temp_dir) / "test.py"
        input_file.write_text("print('Hello, World!')")
        output_dir = Path(self.temp_dir) / "output"
        
        # Mock successful execution
        self.mock_executor.validate_tool_availability.return_value = True
        self.mock_executor.get_memory_args.return_value = ["-J-Xmx4g"]
        self.mock_executor.execute_command.return_value = ExecutionResult(
            success=True,
            return_code=0,
            stdout="CPG generated successfully",
            stderr="",
            execution_time=1.5,
            error_message=None
        )
        
        result = self.generator.generate_cpg("python", input_file, output_dir)
        
        self.assertIsInstance(result, GenerationResult)
        self.assertTrue(result.success)
        self.assertEqual(result.language, "python")
        self.assertEqual(result.input_file, input_file)
        self.assertEqual(result.output_dir, output_dir)
    
    def test_generate_cpg_failure(self):
        """Test failed CPG generation."""
        # Create test input file
        input_file = Path(self.temp_dir) / "test.py"
        input_file.write_text("print('Hello, World!')")
        output_dir = Path(self.temp_dir) / "output"
        
        # Mock failed execution
        self.mock_executor.validate_tool_availability.return_value = True
        self.mock_executor.get_memory_args.return_value = ["-J-Xmx4g"]
        self.mock_executor.execute_command.return_value = ExecutionResult(
            success=False,
            return_code=1,
            stdout="",
            stderr="Error: Failed to generate CPG",
            execution_time=0.5,
            error_message="Command failed"
        )
        
        result = self.generator.generate_cpg("python", input_file, output_dir)
        
        self.assertIsInstance(result, GenerationResult)
        self.assertFalse(result.success)
        self.assertEqual(result.return_code, 1)
        self.assertIn("Command failed", result.error_message)
    
    def test_get_available_languages(self):
        """Test getting available languages."""
        # Mock some tools as available
        def mock_validate(tool_path):
            return tool_path.name in ["pysrc2cpg.bat", "javasrc2cpg.bat"]
        
        self.mock_executor.validate_tool_availability.side_effect = mock_validate
        
        available = self.generator.get_available_languages()
        
        self.assertIsInstance(available, list)
        self.assertIn("python", available)
        self.assertIn("java", available)
    
    def test_validate_output_success(self):
        """Test output validation for successful generation."""
        # Create mock output files
        output_dir = Path(self.temp_dir) / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "test.cpg"
        output_file.write_text("mock cpg content")
        
        result = GenerationResult(
            language="python",
            input_file=Path("test.py"),
            output_dir=output_dir,
            success=True,
            execution_time=1.0,
            memory_usage=None,
            stdout="",
            stderr="",
            return_code=0,
            output_files=[output_file]
        )
        
        is_valid = self.generator.validate_output(result)
        self.assertTrue(is_valid)
    
    def test_validate_output_failure(self):
        """Test output validation for failed generation."""
        result = GenerationResult(
            language="python",
            input_file=Path("test.py"),
            output_dir=Path("output"),
            success=False,
            execution_time=0.0,
            memory_usage=None,
            stdout="",
            stderr="Error",
            return_code=1,
            output_files=[]
        )
        
        is_valid = self.generator.validate_output(result)
        self.assertFalse(is_valid)


class TestTestFileManager(unittest.TestCase):
    """Test cases for TestFileManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = TestFileManager(self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test TestFileManager initialization."""
        self.assertEqual(self.manager.base_dir, self.temp_dir)
    
    def test_create_test_environment(self):
        """Test creating test environment."""
        languages = ["python", "java"]
        
        with self.manager.create_test_environment(languages) as generator:
            self.assertIsInstance(generator, TestFileGenerator)
            
            # Check that files were created
            created_files = generator.get_created_files()
            self.assertGreater(len(created_files), 0)
    
    def test_validate_test_files(self):
        """Test validating multiple test files."""
        # Create test files
        test_files = []
        
        # Python file
        py_file = os.path.join(self.temp_dir, "test.py")
        with open(py_file, 'w') as f:
            f.write("def hello():\n    print('Hello')\n")
        test_files.append(py_file)
        
        # Java file
        java_file = os.path.join(self.temp_dir, "Test.java")
        with open(java_file, 'w') as f:
            f.write("public class Test { public static void main(String[] args) {} }")
        test_files.append(java_file)
        
        # Invalid file
        invalid_file = os.path.join(self.temp_dir, "invalid.xyz")
        with open(invalid_file, 'w') as f:
            f.write("invalid content")
        test_files.append(invalid_file)
        
        results = self.manager.validate_test_files(test_files)
        
        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), 3)
        self.assertTrue(results[py_file])
        self.assertTrue(results[java_file])
        self.assertFalse(results[invalid_file])


if __name__ == '__main__':
    unittest.main()