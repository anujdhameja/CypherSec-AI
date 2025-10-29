"""
Test fixtures and utilities for Joern verification tests.
"""

import tempfile
import os
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import Mock

from joern_verification.core.interfaces import GenerationResult, AnalysisReport, LanguageSupport
from joern_verification.discovery.scanner import LanguageTool
from joern_verification.generation.command_executor import ExecutionResult


class TestFixtures:
    """Collection of test fixtures for unit tests."""
    
    @staticmethod
    def create_temp_directory() -> str:
        """Create a temporary directory for testing."""
        return tempfile.mkdtemp(prefix="joern_test_")
    
    @staticmethod
    def create_mock_joern_installation(base_dir: str, tools: Optional[List[str]] = None) -> str:
        """
        Create a mock Joern installation directory with tool files.
        
        Args:
            base_dir: Base directory for the installation
            tools: List of tool names to create (uses default if None)
            
        Returns:
            Path to the joern-cli directory
        """
        if tools is None:
            tools = [
                "c2cpg.bat",
                "javasrc2cpg.bat", 
                "pysrc2cpg.bat",
                "csharpsrc2cpg.bat",
                "jssrc2cpg.bat",
                "php2cpg.bat"
            ]
        
        joern_cli_path = os.path.join(base_dir, "joern-cli")
        os.makedirs(joern_cli_path, exist_ok=True)
        
        for tool in tools:
            tool_path = os.path.join(joern_cli_path, tool)
            with open(tool_path, 'w') as f:
                f.write(f"@echo off\necho Mock {tool}")
        
        return joern_cli_path
    
    @staticmethod
    def create_test_source_file(directory: str, language: str, content: Optional[str] = None) -> str:
        """
        Create a test source file for a specific language.
        
        Args:
            directory: Directory to create the file in
            language: Programming language
            content: File content (uses default if None)
            
        Returns:
            Path to the created file
        """
        extensions = {
            'python': '.py',
            'java': '.java',
            'c': '.c',
            'cpp': '.cpp',
            'csharp': '.cs',
            'javascript': '.js',
            'php': '.php',
            'kotlin': '.kt',
            'ruby': '.rb',
            'swift': '.swift',
            'go': '.go'
        }
        
        default_content = {
            'python': 'def hello():\n    print("Hello, World!")\n',
            'java': 'public class Test {\n    public static void main(String[] args) {\n        System.out.println("Hello, World!");\n    }\n}',
            'c': '#include <stdio.h>\nint main() {\n    printf("Hello, World!\\n");\n    return 0;\n}',
            'cpp': '#include <iostream>\nint main() {\n    std::cout << "Hello, World!" << std::endl;\n    return 0;\n}',
            'csharp': 'using System;\nclass Program {\n    static void Main() {\n        Console.WriteLine("Hello, World!");\n    }\n}',
            'javascript': 'function hello() {\n    console.log("Hello, World!");\n}\nhello();',
            'php': '<?php\necho "Hello, World!";\n?>',
            'kotlin': 'fun main() {\n    println("Hello, World!")\n}',
            'ruby': 'puts "Hello, World!"',
            'swift': 'print("Hello, World!")',
            'go': 'package main\nimport "fmt"\nfunc main() {\n    fmt.Println("Hello, World!")\n}'
        }
        
        extension = extensions.get(language, '.txt')
        filename = f"test{extension}"
        file_path = os.path.join(directory, filename)
        
        file_content = content or default_content.get(language, f"// Test file for {language}")
        
        with open(file_path, 'w') as f:
            f.write(file_content)
        
        return file_path
    
    @staticmethod
    def create_mock_generation_result(
        language: str = "python",
        success: bool = True,
        execution_time: float = 1.0,
        output_files: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None,
        error_message: Optional[str] = None
    ) -> GenerationResult:
        """Create a mock GenerationResult for testing."""
        if output_files is None:
            output_files = [Path(f"output/{language}_test.cpg")] if success else []
        else:
            output_files = [Path(f) for f in output_files]
        
        return GenerationResult(
            language=language,
            input_file=f"test.{language}",
            output_dir="output",
            success=success,
            execution_time=execution_time,
            memory_usage=512 if success else None,
            stdout="CPG generated successfully" if success else "",
            stderr="" if success else "Error occurred",
            return_code=0 if success else 1,
            output_files=output_files,
            warnings=warnings or [],
            error_message=error_message
        )
    
    @staticmethod
    def create_mock_analysis_report(
        language: str = "python",
        category: str = "success",
        errors: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None,
        recommendations: Optional[List[str]] = None
    ) -> AnalysisReport:
        """Create a mock AnalysisReport for testing."""
        return AnalysisReport(
            language=language,
            category=category,
            metrics={
                'execution_time': 1.0,
                'memory_usage': 512,
                'output_file_count': 1,
                'success_rate': 1.0 if category == "success" else 0.0
            },
            errors=errors or [],
            warnings=warnings or [],
            recommendations=recommendations or [f"Language {language} is working correctly"]
        )
    
    @staticmethod
    def create_mock_language_tool(
        name: str = "pysrc2cpg.bat",
        language: str = "python",
        executable: bool = True
    ) -> LanguageTool:
        """Create a mock LanguageTool for testing."""
        return LanguageTool(
            name=name,
            path=f"/mock/path/{name}",
            language=language,
            executable=executable
        )
    
    @staticmethod
    def create_mock_language_support(
        language: str = "python",
        supported: bool = True,
        tool_available: bool = True
    ) -> LanguageSupport:
        """Create a mock LanguageSupport for testing."""
        return LanguageSupport(
            language=language,
            tool_available=tool_available,
            tool_path=f"/mock/path/{language}src2cpg.bat" if tool_available else "",
            supported=supported,
            alternative_tools=["tree-sitter", "ast-parser"] if not supported else [],
            notes=f"Language {language} support status"
        )
    
    @staticmethod
    def create_mock_execution_result(
        success: bool = True,
        return_code: int = 0,
        stdout: str = "Success",
        stderr: str = "",
        execution_time: float = 1.0,
        error_message: Optional[str] = None
    ) -> ExecutionResult:
        """Create a mock ExecutionResult for testing."""
        return ExecutionResult(
            success=success,
            return_code=return_code,
            stdout=stdout,
            stderr=stderr,
            execution_time=execution_time,
            error_message=error_message
        )


class MockCommandExecutor:
    """Mock command executor for testing CPG generation."""
    
    def __init__(self, simulate_success: bool = True):
        """
        Initialize mock executor.
        
        Args:
            simulate_success: Whether to simulate successful command execution
        """
        self.simulate_success = simulate_success
        self.executed_commands = []
    
    def execute_command(
        self,
        command: List[str],
        working_dir: Optional[Path] = None,
        timeout: Optional[int] = None
    ) -> ExecutionResult:
        """Mock command execution."""
        self.executed_commands.append({
            'command': command,
            'working_dir': working_dir,
            'timeout': timeout
        })
        
        if self.simulate_success:
            return TestFixtures.create_mock_execution_result(
                success=True,
                stdout="CPG generated successfully",
                execution_time=1.5
            )
        else:
            return TestFixtures.create_mock_execution_result(
                success=False,
                return_code=1,
                stderr="Command execution failed",
                error_message="Mock execution failure"
            )
    
    def validate_tool_availability(self, tool_path: Path) -> bool:
        """Mock tool availability validation."""
        # Simulate that common tools are available
        available_tools = [
            "pysrc2cpg.bat",
            "javasrc2cpg.bat",
            "c2cpg.bat",
            "csharpsrc2cpg.bat"
        ]
        return tool_path.name in available_tools
    
    def get_memory_args(self, memory: Optional[str] = None) -> List[str]:
        """Mock memory arguments generation."""
        memory = memory or "4g"
        return [f"-J-Xmx{memory}"]


class TestDataGenerator:
    """Generator for test data scenarios."""
    
    @staticmethod
    def generate_mixed_results(count: int = 5) -> List[GenerationResult]:
        """Generate a mix of successful and failed results for testing."""
        results = []
        languages = ["python", "java", "cpp", "javascript", "csharp"]
        
        for i in range(count):
            language = languages[i % len(languages)]
            success = i % 3 != 0  # 2/3 success rate
            
            result = TestFixtures.create_mock_generation_result(
                language=language,
                success=success,
                execution_time=1.0 + i * 0.5,
                warnings=["Minor warning"] if success and i % 2 == 0 else None,
                error_message="Mock error" if not success else None
            )
            results.append(result)
        
        return results
    
    @staticmethod
    def generate_language_scenarios() -> Dict[str, Dict[str, any]]:
        """Generate different language testing scenarios."""
        return {
            "fully_supported": {
                "languages": ["python", "java", "c"],
                "success_rate": 1.0,
                "has_warnings": False
            },
            "partially_supported": {
                "languages": ["cpp", "javascript"],
                "success_rate": 0.7,
                "has_warnings": True
            },
            "unsupported": {
                "languages": ["rust", "haskell"],
                "success_rate": 0.0,
                "has_warnings": False
            }
        }
    
    @staticmethod
    def generate_error_scenarios() -> List[Dict[str, any]]:
        """Generate different error scenarios for testing."""
        return [
            {
                "name": "compilation_error",
                "stderr": "Error: Compilation failed at line 15\nSyntax error: missing semicolon",
                "return_code": 1,
                "category": "failure"
            },
            {
                "name": "memory_error",
                "stderr": "Error: OutOfMemoryError\nJava heap space",
                "return_code": 1,
                "category": "failure"
            },
            {
                "name": "timeout_error",
                "stderr": "Error: Process timed out after 300 seconds",
                "return_code": 124,
                "category": "failure"
            },
            {
                "name": "partial_success",
                "stderr": "Warning: Some language constructs not supported\nWarning: Incomplete analysis",
                "return_code": 0,
                "category": "partial_success"
            }
        ]


class TestEnvironmentManager:
    """Manager for test environment setup and cleanup."""
    
    def __init__(self):
        """Initialize test environment manager."""
        self.temp_directories = []
        self.created_files = []
    
    def create_test_workspace(self, languages: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Create a complete test workspace with Joern installation and test files.
        
        Args:
            languages: Languages to create test files for
            
        Returns:
            Dictionary with paths to created resources
        """
        if languages is None:
            languages = ["python", "java", "c", "cpp"]
        
        # Create base directory
        base_dir = TestFixtures.create_temp_directory()
        self.temp_directories.append(base_dir)
        
        # Create mock Joern installation
        joern_cli_path = TestFixtures.create_mock_joern_installation(base_dir)
        
        # Create test files directory
        test_files_dir = os.path.join(base_dir, "test_files")
        os.makedirs(test_files_dir, exist_ok=True)
        
        # Create test files for each language
        test_files = {}
        for language in languages:
            file_path = TestFixtures.create_test_source_file(test_files_dir, language)
            test_files[language] = file_path
            self.created_files.append(file_path)
        
        # Create output directory
        output_dir = os.path.join(base_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        return {
            'base_dir': base_dir,
            'joern_cli_path': joern_cli_path,
            'test_files_dir': test_files_dir,
            'output_dir': output_dir,
            'test_files': test_files
        }
    
    def cleanup(self):
        """Clean up all created test resources."""
        import shutil
        
        for temp_dir in self.temp_directories:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
        
        self.temp_directories.clear()
        self.created_files.clear()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()