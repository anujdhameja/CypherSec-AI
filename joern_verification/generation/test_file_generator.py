"""
Test file generator for creating language-specific test files.
Handles file creation, validation, and cleanup for CPG generation testing.
"""

import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

from .test_templates import TestTemplates


class TestFileGenerator:
    """Generates and manages test files for different programming languages."""
    
    def __init__(self, base_output_dir: Optional[str] = None):
        """
        Initialize the test file generator.
        
        Args:
            base_output_dir: Base directory for test files. If None, uses temp directory.
        """
        self.templates = TestTemplates()
        self.base_output_dir = base_output_dir or tempfile.mkdtemp(prefix="joern_test_")
        self.created_files: List[str] = []
        self.created_dirs: List[str] = []
        
        # Ensure base directory exists
        Path(self.base_output_dir).mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
    
    def generate_test_file(self, language: str, output_path: Optional[str] = None) -> Tuple[bool, str]:
        """
        Generate a test file for the specified language.
        
        Args:
            language: Programming language name (e.g., 'python', 'java', 'c')
            output_path: Custom output path. If None, generates default path.
            
        Returns:
            Tuple of (success: bool, file_path: str)
        """
        try:
            # Get template content
            content = self.get_test_content(language)
            if not content:
                return False, f"No template available for language: {language}"
            
            # Determine output path
            if output_path is None:
                output_path = self._get_default_file_path(language)
            
            # Ensure directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                self.created_dirs.append(output_dir)
            
            # Write file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            self.created_files.append(output_path)
            self.logger.info(f"Generated test file for {language}: {output_path}")
            
            # Validate syntax if possible
            is_valid = self.validate_syntax(language, content)
            if not is_valid:
                self.logger.warning(f"Syntax validation failed for {language} test file")
            
            return True, output_path
            
        except Exception as e:
            self.logger.error(f"Failed to generate test file for {language}: {str(e)}")
            return False, str(e)
    
    def get_test_content(self, language: str) -> Optional[str]:
        """
        Get test content for the specified language.
        
        Args:
            language: Programming language name
            
        Returns:
            Test content string or None if not available
        """
        templates = self.templates.get_all_templates()
        return templates.get(language.lower())
    
    def validate_syntax(self, language: str, content: str) -> bool:
        """
        Validate syntax of test content for the specified language.
        
        Args:
            language: Programming language name
            content: Source code content to validate
            
        Returns:
            True if syntax is valid, False otherwise
        """
        try:
            # Basic validation - check for common syntax issues
            if not content or not content.strip():
                return False
            
            # Language-specific basic validation
            if language.lower() == 'python':
                return self._validate_python_syntax(content)
            elif language.lower() == 'java':
                return self._validate_java_syntax(content)
            elif language.lower() in ['c', 'cpp']:
                return self._validate_c_syntax(content)
            elif language.lower() == 'javascript':
                return self._validate_javascript_syntax(content)
            
            # For other languages, do basic checks
            return self._validate_basic_syntax(content)
            
        except Exception as e:
            self.logger.error(f"Syntax validation error for {language}: {str(e)}")
            return False
    
    def _validate_python_syntax(self, content: str) -> bool:
        """Validate Python syntax using compile()."""
        try:
            compile(content, '<string>', 'exec')
            return True
        except SyntaxError:
            return False
    
    def _validate_java_syntax(self, content: str) -> bool:
        """Basic Java syntax validation."""
        # Check for basic Java structure
        required_elements = ['class ', 'public ', '{', '}']
        return all(element in content for element in required_elements)
    
    def _validate_c_syntax(self, content: str) -> bool:
        """Basic C/C++ syntax validation."""
        # Check for basic C structure
        required_elements = ['#include', 'int main', '{', '}']
        return all(element in content for element in required_elements)
    
    def _validate_javascript_syntax(self, content: str) -> bool:
        """Basic JavaScript syntax validation."""
        # Check for basic JavaScript structure
        return '{' in content and '}' in content and 'function' in content or 'class' in content
    
    def _validate_basic_syntax(self, content: str) -> bool:
        """Basic syntax validation for other languages."""
        # Check for balanced braces
        open_braces = content.count('{')
        close_braces = content.count('}')
        return open_braces == close_braces and len(content.strip()) > 10
    
    def _get_default_file_path(self, language: str) -> str:
        """Generate default file path for a language."""
        extensions = self.templates.get_file_extensions()
        extension = extensions.get(language.lower(), '.txt')
        filename = f"test_sample{extension}"
        return os.path.join(self.base_output_dir, language.lower(), filename)
    
    def generate_all_test_files(self) -> Dict[str, Tuple[bool, str]]:
        """
        Generate test files for all supported languages.
        
        Returns:
            Dictionary mapping language names to (success, path/error) tuples
        """
        results = {}
        templates = self.templates.get_all_templates()
        
        for language in templates.keys():
            success, path_or_error = self.generate_test_file(language)
            results[language] = (success, path_or_error)
        
        return results
    
    def cleanup_test_files(self) -> None:
        """Clean up all created test files and directories."""
        try:
            # Remove created files
            for file_path in self.created_files:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    self.logger.debug(f"Removed test file: {file_path}")
            
            # Remove created directories (in reverse order)
            for dir_path in reversed(self.created_dirs):
                if os.path.exists(dir_path) and not os.listdir(dir_path):
                    os.rmdir(dir_path)
                    self.logger.debug(f"Removed test directory: {dir_path}")
            
            # Remove base directory if it was temporary
            if self.base_output_dir.startswith(tempfile.gettempdir()):
                if os.path.exists(self.base_output_dir):
                    shutil.rmtree(self.base_output_dir)
                    self.logger.debug(f"Removed base test directory: {self.base_output_dir}")
            
            self.created_files.clear()
            self.created_dirs.clear()
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported programming languages."""
        return list(self.templates.get_all_templates().keys())
    
    def get_created_files(self) -> List[str]:
        """Get list of all created test files."""
        return self.created_files.copy()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic cleanup."""
        self.cleanup_test_files()


class TestFileManager:
    """Higher-level manager for test file operations."""
    
    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize test file manager.
        
        Args:
            base_dir: Base directory for test files
        """
        self.base_dir = base_dir
        self.logger = logging.getLogger(__name__)
    
    def create_test_environment(self, languages: Optional[List[str]] = None) -> TestFileGenerator:
        """
        Create a test environment with files for specified languages.
        
        Args:
            languages: List of languages to create files for. If None, creates for all.
            
        Returns:
            TestFileGenerator instance
        """
        generator = TestFileGenerator(self.base_dir)
        
        if languages is None:
            languages = generator.get_supported_languages()
        
        results = {}
        for language in languages:
            success, path_or_error = generator.generate_test_file(language)
            results[language] = (success, path_or_error)
            
            if success:
                self.logger.info(f"Created test file for {language}: {path_or_error}")
            else:
                self.logger.error(f"Failed to create test file for {language}: {path_or_error}")
        
        return generator
    
    def validate_test_files(self, file_paths: List[str]) -> Dict[str, bool]:
        """
        Validate multiple test files.
        
        Args:
            file_paths: List of file paths to validate
            
        Returns:
            Dictionary mapping file paths to validation results
        """
        results = {}
        generator = TestFileGenerator()
        
        for file_path in file_paths:
            try:
                # Determine language from file extension
                extension = Path(file_path).suffix.lower()
                language = self._extension_to_language(extension)
                
                if language:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    results[file_path] = generator.validate_syntax(language, content)
                else:
                    results[file_path] = False
                    
            except Exception as e:
                self.logger.error(f"Error validating {file_path}: {str(e)}")
                results[file_path] = False
        
        return results
    
    def _extension_to_language(self, extension: str) -> Optional[str]:
        """Map file extension to language name."""
        extension_map = {
            '.c': 'c',
            '.cpp': 'cpp',
            '.cs': 'csharp',
            '.py': 'python',
            '.java': 'java',
            '.php': 'php',
            '.js': 'javascript',
            '.kt': 'kotlin',
            '.rb': 'ruby',
            '.swift': 'swift',
            '.go': 'go'
        }
        return extension_map.get(extension)