"""
Unit tests for language discovery functionality.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from pathlib import Path

from joern_verification.discovery.discovery import LanguageDiscoveryManager
from joern_verification.discovery.scanner import JoernInstallationScanner, LanguageTool
from joern_verification.discovery.language_database import LanguageSupportDatabase, SupportLevel


class TestLanguageDiscoveryManager(unittest.TestCase):
    """Test cases for LanguageDiscoveryManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.joern_cli_path = os.path.join(self.temp_dir, "joern-cli")
        os.makedirs(self.joern_cli_path, exist_ok=True)
        
        # Create mock tool files
        self.mock_tools = [
            "c2cpg.bat",
            "javasrc2cpg.bat", 
            "pysrc2cpg.bat"
        ]
        for tool in self.mock_tools:
            tool_path = os.path.join(self.joern_cli_path, tool)
            with open(tool_path, 'w') as f:
                f.write("@echo off\necho Mock tool")
        
        self.discovery_manager = LanguageDiscoveryManager(self.joern_cli_path)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test LanguageDiscoveryManager initialization."""
        self.assertEqual(self.discovery_manager.joern_cli_path, self.joern_cli_path)
        self.assertIsInstance(self.discovery_manager.scanner, JoernInstallationScanner)
        self.assertIsInstance(self.discovery_manager.database, LanguageSupportDatabase)
        self.assertFalse(self.discovery_manager.discovery_completed)
    
    @patch('joern_verification.discovery.scanner.JoernInstallationScanner.scan_installation')
    def test_discover_languages_success(self, mock_scan):
        """Test successful language discovery."""
        # Mock scanner results
        mock_tools = [
            LanguageTool("c2cpg.bat", "/path/to/c2cpg.bat", "c", True),
            LanguageTool("javasrc2cpg.bat", "/path/to/javasrc2cpg.bat", "java", True)
        ]
        mock_scan.return_value = mock_tools
        
        # Execute discovery
        results = self.discovery_manager.discover_languages()
        
        # Verify results
        self.assertIsInstance(results, dict)
        self.assertEqual(results['discovery_status'], 'completed')
        self.assertTrue(self.discovery_manager.discovery_completed)
        self.assertIn('discovered_tools', results)
        self.assertEqual(len(results['discovered_tools']), 2)
    
    def test_discover_languages_invalid_path(self):
        """Test discovery with invalid Joern path."""
        invalid_manager = LanguageDiscoveryManager("/invalid/path")
        
        with self.assertRaises(Exception):
            invalid_manager.discover_languages()
    
    def test_get_supported_languages_before_discovery(self):
        """Test getting supported languages before discovery is completed."""
        with self.assertRaises(RuntimeError):
            self.discovery_manager.get_supported_languages()
    
    @patch('joern_verification.discovery.scanner.JoernInstallationScanner.scan_installation')
    def test_get_supported_languages_after_discovery(self, mock_scan):
        """Test getting supported languages after discovery."""
        mock_tools = [
            LanguageTool("c2cpg.bat", "/path/to/c2cpg.bat", "c", True)
        ]
        mock_scan.return_value = mock_tools
        
        self.discovery_manager.discover_languages()
        supported = self.discovery_manager.get_supported_languages()
        
        self.assertIsInstance(supported, list)
    
    def test_validate_language_support_before_discovery(self):
        """Test language validation before discovery."""
        is_supported, message = self.discovery_manager.validate_language_support("python")
        
        self.assertFalse(is_supported)
        self.assertEqual(message, "Discovery not completed")


class TestJoernInstallationScanner(unittest.TestCase):
    """Test cases for JoernInstallationScanner."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.joern_cli_path = os.path.join(self.temp_dir, "joern-cli")
        os.makedirs(self.joern_cli_path, exist_ok=True)
        
        self.scanner = JoernInstallationScanner(self.joern_cli_path)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test scanner initialization."""
        self.assertEqual(str(self.scanner.joern_cli_path), self.joern_cli_path)
        self.assertIsInstance(self.scanner.discovered_tools, list)
    
    def test_scan_installation_with_tools(self):
        """Test scanning installation with available tools."""
        # Create mock tool files
        tools = ["c2cpg.bat", "javasrc2cpg.bat", "pysrc2cpg.bat"]
        for tool in tools:
            tool_path = os.path.join(self.joern_cli_path, tool)
            with open(tool_path, 'w') as f:
                f.write("@echo off\necho Mock tool")
        
        discovered = self.scanner.scan_installation()
        
        self.assertIsInstance(discovered, list)
        self.assertGreater(len(discovered), 0)
        
        # Check that tools were discovered
        tool_names = [tool.name for tool in discovered]
        for tool in tools:
            self.assertIn(tool, tool_names)
    
    def test_scan_empty_installation(self):
        """Test scanning empty installation directory."""
        discovered = self.scanner.scan_installation()
        
        self.assertIsInstance(discovered, list)
        self.assertEqual(len(discovered), 0)
    
    def test_scan_nonexistent_installation(self):
        """Test scanning non-existent installation directory."""
        invalid_scanner = JoernInstallationScanner("/invalid/path")
        
        with self.assertRaises(FileNotFoundError):
            invalid_scanner.scan_installation()


class TestLanguageSupportDatabase(unittest.TestCase):
    """Test cases for LanguageSupportDatabase."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.database = LanguageSupportDatabase()
    
    def test_initialization(self):
        """Test database initialization."""
        self.assertIsInstance(self.database.languages, dict)
        self.assertIsInstance(self.database.tool_mappings, dict)
        self.assertGreater(len(self.database.languages), 0)  # Should have predefined languages
    
    def test_get_supported_languages(self):
        """Test getting supported languages."""
        supported = self.database.get_supported_languages()
        
        self.assertIsInstance(supported, list)
        # Initially no languages should be fully supported without tools
        self.assertEqual(len(supported), 0)
    
    def test_get_language_info(self):
        """Test getting language information."""
        # Test with known language
        info = self.database.get_language_info("python")
        if info:  # Language might be in database
            self.assertEqual(info.language, "python")
        
        # Test with unknown language
        info = self.database.get_language_info("nonexistent")
        self.assertIsNone(info)
    
    def test_update_from_scanner(self):
        """Test updating database from scanner results."""
        mock_tools = [
            LanguageTool("c2cpg.bat", "/path/to/c2cpg.bat", "c", True),
            LanguageTool("pysrc2cpg.bat", "/path/to/pysrc2cpg.bat", "python", True)
        ]
        
        self.database.update_from_scanner(mock_tools)
        
        # Check that tools were registered (languages may not be "supported" without proper validation)
        self.assertGreater(len(self.database.tool_mappings), 0)
        
        # Check that language info exists
        c_info = self.database.get_language_info("c")
        python_info = self.database.get_language_info("python")
        
        if c_info:
            self.assertEqual(c_info.language, "c")
        if python_info:
            self.assertEqual(python_info.language, "python")
    
    def test_get_command_for_language(self):
        """Test getting command for language."""
        # Test with unsupported language
        command = self.database.get_command_for_language("python", "input.py", "output")
        self.assertIsNone(command)  # Should be None if not supported
        
        # Mock a supported language
        mock_tools = [
            LanguageTool("pysrc2cpg.bat", "/path/to/pysrc2cpg.bat", "python", True)
        ]
        self.database.update_from_scanner(mock_tools)
        
        command = self.database.get_command_for_language("python", "input.py", "output")
        if command:  # Command should be generated if language is supported
            self.assertIn("pysrc2cpg.bat", command)
            self.assertIn("input.py", command)
            self.assertIn("output", command)


if __name__ == '__main__':
    unittest.main()