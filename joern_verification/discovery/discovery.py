"""
Main Discovery Module

This module provides the main interface for language discovery functionality,
combining the scanner and database components.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .scanner import JoernInstallationScanner, LanguageTool
from .language_database import LanguageSupportDatabase, LanguageInfo, SupportLevel

logger = logging.getLogger(__name__)


class LanguageDiscoveryManager:
    """Main manager for language discovery operations."""
    
    def __init__(self, joern_cli_path: str):
        """
        Initialize discovery manager.
        
        Args:
            joern_cli_path: Path to joern-cli directory
        """
        self.joern_cli_path = joern_cli_path
        self.scanner = JoernInstallationScanner(joern_cli_path)
        self.database = LanguageSupportDatabase()
        self.discovery_completed = False
    
    def discover_languages(self) -> Dict[str, any]:
        """
        Perform complete language discovery process.
        
        Returns:
            Dictionary containing discovery results
            
        Raises:
            FileNotFoundError: If Joern installation not found
            PermissionError: If installation directory not accessible
        """
        logger.info("Starting language discovery process")
        
        try:
            # Scan Joern installation
            discovered_tools = self.scanner.scan_installation()
            
            # Update database with discovered tools
            self.database.update_from_scanner(discovered_tools)
            
            self.discovery_completed = True
            
            # Generate discovery results
            results = self._generate_discovery_results(discovered_tools)
            
            logger.info("Language discovery completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Language discovery failed: {e}")
            raise
    
    def _generate_discovery_results(self, discovered_tools: List[LanguageTool]) -> Dict[str, any]:
        """
        Generate comprehensive discovery results.
        
        Args:
            discovered_tools: List of discovered tools
            
        Returns:
            Dictionary containing discovery results
        """
        scanner_summary = self.scanner.get_installation_summary()
        database_summary = self.database.get_database_summary()
        
        # Combine results
        results = {
            'discovery_status': 'completed',
            'joern_installation': scanner_summary,
            'language_database': database_summary,
            'discovered_tools': [
                {
                    'name': tool.name,
                    'path': tool.path,
                    'language': tool.language,
                    'executable': tool.executable
                }
                for tool in discovered_tools
            ],
            'language_support_details': self._get_language_support_details()
        }
        
        return results
    
    def _get_language_support_details(self) -> Dict[str, Dict[str, any]]:
        """
        Get detailed language support information.
        
        Returns:
            Dictionary mapping language names to support details
        """
        details = {}
        
        for lang_name, lang_info in self.database.languages.items():
            details[lang_name] = {
                'support_level': lang_info.support_level.value,
                'primary_tool': lang_info.primary_tool,
                'file_extensions': lang_info.file_extensions,
                'alternative_tools': lang_info.alternative_tools,
                'command_template': lang_info.command_template,
                'notes': lang_info.notes
            }
        
        return details
    
    def get_supported_languages(self) -> List[str]:
        """
        Get list of fully supported languages.
        
        Returns:
            List of supported language names
        """
        if not self.discovery_completed:
            raise RuntimeError("Discovery must be completed before getting supported languages")
        
        return self.database.get_supported_languages()
    
    def get_language_info(self, language: str) -> Optional[LanguageInfo]:
        """
        Get information about a specific language.
        
        Args:
            language: Programming language name
            
        Returns:
            LanguageInfo object or None if not found
        """
        return self.database.get_language_info(language)
    
    def get_command_for_language(self, language: str, input_path: str, output_dir: str) -> Optional[str]:
        """
        Get CPG generation command for a specific language.
        
        Args:
            language: Programming language name
            input_path: Path to input source file
            output_dir: Output directory for CPG
            
        Returns:
            Complete command string or None if not supported
        """
        if not self.discovery_completed:
            raise RuntimeError("Discovery must be completed before getting commands")
        
        return self.database.get_command_for_language(language, input_path, output_dir)
    
    def validate_language_support(self, language: str) -> Tuple[bool, str]:
        """
        Validate if a language is supported and provide status message.
        
        Args:
            language: Programming language name
            
        Returns:
            Tuple of (is_supported, status_message)
        """
        if not self.discovery_completed:
            return False, "Discovery not completed"
        
        lang_info = self.database.get_language_info(language)
        if not lang_info:
            return False, f"Language '{language}' not recognized"
        
        if lang_info.support_level == SupportLevel.FULLY_SUPPORTED:
            return True, f"Language '{language}' is fully supported"
        elif lang_info.support_level == SupportLevel.PARTIALLY_SUPPORTED:
            return False, f"Language '{language}' is partially supported (tool not executable)"
        else:
            return False, f"Language '{language}' is not supported"
    
    def get_alternative_tools(self, language: str) -> List[str]:
        """
        Get alternative tools for a language.
        
        Args:
            language: Programming language name
            
        Returns:
            List of alternative tool names
        """
        return self.database.get_alternative_tools(language)
    
    def export_results(self, output_path: str):
        """
        Export discovery results to JSON file.
        
        Args:
            output_path: Path to output file
        """
        if not self.discovery_completed:
            raise RuntimeError("Discovery must be completed before exporting results")
        
        self.database.export_to_json(output_path)
        logger.info(f"Discovery results exported to: {output_path}")
    
    def get_discovery_summary(self) -> str:
        """
        Get human-readable summary of discovery results.
        
        Returns:
            Formatted summary string
        """
        if not self.discovery_completed:
            return "Discovery not completed"
        
        supported = self.database.get_supported_languages()
        total_languages = len(self.database.languages)
        total_tools = len(self.database.tool_mappings)
        
        summary = f"""
Language Discovery Summary:
==========================
Joern CLI Path: {self.joern_cli_path}
Total Languages: {total_languages}
Supported Languages: {len(supported)}
Total Tools Found: {total_tools}

Supported Languages:
{', '.join(supported) if supported else 'None'}

Unsupported Languages:
{', '.join([
    name for name, info in self.database.languages.items()
    if info.support_level != SupportLevel.FULLY_SUPPORTED
]) or 'None'}
"""
        
        return summary.strip()