"""
Joern Installation Scanner

This module scans the Joern installation directory to discover available
language tools and validate their availability.
"""

import os
import re
import stat
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class LanguageTool:
    """Represents a discovered language tool in Joern installation."""
    name: str
    path: str
    language: str
    executable: bool
    version: Optional[str] = None


class JoernInstallationScanner:
    """Scans Joern installation to discover available language tools."""
    
    # Mapping of tool patterns to programming languages
    TOOL_LANGUAGE_MAP = {
        r'^c2cpg\.(bat|sh)$': ['C', 'C++'],
        r'^csharpsrc2cpg\.(bat|sh)$': ['C#'],
        r'^javasrc2cpg\.(bat|sh)$': ['Java'],
        r'^jssrc2cpg\.(bat|sh)$': ['JavaScript'],
        r'^kotlin2cpg\.(bat|sh)$': ['Kotlin'],
        r'^php2cpg\.(bat|sh)$': ['PHP'],
        r'^pysrc2cpg\.(bat|sh)$': ['Python'],
        r'^rubysrc2cpg\.(bat|sh)$': ['Ruby'],
        r'^swiftsrc2cpg\.(bat|sh)$': ['Swift'],
        r'^gosrc2cpg\.(bat|sh)$': ['Go'],
        r'^ghidra2cpg\.(bat|sh)$': ['Binary/Assembly'],
        r'^jimple2cpg\.(bat|sh)$': ['Java Bytecode']
    }
    
    def __init__(self, joern_cli_path: str):
        """
        Initialize scanner with Joern CLI directory path.
        
        Args:
            joern_cli_path: Path to joern-cli directory
        """
        self.joern_cli_path = Path(joern_cli_path)
        self.discovered_tools: List[LanguageTool] = []
        
    def scan_installation(self) -> List[LanguageTool]:
        """
        Scan Joern installation directory for available language tools.
        
        Returns:
            List of discovered language tools
            
        Raises:
            FileNotFoundError: If joern-cli directory doesn't exist
            PermissionError: If directory is not accessible
        """
        if not self.joern_cli_path.exists():
            raise FileNotFoundError(f"Joern CLI directory not found: {self.joern_cli_path}")
            
        if not self.joern_cli_path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {self.joern_cli_path}")
            
        logger.info(f"Scanning Joern installation at: {self.joern_cli_path}")
        
        self.discovered_tools = []
        
        try:
            # Scan all files in joern-cli directory
            for file_path in self.joern_cli_path.iterdir():
                if file_path.is_file():
                    tool = self._analyze_tool_file(file_path)
                    if tool:
                        self.discovered_tools.append(tool)
                        
        except PermissionError as e:
            logger.error(f"Permission denied accessing {self.joern_cli_path}: {e}")
            raise
            
        logger.info(f"Discovered {len(self.discovered_tools)} language tools")
        return self.discovered_tools
    
    def _analyze_tool_file(self, file_path: Path) -> Optional[LanguageTool]:
        """
        Analyze a file to determine if it's a language tool.
        
        Args:
            file_path: Path to the file to analyze
            
        Returns:
            LanguageTool if file is a recognized language tool, None otherwise
        """
        filename = file_path.name
        
        # Check if filename matches any known language tool pattern
        for pattern, languages in self.TOOL_LANGUAGE_MAP.items():
            if re.search(pattern, filename, re.IGNORECASE):
                # Determine primary language (first in list)
                primary_language = languages[0]
                
                # Check if file is executable
                is_executable = self._is_executable(file_path)
                
                tool = LanguageTool(
                    name=filename,
                    path=str(file_path),
                    language=primary_language,
                    executable=is_executable
                )
                
                logger.debug(f"Found language tool: {filename} -> {primary_language} (executable: {is_executable})")
                return tool
                
        return None
    
    def _is_executable(self, file_path: Path) -> bool:
        """
        Check if a file is executable.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            True if file is executable, False otherwise
        """
        try:
            file_stat = file_path.stat()
            
            # On Windows, .bat files are executable by default
            if file_path.suffix.lower() == '.bat':
                return True
                
            # On Unix-like systems, check execute permission
            if hasattr(stat, 'S_IXUSR'):
                return bool(file_stat.st_mode & stat.S_IXUSR)
                
            # Fallback: assume executable if we can read it
            return os.access(file_path, os.R_OK)
            
        except (OSError, AttributeError):
            return False
    
    def get_installation_summary(self) -> Dict[str, any]:
        """
        Get summary of Joern installation analysis.
        
        Returns:
            Dictionary containing installation summary
        """
        executable_tools = [tool for tool in self.discovered_tools if tool.executable]
        non_executable_tools = [tool for tool in self.discovered_tools if not tool.executable]
        
        # Group tools by language
        languages = {}
        for tool in self.discovered_tools:
            if tool.language not in languages:
                languages[tool.language] = []
            languages[tool.language].append(tool.name)
        
        return {
            'installation_path': str(self.joern_cli_path),
            'total_tools': len(self.discovered_tools),
            'executable_tools': len(executable_tools),
            'non_executable_tools': len(non_executable_tools),
            'languages_found': list(languages.keys()),
            'tools_by_language': languages,
            'executable_tool_names': [tool.name for tool in executable_tools],
            'non_executable_tool_names': [tool.name for tool in non_executable_tools]
        }
    
    def validate_tool_availability(self, tool_name: str) -> bool:
        """
        Validate if a specific tool is available and executable.
        
        Args:
            tool_name: Name of the tool to validate
            
        Returns:
            True if tool is available and executable, False otherwise
        """
        for tool in self.discovered_tools:
            if tool.name == tool_name:
                return tool.executable
        return False
    
    def get_tools_for_language(self, language: str) -> List[LanguageTool]:
        """
        Get all tools that support a specific language.
        
        Args:
            language: Programming language name
            
        Returns:
            List of tools that support the language
        """
        return [tool for tool in self.discovered_tools if tool.language == language]
    
    def get_executable_tools(self) -> List[LanguageTool]:
        """
        Get all executable tools.
        
        Returns:
            List of executable tools
        """
        return [tool for tool in self.discovered_tools if tool.executable]
    
    def export_results(self, output_path: str):
        """
        Export discovery results to JSON file.
        
        Args:
            output_path: Path to output file
        """
        import json
        
        results = {
            'installation_summary': self.get_installation_summary(),
            'discovered_tools': [
                {
                    'name': tool.name,
                    'path': tool.path,
                    'language': tool.language,
                    'executable': tool.executable,
                    'version': tool.version
                }
                for tool in self.discovered_tools
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Scanner results exported to: {output_path}")