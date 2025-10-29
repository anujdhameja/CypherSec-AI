"""
Language Support Database

This module manages language support information, mapping discovered tools
to their corresponding programming languages and storing command templates.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum
import logging

from .scanner import LanguageTool

logger = logging.getLogger(__name__)


class SupportLevel(Enum):
    """Enumeration of language support levels."""
    FULLY_SUPPORTED = "fully_supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    NOT_SUPPORTED = "not_supported"
    UNKNOWN = "unknown"


@dataclass
class LanguageInfo:
    """Information about a programming language and its support."""
    name: str
    file_extensions: List[str]
    support_level: SupportLevel
    primary_tool: Optional[str] = None
    alternative_tools: List[str] = None
    command_template: Optional[str] = None
    notes: Optional[str] = None
    
    def __post_init__(self):
        if self.alternative_tools is None:
            self.alternative_tools = []


@dataclass
class ToolMapping:
    """Mapping between a tool and its language support."""
    tool_name: str
    tool_path: str
    supported_languages: List[str]
    command_template: str
    memory_args: str = "-J-Xmx4g"
    executable: bool = True


class LanguageSupportDatabase:
    """Database for managing language support information."""
    
    # Default language configurations
    DEFAULT_LANGUAGE_CONFIG = {
        'C': {
            'file_extensions': ['.c', '.h'],
            'command_template': 'c2cpg.bat {memory_args} {input_path} --output {output_dir}',
            'alternative_tools': ['tree-sitter-c', 'clang-ast-dump']
        },
        'C++': {
            'file_extensions': ['.cpp', '.cxx', '.cc', '.hpp', '.hxx', '.hh'],
            'command_template': 'c2cpg.bat {memory_args} {input_path} --output {output_dir}',
            'alternative_tools': ['tree-sitter-cpp', 'clang-ast-dump']
        },
        'C#': {
            'file_extensions': ['.cs'],
            'command_template': 'csharpsrc2cpg.bat {memory_args} {input_path} --output {output_dir}',
            'alternative_tools': ['roslyn-analyzers', 'tree-sitter-c-sharp']
        },
        'Java': {
            'file_extensions': ['.java'],
            'command_template': 'javasrc2cpg.bat {memory_args} {input_path} --output {output_dir}',
            'alternative_tools': ['tree-sitter-java', 'eclipse-jdt']
        },
        'JavaScript': {
            'file_extensions': ['.js', '.mjs', '.jsx'],
            'command_template': 'jssrc2cpg.bat {memory_args} {input_path} --output {output_dir}',
            'alternative_tools': ['tree-sitter-javascript', 'babel-parser', 'esprima']
        },
        'Kotlin': {
            'file_extensions': ['.kt', '.kts'],
            'command_template': 'kotlin2cpg.bat {memory_args} {input_path} --output {output_dir}',
            'alternative_tools': ['tree-sitter-kotlin', 'kotlinc-ast']
        },
        'PHP': {
            'file_extensions': ['.php', '.phtml', '.php3', '.php4', '.php5'],
            'command_template': 'php2cpg.bat {memory_args} {input_path} --output {output_dir}',
            'alternative_tools': ['tree-sitter-php', 'php-parser']
        },
        'Python': {
            'file_extensions': ['.py', '.pyw', '.pyi'],
            'command_template': 'pysrc2cpg.bat {memory_args} {input_path} --output {output_dir}',
            'alternative_tools': ['tree-sitter-python', 'ast', 'libcst']
        },
        'Ruby': {
            'file_extensions': ['.rb', '.rbw'],
            'command_template': 'rubysrc2cpg.bat {memory_args} {input_path} --output {output_dir}',
            'alternative_tools': ['tree-sitter-ruby', 'ripper']
        },
        'Swift': {
            'file_extensions': ['.swift'],
            'command_template': 'swiftsrc2cpg.bat {memory_args} {input_path} --output {output_dir}',
            'alternative_tools': ['tree-sitter-swift', 'swift-syntax']
        },
        'Go': {
            'file_extensions': ['.go'],
            'command_template': 'gosrc2cpg.bat {memory_args} {input_path} --output {output_dir}',
            'alternative_tools': ['tree-sitter-go', 'go-ast']
        },
        'Binary/Assembly': {
            'file_extensions': ['.exe', '.dll', '.so', '.dylib'],
            'command_template': 'ghidra2cpg.bat {memory_args} {input_path} --output {output_dir}',
            'alternative_tools': ['radare2', 'ida-pro', 'binary-ninja']
        },
        'Java Bytecode': {
            'file_extensions': ['.class', '.jar'],
            'command_template': 'jimple2cpg.bat {memory_args} {input_path} --output {output_dir}',
            'alternative_tools': ['soot', 'asm-bytecode']
        }
    }
    
    def __init__(self):
        """Initialize the language support database."""
        self.languages: Dict[str, LanguageInfo] = {}
        self.tool_mappings: Dict[str, ToolMapping] = {}
        self._initialize_default_languages()
    
    def _initialize_default_languages(self):
        """Initialize database with default language configurations."""
        for lang_name, config in self.DEFAULT_LANGUAGE_CONFIG.items():
            language_info = LanguageInfo(
                name=lang_name,
                file_extensions=config['file_extensions'],
                support_level=SupportLevel.UNKNOWN,
                command_template=config['command_template'],
                alternative_tools=config['alternative_tools']
            )
            self.languages[lang_name] = language_info
    
    def update_from_scanner(self, discovered_tools: List[LanguageTool]):
        """
        Update database with information from scanner results.
        
        Args:
            discovered_tools: List of tools discovered by scanner
        """
        logger.info(f"Updating database with {len(discovered_tools)} discovered tools")
        
        # Clear existing tool mappings
        self.tool_mappings.clear()
        
        # Process each discovered tool
        for tool in discovered_tools:
            self._process_discovered_tool(tool)
        
        # Update language support levels based on available tools
        self._update_support_levels()
        
        logger.info(f"Database updated with {len(self.tool_mappings)} tool mappings")
    
    def _process_discovered_tool(self, tool: LanguageTool):
        """
        Process a single discovered tool and update mappings.
        
        Args:
            tool: Discovered language tool
        """
        # Create tool mapping
        tool_mapping = ToolMapping(
            tool_name=tool.name,
            tool_path=tool.path,
            supported_languages=[tool.language],
            command_template=self._get_command_template(tool.language),
            executable=tool.executable
        )
        
        self.tool_mappings[tool.name] = tool_mapping
        
        # Update language info with primary tool
        if tool.language in self.languages:
            lang_info = self.languages[tool.language]
            if tool.executable and not lang_info.primary_tool:
                lang_info.primary_tool = tool.name
        
        logger.debug(f"Processed tool: {tool.name} -> {tool.language}")
    
    def _get_command_template(self, language: str) -> str:
        """
        Get command template for a language.
        
        Args:
            language: Programming language name
            
        Returns:
            Command template string
        """
        if language in self.languages:
            return self.languages[language].command_template or ""
        return ""
    
    def _update_support_levels(self):
        """Update support levels for all languages based on available tools."""
        for lang_name, lang_info in self.languages.items():
            if lang_info.primary_tool and lang_info.primary_tool in self.tool_mappings:
                tool_mapping = self.tool_mappings[lang_info.primary_tool]
                if tool_mapping.executable:
                    lang_info.support_level = SupportLevel.FULLY_SUPPORTED
                else:
                    lang_info.support_level = SupportLevel.PARTIALLY_SUPPORTED
            else:
                lang_info.support_level = SupportLevel.NOT_SUPPORTED
    
    def get_supported_languages(self) -> List[str]:
        """
        Get list of fully supported languages.
        
        Returns:
            List of language names with full support
        """
        return [
            name for name, info in self.languages.items()
            if info.support_level == SupportLevel.FULLY_SUPPORTED
        ]
    
    def get_language_info(self, language: str) -> Optional[LanguageInfo]:
        """
        Get information about a specific language.
        
        Args:
            language: Programming language name
            
        Returns:
            LanguageInfo object or None if not found
        """
        return self.languages.get(language)
    
    def get_tool_mapping(self, tool_name: str) -> Optional[ToolMapping]:
        """
        Get tool mapping for a specific tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            ToolMapping object or None if not found
        """
        return self.tool_mappings.get(tool_name)
    
    def get_command_for_language(self, language: str, input_path: str, output_dir: str) -> Optional[str]:
        """
        Generate command string for a specific language.
        
        Args:
            language: Programming language name
            input_path: Path to input source file
            output_dir: Output directory for CPG
            
        Returns:
            Complete command string or None if language not supported
        """
        lang_info = self.get_language_info(language)
        if not lang_info or not lang_info.primary_tool:
            return None
        
        tool_mapping = self.get_tool_mapping(lang_info.primary_tool)
        if not tool_mapping or not tool_mapping.executable:
            return None
        
        # Format command template
        command = tool_mapping.command_template.format(
            memory_args=tool_mapping.memory_args,
            input_path=input_path,
            output_dir=output_dir
        )
        
        return command
    
    def get_file_extensions_for_language(self, language: str) -> List[str]:
        """
        Get file extensions for a specific language.
        
        Args:
            language: Programming language name
            
        Returns:
            List of file extensions
        """
        lang_info = self.get_language_info(language)
        return lang_info.file_extensions if lang_info else []
    
    def get_alternative_tools(self, language: str) -> List[str]:
        """
        Get alternative tools for a specific language.
        
        Args:
            language: Programming language name
            
        Returns:
            List of alternative tool names
        """
        lang_info = self.get_language_info(language)
        return lang_info.alternative_tools if lang_info else []
    
    def export_to_json(self, file_path: str):
        """
        Export database to JSON file.
        
        Args:
            file_path: Path to output JSON file
        """
        export_data = {
            'languages': {
                name: asdict(info) for name, info in self.languages.items()
            },
            'tool_mappings': {
                name: asdict(mapping) for name, mapping in self.tool_mappings.items()
            }
        }
        
        # Convert enums to strings for JSON serialization
        for lang_data in export_data['languages'].values():
            if 'support_level' in lang_data:
                lang_data['support_level'] = lang_data['support_level'].value
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Database exported to: {file_path}")
    
    def get_database_summary(self) -> Dict[str, any]:
        """
        Get summary of the language support database.
        
        Returns:
            Dictionary containing database summary
        """
        support_counts = {}
        for level in SupportLevel:
            support_counts[level.value] = sum(
                1 for info in self.languages.values()
                if info.support_level == level
            )
        
        return {
            'total_languages': len(self.languages),
            'total_tools': len(self.tool_mappings),
            'support_levels': support_counts,
            'supported_languages': self.get_supported_languages(),
            'executable_tools': [
                name for name, mapping in self.tool_mappings.items()
                if mapping.executable
            ]
        }