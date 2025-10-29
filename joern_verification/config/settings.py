"""
Configuration management for the Joern multi-language verification system.

This module handles configuration for supported languages, tool paths,
and system settings.
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any


@dataclass
class LanguageConfig:
    """Configuration for a specific programming language."""
    name: str
    file_extension: str
    tool_name: str
    command_template: str
    memory_allocation: str = "-J-Xmx4g"
    timeout_seconds: int = 300
    alternative_tools: List[str] = None
    
    def __post_init__(self):
        if self.alternative_tools is None:
            self.alternative_tools = []


@dataclass
class SystemConfig:
    """System-wide configuration settings."""
    joern_path: str
    output_base_dir: str
    temp_dir: str
    max_concurrent_tests: int = 1
    cleanup_temp_files: bool = True
    verbose_logging: bool = False


class ConfigurationManager:
    """Manages configuration for the verification system."""
    
    # Default language configurations based on discovered Joern tools
    DEFAULT_LANGUAGES = {
        "c": LanguageConfig(
            name="C",
            file_extension=".c",
            tool_name="c2cpg.bat",
            command_template="{tool_path} {memory_allocation} {input_file} --output {output_dir}"
        ),
        "cpp": LanguageConfig(
            name="C++",
            file_extension=".cpp",
            tool_name="c2cpg.bat",
            command_template="{tool_path} {memory_allocation} {input_file} --output {output_dir}"
        ),
        "csharp": LanguageConfig(
            name="C#",
            file_extension=".cs",
            tool_name="csharpsrc2cpg.bat",
            command_template="{tool_path} {memory_allocation} {input_file} --output {output_dir}"
        ),
        "java": LanguageConfig(
            name="Java",
            file_extension=".java",
            tool_name="javasrc2cpg.bat",
            command_template="{tool_path} {memory_allocation} {input_file} --output {output_dir}"
        ),
        "javascript": LanguageConfig(
            name="JavaScript",
            file_extension=".js",
            tool_name="jssrc2cpg.bat",
            command_template="{tool_path} {memory_allocation} {input_file} --output {output_dir}"
        ),
        "kotlin": LanguageConfig(
            name="Kotlin",
            file_extension=".kt",
            tool_name="kotlin2cpg.bat",
            command_template="{tool_path} {memory_allocation} {input_file} --output {output_dir}"
        ),
        "php": LanguageConfig(
            name="PHP",
            file_extension=".php",
            tool_name="php2cpg.bat",
            command_template="{tool_path} {memory_allocation} {input_file} --output {output_dir}"
        ),
        "python": LanguageConfig(
            name="Python",
            file_extension=".py",
            tool_name="pysrc2cpg.bat",
            command_template="{tool_path} {memory_allocation} {input_file} --output {output_dir}",
            alternative_tools=["tree-sitter", "ast"]
        ),
        "ruby": LanguageConfig(
            name="Ruby",
            file_extension=".rb",
            tool_name="rubysrc2cpg.bat",
            command_template="{tool_path} {memory_allocation} {input_file} --output {output_dir}"
        ),
        "swift": LanguageConfig(
            name="Swift",
            file_extension=".swift",
            tool_name="swiftsrc2cpg.bat",
            command_template="{tool_path} {memory_allocation} {input_file} --output {output_dir}"
        ),
        "go": LanguageConfig(
            name="Go",
            file_extension=".go",
            tool_name="gosrc2cpg.bat",
            command_template="{tool_path} {memory_allocation} {input_file} --output {output_dir}"
        )
    }
    
    def __init__(self, config_file: Optional[Path] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Optional path to configuration file
        """
        self.config_file = config_file or Path("joern_verification_config.json")
        self.languages: Dict[str, LanguageConfig] = {}
        self.system_config: Optional[SystemConfig] = None
        self._load_configuration()
    
    def _load_configuration(self):
        """Load configuration from file or use defaults."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                self._parse_configuration(config_data)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error loading configuration: {e}")
                self._use_default_configuration()
        else:
            self._use_default_configuration()
    
    def _parse_configuration(self, config_data: Dict[str, Any]):
        """Parse configuration data from JSON."""
        # Load language configurations
        languages_data = config_data.get("languages", {})
        for lang_key, lang_data in languages_data.items():
            self.languages[lang_key] = LanguageConfig(**lang_data)
        
        # Load system configuration
        system_data = config_data.get("system", {})
        if system_data:
            self.system_config = SystemConfig(**system_data)
        
        # Fill in missing languages with defaults
        for lang_key, default_config in self.DEFAULT_LANGUAGES.items():
            if lang_key not in self.languages:
                self.languages[lang_key] = default_config
    
    def _use_default_configuration(self):
        """Use default configuration settings."""
        self.languages = self.DEFAULT_LANGUAGES.copy()
        self.system_config = SystemConfig(
            joern_path="joern/joern-cli",
            output_base_dir="verification_output",
            temp_dir="temp_test_files",
            max_concurrent_tests=1,
            cleanup_temp_files=True,
            verbose_logging=False
        )
    
    def get_language_config(self, language: str) -> Optional[LanguageConfig]:
        """
        Get configuration for a specific language.
        
        Args:
            language: Language identifier
            
        Returns:
            LanguageConfig object or None if not found
        """
        return self.languages.get(language.lower())
    
    def get_all_languages(self) -> List[str]:
        """
        Get list of all configured languages.
        
        Returns:
            List of language identifiers
        """
        return list(self.languages.keys())
    
    def get_system_config(self) -> SystemConfig:
        """
        Get system configuration.
        
        Returns:
            SystemConfig object
        """
        return self.system_config
    
    def update_language_config(self, language: str, config: LanguageConfig):
        """
        Update configuration for a specific language.
        
        Args:
            language: Language identifier
            config: New LanguageConfig object
        """
        self.languages[language.lower()] = config
    
    def update_system_config(self, config: SystemConfig):
        """
        Update system configuration.
        
        Args:
            config: New SystemConfig object
        """
        self.system_config = config
    
    def save_configuration(self):
        """Save current configuration to file."""
        config_data = {
            "languages": {
                lang: asdict(config) for lang, config in self.languages.items()
            },
            "system": asdict(self.system_config) if self.system_config else {}
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    def get_joern_tool_path(self, language: str) -> Optional[Path]:
        """
        Get the full path to a Joern tool for a specific language.
        
        Args:
            language: Language identifier
            
        Returns:
            Path to the tool or None if not configured
        """
        lang_config = self.get_language_config(language)
        if not lang_config or not self.system_config:
            return None
        
        joern_cli_path = Path(self.system_config.joern_path)
        tool_path = joern_cli_path / lang_config.tool_name
        
        return tool_path if tool_path.exists() else None
    
    def validate_configuration(self) -> List[str]:
        """
        Validate the current configuration and return any issues found.
        
        Returns:
            List of validation error messages
        """
        errors = []
        
        if not self.system_config:
            errors.append("System configuration is missing")
            return errors
        
        # Check if Joern path exists
        joern_path = Path(self.system_config.joern_path)
        if not joern_path.exists():
            errors.append(f"Joern path does not exist: {joern_path}")
        
        # Check if output directories can be created
        output_dir = Path(self.system_config.output_base_dir)
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create output directory: {e}")
        
        temp_dir = Path(self.system_config.temp_dir)
        try:
            temp_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create temp directory: {e}")
        
        # Validate language configurations
        for lang, config in self.languages.items():
            tool_path = self.get_joern_tool_path(lang)
            if tool_path and not tool_path.exists():
                errors.append(f"Tool for {lang} not found: {tool_path}")
        
        return errors
    
    def override_joern_path(self, path: str):
        """Override the Joern CLI path."""
        if self.system_config:
            self.system_config.joern_path = path
    
    def override_output_dir(self, path: str):
        """Override the output directory."""
        if self.system_config:
            self.system_config.output_base_dir = path
    
    def override_temp_dir(self, path: str):
        """Override the temporary directory."""
        if self.system_config:
            self.system_config.temp_dir = path
    
    def export_config(self, output_path: Path) -> bool:
        """
        Export current configuration to a file.
        
        Args:
            output_path: Path to export configuration to
            
        Returns:
            True if export was successful, False otherwise
        """
        try:
            config_data = {
                "languages": {
                    lang: asdict(config) for lang, config in self.languages.items()
                },
                "system": asdict(self.system_config) if self.system_config else {}
            }
            
            with open(output_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Failed to export configuration: {e}")
            return False