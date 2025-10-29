#!/usr/bin/env python3
"""
Configuration Generator for Joern Multi-Language Verification System

This script helps generate customized configuration files based on user requirements
and system capabilities.
"""

import json
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional


class ConfigurationGenerator:
    """Generates configuration files for different use cases."""
    
    def __init__(self):
        self.base_config = {
            "system": {
                "joern_path": "joern-cli",
                "output_base_dir": "verification_output",
                "temp_dir": "temp_test_files",
                "max_concurrent_tests": 1,
                "cleanup_temp_files": True,
                "verbose_logging": False
            },
            "languages": {}
        }
        
        self.language_templates = {
            "python": {
                "name": "Python",
                "file_extension": ".py",
                "tool_name": "pysrc2cpg.bat",
                "command_template": "{tool_path} {memory_allocation} {input_file} --output {output_dir}",
                "memory_allocation": "-J-Xmx4g",
                "timeout_seconds": 300,
                "alternative_tools": ["tree-sitter", "ast", "libcst"]
            },
            "java": {
                "name": "Java",
                "file_extension": ".java",
                "tool_name": "javasrc2cpg.bat",
                "command_template": "{tool_path} {memory_allocation} {input_file} --output {output_dir}",
                "memory_allocation": "-J-Xmx4g",
                "timeout_seconds": 300,
                "alternative_tools": ["eclipse-jdt", "spoon", "javaparser"]
            },
            "c": {
                "name": "C",
                "file_extension": ".c",
                "tool_name": "c2cpg.bat",
                "command_template": "{tool_path} {memory_allocation} {input_file} --output {output_dir}",
                "memory_allocation": "-J-Xmx4g",
                "timeout_seconds": 300,
                "alternative_tools": ["clang", "gcc", "tree-sitter"]
            },
            "cpp": {
                "name": "C++",
                "file_extension": ".cpp",
                "tool_name": "c2cpg.bat",
                "command_template": "{tool_path} {memory_allocation} {input_file} --output {output_dir}",
                "memory_allocation": "-J-Xmx4g",
                "timeout_seconds": 300,
                "alternative_tools": ["clang++", "g++", "tree-sitter"]
            },
            "csharp": {
                "name": "C#",
                "file_extension": ".cs",
                "tool_name": "csharpsrc2cpg.bat",
                "command_template": "{tool_path} {memory_allocation} {input_file} --output {output_dir}",
                "memory_allocation": "-J-Xmx4g",
                "timeout_seconds": 300,
                "alternative_tools": ["roslyn", "mono"]
            },
            "javascript": {
                "name": "JavaScript",
                "file_extension": ".js",
                "tool_name": "jssrc2cpg.bat",
                "command_template": "{tool_path} {memory_allocation} {input_file} --output {output_dir}",
                "memory_allocation": "-J-Xmx4g",
                "timeout_seconds": 300,
                "alternative_tools": ["babel", "acorn", "tree-sitter"]
            },
            "kotlin": {
                "name": "Kotlin",
                "file_extension": ".kt",
                "tool_name": "kotlin2cpg.bat",
                "command_template": "{tool_path} {memory_allocation} {input_file} --output {output_dir}",
                "memory_allocation": "-J-Xmx4g",
                "timeout_seconds": 300,
                "alternative_tools": ["kotlinc"]
            },
            "php": {
                "name": "PHP",
                "file_extension": ".php",
                "tool_name": "php2cpg.bat",
                "command_template": "{tool_path} {memory_allocation} {input_file} --output {output_dir}",
                "memory_allocation": "-J-Xmx4g",
                "timeout_seconds": 300,
                "alternative_tools": ["php-parser", "tree-sitter"]
            },
            "ruby": {
                "name": "Ruby",
                "file_extension": ".rb",
                "tool_name": "rubysrc2cpg.bat",
                "command_template": "{tool_path} {memory_allocation} {input_file} --output {output_dir}",
                "memory_allocation": "-J-Xmx4g",
                "timeout_seconds": 300,
                "alternative_tools": ["parser", "tree-sitter"]
            },
            "swift": {
                "name": "Swift",
                "file_extension": ".swift",
                "tool_name": "swiftsrc2cpg.bat",
                "command_template": "{tool_path} {memory_allocation} {input_file} --output {output_dir}",
                "memory_allocation": "-J-Xmx4g",
                "timeout_seconds": 300,
                "alternative_tools": ["swift-syntax"]
            },
            "go": {
                "name": "Go",
                "file_extension": ".go",
                "tool_name": "gosrc2cpg.bat",
                "command_template": "{tool_path} {memory_allocation} {input_file} --output {output_dir}",
                "memory_allocation": "-J-Xmx4g",
                "timeout_seconds": 300,
                "alternative_tools": ["go-ast", "tree-sitter"]
            }
        }
    
    def generate_basic_config(self, languages: List[str], joern_path: str = "joern-cli") -> Dict[str, Any]:
        """Generate a basic configuration with specified languages."""
        config = self.base_config.copy()
        config["system"]["joern_path"] = joern_path
        
        for lang in languages:
            if lang in self.language_templates:
                config["languages"][lang] = self.language_templates[lang].copy()
        
        return config
    
    def generate_development_config(self, languages: List[str], joern_path: str = "joern-cli") -> Dict[str, Any]:
        """Generate a development-optimized configuration."""
        config = self.generate_basic_config(languages, joern_path)
        
        # Development-specific settings
        config["system"].update({
            "output_base_dir": "dev_verification_output",
            "temp_dir": "dev_temp_files",
            "cleanup_temp_files": False,
            "verbose_logging": True
        })
        
        # Reduce timeouts and memory for faster development cycles
        for lang_config in config["languages"].values():
            lang_config["timeout_seconds"] = 120
            lang_config["memory_allocation"] = "-J-Xmx2g"
        
        return config
    
    def generate_production_config(self, languages: List[str], joern_path: str = "/opt/joern/joern-cli") -> Dict[str, Any]:
        """Generate a production-optimized configuration."""
        config = self.generate_basic_config(languages, joern_path)
        
        # Production-specific settings
        config["system"].update({
            "joern_path": joern_path,
            "output_base_dir": "/var/log/joern_verification",
            "temp_dir": "/tmp/joern_verification_temp",
            "max_concurrent_tests": 4,
            "cleanup_temp_files": True,
            "verbose_logging": False
        })
        
        # Increase timeouts and memory for production workloads
        for lang_config in config["languages"].values():
            lang_config["timeout_seconds"] = 600
            lang_config["memory_allocation"] = "-J-Xmx8g"
        
        return config
    
    def generate_ci_config(self, languages: List[str], joern_path: str = "joern-cli") -> Dict[str, Any]:
        """Generate a CI/CD-optimized configuration."""
        config = self.generate_basic_config(languages, joern_path)
        
        # CI-specific settings
        config["system"].update({
            "output_base_dir": "ci_verification_output",
            "temp_dir": "ci_temp_files",
            "max_concurrent_tests": 2,
            "cleanup_temp_files": True,
            "verbose_logging": False
        })
        
        # Balanced timeouts and memory for CI environments
        for lang_config in config["languages"].values():
            lang_config["timeout_seconds"] = 180
            lang_config["memory_allocation"] = "-J-Xmx4g"
        
        return config
    
    def generate_minimal_config(self, language: str = "python", joern_path: str = "joern-cli") -> Dict[str, Any]:
        """Generate a minimal configuration for testing."""
        config = {
            "system": {
                "joern_path": joern_path,
                "output_base_dir": "output",
                "temp_dir": "temp",
                "max_concurrent_tests": 1,
                "cleanup_temp_files": True,
                "verbose_logging": False
            },
            "languages": {}
        }
        
        if language in self.language_templates:
            lang_config = self.language_templates[language].copy()
            lang_config["timeout_seconds"] = 60
            lang_config["memory_allocation"] = "-J-Xmx2g"
            config["languages"][language] = lang_config
        
        return config
    
    def customize_config(self, config: Dict[str, Any], customizations: Dict[str, Any]) -> Dict[str, Any]:
        """Apply custom settings to a configuration."""
        def deep_update(base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> Dict[str, Any]:
            for key, value in update_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
            return base_dict
        
        return deep_update(config.copy(), customizations)
    
    def detect_available_languages(self, joern_path: str) -> List[str]:
        """Detect available languages based on Joern installation."""
        available_languages = []
        joern_dir = Path(joern_path)
        
        if not joern_dir.exists():
            print(f"Warning: Joern directory not found at {joern_path}")
            return list(self.language_templates.keys())  # Return all as fallback
        
        for lang, template in self.language_templates.items():
            tool_path = joern_dir / template["tool_name"]
            if tool_path.exists():
                available_languages.append(lang)
        
        return available_languages
    
    def save_config(self, config: Dict[str, Any], output_path: str, add_comments: bool = True) -> None:
        """Save configuration to file with optional comments."""
        if add_comments:
            # Add helpful comments to the configuration
            config_with_comments = {
                "_description": "Joern Multi-Language Verification System Configuration",
                "_generated_by": "Configuration Generator Script",
                "_usage": "Copy this file to joern_verification_config.json and customize as needed",
                **config
            }
        else:
            config_with_comments = config
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config_with_comments, f, indent=2, ensure_ascii=False)
        
        print(f"Configuration saved to: {output_path}")


def interactive_mode() -> Dict[str, Any]:
    """Interactive configuration generation."""
    generator = ConfigurationGenerator()
    
    print("=== Joern Multi-Language Verification Configuration Generator ===\n")
    
    # Get configuration type
    print("Configuration Types:")
    print("1. Basic - Standard configuration for general use")
    print("2. Development - Optimized for development workflow")
    print("3. Production - High-performance production settings")
    print("4. CI/CD - Streamlined for continuous integration")
    print("5. Minimal - Bare minimum for testing")
    print("6. Custom - Start with basic and customize")
    
    while True:
        try:
            config_type = int(input("\nSelect configuration type (1-6): "))
            if 1 <= config_type <= 6:
                break
            else:
                print("Please enter a number between 1 and 6.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Get Joern path
    default_joern_path = "joern-cli"
    joern_path = input(f"\nJoern CLI path [{default_joern_path}]: ").strip()
    if not joern_path:
        joern_path = default_joern_path
    
    # Detect available languages
    available_languages = generator.detect_available_languages(joern_path)
    if available_languages:
        print(f"\nDetected languages: {', '.join(available_languages)}")
    else:
        available_languages = list(generator.language_templates.keys())
        print(f"\nAvailable languages: {', '.join(available_languages)}")
    
    # Get languages to include
    if config_type == 5:  # Minimal
        print("\nFor minimal configuration, select one language:")
        for i, lang in enumerate(available_languages, 1):
            print(f"{i}. {lang}")
        
        while True:
            try:
                lang_choice = int(input("Select language: ")) - 1
                if 0 <= lang_choice < len(available_languages):
                    selected_languages = [available_languages[lang_choice]]
                    break
                else:
                    print(f"Please enter a number between 1 and {len(available_languages)}.")
            except ValueError:
                print("Please enter a valid number.")
    else:
        print(f"\nSelect languages (comma-separated numbers or 'all' for all languages):")
        for i, lang in enumerate(available_languages, 1):
            print(f"{i}. {lang}")
        
        lang_input = input("Languages: ").strip()
        if lang_input.lower() == 'all':
            selected_languages = available_languages
        else:
            try:
                lang_indices = [int(x.strip()) - 1 for x in lang_input.split(',')]
                selected_languages = [available_languages[i] for i in lang_indices if 0 <= i < len(available_languages)]
            except (ValueError, IndexError):
                print("Invalid selection, using all languages.")
                selected_languages = available_languages
    
    # Generate configuration
    config_generators = {
        1: generator.generate_basic_config,
        2: generator.generate_development_config,
        3: generator.generate_production_config,
        4: generator.generate_ci_config,
        5: generator.generate_minimal_config,
        6: generator.generate_basic_config
    }
    
    if config_type == 5:  # Minimal
        config = generator.generate_minimal_config(selected_languages[0], joern_path)
    else:
        config = config_generators[config_type](selected_languages, joern_path)
    
    # Custom modifications for type 6
    if config_type == 6:
        print("\nCustomization options:")
        print("1. Memory allocation (current: 4g)")
        print("2. Timeout seconds (current: 300)")
        print("3. Concurrent tests (current: 1)")
        print("4. Verbose logging (current: False)")
        
        memory = input("Memory allocation [4g]: ").strip()
        if memory:
            for lang_config in config["languages"].values():
                lang_config["memory_allocation"] = f"-J-Xmx{memory}"
        
        timeout = input("Timeout seconds [300]: ").strip()
        if timeout:
            try:
                timeout_int = int(timeout)
                for lang_config in config["languages"].values():
                    lang_config["timeout_seconds"] = timeout_int
            except ValueError:
                print("Invalid timeout, keeping default.")
        
        concurrent = input("Max concurrent tests [1]: ").strip()
        if concurrent:
            try:
                concurrent_int = int(concurrent)
                config["system"]["max_concurrent_tests"] = concurrent_int
            except ValueError:
                print("Invalid concurrent value, keeping default.")
        
        verbose = input("Verbose logging (y/N): ").strip().lower()
        if verbose in ['y', 'yes']:
            config["system"]["verbose_logging"] = True
    
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Generate configuration files for Joern Multi-Language Verification System"
    )
    
    parser.add_argument(
        "--type", "-t",
        choices=["basic", "development", "production", "ci", "minimal"],
        help="Configuration type"
    )
    
    parser.add_argument(
        "--languages", "-l",
        nargs="+",
        help="Languages to include in configuration"
    )
    
    parser.add_argument(
        "--joern-path", "-j",
        default="joern-cli",
        help="Path to Joern CLI installation"
    )
    
    parser.add_argument(
        "--output", "-o",
        default="generated_config.json",
        help="Output file path"
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    
    parser.add_argument(
        "--detect-languages",
        action="store_true",
        help="Detect available languages and exit"
    )
    
    parser.add_argument(
        "--no-comments",
        action="store_true",
        help="Don't add comments to generated configuration"
    )
    
    args = parser.parse_args()
    
    generator = ConfigurationGenerator()
    
    # Detect languages mode
    if args.detect_languages:
        available = generator.detect_available_languages(args.joern_path)
        print("Available languages:")
        for lang in available:
            print(f"  - {lang}")
        return
    
    # Interactive mode
    if args.interactive or (not args.type and not args.languages):
        config = interactive_mode()
        generator.save_config(config, args.output, not args.no_comments)
        return
    
    # Command-line mode
    if not args.type:
        print("Error: --type is required in non-interactive mode")
        sys.exit(1)
    
    if not args.languages:
        # Use all available languages
        args.languages = generator.detect_available_languages(args.joern_path)
        if not args.languages:
            args.languages = list(generator.language_templates.keys())
    
    # Generate configuration based on type
    config_generators = {
        "basic": generator.generate_basic_config,
        "development": generator.generate_development_config,
        "production": generator.generate_production_config,
        "ci": generator.generate_ci_config,
        "minimal": lambda langs, path: generator.generate_minimal_config(langs[0] if langs else "python", path)
    }
    
    config = config_generators[args.type](args.languages, args.joern_path)
    generator.save_config(config, args.output, not args.no_comments)
    
    print(f"\nGenerated {args.type} configuration with languages: {', '.join(args.languages)}")


if __name__ == "__main__":
    main()