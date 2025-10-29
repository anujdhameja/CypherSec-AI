#!/usr/bin/env python3
"""
Test script for language discovery functionality.
"""

import sys
import logging
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from joern_verification.discovery import LanguageDiscoveryManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Test the language discovery functionality."""
    print("Testing Joern Language Discovery")
    print("=" * 40)
    
    # Initialize discovery manager
    joern_cli_path = "joern/joern-cli"
    
    try:
        manager = LanguageDiscoveryManager(joern_cli_path)
        
        # Perform discovery
        print("Starting language discovery...")
        results = manager.discover_languages()
        
        # Print summary
        print("\nDiscovery Summary:")
        print(manager.get_discovery_summary())
        
        # Print supported languages
        supported = manager.get_supported_languages()
        print(f"\nSupported Languages ({len(supported)}):")
        for lang in supported:
            print(f"  - {lang}")
        
        # Test command generation for a few languages
        print("\nSample Commands:")
        test_languages = ['C', 'Python', 'Java', 'JavaScript']
        for lang in test_languages:
            command = manager.get_command_for_language(
                lang, f"test.{lang.lower()}", "output"
            )
            if command:
                print(f"  {lang}: {command}")
            else:
                print(f"  {lang}: Not supported")
        
        # Export results
        output_file = "language_discovery_results.json"
        manager.export_results(output_file)
        print(f"\nResults exported to: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"Error during discovery: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)