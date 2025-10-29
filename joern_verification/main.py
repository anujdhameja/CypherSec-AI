"""
Main entry point for the Joern multi-language verification system.

This module provides the primary interface for running language verification
tests and generating comprehensive reports.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

from joern_verification.config.settings import ConfigurationManager
from joern_verification.core.exceptions import JoernVerificationError


class ProgressTracker:
    """Simple progress tracking for verification workflow."""
    
    def __init__(self, total_steps: int = 7):
        """Initialize progress tracker with total steps."""
        self.total_steps = total_steps
        self.current_step = 0
        self.step_names = [
            "Language Discovery",
            "Test File Generation", 
            "CPG Generation",
            "Results Analysis",
            "Alternative Identification",
            "Report Generation",
            "Cleanup"
        ]
    
    def update(self, message: str):
        """Update progress with a message."""
        print(f"[{self.current_step}/{self.total_steps}] {message}")
    
    def next_step(self, step_name: Optional[str] = None):
        """Move to next step."""
        self.current_step += 1
        if step_name:
            print(f"\n[{self.current_step}/{self.total_steps}] {step_name}")
        elif self.current_step <= len(self.step_names):
            print(f"\n[{self.current_step}/{self.total_steps}] {self.step_names[self.current_step-1]}")
    
    def complete(self):
        """Mark progress as complete."""
        print(f"\n[{self.total_steps}/{self.total_steps}] Verification Complete!")


class VerificationStatus:
    """Track status of verification process."""
    
    def __init__(self):
        """Initialize verification status."""
        self.start_time = None
        self.end_time = None
        self.languages_tested = []
        self.languages_successful = []
        self.languages_failed = []
        self.errors = []
        self.warnings = []
    
    def start(self):
        """Mark verification as started."""
        from datetime import datetime
        self.start_time = datetime.now()
    
    def finish(self):
        """Mark verification as finished."""
        from datetime import datetime
        self.end_time = datetime.now()
    
    def add_language_result(self, language: str, success: bool, category: str):
        """Add result for a language."""
        self.languages_tested.append(language)
        if success:
            self.languages_successful.append(language)
        else:
            self.languages_failed.append(language)
    
    def add_error(self, error: str):
        """Add an error message."""
        self.errors.append(error)
    
    def add_warning(self, warning: str):
        """Add a warning message."""
        self.warnings.append(warning)
    
    def get_duration(self) -> float:
        """Get verification duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    def get_summary(self) -> str:
        """Get status summary."""
        total = len(self.languages_tested)
        successful = len(self.languages_successful)
        failed = len(self.languages_failed)
        duration = self.get_duration()
        
        return f"""
Verification Status Summary:
- Duration: {duration:.1f} seconds
- Languages Tested: {total}
- Successful: {successful}
- Failed: {failed}
- Success Rate: {(successful/total*100) if total > 0 else 0:.1f}%
- Errors: {len(self.errors)}
- Warnings: {len(self.warnings)}
"""


class JoernVerificationSystem:
    """Main orchestrator for the verification system."""
    
    def __init__(self, config_file: Optional[Path] = None):
        """
        Initialize the verification system.
        
        Args:
            config_file: Optional path to configuration file
        """
        self.config_manager = ConfigurationManager(config_file)
        self.progress_tracker = None
        self.status = VerificationStatus()
        self.setup_directories()
    
    def _map_target_languages(self, target_languages: List[str], supported_languages: List[str]) -> List[str]:
        """
        Map target language names to discovered language names.
        
        Args:
            target_languages: List of requested language names (lowercase)
            supported_languages: List of discovered language names (title case)
            
        Returns:
            List of mapped target languages that are supported
        """
        # Create mapping between config names (lowercase) and discovered names (title case)
        language_mapping = {
            'c': 'C',
            'cpp': 'C++', 
            'c++': 'C++',
            'csharp': 'C#',
            'c#': 'C#',
            'java': 'Java',
            'javascript': 'JavaScript',
            'js': 'JavaScript',
            'kotlin': 'Kotlin',
            'php': 'PHP',
            'python': 'Python',
            'py': 'Python',
            'ruby': 'Ruby',
            'rb': 'Ruby',
            'swift': 'Swift',
            'go': 'Go'
        }
        
        # Map target languages to discovered language names
        mapped_targets = []
        for lang in target_languages:
            mapped_lang = language_mapping.get(lang.lower())
            if mapped_lang and mapped_lang in supported_languages:
                mapped_targets.append(lang)  # Keep original name for test generation
        
        return mapped_targets
    
    def setup_directories(self):
        """Create necessary directories for the verification system."""
        system_config = self.config_manager.get_system_config()
        
        # Create output directory
        output_dir = Path(system_config.output_base_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create temp directory
        temp_dir = Path(system_config.temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
    
    def validate_setup(self) -> bool:
        """
        Validate that the system is properly configured.
        
        Returns:
            True if setup is valid, False otherwise
        """
        errors = self.config_manager.validate_configuration()
        
        if errors:
            print("Configuration validation errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        print("Configuration validation passed.")
        return True
    
    def list_supported_languages(self) -> List[str]:
        """
        Get list of all configured languages.
        
        Returns:
            List of supported language names
        """
        return self.config_manager.get_all_languages()
    
    def run_verification(self, languages: Optional[List[str]] = None, progress_callback=None) -> bool:
        """
        Run verification tests for specified languages.
        
        Args:
            languages: Optional list of languages to test. If None, test all.
            progress_callback: Optional callback function for progress updates
            
        Returns:
            True if verification completed successfully, False otherwise
        """
        if not self.validate_setup():
            return False
        
        target_languages = languages or self.list_supported_languages()
        
        # Initialize progress tracking
        self.progress_tracker = ProgressTracker()
        self.status.start()
        
        print(f"Starting verification for languages: {', '.join(target_languages)}")
        print("="*60)
        
        try:
            # Import required components
            from joern_verification.discovery.discovery import LanguageDiscoveryManager
            from joern_verification.generation.test_file_generator import TestFileGenerator
            from joern_verification.generation.cpg_generator import CPGGenerator
            from joern_verification.analysis.result_analyzer import ResultAnalyzer
            from joern_verification.reporting.report_generator import ReportGenerator
            from joern_verification.alternatives.recommender import AlternativeToolRecommender
            from pathlib import Path
            import tempfile
            import shutil
            
            # Get system configuration
            system_config = self.config_manager.get_system_config()
            
            # Step 1: Language Discovery
            self.progress_tracker.next_step("Language Discovery")
            if progress_callback:
                progress_callback("Starting language discovery...")
            
            discovery_manager = LanguageDiscoveryManager(system_config.joern_path)
            discovery_results = discovery_manager.discover_languages()
            
            supported_languages = discovery_manager.get_supported_languages()
            self.progress_tracker.update(f"Found {len(supported_languages)} supported languages: {', '.join(supported_languages)}")
            
            # Map target languages to supported ones
            available_targets = self._map_target_languages(target_languages, supported_languages)
            if not available_targets:
                error_msg = "No supported languages found in target list"
                self.status.add_error(error_msg)
                print(f"Error: {error_msg}")
                return False
            
            if len(available_targets) != len(target_languages):
                missing = set(target_languages) - set(available_targets)
                warning_msg = f"Some languages not supported: {', '.join(missing)}"
                self.status.add_warning(warning_msg)
                self.progress_tracker.update(f"Warning: {warning_msg}")
            
            # Step 2: Generate Test Files
            self.progress_tracker.next_step("Test File Generation")
            if progress_callback:
                progress_callback("Generating test files...")
            
            temp_dir = Path(tempfile.mkdtemp(prefix="joern_verification_"))
            test_generator = TestFileGenerator(str(temp_dir))
            
            test_results = {}
            for language in available_targets:
                success, path_or_error = test_generator.generate_test_file(language)
                test_results[language] = (success, path_or_error)
                if success:
                    self.progress_tracker.update(f"✓ Generated test file for {language}")
                else:
                    error_msg = f"Failed to generate test file for {language}: {path_or_error}"
                    self.status.add_error(error_msg)
                    self.progress_tracker.update(f"✗ {error_msg}")
            
            # Step 3: CPG Generation
            self.progress_tracker.next_step("CPG Generation")
            if progress_callback:
                progress_callback("Running CPG generation...")
            
            cpg_generator = CPGGenerator(Path(system_config.joern_path))
            generation_results = []
            
            for i, language in enumerate(available_targets, 1):
                if not test_results[language][0]:
                    continue  # Skip if test file generation failed
                
                self.progress_tracker.update(f"Processing {language} ({i}/{len(available_targets)})...")
                test_file_path = Path(test_results[language][1])
                output_dir = temp_dir / f"{language}_output"
                
                # Generate CPG with timeout
                result = cpg_generator.generate_cpg(
                    language=language,
                    input_file=test_file_path,
                    output_dir=output_dir,
                    memory="4g",
                    timeout=300  # 5 minute timeout
                )
                
                generation_results.append(result)
                
                # Track result in status
                self.status.add_language_result(language, result.success, "unknown")
                
                # Report result
                if result.success:
                    self.progress_tracker.update(f"  ✓ CPG generation successful ({result.execution_time:.2f}s)")
                else:
                    error_msg = result.error_message or 'Unknown error'
                    self.status.add_error(f"{language}: {error_msg}")
                    self.progress_tracker.update(f"  ✗ CPG generation failed: {error_msg}")
                
                if progress_callback:
                    progress_callback(f"Completed {language}")
            
            # Step 4: Analyze Results
            self.progress_tracker.next_step("Results Analysis")
            if progress_callback:
                progress_callback("Analyzing results...")
            
            analyzer = ResultAnalyzer()
            analysis_reports = []
            
            for result in generation_results:
                analysis = analyzer.analyze_result(result)
                analysis_reports.append(analysis)
                
                # Update status with final category
                self.status.add_language_result(result.language, result.success, analysis.category)
                self.progress_tracker.update(f"{result.language}: {analysis.category}")
            
            # Step 5: Generate Alternatives for Failed Languages
            self.progress_tracker.next_step("Alternative Identification")
            if progress_callback:
                progress_callback("Identifying alternatives...")
            
            recommender = AlternativeToolRecommender()
            failed_languages = [r.language for r in generation_results if not r.success]
            
            for language in failed_languages:
                alternatives = recommender.get_alternatives_for_language(language)
                if alternatives:
                    self.progress_tracker.update(f"{language}: {len(alternatives)} alternatives available")
                else:
                    self.progress_tracker.update(f"{language}: No alternatives found")
            
            # Step 6: Generate Report
            self.progress_tracker.next_step("Report Generation")
            if progress_callback:
                progress_callback("Generating report...")
            
            report_generator = ReportGenerator()
            output_path = Path(system_config.output_base_dir) / "verification_report"
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            report_success = report_generator.generate_report(
                analyses=analysis_reports,
                output_path=output_path
            )
            
            if report_success:
                self.progress_tracker.update(f"✓ Report generated: {output_path}")
            else:
                error_msg = "Failed to generate report"
                self.status.add_error(error_msg)
                self.progress_tracker.update(f"✗ {error_msg}")
            
            # Step 7: Cleanup
            self.progress_tracker.next_step("Cleanup")
            try:
                test_generator.cleanup_test_files()
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                self.progress_tracker.update("✓ Cleanup completed")
            except Exception as e:
                warning_msg = f"Cleanup failed: {e}"
                self.status.add_warning(warning_msg)
                self.progress_tracker.update(f"Warning: {warning_msg}")
            
            # Complete progress tracking
            self.progress_tracker.complete()
            self.status.finish()
            
            # Summary
            print("\n" + "="*60)
            print("VERIFICATION SUMMARY")
            print("="*60)
            
            summary = report_generator.format_summary(analysis_reports)
            print(summary)
            
            # Print status summary
            print(self.status.get_summary())
            
            # Determine overall success
            successful_count = sum(1 for a in analysis_reports if a.category in ['success', 'success_with_warnings'])
            overall_success = successful_count > 0 and report_success
            
            if progress_callback:
                progress_callback("Verification completed")
            
            return overall_success
            
        except Exception as e:
            error_msg = f"Error during verification: {e}"
            self.status.add_error(error_msg)
            self.status.finish()
            print(error_msg)
            import traceback
            traceback.print_exc()
            return False
    
    def run_verification_with_options(self, options: Dict[str, Any]) -> bool:
        """
        Run verification with enhanced options from command line.
        
        Args:
            options: Dictionary of verification options
            
        Returns:
            True if verification completed successfully, False otherwise
        """
        # Extract options
        languages = options.get('languages')
        timeout = options.get('timeout', 300)
        memory = options.get('memory', '4g')
        dry_run = options.get('dry_run', False)
        parallel = options.get('parallel', False)
        report_formats = options.get('report_formats', ['json', 'markdown'])
        summary_only = options.get('summary_only', False)
        skip_cleanup = options.get('skip_cleanup', False)
        continue_on_error = options.get('continue_on_error', False)
        
        if not self.validate_setup():
            return False
        
        target_languages = languages or self.list_supported_languages()
        
        # Initialize progress tracking
        self.progress_tracker = ProgressTracker()
        self.status.start()
        
        print(f"Starting verification for languages: {', '.join(target_languages)}")
        if dry_run:
            print("DRY RUN MODE: No actual CPG generation will be performed")
        print("="*60)
        
        try:
            # Import required components
            from joern_verification.discovery.discovery import LanguageDiscoveryManager
            from joern_verification.generation.test_file_generator import TestFileGenerator
            from joern_verification.generation.cpg_generator import CPGGenerator
            from joern_verification.analysis.result_analyzer import ResultAnalyzer
            from joern_verification.reporting.report_generator import ReportGenerator
            from joern_verification.alternatives.recommender import AlternativeToolRecommender
            from pathlib import Path
            import tempfile
            import shutil
            
            # Get system configuration
            system_config = self.config_manager.get_system_config()
            
            # Step 1: Language Discovery
            self.progress_tracker.next_step("Language Discovery")
            
            discovery_manager = LanguageDiscoveryManager(system_config.joern_path)
            discovery_results = discovery_manager.discover_languages()
            
            supported_languages = discovery_manager.get_supported_languages()
            self.progress_tracker.update(f"Found {len(supported_languages)} supported languages: {', '.join(supported_languages)}")
            
            # Map target languages to supported ones
            available_targets = self._map_target_languages(target_languages, supported_languages)
            if not available_targets:
                error_msg = f"No supported languages found in target list. Available: {', '.join(supported_languages)}"
                self.status.add_error(error_msg)
                print(f"Error: {error_msg}")
                return False
            
            if len(available_targets) != len(target_languages):
                missing = set(target_languages) - set(available_targets)
                warning_msg = f"Some languages not supported: {', '.join(missing)}"
                self.status.add_warning(warning_msg)
                self.progress_tracker.update(f"Warning: {warning_msg}")
            
            # Step 2: Generate Test Files
            self.progress_tracker.next_step("Test File Generation")
            
            temp_dir = Path(tempfile.mkdtemp(prefix="joern_verification_"))
            test_generator = TestFileGenerator(str(temp_dir))
            
            test_results = {}
            for language in available_targets:
                success, path_or_error = test_generator.generate_test_file(language)
                test_results[language] = (success, path_or_error)
                if success:
                    self.progress_tracker.update(f"✓ Generated test file for {language}")
                else:
                    error_msg = f"Failed to generate test file for {language}: {path_or_error}"
                    self.status.add_error(error_msg)
                    self.progress_tracker.update(f"✗ {error_msg}")
                    if not continue_on_error:
                        return False
            
            # Step 3: CPG Generation (or simulation in dry run)
            self.progress_tracker.next_step("CPG Generation")
            
            generation_results = []
            
            if dry_run:
                # Simulate CPG generation for dry run
                self.progress_tracker.update("Simulating CPG generation (dry run mode)")
                for language in available_targets:
                    if not test_results[language][0]:
                        continue
                    
                    # Create mock result
                    from joern_verification.generation.cpg_generator import GenerationResult
                    mock_result = GenerationResult(
                        language=language,
                        input_file=Path(test_results[language][1]),
                        output_dir=temp_dir / f"{language}_output",
                        success=True,  # Assume success in dry run
                        execution_time=1.0,  # Mock execution time
                        memory_usage=None,
                        stdout="Dry run - no actual execution",
                        stderr="",
                        return_code=0,
                        output_files=[],
                        warnings=[]
                    )
                    generation_results.append(mock_result)
                    self.progress_tracker.update(f"✓ {language}: Simulated (dry run)")
            else:
                # Actual CPG generation
                cpg_generator = CPGGenerator(Path(system_config.joern_path))
                
                for i, language in enumerate(available_targets, 1):
                    if not test_results[language][0]:
                        continue
                    
                    self.progress_tracker.update(f"Processing {language} ({i}/{len(available_targets)})...")
                    test_file_path = Path(test_results[language][1])
                    output_dir = temp_dir / f"{language}_output"
                    
                    # Generate CPG with custom options
                    result = cpg_generator.generate_cpg(
                        language=language,
                        input_file=test_file_path,
                        output_dir=output_dir,
                        memory=memory,
                        timeout=timeout
                    )
                    
                    generation_results.append(result)
                    
                    # Track result in status
                    self.status.add_language_result(language, result.success, "unknown")
                    
                    # Report result
                    if result.success:
                        self.progress_tracker.update(f"  ✓ CPG generation successful ({result.execution_time:.2f}s)")
                    else:
                        error_msg = result.error_message or 'Unknown error'
                        self.status.add_error(f"{language}: {error_msg}")
                        self.progress_tracker.update(f"  ✗ CPG generation failed: {error_msg}")
                        if not continue_on_error:
                            return False
            
            # Step 4: Analyze Results
            self.progress_tracker.next_step("Results Analysis")
            
            analyzer = ResultAnalyzer()
            analysis_reports = []
            
            for result in generation_results:
                analysis = analyzer.analyze_result(result)
                analysis_reports.append(analysis)
                
                # Update status with final category
                self.status.add_language_result(result.language, result.success, analysis.category)
                self.progress_tracker.update(f"{result.language}: {analysis.category}")
            
            # Step 5: Generate Alternatives for Failed Languages
            self.progress_tracker.next_step("Alternative Identification")
            
            recommender = AlternativeToolRecommender()
            failed_languages = [r.language for r in generation_results if not r.success]
            
            for language in failed_languages:
                alternatives = recommender.get_alternatives_for_language(language)
                if alternatives:
                    self.progress_tracker.update(f"{language}: {len(alternatives)} alternatives available")
                else:
                    self.progress_tracker.update(f"{language}: No alternatives found")
            
            # Step 6: Generate Report (if not disabled)
            report_success = True
            if report_formats:
                self.progress_tracker.next_step("Report Generation")
                
                report_generator = ReportGenerator()
                output_path = Path(system_config.output_base_dir) / "verification_report"
                
                # Ensure output directory exists
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Configure report options
                from joern_verification.reporting.models import ReportConfiguration
                report_config = ReportConfiguration(
                    output_formats=report_formats,
                    include_detailed_results=not summary_only,
                    include_performance_analysis=not summary_only,
                    include_error_analysis=not summary_only
                )
                
                report_success = report_generator.generate_report(
                    analyses=analysis_reports,
                    output_path=output_path,
                    config=report_config
                )
                
                if report_success:
                    for fmt in report_formats:
                        format_path = output_path.with_suffix(f'.{fmt}')
                        self.progress_tracker.update(f"✓ {fmt.upper()} report: {format_path}")
                else:
                    error_msg = "Failed to generate report"
                    self.status.add_error(error_msg)
                    self.progress_tracker.update(f"✗ {error_msg}")
            
            # Step 7: Cleanup (if not skipped)
            if not skip_cleanup:
                self.progress_tracker.next_step("Cleanup")
                try:
                    test_generator.cleanup_test_files()
                    if temp_dir.exists():
                        shutil.rmtree(temp_dir)
                    self.progress_tracker.update("✓ Cleanup completed")
                except Exception as e:
                    warning_msg = f"Cleanup failed: {e}"
                    self.status.add_warning(warning_msg)
                    self.progress_tracker.update(f"Warning: {warning_msg}")
            else:
                self.progress_tracker.update(f"Skipping cleanup - files preserved in: {temp_dir}")
            
            # Complete progress tracking
            self.progress_tracker.complete()
            self.status.finish()
            
            # Summary
            print("\n" + "="*60)
            print("VERIFICATION SUMMARY")
            print("="*60)
            
            summary = report_generator.format_summary(analysis_reports)
            print(summary)
            
            # Print status summary
            print(self.status.get_summary())
            
            # Determine overall success
            successful_count = sum(1 for a in analysis_reports if a.category in ['success', 'success_with_warnings'])
            overall_success = successful_count > 0 and report_success
            
            return overall_success
            
        except Exception as e:
            error_msg = f"Error during verification: {e}"
            self.status.add_error(error_msg)
            self.status.finish()
            print(error_msg)
            import traceback
            traceback.print_exc()
            return False


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Joern Multi-Language Verification System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic verification
  python -m joern_verification.main
  
  # Validate setup only
  python -m joern_verification.main --validate
  
  # List supported languages
  python -m joern_verification.main --list-languages
  
  # Test specific languages
  python -m joern_verification.main --languages python java c
  
  # Use custom configuration
  python -m joern_verification.main --config custom_config.json
  
  # Generate specific report formats
  python -m joern_verification.main --report-format json markdown
  
  # Set timeout and memory limits
  python -m joern_verification.main --timeout 600 --memory 8g
  
  # Dry run (no actual CPG generation)
  python -m joern_verification.main --dry-run
        """
    )
    
    # Configuration options
    config_group = parser.add_argument_group('Configuration')
    config_group.add_argument(
        "--config",
        type=Path,
        help="Path to configuration file"
    )
    config_group.add_argument(
        "--joern-path",
        type=Path,
        help="Override Joern CLI installation path"
    )
    config_group.add_argument(
        "--output-dir",
        type=Path,
        help="Override output directory for reports and results"
    )
    config_group.add_argument(
        "--temp-dir",
        type=Path,
        help="Override temporary directory for test files"
    )
    
    # Execution options
    execution_group = parser.add_argument_group('Execution')
    execution_group.add_argument(
        "--languages",
        nargs="+",
        help="Specific languages to verify (default: all supported)"
    )
    execution_group.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout for CPG generation in seconds (default: 300)"
    )
    execution_group.add_argument(
        "--memory",
        default="4g",
        help="Memory allocation for JVM (default: 4g)"
    )
    execution_group.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel processing where safe (experimental)"
    )
    execution_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform dry run without actual CPG generation"
    )
    
    # Report options
    report_group = parser.add_argument_group('Reporting')
    report_group.add_argument(
        "--report-format",
        nargs="+",
        choices=["json", "markdown", "html", "csv"],
        default=["json", "markdown"],
        help="Report output formats (default: json markdown)"
    )
    report_group.add_argument(
        "--no-report",
        action="store_true",
        help="Skip report generation"
    )
    report_group.add_argument(
        "--summary-only",
        action="store_true",
        help="Generate summary report only (faster)"
    )
    
    # Information options
    info_group = parser.add_argument_group('Information')
    info_group.add_argument(
        "--validate",
        action="store_true",
        help="Validate configuration and setup only"
    )
    info_group.add_argument(
        "--list-languages",
        action="store_true",
        help="List all supported languages and exit"
    )
    info_group.add_argument(
        "--discover",
        action="store_true",
        help="Run language discovery only and show results"
    )
    info_group.add_argument(
        "--version",
        action="version",
        version="Joern Multi-Language Verification System v1.0.0"
    )
    
    # Logging options
    logging_group = parser.add_argument_group('Logging')
    logging_group.add_argument(
        "--verbose", "-v",
        action="count",
        default=0,
        help="Increase verbosity (use -v, -vv, or -vvv)"
    )
    logging_group.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress all output except errors"
    )
    logging_group.add_argument(
        "--log-file",
        type=Path,
        help="Write logs to file"
    )
    
    # Advanced options
    advanced_group = parser.add_argument_group('Advanced')
    advanced_group.add_argument(
        "--skip-cleanup",
        action="store_true",
        help="Skip cleanup of temporary files (for debugging)"
    )
    advanced_group.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue verification even if some languages fail"
    )
    advanced_group.add_argument(
        "--export-config",
        type=Path,
        help="Export current configuration to file and exit"
    )
    
    return parser


def setup_logging(args):
    """Setup logging based on command line arguments."""
    import logging
    
    # Determine log level
    if args.quiet:
        level = logging.ERROR
    elif args.verbose == 0:
        level = logging.INFO
    elif args.verbose == 1:
        level = logging.DEBUG
    else:  # args.verbose >= 2
        level = logging.DEBUG
    
    # Configure logging
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    if args.log_file:
        logging.basicConfig(
            level=level,
            format=log_format,
            filename=args.log_file,
            filemode='w'
        )
        # Also log to console unless quiet
        if not args.quiet:
            console = logging.StreamHandler()
            console.setLevel(level)
            formatter = logging.Formatter('%(levelname)s - %(message)s')
            console.setFormatter(formatter)
            logging.getLogger('').addHandler(console)
    else:
        logging.basicConfig(level=level, format=log_format)


def main():
    """Main entry point for command-line execution."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args)
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize verification system
        verification_system = JoernVerificationSystem(args.config)
        
        # Apply command-line overrides to configuration
        if args.joern_path:
            verification_system.config_manager.override_joern_path(str(args.joern_path))
        if args.output_dir:
            verification_system.config_manager.override_output_dir(str(args.output_dir))
        if args.temp_dir:
            verification_system.config_manager.override_temp_dir(str(args.temp_dir))
        
        # Handle export configuration
        if args.export_config:
            success = verification_system.config_manager.export_config(args.export_config)
            if success:
                print(f"Configuration exported to: {args.export_config}")
                sys.exit(0)
            else:
                print("Failed to export configuration")
                sys.exit(1)
        
        # Handle information commands
        if args.validate:
            print("Validating configuration and setup...")
            success = verification_system.validate_setup()
            if success:
                print("✓ Configuration validation passed")
            sys.exit(0 if success else 1)
        
        elif args.list_languages:
            print("Discovering supported languages...")
            languages = verification_system.list_supported_languages()
            print(f"\nSupported languages ({len(languages)}):")
            for lang in sorted(languages):
                config = verification_system.config_manager.get_language_config(lang)
                if config:
                    print(f"  ✓ {config.name:<12} - {config.tool_name}")
                else:
                    print(f"  ? {lang:<12} - Configuration not found")
            sys.exit(0)
        
        elif args.discover:
            print("Running language discovery...")
            from joern_verification.discovery.discovery import LanguageDiscoveryManager
            system_config = verification_system.config_manager.get_system_config()
            discovery_manager = LanguageDiscoveryManager(system_config.joern_path)
            
            try:
                results = discovery_manager.discover_languages()
                print("\nDiscovery Results:")
                print("="*50)
                print(discovery_manager.get_discovery_summary())
                sys.exit(0)
            except Exception as e:
                print(f"Discovery failed: {e}")
                sys.exit(1)
        
        else:
            # Prepare verification options
            verification_options = {
                'languages': args.languages,
                'timeout': args.timeout,
                'memory': args.memory,
                'dry_run': args.dry_run,
                'parallel': args.parallel,
                'report_formats': args.report_format if not args.no_report else [],
                'summary_only': args.summary_only,
                'skip_cleanup': args.skip_cleanup,
                'continue_on_error': args.continue_on_error
            }
            
            # Run verification
            if args.dry_run:
                print("Running in dry-run mode (no actual CPG generation)")
            
            success = verification_system.run_verification_with_options(verification_options)
            
            if success:
                print("\n✓ Verification completed successfully")
                sys.exit(0)
            else:
                print("\n✗ Verification completed with errors")
                sys.exit(1)
    
    except JoernVerificationError as e:
        logger.error(f"Verification error: {e}")
        if not args.quiet:
            print(f"Verification error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Verification interrupted by user")
        if not args.quiet:
            print("\nVerification interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        if not args.quiet:
            print(f"Unexpected error: {e}")
            if args.verbose > 0:
                import traceback
                traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()