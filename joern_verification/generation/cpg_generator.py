"""CPG generation engine for multiple programming languages."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any
from .command_executor import CommandExecutor, ExecutionResult


@dataclass
class GenerationResult:
    """Result of CPG generation for a specific language."""
    language: str
    input_file: Path
    output_dir: Path
    success: bool
    execution_time: float
    memory_usage: Optional[int]
    stdout: str
    stderr: str
    return_code: int
    output_files: List[Path]
    error_message: Optional[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class CPGGenerator:
    """Language-specific CPG generator using Joern tools."""
    
    # Language tool mapping based on discovered tools
    LANGUAGE_TOOLS = {
        'c': 'c2cpg.bat',
        'cpp': 'c2cpg.bat', 
        'c++': 'c2cpg.bat',
        'csharp': 'csharpsrc2cpg.bat',
        'c#': 'csharpsrc2cpg.bat',
        'java': 'javasrc2cpg.bat',
        'javascript': 'jssrc2cpg.bat',
        'js': 'jssrc2cpg.bat',
        'kotlin': 'kotlin2cpg.bat',
        'php': 'php2cpg.bat',
        'python': 'pysrc2cpg.bat',
        'py': 'pysrc2cpg.bat',
        'ruby': 'rubysrc2cpg.bat',
        'rb': 'rubysrc2cpg.bat',
        'swift': 'swiftsrc2cpg.bat',
        'go': 'gosrc2cpg.bat'
    }
    
    def __init__(self, joern_cli_path: Path, executor: Optional[CommandExecutor] = None):
        """
        Initialize CPG generator.
        
        Args:
            joern_cli_path: Path to joern-cli directory
            executor: Command executor instance (creates default if None)
        """
        self.joern_cli_path = Path(joern_cli_path)
        self.executor = executor or CommandExecutor()
        self.logger = logging.getLogger(__name__)
        
        # Validate joern-cli path
        if not self.joern_cli_path.exists():
            raise ValueError(f"Joern CLI path does not exist: {joern_cli_path}")
            
    def get_available_languages(self) -> List[str]:
        """
        Get list of languages that have available tools.
        
        Returns:
            List of supported language names
        """
        available = []
        for language, tool_name in self.LANGUAGE_TOOLS.items():
            tool_path = self.joern_cli_path / tool_name
            if self.executor.validate_tool_availability(tool_path):
                available.append(language)
        return available
    
    def get_tool_path(self, language: str) -> Optional[Path]:
        """
        Get the tool path for a specific language.
        
        Args:
            language: Programming language name
            
        Returns:
            Path to the language tool or None if not supported
        """
        language_lower = language.lower()
        if language_lower not in self.LANGUAGE_TOOLS:
            return None
            
        tool_name = self.LANGUAGE_TOOLS[language_lower]
        tool_path = self.joern_cli_path / tool_name
        
        if self.executor.validate_tool_availability(tool_path):
            return tool_path
        return None
    
    def build_command(
        self,
        language: str,
        input_file: Path,
        output_dir: Path,
        memory: Optional[str] = None
    ) -> Optional[List[str]]:
        """
        Build command for CPG generation.
        
        Args:
            language: Programming language
            input_file: Source file to process
            output_dir: Output directory for CPG
            memory: Memory allocation (e.g., "4g")
            
        Returns:
            Command as list of strings or None if language not supported
        """
        tool_path = self.get_tool_path(language)
        if not tool_path:
            return None
            
        # Build base command
        command = [str(tool_path)]
        
        # Add memory arguments
        memory_args = self.executor.get_memory_args(memory)
        command.extend(memory_args)
        
        # Add input file
        command.append(str(input_file))
        
        # Add output directory
        command.extend(["--output", str(output_dir)])
        
        return command
    
    def generate_cpg(
        self,
        language: str,
        input_file: Path,
        output_dir: Path,
        memory: Optional[str] = None,
        timeout: Optional[int] = None
    ) -> GenerationResult:
        """
        Generate CPG for a source file.
        
        Args:
            language: Programming language
            input_file: Source file to process
            output_dir: Output directory for CPG
            memory: Memory allocation (e.g., "4g")
            timeout: Execution timeout in seconds
            
        Returns:
            GenerationResult with execution details
        """
        self.logger.info(f"Generating CPG for {language}: {input_file}")
        
        # Validate inputs
        if not input_file.exists():
            return GenerationResult(
                language=language,
                input_file=input_file,
                output_dir=output_dir,
                success=False,
                execution_time=0.0,
                memory_usage=None,
                stdout="",
                stderr="Input file does not exist",
                return_code=-1,
                output_files=[],
                error_message=f"Input file does not exist: {input_file}"
            )
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Build command
        command = self.build_command(language, input_file, output_dir, memory)
        if not command:
            return GenerationResult(
                language=language,
                input_file=input_file,
                output_dir=output_dir,
                success=False,
                execution_time=0.0,
                memory_usage=None,
                stdout="",
                stderr=f"Language not supported: {language}",
                return_code=-1,
                output_files=[],
                error_message=f"Language not supported or tool not available: {language}"
            )
        
        # Execute command
        exec_result = self.executor.execute_command(
            command=command,
            working_dir=output_dir,
            timeout=timeout
        )
        
        # Find output files
        output_files = self._find_output_files(output_dir)
        
        # Extract warnings from stderr
        warnings = self._extract_warnings(exec_result.stderr)
        
        # Create generation result
        result = GenerationResult(
            language=language,
            input_file=input_file,
            output_dir=output_dir,
            success=exec_result.success,
            execution_time=exec_result.execution_time,
            memory_usage=None,  # TODO: Implement memory usage tracking
            stdout=exec_result.stdout,
            stderr=exec_result.stderr,
            return_code=exec_result.return_code,
            output_files=output_files,
            error_message=exec_result.error_message,
            warnings=warnings
        )
        
        if result.success:
            self.logger.info(f"CPG generation successful for {language}")
            self.logger.info(f"Generated {len(output_files)} output files")
        else:
            self.logger.error(f"CPG generation failed for {language}: {result.error_message}")
            
        return result
    
    def _find_output_files(self, output_dir: Path) -> List[Path]:
        """
        Find generated output files in the output directory.
        
        Args:
            output_dir: Directory to search for output files
            
        Returns:
            List of generated file paths
        """
        output_files = []
        
        if not output_dir.exists():
            return output_files
            
        # Common CPG file extensions
        cpg_extensions = ['.cpg', '.bin', '.dot', '.json', '.xml']
        
        for file_path in output_dir.rglob('*'):
            if file_path.is_file():
                # Include CPG files and any other generated files
                if (file_path.suffix.lower() in cpg_extensions or 
                    'cpg' in file_path.name.lower()):
                    output_files.append(file_path)
        
        return sorted(output_files)
    
    def _extract_warnings(self, stderr: str) -> List[str]:
        """
        Extract warning messages from stderr output.
        
        Args:
            stderr: Standard error output
            
        Returns:
            List of warning messages
        """
        warnings = []
        
        if not stderr:
            return warnings
            
        lines = stderr.split('\n')
        for line in lines:
            line = line.strip()
            if line and any(keyword in line.lower() for keyword in ['warn', 'warning']):
                warnings.append(line)
                
        return warnings
    
    def validate_output(self, result: GenerationResult) -> bool:
        """
        Validate that CPG generation produced expected output.
        
        Args:
            result: Generation result to validate
            
        Returns:
            True if output is valid
        """
        if not result.success:
            return False
            
        # Check if output files were generated
        if not result.output_files:
            self.logger.warning(f"No output files generated for {result.language}")
            return False
            
        # Check if output files exist and have content
        for output_file in result.output_files:
            if not output_file.exists():
                self.logger.error(f"Output file does not exist: {output_file}")
                return False
                
            if output_file.stat().st_size == 0:
                self.logger.warning(f"Output file is empty: {output_file}")
                
        return True
    
    def get_command_template(self, language: str) -> Optional[str]:
        """
        Get command template for a language.
        
        Args:
            language: Programming language
            
        Returns:
            Command template string or None if not supported
        """
        tool_path = self.get_tool_path(language)
        if not tool_path:
            return None
            
        return f"{tool_path.name} -J-Xmx4g <input_file> --output <output_dir>"