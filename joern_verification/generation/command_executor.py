"""Command execution framework for running Joern language tools."""

import subprocess
import threading
import time
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from pathlib import Path


@dataclass
class ExecutionResult:
    """Result of command execution."""
    command: str
    success: bool
    return_code: int
    stdout: str
    stderr: str
    execution_time: float
    timeout_occurred: bool
    error_message: Optional[str] = None


class CommandExecutor:
    """Framework for executing Joern language tools with proper resource management."""
    
    def __init__(self, default_timeout: int = 300, max_memory: str = "4g"):
        """
        Initialize command executor.
        
        Args:
            default_timeout: Default timeout in seconds for command execution
            max_memory: Maximum memory allocation (e.g., "4g", "2g")
        """
        self.default_timeout = default_timeout
        self.max_memory = max_memory
        self.logger = logging.getLogger(__name__)
        
    def execute_command(
        self,
        command: List[str],
        working_dir: Optional[Path] = None,
        timeout: Optional[int] = None,
        env_vars: Optional[Dict[str, str]] = None
    ) -> ExecutionResult:
        """
        Execute a command with timeout and resource management.
        
        Args:
            command: Command and arguments as list
            working_dir: Working directory for command execution
            timeout: Timeout in seconds (uses default if None)
            env_vars: Additional environment variables
            
        Returns:
            ExecutionResult with execution details
        """
        timeout = timeout or self.default_timeout
        command_str = " ".join(command)
        
        self.logger.info(f"Executing command: {command_str}")
        self.logger.info(f"Working directory: {working_dir}")
        self.logger.info(f"Timeout: {timeout}s")
        
        start_time = time.time()
        
        try:
            # Prepare environment
            env = None
            if env_vars:
                import os
                env = os.environ.copy()
                env.update(env_vars)
            
            # Execute command with timeout
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=working_dir,
                env=env
            )
            
            try:
                stdout, stderr = process.communicate(timeout=timeout)
                execution_time = time.time() - start_time
                
                result = ExecutionResult(
                    command=command_str,
                    success=process.returncode == 0,
                    return_code=process.returncode,
                    stdout=stdout,
                    stderr=stderr,
                    execution_time=execution_time,
                    timeout_occurred=False
                )
                
                if result.success:
                    self.logger.info(f"Command completed successfully in {execution_time:.2f}s")
                else:
                    self.logger.warning(f"Command failed with return code {process.returncode}")
                    result.error_message = f"Command failed with return code {process.returncode}"
                
                return result
                
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                execution_time = time.time() - start_time
                
                self.logger.error(f"Command timed out after {timeout}s")
                
                return ExecutionResult(
                    command=command_str,
                    success=False,
                    return_code=-1,
                    stdout="",
                    stderr="Command timed out",
                    execution_time=execution_time,
                    timeout_occurred=True,
                    error_message=f"Command timed out after {timeout}s"
                )
                
        except FileNotFoundError as e:
            execution_time = time.time() - start_time
            error_msg = f"Command not found: {e}"
            self.logger.error(error_msg)
            
            return ExecutionResult(
                command=command_str,
                success=False,
                return_code=-1,
                stdout="",
                stderr=str(e),
                execution_time=execution_time,
                timeout_occurred=False,
                error_message=error_msg
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Unexpected error executing command: {e}"
            self.logger.error(error_msg)
            
            return ExecutionResult(
                command=command_str,
                success=False,
                return_code=-1,
                stdout="",
                stderr=str(e),
                execution_time=execution_time,
                timeout_occurred=False,
                error_message=error_msg
            )
    
    def validate_tool_availability(self, tool_path: Path) -> bool:
        """
        Validate that a tool is available and executable.
        
        Args:
            tool_path: Path to the tool executable
            
        Returns:
            True if tool is available and executable
        """
        if not tool_path.exists():
            self.logger.error(f"Tool not found: {tool_path}")
            return False
            
        if not tool_path.is_file():
            self.logger.error(f"Tool path is not a file: {tool_path}")
            return False
            
        # On Windows, check if it's a .bat or .exe file
        if tool_path.suffix.lower() not in ['.bat', '.exe', '.cmd']:
            self.logger.warning(f"Tool may not be executable: {tool_path}")
            
        return True
    
    def get_memory_args(self, memory: Optional[str] = None) -> List[str]:
        """
        Get JVM memory arguments for Joern tools.
        
        Args:
            memory: Memory allocation (e.g., "4g", "2g")
            
        Returns:
            List of JVM memory arguments
        """
        memory = memory or self.max_memory
        return [f"-J-Xmx{memory}"]