"""
Custom exceptions for the Joern multi-language verification system.
"""


class JoernVerificationError(Exception):
    """Base exception for all verification system errors."""
    pass


class LanguageNotSupportedError(JoernVerificationError):
    """Raised when a requested language is not supported."""
    pass


class ToolNotFoundError(JoernVerificationError):
    """Raised when a required Joern tool is not found."""
    pass


class CPGGenerationError(JoernVerificationError):
    """Raised when CPG generation fails."""
    pass


class ConfigurationError(JoernVerificationError):
    """Raised when there are configuration issues."""
    pass


class TestFileGenerationError(JoernVerificationError):
    """Raised when test file generation fails."""
    pass