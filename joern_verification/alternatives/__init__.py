"""
Alternative tool recommendation system for Joern multi-language verification.

This module provides alternative CPG/AST generation tools for languages that
are not supported by Joern or have limited support.
"""

from .database import AlternativeToolDatabase
from .recommender import AlternativeToolRecommender, RecommendationContext, ToolRecommendation
from .models import AlternativeTool, ToolCapability, InstallationMethod
from .integration import AlternativeToolIntegrator

__all__ = [
    'AlternativeToolDatabase',
    'AlternativeToolRecommender',
    'RecommendationContext',
    'ToolRecommendation',
    'AlternativeTool',
    'ToolCapability',
    'InstallationMethod',
    'AlternativeToolIntegrator'
]