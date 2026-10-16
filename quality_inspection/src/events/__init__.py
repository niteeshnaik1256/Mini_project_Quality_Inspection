"""
Quality Inspector Event Module.
Provides standardized CVEvent schema for compatibility with multi-role Smart Factory system.
"""

from .quality_event import QualityCVEvent, QualityEventLogger

__all__ = ["QualityCVEvent", "QualityEventLogger"]
