"""Unified SQLite database layer for ArtSeek v2."""

from .connection import Database
from .scope_resolver import ScopeResolver

__all__ = ["Database", "ScopeResolver"]
