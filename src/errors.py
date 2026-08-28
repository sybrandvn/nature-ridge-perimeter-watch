"""Shared exception types for configuration and persistence errors."""

from __future__ import annotations


class ConfigError(ValueError):
    """Raised when environment or YAML configuration is missing or malformed."""


class DbError(RuntimeError):
    """Raised for schema, constraint, or persistence failures in src.db."""
