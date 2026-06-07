"""Compatibility layer for the old verifier module.

Use validator.MessageValidator for deterministic validation.
"""

from __future__ import annotations

from schemas import ValidationIssue, ValidationResult
from validator import MessageValidator

__all__ = ["MessageValidator", "ValidationIssue", "ValidationResult"]
