# Original code (CMP 3005 term project).
"""Shared pytest fixtures: paths are resolved from the repository root."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
AUGERAT = ["A-n32-k5", "A-n45-k6", "A-n60-k9", "A-n80-k10"]


@pytest.fixture
def root() -> Path:
    """Repository root directory."""
    return ROOT
