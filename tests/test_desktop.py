"""Tests for os.desktop.aios_desktop module."""

from __future__ import annotations

import os
import sys

# add os/desktop to path so it can import aios_display
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'os', 'desktop')))

import importlib.util

import pytest

# We dynamically import aios_desktop because the 'os' module name conflicts with the standard library
spec = importlib.util.spec_from_file_location("aios_desktop", os.path.join(os.path.dirname(__file__), "..", "os", "desktop", "aios_desktop.py"))
aios_desktop = importlib.util.module_from_spec(spec)
sys.modules["aios_desktop"] = aios_desktop
spec.loader.exec_module(aios_desktop)

clamp = aios_desktop.clamp

def test_clamp_inside_range() -> None:
    """Test value strictly inside the range."""
    assert clamp(5.0, 0.0, 10.0) == 5.0
    assert clamp(5, 0, 10) == 5

def test_clamp_below_lower_bound() -> None:
    """Test value less than the lower bound."""
    assert clamp(-5.0, 0.0, 10.0) == 0.0
    assert clamp(-1, 0, 10) == 0

def test_clamp_above_upper_bound() -> None:
    """Test value greater than the upper bound."""
    assert clamp(15.0, 0.0, 10.0) == 10.0
    assert clamp(11, 0, 10) == 10

def test_clamp_at_bounds() -> None:
    """Test value equal to bounds."""
    assert clamp(0.0, 0.0, 10.0) == 0.0
    assert clamp(10.0, 0.0, 10.0) == 10.0

def test_clamp_negative_values() -> None:
    """Test negative bounds and values."""
    assert clamp(-5.0, -10.0, 0.0) == -5.0
    assert clamp(-15.0, -10.0, 0.0) == -10.0
    assert clamp(5.0, -10.0, 0.0) == 0.0

def test_clamp_equal_bounds() -> None:
    """Test edge case where lower and upper bounds are equal."""
    assert clamp(5.0, 10.0, 10.0) == 10.0
    assert clamp(15.0, 10.0, 10.0) == 10.0
    assert clamp(10.0, 10.0, 10.0) == 10.0

def test_clamp_float_infinity() -> None:
    """Test behavior with positive and negative infinity."""
    import math
    assert clamp(5.0, -math.inf, math.inf) == 5.0
    assert clamp(math.inf, 0.0, 10.0) == 10.0
    assert clamp(-math.inf, 0.0, 10.0) == 0.0
