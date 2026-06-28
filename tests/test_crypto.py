"""Tests for binsys._crypto module."""

from __future__ import annotations

import os
import pytest

import binsys._crypto as crypto_module
from binsys._crypto import (
    _app_lock_hash,
    _check_rate_limit,
    _clear_failures,
    _load_app_locks,
    _record_failure,
    _save_app_locks,
)


def test_app_lock_hash() -> None:
    """Test password hashing with PBKDF2."""
    salt = os.urandom(16)
    hash1 = _app_lock_hash("password123", salt)
    assert isinstance(hash1, str)
    assert len(hash1) == 64  # SHA256 hex digest

    # Same password + same salt should produce same hash
    hash2 = _app_lock_hash("password123", salt)
    assert hash1 == hash2

    # Different password should produce different hash
    hash3 = _app_lock_hash("different", salt)
    assert hash1 != hash3

    # Same password + different salt should produce different hash
    hash4 = _app_lock_hash("password123", os.urandom(16))
    assert hash1 != hash4


def test_app_lock_hash_with_keyfile(tmp_path) -> None:
    """Test password hashing with keyfile and PBKDF2."""
    salt = os.urandom(16)
    keyfile = tmp_path / "keyfile.txt"
    keyfile.write_text("secret key content")

    hash1 = _app_lock_hash("password123", salt, str(keyfile))
    assert isinstance(hash1, str)
    assert len(hash1) == 64

    # Same password + salt + keyfile should produce same hash
    hash2 = _app_lock_hash("password123", salt, str(keyfile))
    assert hash1 == hash2

    # Different keyfile content should produce different hash
    keyfile.write_text("different content")
    hash3 = _app_lock_hash("password123", salt, str(keyfile))
    assert hash1 != hash3


def test_rate_limiting(monkeypatch) -> None:
    """Test authentication rate limiting."""
    # Clear any existing state
    crypto_module._auth_failures.clear()

    name = "test-system"

    # First 5 attempts should work (just record, not block)
    for i in range(5):
        _check_rate_limit(name)  # Should not raise
        _record_failure(name)

    # 6th attempt should be blocked
    with pytest.raises(RuntimeError, match="Too many failed attempts"):
        _check_rate_limit(name)

    # Clear and verify it works again
    _clear_failures(name)
    _check_rate_limit(name)  # Should work now


def test_save_and_load_app_locks(tmp_path, monkeypatch) -> None:
    """Test saving and loading app locks."""

    # Mock STORE to use tmp_path
    monkeypatch.setattr("binsys._crypto.STORE", tmp_path)

    locks = {
        "system1": {
            "hash": "abc123",
            "salt": "f00f",
            "keyfile": None,
            "unlocked": False,
        },
        "system2": {
            "hash": "def456",
            "salt": "baaa",
            "keyfile": "/path/to/keyfile",
            "unlocked": True,
        },
    }

    _save_app_locks(locks)

    # Verify file was created
    lock_file = tmp_path / "app_locks.json"
    assert lock_file.exists()

    # Load and verify
    loaded = _load_app_locks()
    assert loaded == locks


def test_app_locks_file_not_exists(tmp_path, monkeypatch) -> None:
    """Test loading app locks when file doesn't exist."""
    monkeypatch.setattr("binsys._crypto.STORE", tmp_path)

    # Should return empty dict
    result = _load_app_locks()
    assert result == {}


def test_app_locks_invalid_json(tmp_path, monkeypatch) -> None:
    """Test loading app locks with invalid JSON."""
    monkeypatch.setattr("binsys._crypto.STORE", tmp_path)

    # Create invalid JSON file
    lock_file = tmp_path / "app_locks.json"
    lock_file.write_text("not valid json")

    # Should return empty dict
    result = _load_app_locks()
    assert result == {}
