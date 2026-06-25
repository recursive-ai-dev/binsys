"""Tests for binsys._util module."""

from __future__ import annotations

import shutil

import pytest

from binsys._util import (
    DEFAULT_KEYBINDINGS,
    QEMU_ARCHES,
    SIZE_PRESETS,
    TYPES,
    _size_to_bytes,
    _unique_snap_name,
    _validate_name,
    _validate_positive_int,
    _validate_size,
    check_dependencies,
    human,
    resolve_size,
    sanitize_filename,
)


def test_default_keybindings_present() -> None:
    assert "new" in DEFAULT_KEYBINDINGS
    assert "delete" in DEFAULT_KEYBINDINGS
    assert "run" in DEFAULT_KEYBINDINGS
    assert "quit" in DEFAULT_KEYBINDINGS


def test_qemu_arches_x86_64() -> None:
    bin_name, opts = QEMU_ARCHES["x86_64"]
    assert bin_name == "qemu-system-x86_64"
    assert opts == []


def test_qemu_arches_aarch64() -> None:
    bin_name, opts = QEMU_ARCHES["aarch64"]
    assert bin_name == "qemu-system-aarch64"
    assert "-machine" in opts


def test_size_presets() -> None:
    assert SIZE_PRESETS["nano"] == "256M"
    assert SIZE_PRESETS["mini"] == "512M"
    assert SIZE_PRESETS["small"] == "1G"
    assert SIZE_PRESETS["medium"] == "2G"
    assert SIZE_PRESETS["large"] == "4G"
    assert SIZE_PRESETS["xl"] == "8G"
    assert SIZE_PRESETS["huge"] == "16G"


def test_types() -> None:
    assert "ext4" in TYPES
    assert "overlay" in TYPES
    assert "squashfs" in TYPES
    assert "fat32" in TYPES
    assert "frugal" in TYPES
    assert "iso" in TYPES


def test_resolve_size_none() -> None:
    assert resolve_size(None) == "1G"


def test_resolve_size_preset() -> None:
    assert resolve_size("nano") == "256M"


def test_resolve_size_custom() -> None:
    assert resolve_size("4G") == "4G"


def test_resolve_size_strips_whitespace() -> None:
    assert resolve_size("  2G  ") == "2G"


def test_size_to_bytes_b() -> None:
    assert _size_to_bytes("100B") == 100


def test_size_to_bytes_k() -> None:
    assert _size_to_bytes("1K") == 1000


def test_size_to_bytes_kib() -> None:
    assert _size_to_bytes("1KiB") == 1024


def test_size_to_bytes_m() -> None:
    assert _size_to_bytes("1M") == 1000**2


def test_size_to_bytes_mib() -> None:
    assert _size_to_bytes("1MiB") == 1024**2


def test_size_to_bytes_g() -> None:
    assert _size_to_bytes("1G") == 1000**3


def test_size_to_bytes_gib() -> None:
    assert _size_to_bytes("1GiB") == 1024**3


def test_size_to_bytes_t() -> None:
    assert _size_to_bytes("1T") == 1000**4


def test_size_to_bytes_case_insensitive() -> None:
    assert _size_to_bytes("1g") == 1000**3
    assert _size_to_bytes("1m") == 1000**2
    assert _size_to_bytes("1k") == 1000


def test_human_bytes() -> None:
    assert human(0) == "0B"


def test_human_kib() -> None:
    assert human(1024) == "1KiB"


def test_human_mib() -> None:
    assert human(1024**2) == "1MiB"


def test_human_gib() -> None:
    assert human(1024**3) == "1GiB"


def test_human_tib() -> None:
    assert human(1024**4) == "1TiB"


def test_human_rounding() -> None:
    assert human(2048) == "2KiB"


def test_sanitize_filename() -> None:
    assert sanitize_filename("hello world") == "hello world"


def test_sanitize_filename_removes_slashes() -> None:
    assert sanitize_filename("a/b/c") == "a_b_c"


def test_sanitize_filename_removes_controls() -> None:
    assert "\x00" not in sanitize_filename("bad\x00name")


def test_sanitize_filename_empty_becomes_unnamed() -> None:
    assert sanitize_filename("") == "unnamed"


def test_validate_name_valid() -> None:
    _validate_name("my-system_1.test")  # should not raise


def test_validate_name_invalid() -> None:
    with pytest.raises(RuntimeError, match="invalid name"):
        _validate_name(" bad name ")


def test_validate_name_invalid_chars() -> None:
    with pytest.raises(RuntimeError, match="invalid name"):
        _validate_name("hello/world")


def test_unique_snap_name_first(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("binsys._util.IMAGES", tmp_path)
    result = _unique_snap_name("test")
    assert result == "test"


def test_unique_snap_name_second(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("binsys._util.IMAGES", tmp_path)
    (tmp_path / "test").mkdir(parents=True)
    result = _unique_snap_name("test")
    assert result == "test-2"


def test_unique_snap_name_third(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("binsys._util.IMAGES", tmp_path)
    (tmp_path / "test").mkdir(parents=True)
    (tmp_path / "test-2").mkdir(parents=True)
    result = _unique_snap_name("test")
    assert result == "test-3"


def test_validate_size_valid() -> None:
    assert _validate_size("1G") == 1000**3
    assert _validate_size("512M") == 512 * 1000**2
    assert _validate_size("100K") == 100 * 1000


def test_validate_size_invalid() -> None:
    with pytest.raises(RuntimeError, match="invalid size"):
        _validate_size("invalid")
    with pytest.raises(RuntimeError, match="invalid size"):
        _validate_size("-1G")
    with pytest.raises(RuntimeError, match="invalid size"):
        _validate_size("0B")


def test_validate_positive_int_valid() -> None:
    assert _validate_positive_int("42") == 42
    assert _validate_positive_int("1") == 1


def test_validate_positive_int_invalid() -> None:
    with pytest.raises(RuntimeError, match="invalid value"):
        _validate_positive_int("-1")
    with pytest.raises(RuntimeError, match="invalid value"):
        _validate_positive_int("0")
    with pytest.raises(RuntimeError, match="invalid value"):
        _validate_positive_int("abc")


def test_check_dependencies() -> None:
    result = check_dependencies()
    assert isinstance(result, dict)
    # Should return empty dict or dict with missing deps
    for category, bins in result.items():
        assert isinstance(bins, list)
        for b in bins:
            assert not shutil.which(b)
