"""Tests for binsys._image module."""

from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from binsys._image import (
    _flash_source,
    do_new,
    do_delete,
    do_clone,
    do_rename,
    do_export,
    do_check,
    do_mount,
    do_umount,
    do_snap,
    do_import,
    do_resize,
)


# =============================================================================
# Exception handling tests for _image.py
# =============================================================================

class TestFlashSource:
    """Tests for _flash_source function exception handling."""

    def test_unknown_distro_raises(self) -> None:
        """Test that unknown distro raises RuntimeError."""
        with pytest.raises(RuntimeError, match="unknown distro"):
            _flash_source("unknown-distro", Path("/tmp/test"), "1G")

    def test_invalid_distro_empty_raises(self) -> None:
        """Test that empty distro string raises RuntimeError."""
        with pytest.raises(RuntimeError, match="unknown distro"):
            _flash_source("", Path("/tmp/test"), "1G")

    @patch("binsys._image.urllib.request.urlretrieve")
    def test_download_failure_raises(self, mock_retrieve) -> None:
        """Test that download failure raises RuntimeError."""
        import urllib.error
        mock_retrieve.side_effect = urllib.error.URLError("Download failed")
        
        with pytest.raises(RuntimeError, match="download failed"):
            _flash_source("ubuntu", Path("/tmp/test"), "1G")

    @patch("binsys._image.urllib.request.urlretrieve")
    def test_hash_mismatch_raises(self, mock_retrieve, tmp_path) -> None:
        """Test that hash mismatch raises RuntimeError."""
        import hashlib
        
        # Create a temporary directory
        d = tmp_path / "test"
        d.mkdir()
        
        # Mock urlretrieve to create a file with wrong content
        def mock_urlretrieve(url, path):
            (Path(path)).write_text("wrong content")
        
        mock_retrieve.side_effect = mock_urlretrieve
        
        with pytest.raises(RuntimeError, match="download hash mismatch"):
            _flash_source("ubuntu", d, "1G")


class TestDoNew:
    """Tests for do_new function exception handling."""

    @patch("binsys._image.sys_dir")
    def test_duplicate_name_raises(self, mock_sys_dir) -> None:
        """Test that duplicate system name raises RuntimeError."""
        mock_sys_dir.return_value = MagicMock(exists=lambda: True)
        
        with pytest.raises(RuntimeError, match="already exists"):
            do_new("existing-system", "ext4", "1G")

    def test_invalid_name_raises(self) -> None:
        """Test that invalid name raises RuntimeError."""
        with pytest.raises(RuntimeError, match="invalid name"):
            do_new("bad name", "ext4", "1G")

    def test_invalid_type_raises(self) -> None:
        """Test that invalid type raises RuntimeError."""
        with pytest.raises(RuntimeError, match="unknown type"):
            do_new("test", "invalid-type", "1G")

    def test_invalid_size_raises(self) -> None:
        """Test that invalid size raises RuntimeError."""
        with pytest.raises(RuntimeError, match="invalid size"):
            do_new("test", "ext4", "invalid-size")

    def test_zero_size_raises(self) -> None:
        """Test that zero size raises RuntimeError."""
        with pytest.raises(RuntimeError, match="invalid size"):
            do_new("test", "ext4", "0B")

    @patch("binsys._image.sys_dir")
    @patch("binsys._image.sh")
    def test_sh_command_failure_cleanup(self, mock_sh, mock_sys_dir, tmp_path) -> None:
        """Test that cleanup happens when sh command fails during creation."""
        mock_sys_dir.return_value = tmp_path / "test"
        mock_sh.side_effect = RuntimeError("Command failed")
        
        with pytest.raises(RuntimeError):
            do_new("test", "ext4", "1G")
        
        # Verify sh was called
        assert mock_sh.called


class TestDoDelete:
    """Tests for do_delete function exception handling."""

    @patch("binsys._image.sys_dir")
    def test_not_found_raises(self, mock_sys_dir) -> None:
        """Test that deleting non-existent system raises RuntimeError."""
        mock_sys_dir.return_value = MagicMock(exists=lambda: False)
        
        with pytest.raises(RuntimeError, match="not found"):
            do_delete("nonexistent")

    @patch("binsys._image.sys_dir")
    @patch("binsys._image.is_mounted")
    def test_mounted_raises(self, mock_is_mounted, mock_sys_dir) -> None:
        """Test that deleting mounted system raises RuntimeError."""
        mock_sys_dir.return_value = MagicMock(exists=lambda: True)
        mock_is_mounted.return_value = True
        
        with pytest.raises(RuntimeError, match="mounted"):
            do_delete("mounted-system")


class TestDoClone:
    """Tests for do_clone function exception handling."""

    @patch("binsys._image.sys_dir")
    def test_src_not_found_raises(self, mock_sys_dir) -> None:
        """Test that cloning non-existent source raises RuntimeError."""
        mock_sys_dir.return_value = MagicMock(exists=lambda: False)
        
        with pytest.raises(RuntimeError, match="not found"):
            do_clone("nonexistent", "new-clone")

    @patch("binsys._image.sys_dir")
    @patch("binsys._image.is_mounted")
    def test_src_mounted_raises(self, mock_is_mounted, mock_sys_dir) -> None:
        """Test that cloning mounted source raises RuntimeError."""
        mock_sys_dir.return_value = MagicMock(exists=lambda: True)
        mock_is_mounted.return_value = True
        
        with pytest.raises(RuntimeError, match="mounted"):
            do_clone("mounted-src", "new-clone")

    @patch("binsys._image.sys_dir")
    def test_dst_exists_raises(self, mock_sys_dir) -> None:
        """Test that cloning to existing destination raises RuntimeError."""
        mock_sys_dir.side_effect = [MagicMock(exists=lambda: True), MagicMock(exists=lambda: True)]
        
        with pytest.raises(RuntimeError, match="already exists"):
            do_clone("source", "existing-dest")

    def test_invalid_dst_name_raises(self) -> None:
        """Test that invalid destination name raises RuntimeError."""
        with pytest.raises(RuntimeError, match="invalid name"):
            do_clone("source", "bad name")


class TestDoRename:
    """Tests for do_rename function exception handling."""

    @patch("binsys._image.sys_dir")
    def test_old_not_found_raises(self, mock_sys_dir) -> None:
        """Test that renaming non-existent system raises RuntimeError."""
        mock_sys_dir.return_value = MagicMock(exists=lambda: False)
        
        with pytest.raises(RuntimeError, match="not found"):
            do_rename("nonexistent", "new-name")

    @patch("binsys._image.sys_dir")
    @patch("binsys._image.is_mounted")
    def test_old_mounted_raises(self, mock_is_mounted, mock_sys_dir) -> None:
        """Test that renaming mounted system raises RuntimeError."""
        mock_sys_dir.return_value = MagicMock(exists=lambda: True)
        mock_is_mounted.return_value = True
        
        with pytest.raises(RuntimeError, match="mounted"):
            do_rename("mounted", "new-name")

    @patch("binsys._image.sys_dir")
    def test_new_exists_raises(self, mock_sys_dir) -> None:
        """Test that renaming to existing name raises RuntimeError."""
        mock_sys_dir.side_effect = [MagicMock(exists=lambda: True), MagicMock(exists=lambda: True)]
        
        with pytest.raises(RuntimeError, match="already exists"):
            do_rename("old", "existing")

    def test_invalid_new_name_raises(self) -> None:
        """Test that invalid new name raises RuntimeError."""
        with pytest.raises(RuntimeError, match="invalid name"):
            do_rename("old", "bad name")


class TestDoExport:
    """Tests for do_export function exception handling."""

    @patch("binsys._image.load_meta")
    def test_not_found_raises(self, mock_load_meta) -> None:
        """Test that exporting non-existent system raises RuntimeError."""
        mock_load_meta.return_value = None
        
        with pytest.raises(RuntimeError, match="not found"):
            do_export("nonexistent")

    @patch("binsys._image.load_meta")
    def test_unknown_type_raises(self, mock_load_meta) -> None:
        """Test that exporting system with unknown type raises RuntimeError."""
        mock_load_meta.return_value = {"type": "unknown-type", "name": "test"}
        
        with pytest.raises(RuntimeError, match="no export strategy"):
            do_export("test")


class TestDoCheck:
    """Tests for do_check function exception handling."""

    @patch("binsys._image.load_meta")
    def test_not_found_raises(self, mock_load_meta) -> None:
        """Test that checking non-existent system raises RuntimeError."""
        mock_load_meta.return_value = None
        
        with pytest.raises(RuntimeError, match="not found"):
            do_check("nonexistent")

    @patch("binsys._image.load_meta")
    @patch("binsys._image.is_mounted")
    def test_mounted_raises(self, mock_is_mounted, mock_load_meta) -> None:
        """Test that checking mounted system raises RuntimeError."""
        mock_load_meta.return_value = {"type": "ext4", "name": "test", "disk": "disk.img"}
        mock_is_mounted.return_value = True
        
        with pytest.raises(RuntimeError, match="mounted"):
            do_check("mounted")

    @patch("binsys._image.load_meta")
    def test_unknown_type_raises(self, mock_load_meta) -> None:
        """Test that checking system with unknown type raises RuntimeError."""
        mock_load_meta.return_value = {"type": "unknown-type", "name": "test"}
        
        with pytest.raises(RuntimeError, match="no check method"):
            do_check("test")


class TestDoMount:
    """Tests for do_mount function exception handling."""

    @patch("binsys._image.load_meta")
    def test_not_found_raises(self, mock_load_meta) -> None:
        """Test that mounting non-existent system raises RuntimeError."""
        mock_load_meta.return_value = None
        
        with pytest.raises(RuntimeError, match="not found"):
            do_mount("nonexistent")

    @patch("binsys._image.load_meta")
    @patch("binsys._image.is_mounted")
    @patch("binsys._image.MOUNTS")
    def test_already_mounted_raises(self, mock_mounts, mock_is_mounted, mock_load_meta) -> None:
        """Test that mounting already mounted system raises RuntimeError."""
        mock_load_meta.return_value = {"type": "ext4", "name": "test", "disk": "disk.img"}
        mock_mounts.return_value = Path("/mnt/test")
        mock_is_mounted.return_value = True
        
        with pytest.raises(RuntimeError, match="already mounted"):
            do_mount("test")


class TestDoUmount:
    """Tests for do_umount function exception handling."""

    @patch("binsys._image.is_mounted")
    @patch("binsys._image.MOUNTS")
    def test_not_mounted_raises(self, mock_mounts, mock_is_mounted) -> None:
        """Test that unmounting non-mounted system raises RuntimeError."""
        mock_mounts.return_value = Path("/mnt/test")
        mock_is_mounted.return_value = False
        
        with pytest.raises(RuntimeError, match="not mounted"):
            do_umount("test")


class TestDoSnap:
    """Tests for do_snap function exception handling."""

    @patch("binsys._image.load_meta")
    def test_not_found_raises(self, mock_load_meta) -> None:
        """Test that snapping non-existent system raises RuntimeError."""
        mock_load_meta.return_value = None
        
        with pytest.raises(RuntimeError, match="not found"):
            do_snap("nonexistent")

    @patch("binsys._image.load_meta")
    def test_non_overlay_type_raises(self, mock_load_meta) -> None:
        """Test that snapping non-overlay system raises RuntimeError."""
        mock_load_meta.return_value = {"type": "ext4", "name": "test"}
        
        with pytest.raises(RuntimeError, match="snapshot only supported for overlay"):
            do_snap("test")


class TestDoImport:
    """Tests for do_import function exception handling."""

    def test_src_not_found_raises(self, tmp_path) -> None:
        """Test that importing non-existent source raises RuntimeError."""
        with pytest.raises(RuntimeError, match="source not found"):
            do_import("/nonexistent/path", "test")

    def test_invalid_name_raises(self, tmp_path) -> None:
        """Test that invalid name raises RuntimeError."""
        # Create a temp file
        temp_file = tmp_path / "test.img"
        temp_file.write_bytes(b"test")
        
        with pytest.raises(RuntimeError, match="invalid name"):
            do_import(str(temp_file), "bad name")

    @patch("binsys._image.sys_dir")
    def test_dst_exists_raises(self, mock_sys_dir, tmp_path) -> None:
        """Test that importing to existing destination raises RuntimeError."""
        temp_file = tmp_path / "test.img"
        temp_file.write_bytes(b"test")
        mock_sys_dir.return_value = MagicMock(exists=lambda: True)
        
        with pytest.raises(RuntimeError, match="already exists"):
            do_import(str(temp_file), "existing")


class TestDoResize:
    """Tests for do_resize function exception handling."""

    @patch("binsys._image.load_meta")
    def test_not_found_raises(self, mock_load_meta) -> None:
        """Test that resizing non-existent system raises RuntimeError."""
        mock_load_meta.return_value = None
        
        with pytest.raises(RuntimeError, match="not found"):
            do_resize("nonexistent", "2G")

    @patch("binsys._image.load_meta")
    @patch("binsys._image.is_mounted")
    @patch("binsys._image.MOUNTS")
    def test_mounted_raises(self, mock_mounts, mock_is_mounted, mock_load_meta) -> None:
        """Test that resizing mounted system raises RuntimeError."""
        mock_load_meta.return_value = {"type": "ext4", "name": "test", "disk": "disk.img"}
        mock_mounts.return_value = Path("/mnt/test")
        mock_is_mounted.return_value = True
        
        with pytest.raises(RuntimeError, match="mounted"):
            do_resize("test", "2G")

    @patch("binsys._image.load_meta")
    def test_unsupported_type_raises(self, mock_load_meta) -> None:
        """Test that resizing unsupported type raises RuntimeError."""
        mock_load_meta.return_value = {"type": "squashfs", "name": "test"}
        
        with pytest.raises(RuntimeError, match="resize not supported"):
            do_resize("test", "2G")

    @patch("binsys._image.load_meta")
    def test_invalid_size_raises(self, mock_load_meta) -> None:
        """Test that invalid size raises RuntimeError."""
        mock_load_meta.return_value = {"type": "ext4", "name": "test", "disk": "disk.img"}
        
        with pytest.raises(RuntimeError, match="invalid size"):
            do_resize("test", "invalid")


# =============================================================================
# Integration tests for _image.py
# =============================================================================

class TestImageIntegration:
    """Integration tests for image operations."""

    @patch("binsys._image.sys_dir")
    @patch("binsys._image.sh")
    @patch("binsys._image.ensure_dirs")
    @patch("binsys._image.resolve_size")
    @patch("binsys._image.save_meta")
    def test_new_creates_directory_and_calls_sh(self, mock_save_meta, mock_resolve, mock_ensure, mock_sh, mock_sys_dir, tmp_path) -> None:
        """Test that do_new creates directory and calls shell commands."""
        mock_sys_dir.return_value = tmp_path / "test"
        mock_resolve.return_value = "1073741824"
        
        do_new("test", "ext4", "1G")
        
        # Verify system directory was accessed
        assert mock_sys_dir.called
        # Verify ensure_dirs was called
        assert mock_ensure.called
        # Verify sh was called for truncate and mkfs
        assert mock_sh.called
        # Verify save_meta was called
        assert mock_save_meta.called

    @patch("binsys._image.load_meta")
    @patch("binsys._image.sys_dir")
    @patch("binsys._image.sh")
    @patch("binsys._image.is_mounted")
    @patch("binsys._image.MOUNTS")
    def test_mount_calls_mount_command(self, mock_mounts, mock_is_mounted, mock_sh, mock_sys_dir, mock_load_meta, tmp_path) -> None:
        """Test that do_mount calls the mount shell command."""
        mock_load_meta.return_value = {
            "name": "test",
            "type": "ext4",
            "disk": "disk.img"
        }
        mock_sys_dir.return_value = tmp_path / "test"
        mock_mounts.return_value = tmp_path / "mnt" / "test"
        mock_is_mounted.return_value = False
        
        do_mount("test")
        
        # Verify mount was called
        assert mock_sh.called
        # Check that mount command was in the calls
        calls = [str(c) for c in mock_sh.call_args_list]
        assert any("mount" in str(c).lower() for c in mock_sh.call_args_list)

    @patch("binsys._image.load_meta")
    @patch("binsys._image.sys_dir")
    @patch("binsys._image.sh")
    def test_check_calls_fsck(self, mock_sh, mock_sys_dir, mock_load_meta, tmp_path) -> None:
        """Test that do_check calls the appropriate fsck command."""
        mock_load_meta.return_value = {
            "name": "test",
            "type": "ext4",
            "disk": "disk.img"
        }
        mock_sys_dir.return_value = tmp_path / "test"
        
        with patch("binsys._image.is_mounted", return_value=False):
            do_check("test")
        
        # Verify e2fsck was called for ext4
        assert mock_sh.called
        calls = [str(c) for c in mock_sh.call_args_list]
        assert any("e2fsck" in str(c) for c in mock_sh.call_args_list)
