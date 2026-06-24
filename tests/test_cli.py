"""Tests for CLI argument parsing and command dispatch."""

from __future__ import annotations

from binsys.cli import build_parser


def test_build_parser() -> None:
    p = build_parser()
    assert p.prog == "binsys"


def test_parser_default_tui() -> None:
    p = build_parser()
    args = p.parse_args([])
    assert args.command is None


def test_parser_new() -> None:
    p = build_parser()
    args = p.parse_args(["new", "test-system"])
    assert args.command == "new"
    assert args.name == "test-system"
    assert args.type == "ext4"
    assert args.size == "1G"
    assert args.encrypt is False


def test_parser_new_with_options() -> None:
    p = build_parser()
    args = p.parse_args(["new", "test", "--type", "overlay", "--size", "4G", "--encrypt"])
    assert args.name == "test"
    assert args.type == "overlay"
    assert args.size == "4G"
    assert args.encrypt is True


def test_parser_new_save_size() -> None:
    p = build_parser()
    args = p.parse_args(["new", "test", "--type", "overlay", "--save-size", "1G"])
    assert args.save_size == "1G"


def test_parser_list() -> None:
    p = build_parser()
    args = p.parse_args(["list"])
    assert args.command == "list"


def test_parser_ls_alias() -> None:
    p = build_parser()
    args = p.parse_args(["ls"])
    assert args.command == "ls"


def test_parser_run() -> None:
    p = build_parser()
    args = p.parse_args(["run", "my-system"])
    assert args.command == "run"
    assert args.name == "my-system"
    assert args.no_kvm is False
    assert args.memory == "2048"


def test_parser_run_no_kvm() -> None:
    p = build_parser()
    args = p.parse_args(["run", "my-system", "--no-kvm"])
    assert args.no_kvm is True


def test_parser_run_gdb() -> None:
    p = build_parser()
    args = p.parse_args(["run", "my-system", "-g"])
    assert args.gdb is True


def test_parser_delete() -> None:
    p = build_parser()
    args = p.parse_args(["delete", "old-system"])
    assert args.command == "delete"
    assert args.name == "old-system"


def test_parser_clone() -> None:
    p = build_parser()
    args = p.parse_args(["clone", "src", "dst"])
    assert args.src == "src"
    assert args.dst == "dst"


def test_parser_clone_default_dst() -> None:
    p = build_parser()
    args = p.parse_args(["clone", "src"])
    assert args.src == "src"
    assert args.dst is None


def test_parser_rename() -> None:
    p = build_parser()
    args = p.parse_args(["rename", "old", "new"])
    assert args.old == "old"
    assert args.new == "new"


def test_parser_export() -> None:
    p = build_parser()
    args = p.parse_args(["export", "my-system"])
    assert args.command == "export"
    assert args.name == "my-system"


def test_parser_mount() -> None:
    p = build_parser()
    args = p.parse_args(["mount", "my-system"])
    assert args.name == "my-system"


def test_parser_umount() -> None:
    p = build_parser()
    args = p.parse_args(["umount", "my-system"])
    assert args.name == "my-system"


def test_parser_info() -> None:
    p = build_parser()
    args = p.parse_args(["info", "my-system"])
    assert args.name == "my-system"


def test_parser_resize() -> None:
    p = build_parser()
    args = p.parse_args(["resize", "my-system", "4G"])
    assert args.name == "my-system"
    assert args.size == "4G"


def test_parser_encrypt() -> None:
    p = build_parser()
    args = p.parse_args(["encrypt", "my-system", "--hash", "sha512"])
    assert args.hash_algo == "sha512"


def test_parser_import() -> None:
    p = build_parser()
    args = p.parse_args(["import", "/path/to/image.img", "my-system"])
    assert args.src == "/path/to/image.img"
    assert args.name == "my-system"


def test_parser_hash() -> None:
    p = build_parser()
    args = p.parse_args(["hash", "my-system", "--algo", "md5"])
    assert args.algo == "md5"


def test_parser_frugal_snapshot() -> None:
    p = build_parser()
    args = p.parse_args(["frugal", "snapshot", "my-system", "--label", "before-update"])
    assert args.frugal_cmd == "snapshot"
    assert args.name == "my-system"
    assert args.label == "before-update"


def test_parser_frugal_list() -> None:
    p = build_parser()
    args = p.parse_args(["frugal", "list", "my-system"])
    assert args.frugal_cmd == "list"
    assert args.name == "my-system"


def test_parser_frugal_rollback() -> None:
    p = build_parser()
    args = p.parse_args(["frugal", "rollback", "my-system", "save_20240101.img"])
    assert args.frugal_cmd == "rollback"
    assert args.name == "my-system"
    assert args.snap == "save_20240101.img"


def test_parser_frugal_merge() -> None:
    p = build_parser()
    args = p.parse_args(["frugal", "merge", "my-system"])
    assert args.frugal_cmd == "merge"
    assert args.name == "my-system"


def test_parser_iso_from_system() -> None:
    p = build_parser()
    args = p.parse_args(["iso", "my-system"])
    assert args.source == "my-system"


def test_parser_iso_from_dir() -> None:
    p = build_parser()
    args = p.parse_args(["iso", "/some/dir", "--output", "/tmp/test.iso", "--name", "VOLUME", "--bootable"])
    assert args.source == "/some/dir"
    assert args.output == "/tmp/test.iso"
    assert args.name == "VOLUME"
    assert args.bootable is True


def test_parser_boot() -> None:
    p = build_parser()
    args = p.parse_args(["boot", "my-system", "--size", "8G", "--esp-size", "1G"])
    assert args.name == "my-system"
    assert args.size == "8G"
    assert args.esp_size == "1G"


def test_parser_protect() -> None:
    p = build_parser()
    args = p.parse_args(["protect", "my-system"])
    assert args.name == "my-system"
    assert args.password is None


def test_parser_protect_with_password() -> None:
    p = build_parser()
    args = p.parse_args(["protect", "my-system", "--password", "hunter2"])
    assert args.password == "hunter2"


def test_parser_unprotect() -> None:
    p = build_parser()
    args = p.parse_args(["unprotect", "my-system"])
    assert args.name == "my-system"


def test_parser_auth() -> None:
    p = build_parser()
    args = p.parse_args(["auth", "my-system", "--password", "hunter2"])
    assert args.password == "hunter2"


def test_parser_app_lock() -> None:
    p = build_parser()
    args = p.parse_args(["app-lock", "my-system"])
    assert args.name == "my-system"


def test_parser_wizard_list() -> None:
    p = build_parser()
    args = p.parse_args(["wizard", "--list"])
    assert args.list is True


def test_parser_wizard_name() -> None:
    p = build_parser()
    args = p.parse_args(["wizard", "build-frugal"])
    assert args.name == "build-frugal"


def test_parser_check() -> None:
    p = build_parser()
    args = p.parse_args(["check", "my-system"])
    assert args.name == "my-system"


def test_parser_snap() -> None:
    p = build_parser()
    args = p.parse_args(["snap", "my-system"])
    assert args.name == "my-system"


def test_parser_layouts() -> None:
    p = build_parser()
    args = p.parse_args(["layouts", "my-system"])
    assert args.name == "my-system"


def test_parser_shell() -> None:
    p = build_parser()
    args = p.parse_args(["shell", "my-system"])
    assert args.name == "my-system"


def test_parser_verbose() -> None:
    p = build_parser()
    args = p.parse_args(["--verbose", "list"])
    assert args.verbose is True


def test_parser_version() -> None:
    p = build_parser()
    try:
        p.parse_args(["--version"])
        assert False, "Should have exited"
    except SystemExit:
        pass
