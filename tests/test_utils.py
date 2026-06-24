from __future__ import annotations

import json
from pathlib import Path

from utils import (
    Colors,
    confirm,
    deterministic_id,
    preview_and_apply_renames,
    prompt,
    write_undo_log,
)


def test_deterministic_id() -> None:
    result = deterministic_id("test", "foo", "bar")
    assert result.startswith("test_")
    assert len(result) == 5 + 12  # prefix + underscore + 12 hex chars
    # Same inputs should produce same output
    assert deterministic_id("test", "foo", "bar") == result
    # Different inputs should produce different output
    assert deterministic_id("test", "foo", "baz") != result


def test_deterministic_id_single_part() -> None:
    result = deterministic_id("id", "hello")
    assert result.startswith("id_")
    assert len(result) == 15  # 2 + 1 + 12


def test_colors_ansi() -> None:
    assert Colors.RESET == "\033[0m"
    assert Colors.BOLD == "\033[1m"
    assert Colors.RED == "\033[91m"
    assert Colors.GREEN == "\033[92m"
    assert Colors.YELLOW == "\033[93m"
    assert Colors.BLUE == "\033[94m"


def test_confirm_default_false(monkeypatch) -> None:
    monkeypatch.setattr("builtins.input", lambda _: "")
    assert confirm("Proceed?") is False


def test_confirm_default_true(monkeypatch) -> None:
    monkeypatch.setattr("builtins.input", lambda _: "")
    assert confirm("Proceed?", default=True) is True


def test_confirm_yes(monkeypatch) -> None:
    monkeypatch.setattr("builtins.input", lambda _: "y")
    assert confirm("Proceed?") is True


def test_confirm_no(monkeypatch) -> None:
    monkeypatch.setattr("builtins.input", lambda _: "n")
    assert confirm("Proceed?") is False


def test_prompt_default(monkeypatch) -> None:
    monkeypatch.setattr("builtins.input", lambda _: "")
    assert prompt("Name", default="default_name") == "default_name"


def test_prompt_custom(monkeypatch) -> None:
    monkeypatch.setattr("builtins.input", lambda _: "my-value")
    result = prompt("Name", default="fallback")
    assert result == "my-value"


def test_write_undo_log(tmp_path: Path) -> None:
    log = tmp_path / "undo.json"
    ops = [("a.txt", "b.txt"), ("c.txt", "d.txt")]
    write_undo_log(log, ops)
    assert log.exists()
    data = json.loads(log.read_text())
    assert data["ops"] == [{"src": "a.txt", "dst": "b.txt"}, {"src": "c.txt", "dst": "d.txt"}]
    assert "created" in data


def test_preview_and_apply_renames_dry_run(tmp_path: Path) -> None:
    src = tmp_path / "old.txt"
    dst = tmp_path / "new.txt"
    src.write_text("content")
    ops = [(src, dst)]
    preview_and_apply_renames(ops, dry_run=True)
    assert src.exists()
    assert not dst.exists()


def test_preview_and_apply_renames_apply(tmp_path: Path) -> None:
    src = tmp_path / "old.txt"
    dst = tmp_path / "new.txt"
    src.write_text("hello")
    ops = [(src, dst)]
    preview_and_apply_renames(ops, dry_run=False)
    assert not src.exists()
    assert dst.exists()
    assert dst.read_text() == "hello"


def test_preview_and_apply_renames_missing_source(tmp_path: Path, capsys) -> None:
    src = tmp_path / "missing.txt"
    dst = tmp_path / "new.txt"
    ops = [(src, dst)]
    preview_and_apply_renames(ops, dry_run=False)
    captured = capsys.readouterr()
    assert "Skipping missing" in captured.out


def test_preview_and_apply_renames_existing_dst(tmp_path: Path, capsys) -> None:
    src = tmp_path / "old.txt"
    dst = tmp_path / "new.txt"
    src.write_text("source")
    dst.write_text("existing")
    ops = [(src, dst)]
    preview_and_apply_renames(ops, dry_run=False)
    captured = capsys.readouterr()
    assert "Skipping existing" in captured.out


def test_preview_and_apply_renames_undo_log(tmp_path: Path) -> None:
    src = tmp_path / "old.txt"
    dst = tmp_path / "new.txt"
    src.write_text("content")
    undo = tmp_path / "undo.json"
    ops = [(src, dst)]
    preview_and_apply_renames(ops, dry_run=False, undo_log=undo)
    assert undo.exists()
    data = json.loads(undo.read_text())
    assert data["ops"] == [{"src": str(dst), "dst": str(src)}]
