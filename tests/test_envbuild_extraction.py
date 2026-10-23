"""Regressions for code and asset closure found by the Harbor GPU matrix."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from envbuild import extract


def write(path, text=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_real_locomanip_retains_shared_packing_code():
    assert "packing" in extract._suite_closure("locomanip")


def test_transitive_suite_closure_drops_all_privileged_code(tmp_path, monkeypatch):
    source = tmp_path / "source/robobench"
    monkeypatch.setattr(extract, "SRC", source)
    write(source / "suites/first/scenes/example.py", "from robobench.suites.second import shared\n")
    write(source / "suites/second/shared.py", "from ..third import base\n")
    write(source / "suites/third/base.py", "from robobench.suites.first.scenes import example\n")
    for suite in ("first", "second", "third", "unrelated"):
        write(source / f"suites/{suite}/__init__.py")
        write(source / f"suites/{suite}/grader/secret.py", "from robobench.suites.unrelated import secret\n")
        write(source / f"suites/{suite}/smokes/solution.py")
        write(source / f"suites/{suite}/assets/unused.bin")
    target = tmp_path / "target"
    extract.minimal_tree(target, "first", "example", "null")
    suites = target / "robobench/suites"
    assert {p.name for p in suites.iterdir()} == {"first", "second", "third"}
    assert (suites / "second/shared.py").is_file()
    assert (suites / "third/base.py").is_file()
    assert not any(p.name in {"grader", "smokes", "assets"} for p in suites.rglob("*"))


def test_missing_manifest_merges_into_downloaded_asset_directory(tmp_path, monkeypatch):
    source = tmp_path / "source/robobench"
    monkeypatch.setattr(extract, "SRC", source)
    relative = Path("robobench/suites/cutting/assets/knife22r")
    write(source.parent / relative / "manifest.json", '{"planes_y": [0]}')
    write(source.parent / relative / "mesh.usd", "older source")
    target = tmp_path / "target"
    write(target / relative / "mesh.usd", "downloaded verified binary")
    missing = target / relative / "manifest.json"
    assert extract.copy_missing(target, [str(missing)]) == ["suites/cutting/assets/knife22r"]
    assert missing.read_text() == '{"planes_y": [0]}'
    assert (target / relative / "mesh.usd").read_text() == "downloaded verified binary"
    assert extract.copy_missing(target, [str(missing)]) == []
    assert extract.copy_missing(target, [str(target / relative / "absent.usd")]) == []
