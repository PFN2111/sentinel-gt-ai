"""Bootstrap helpers: hardlink a models tree and write starter files without downloading."""

from __future__ import annotations

from sentinel.bootstrap import link_tree, write_knowledge, write_settings


def test_link_tree_hardlinks_and_keeps_existing(tmp_path):
    src = tmp_path / "src"
    (src / "nested").mkdir(parents=True)
    (src / "nested" / "a.txt").write_text("alpha", encoding="utf-8")
    (src / "b.txt").write_text("beta", encoding="utf-8")
    dest = tmp_path / "models"
    (dest / "nested").mkdir(parents=True)
    (dest / "nested" / "a.txt").write_text("keep me", encoding="utf-8")
    counts = link_tree(src, dest)
    assert counts["kept"] == 1
    assert counts["hardlink"] + counts["copy"] == 1
    assert (dest / "nested" / "a.txt").read_text(encoding="utf-8") == "keep me"
    assert (dest / "b.txt").read_text(encoding="utf-8") == "beta"


def test_write_knowledge_and_settings_do_not_overwrite(tmp_path):
    docs = {"gt_overview.md": "# Overview\n", "process_map_vibration.md": "# Vibration\n"}
    written = write_knowledge(tmp_path, docs)
    assert set(written) == set(docs)
    (tmp_path / "knowledge" / "gt_overview.md").write_text("local edit\n", encoding="utf-8")
    again = write_knowledge(tmp_path, docs)
    assert again == []
    # the edited file stays; the second call writes nothing that already exists
    assert (tmp_path / "knowledge" / "gt_overview.md").read_text(encoding="utf-8") == "local edit\n"
    assert write_settings(tmp_path, '{"device": "cpu"}\n') == "created"
    assert write_settings(tmp_path, '{"device": "gpu"}\n') == "kept"
    assert "cpu" in (tmp_path / "settings.local.json").read_text(encoding="utf-8")
