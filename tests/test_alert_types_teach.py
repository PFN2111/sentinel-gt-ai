"""Teach alert types: one version per save, alias move, and a stats coldstart drill.

The drill does not load channel-independent weights and does not score the held-out set.
Classes differ by shape (a sine against a ramp). A level offset collapses under the
per-channel L2 norm, so it would not teach the index anything.
"""

from __future__ import annotations

import numpy as np
import pytest

from sentinel.config import load_settings
from sentinel.e2e.labels import LabelStore, Metrics, RetrainWorker
from sentinel.e2e.model import ProfileCache
from sentinel.e2e.pipeline import analyze_case, label_case
from sentinel.e2e.taxonomy import load_taxonomy
from sentinel.e2e.teach import TeachRegistry, move_alias, save_alert_type
from sentinel.e2e.train import load_profile_model, retrain_profile
from sentinel.profiles import alert_type_for_name, load_profiles, resolve_alert_type

HEATER = "Fuel gas heater outlet low"


def _settings(tmp_path):
    return load_settings(
        {
            "data_dir": tmp_path / "data",
            "use_llm": False,
            "use_tspulse": False,
            "embedder": "stats",
            "e2e_mode": "peralert",
            "e2e_retrain_every": 1000,
            "e2e_model_dir": tmp_path / "models",
            "profiles_dir": tmp_path / "profiles",
            "knowledge_dir": tmp_path / "knowledge",
            "residual_mode": "robust_z",
        }
    )


def _bind(monkeypatch, settings) -> None:
    """Point the CLI at the same temp folders the in-process settings use."""
    monkeypatch.setenv("SENTINEL_DATA_DIR", str(settings.data_dir))
    monkeypatch.setenv("SENTINEL_PROFILES_DIR", str(settings.profiles_dir))
    monkeypatch.setenv("SENTINEL_E2E_MODEL_DIR", str(settings.e2e_model_dir))
    monkeypatch.setenv("SENTINEL_KNOWLEDGE_DIR", str(settings.knowledge_dir))
    monkeypatch.setenv("SENTINEL_EMBEDDER", "stats")
    monkeypatch.setenv("SENTINEL_E2E_MODE", "peralert")
    monkeypatch.setenv("SENTINEL_USE_LLM", "0")
    monkeypatch.setenv("SENTINEL_USE_TSPULSE", "0")
    monkeypatch.setenv("SENTINEL_RESIDUAL_MODE", "robust_z")
    monkeypatch.setenv("SENTINEL_RETRAIN_EVERY", "1000")


def _heater_csv(kind: str, seed: int = 0, n: int = 240) -> bytes:
    """One historian file. ``normal`` oscillates; a fault is a ramp that crosses 2."""
    rng = np.random.default_rng(seed + (0 if kind == "normal" else 1000))
    wave = np.sin(np.linspace(0.0, 8.0 * np.pi, n))
    if kind == "normal":
        heater = wave + 0.02 * rng.normal(size=n)
    else:
        heater = np.linspace(-6.0, 6.0, n) + 0.02 * rng.normal(size=n)
    fuel = 20.0 + wave
    flow = 1.2 + 0.05 * wave
    lines = ["timestamp,load_MW,heater_outlet_C,fuel_gas_C,fuel_flow_kg_s"]
    for i in range(n):
        stamp = f"2024-01-01 {i // 60:02d}:{i % 60:02d}"
        lines.append(f"{stamp},80.0,{heater[i]:.5f},{fuel[i]:.5f},{flow[i]:.5f}")
    return ("\n".join(lines) + "\n").encode("utf-8")


class _Engine:
    """The pieces analyze and label touch. No encoder weights."""

    def __init__(self, settings):
        self.settings = settings
        self.labels = LabelStore(settings)
        self.metrics = Metrics(settings)
        self.profile_cache = ProfileCache()
        self.worker = RetrainWorker()
        self.cal = type("Cal", (), {"low_confidence": 0.6, "residual_thresholds": None})()
        self.registry = type("Reg", (), {"current_encoder": staticmethod(lambda: "e0000")})()

    def taxonomy(self):
        return load_taxonomy(self.settings.data_dir)


def _engines(monkeypatch):
    made = {}

    def get_engine(settings):
        key = str(settings.data_dir)
        if key not in made:
            made[key] = _Engine(settings)
        return made[key]

    monkeypatch.setattr("sentinel.e2e.pipeline.get_engine", get_engine)
    return made


def _actions(settings):
    return [row.get("action") for row in TeachRegistry(settings).versions()]


def test_one_teach_version_rolls_back_the_alert_type_and_its_names(tmp_path):
    settings = _settings(tmp_path)
    out = save_alert_type(
        settings,
        HEATER,
        csv_bytes=_heater_csv("normal"),
        signatures=["normal", "heater_fault"],
        no_document=True,
    )
    assert _actions(settings).count("save alert type") == 1
    reg = TeachRegistry(settings)
    assert reg.rollback() == "k0000"
    book = load_profiles(settings)
    assert alert_type_for_name(HEATER, book) is None
    assert not (settings.profiles_dir / f"{out['id']}.json").is_file()


def test_moved_alias_is_in_exactly_one_file(tmp_path):
    settings = _settings(tmp_path)
    sample = _heater_csv("normal")
    left = save_alert_type(
        settings,
        "Left heater stack",
        aliases=["Stack outlet low"],
        csv_bytes=sample,
        signatures=["normal", "heater_fault"],
        no_document=True,
    )
    right = save_alert_type(
        settings,
        "Right heater stack",
        csv_bytes=sample,
        signatures=["normal", "heater_fault"],
        no_document=True,
    )
    move_alias(settings, left["id"], right["id"], "Stack outlet low")
    hits = []
    for path in sorted(settings.profiles_dir.glob("*.json")):
        if path.name.startswith("_"):
            continue
        if "Stack outlet low" in path.read_text(encoding="utf-8"):
            hits.append(path.name)
    assert hits == [f"{right['id']}.json"]
    book = load_profiles(settings)
    assert alert_type_for_name("Stack outlet low", book).id == right["id"]
    assert "Stack outlet low" not in book[left["id"]].alert_aliases()


def test_alert_types_help_text(capsys):
    from sentinel.cli import main

    with pytest.raises(SystemExit) as new_help:
        main(["alert-types", "new", "--help"])
    assert new_help.value.code == 0
    assert "--sample" in capsys.readouterr().out
    with pytest.raises(SystemExit) as top_help:
        main(["--help"])
    assert top_help.value.code == 0
    assert "alert-types" in capsys.readouterr().out
    with pytest.raises(SystemExit) as train_help:
        main(["train", "--help"])
    assert train_help.value.code == 0
    text = capsys.readouterr().out
    assert "--alert-type" in text and "--profile" in text and "--all" in text


def test_unknown_name_prints_the_fix_and_a_label_is_not_a_teach_version(tmp_path, monkeypatch, capsys):
    from sentinel.cli import main

    settings = _settings(tmp_path)
    _bind(monkeypatch, settings)
    _engines(monkeypatch)
    csv = tmp_path / "heater.csv"
    csv.write_bytes(_heater_csv("normal", seed=1))
    code = main(
        ["--data-dir", str(settings.data_dir), "e2e", "run", "--csv", str(csv), "--analytic", HEATER, "--no-rules"]
    )
    out = capsys.readouterr().out
    assert code == 0, out
    assert f"UNKNOWN ALERT TYPE '{HEATER}'" in out
    assert "alert-types add-alias --type <id>" in out
    assert "alert-types new --name" in out and "--sample" in out
    assert "NEEDS_HUMAN\tTrue" in out
    case_id = next(line.split("\t")[1] for line in out.splitlines() if line.startswith("CASE\t"))
    assert "save alert type" not in _actions(settings)
    code = main(
        [
            "--data-dir",
            str(settings.data_dir),
            "alert-types",
            "new",
            "--name",
            HEATER,
            "--from-case",
            case_id,
            "--signatures",
            "normal,heater_fault",
        ]
    )
    made = capsys.readouterr().out
    assert code == 0, made
    assert resolve_alert_type(HEATER, settings=settings).profile is not None
    assert _actions(settings).count("save alert type") == 1
    code = main(
        ["--data-dir", str(settings.data_dir), "e2e", "run", "--csv", str(csv), "--analytic", HEATER, "--no-rules"]
    )
    again = capsys.readouterr().out
    assert code == 0, again
    assert "UNKNOWN ALERT TYPE" not in again
    labeled = next(line.split("\t")[1] for line in again.splitlines() if line.startswith("CASE\t"))
    code = main(
        [
            "--data-dir",
            str(settings.data_dir),
            "e2e",
            "label",
            "--case",
            labeled,
            "--signature",
            "heater_fault",
            "--action",
            "escalate",
        ]
    )
    assert code == 0, capsys.readouterr().out
    assert _actions(settings).count("save alert type") == 1


def test_coldstart_drill_reaches_a_head_on_stats(tmp_path, monkeypatch):
    from sentinel.cli import main

    settings = _settings(tmp_path)
    _bind(monkeypatch, settings)
    assert resolve_alert_type(HEATER, settings=settings).how == "unknown"
    sample = tmp_path / "sample.csv"
    sample.write_bytes(_heater_csv("normal", seed=2))
    code = main(
        [
            "--data-dir",
            str(settings.data_dir),
            "alert-types",
            "new",
            "--name",
            HEATER,
            "--sample",
            str(sample),
            "--signatures",
            "normal,heater_fault",
        ]
    )
    assert code == 0
    choice = resolve_alert_type(HEATER, settings=settings)
    assert choice.profile is not None
    pid = choice.profile.id
    before = _actions(settings)
    eng = _Engine(settings)

    def teach(kind: str, seed: int) -> None:
        report = analyze_case(
            _heater_csv(kind, seed=seed),
            HEATER,
            "",
            settings=settings,
            engine=eng,
            use_llm=False,
            run_rules=False,
            source_name=f"{kind}-{seed}.csv",
        )
        assert report["profile"]["id"] == pid
        action = "monitor_only" if kind == "normal" else "escalate"
        label_case(settings, report["case_id"], kind, action, engine=eng, auto_retrain=False)

    for i, kind in enumerate(["normal", "heater_fault"] * 3):
        teach(kind, 10 + i)
    assert _actions(settings) == before
    small = retrain_profile(settings, pid, trigger="six labels")
    assert small["promoted"] is True and small["gate"]["status"] == "none"
    assert load_profile_model(settings, pid).tier() == "knn"
    for i, kind in enumerate(["normal", "heater_fault"] * 7):
        teach(kind, 40 + i)
    large = retrain_profile(settings, pid, trigger="twenty labels")
    assert large["promoted"] is True and large["gate"]["status"] == "pass", large["gate"]
    assert load_profile_model(settings, pid).tier() == "head"
    assert _actions(settings).count("save alert type") == 1
