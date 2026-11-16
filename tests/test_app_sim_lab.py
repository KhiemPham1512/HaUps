import importlib.util
from pathlib import Path

LAB_CODE = Path(__file__).resolve().parents[1] / "lab" / "outpainting"
spec = importlib.util.spec_from_file_location("app_sim_lab", LAB_CODE / "app_sim_lab.py")
sim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sim)


def test_first_unflagged_stops_early():
    candidates = [(42, "img42"), (123, "img123"), (7, "img7")]
    metrics = {
        "img42": (0.12, 0.01),
        "img123": (0.02, 0.00),
        "img7": (0.01, 0.00),
    }
    seed, image, flagged, tried = sim.choose(candidates, lambda img: metrics[img])
    assert seed == 42
    assert image == "img42"
    assert flagged is False
    assert [t[0] for t in tried] == [42]


def test_all_flagged_picks_lowest_score():
    candidates = [(42, "img42"), (123, "img123"), (7, "img7")]
    metrics = {
        "img42": (0.65, 0.01),
        "img123": (0.42, 0.15),
        "img7": (0.51, 0.02),
    }
    seed, image, flagged, tried = sim.choose(candidates, lambda img: metrics[img])
    assert seed == 123
    assert image == "img123"
    assert flagged is True
    assert [t[0] for t in tried] == [42, 123, 7]


def test_unflagged_with_higher_score_beats_flagged_with_lower_score():
    candidates = [(42, "img42"), (123, "img123"), (7, "img7")]
    metrics = {
        "img42": (0.05, 0.10),  # flagged via detached, low far-zone score
        "img123": (0.30, 0.01),  # unflagged, higher far-zone score
        "img7": (0.01, 0.00),
    }
    seed, image, flagged, tried = sim.choose(candidates, lambda img: metrics[img])
    assert seed == 123
    assert image == "img123"
    assert flagged is False
    assert [t[0] for t in tried] == [42, 123]


def test_detached_alone_flags():
    candidates = [(42, "img42"), (123, "img123")]
    metrics = {
        "img42": (0.0, 0.05),
        "img123": (0.10, 0.08),
    }
    seed, image, flagged, tried = sim.choose(candidates, lambda img: metrics[img])
    assert seed == 42
    assert image == "img42"
    assert flagged is True
    assert [t[0] for t in tried] == [42, 123]
