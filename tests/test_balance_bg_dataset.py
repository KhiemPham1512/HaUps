import importlib.util
from pathlib import Path

LAB_CODE = Path(__file__).resolve().parents[1] / "lab" / "outpainting"
spec = importlib.util.spec_from_file_location("balance_bg_dataset", LAB_CODE / "balance_bg_dataset.py")
bal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bal)


def test_hue_bucket():
    assert bal.hue_bucket((128, 128, 128)) == "grey"
    assert bal.hue_bucket((150, 60, 220)) == "purple"
    assert bal.hue_bucket((40, 90, 230)) == "blue"


def test_select_drops_flattest_caps_hues_and_doubles_non_sky():
    items = [{"file": f"p{i:03}", "detail": 100 - i * 0.1, "hue": "purple", "sky": True} for i in range(200)]
    items += [{"file": "street", "detail": 99.95, "hue": "grey", "sky": False}]
    items += [{"file": "flat", "detail": 0.1, "hue": "blue", "sky": True}]
    chosen = bal.select(items)
    assert "flat" not in chosen
    assert chosen.count("street") == 2
    assert sum(name.startswith("p") for name in chosen) == bal.HUE_CAP
    assert "p000" in chosen and "p150" not in chosen
