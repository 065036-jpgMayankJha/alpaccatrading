import json
from pathlib import Path


def test_universe_has_40_unique_symbols_split_12_and_28():
    config_path = Path(__file__).parents[1] / "config" / "universe.json"
    universe = json.loads(config_path.read_text(encoding="utf-8"))
    growth = universe["growth_satellites"]
    diversified = [
        symbol
        for group in universe["diversified"].values()
        for symbol in group
    ]
    combined = growth + diversified
    assert len(growth) == 12
    assert len(diversified) == 28
    assert len(combined) == 40
    assert len(set(combined)) == 40
    assert all(symbol.isalpha() for symbol in combined)
