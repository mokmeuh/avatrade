import json

import pytest

from ava_mt5.data import build_profile, generate_synthetic


@pytest.fixture(scope="session")
def synth_dir(tmp_path_factory):
    """Petit jeu de données synthétiques (30 jours, un symbole) partagé par les tests."""
    d = tmp_path_factory.mktemp("synth")
    generate_synthetic(d, days=30, symbols=("CrudeOIL",), tick_days=14)
    (d / "spread_profile.json").write_text(json.dumps(build_profile(d)), encoding="utf-8")
    return d
