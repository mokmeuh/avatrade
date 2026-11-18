import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from ava_mt5 import data
from ava_mt5.core import Broker, Settings
from fake_mt5 import FakeMT5


def test_heures_evenements_heure_d_ete():
    ev = lambda kind, a, b: [str(x) for x in data.event_times_utc(  # noqa: E731
        pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC"), kind)]
    # EIA mercredi 10:30 NY : UTC-5 avant le 8 mars 2026, UTC-4 après
    assert ev("eia", "2026-03-04", "2026-03-04 23:00") == ["2026-03-04 15:30:00+00:00"]
    assert ev("eia", "2026-03-11", "2026-03-11 23:00") == ["2026-03-11 14:30:00+00:00"]
    # API mardi 16:30 NY
    assert ev("api", "2026-03-03", "2026-03-03 23:59") == ["2026-03-03 21:30:00+00:00"]


def test_synthetique_reproductible_et_profil(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    data.generate_synthetic(a, days=12, symbols=("CrudeOIL",), tick_days=7)
    data.generate_synthetic(b, days=12, symbols=("CrudeOIL",), tick_days=7)
    h = lambda p: hashlib.md5((p / "CrudeOIL_ticks.csv.gz").read_bytes()[10:]).hexdigest()  # noqa: E731
    assert h(a) == h(b)
    prof = data.profile_command(a)
    p = prof["CrudeOIL"]
    assert p["source"] == "ticks" and p["point"] == 0.01
    assert set(p["hourly_median"]) == {str(h) for h in range(24)} == set(p["hourly_p90"])
    assert all(p["hourly_p90"][k] >= p["hourly_median"][k] for k in p["hourly_median"])
    for kind in ("eia", "api"):
        e = p[kind]
        assert set(e) == {"before_5m", "minute_0_to_3", "normal"} and len(e["minute_0_to_3"]) == 4
        assert e["minute_0_to_3"][0] > e["normal"]          # le spread s'élargit à la publication
    assert json.loads((a / "spread_profile.json").read_text())["CrudeOIL"] == p


def test_profil_sans_ticks_utilise_les_barres(tmp_path):
    data.generate_synthetic(tmp_path, days=8, symbols=("CrudeOIL",), tick_days=3)
    (tmp_path / "CrudeOIL_ticks.csv.gz").unlink()
    assert data.build_profile(tmp_path)["CrudeOIL"]["source"] == "bars"


class FakeCollectMT5(FakeMT5):
    TIMEFRAME_M1 = 1
    TIMEFRAME_M5 = 5
    COPY_TICKS_ALL = -1
    N = 120

    def __init__(self):
        super().__init__()
        self.rates = [{"time": 1_700_000_000 + 60 * i, "open": 70.0, "high": 70.1, "low": 69.9,
                       "close": 70.05, "tick_volume": 5, "spread": 3, "real_volume": 0}
                      for i in range(self.N)]
        self.tick_calls = []

    def copy_rates_from_pos(self, symbol, tf, pos, count):
        end = len(self.rates) - pos
        return self.rates[max(0, end - count):end] if end > 0 else None

    def copy_ticks_range(self, symbol, a, b, flags):
        self.tick_calls.append((a, b))
        return [{"time": 1, "bid": 70.0, "ask": 70.03, "last": 0.0, "volume": 0, "flags": 6,
                 "time_msc": int(a.timestamp() * 1000) + 5, "volume_real": 0.0}]


def test_collecte_conversion_utc_et_symboles_manquants(tmp_path, capsys):
    mt5 = FakeCollectMT5()
    broker = Broker(mt5, Settings())
    broker.connect()
    meta = data.collect(broker, ["CrudeOIL", "Inconnu"], tmp_path, weeks=1, server_offset=2, block=50)
    err = capsys.readouterr().err
    assert "Inconnu" in err and "CrudeOIL" not in meta.get("missing", [])
    m1 = pd.read_csv(tmp_path / "CrudeOIL_M1.csv.gz")
    assert len(m1) == 120 and list(m1.columns) == data.BAR_COLUMNS
    assert m1["time"].iloc[0] == 1_700_000_000 - 2 * 3600      # heure serveur -> UTC
    assert m1["time"].is_monotonic_increasing
    assert (tmp_path / "CrudeOIL_M5.csv.gz").exists() and (tmp_path / "CrudeOIL_ticks.csv.gz").exists()
    assert len(mt5.tick_calls) == 7                              # une tranche par jour
    assert meta["server_offset_h"] == 2 and meta["synthetic"] is False
    assert meta["symbols"]["CrudeOIL"]["contract_size"] == 1000.0
    assert not (tmp_path / "Inconnu_M1.csv.gz").exists()


def test_estimation_decalage_serveur():
    mt5 = FakeMT5()  # tick à 1_700_000_000
    assert data.estimate_server_offset(mt5, "CrudeOIL", now=1_700_000_000 - 3 * 3600 + 20) == 3
    with pytest.raises(RuntimeError, match="server-offset"):
        data.estimate_server_offset(mt5, "CrudeOIL", now=1_700_000_000 - 2 * 86400 - 1800)
