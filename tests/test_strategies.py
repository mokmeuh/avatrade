import itertools

import numpy as np
import pandas as pd
import pytest

from ava_mt5.strategies import (STRATEGIES, Signal, add_tick_features, load_bars, load_ticks,
                                tick_imbalance_mode)

NY = "America/New_York"


def bars(rows, start, spread=3, tick_volume=10):
    idx = pd.date_range(start, periods=len(rows), freq="1min", tz="UTC")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)
    df["tick_volume"] = tick_volume
    df["spread"] = spread
    return df


def noise_rows(n, base=70.0, amp=0.1, seed=0):
    rng = np.random.default_rng(seed)
    out, last = [], base
    for _ in range(n):
        c = base + rng.uniform(-amp, amp)
        out.append((last, max(last, c) + 0.03, min(last, c) - 0.03, c))
        last = c
    return out


def test_chargement_csv(synth_dir):
    m1 = load_bars(synth_dir / "CrudeOIL_M1.csv.gz")
    assert str(m1.index.tz) == "UTC" and m1.index.is_monotonic_increasing
    assert {"open", "high", "low", "close", "tick_volume", "spread"} <= set(m1.columns)
    assert len(load_ticks(synth_dir / "CrudeOIL_ticks.csv.gz")) > 1000


def test_contrat_des_signaux(synth_dir):
    m1 = load_bars(synth_dir / "CrudeOIL_M1.csv.gz")
    m1 = add_tick_features(m1, load_ticks(synth_dir / "CrudeOIL_ticks.csv.gz"))
    for name, s in STRATEGIES.items():
        assert s.name == name
        combos = list(itertools.product(*s.grid.values()))
        assert 1 <= len(combos) <= 50
        for vals in combos:
            params = dict(zip(s.grid, vals))
            for g in (s.signals(m1, params),):
                for sg in g:
                    assert isinstance(sg, Signal) and sg.strategy == name
                    assert sg.side in ("buy", "sell") and sg.sl_dist > 0 and sg.tp_dist > 0
                    assert sg.tp_dist >= 0.5 * sg.sl_dist - 1e-12
                    assert sg.time.tzinfo is not None and sg.max_minutes > 0
                    assert sg.time.second == 0
                assert [x.time for x in g] == sorted(x.time for x in g)


@pytest.mark.parametrize("name", list(STRATEGIES))
def test_pas_de_biais_d_anticipation(synth_dir, name):
    """Truquer ou supprimer les barres futures ne change aucun signal déjà émis."""
    m1 = load_bars(synth_dir / "CrudeOIL_M1.csv.gz")
    m1 = add_tick_features(m1, load_ticks(synth_dir / "CrudeOIL_ticks.csv.gz"))
    s = STRATEGIES[name]
    full = s.signals(m1, {})
    assert full or name == "eia_momentum"
    cut = full[len(full) // 2].time if full else m1.index[len(m1) // 2]
    ref = {x for x in full if x.time <= cut}
    # 1) barres futures supprimées : la barre d'entrée (time == cut) n'est pas nécessaire
    trunc = m1[m1.index < cut]
    assert {x for x in s.signals(trunc, {}) if x.time <= cut} == ref
    # 2) barres futures remplacées par n'importe quoi
    fake = m1.copy()
    rng = np.random.default_rng(5)
    fut = fake.index >= cut
    for col in ("open", "high", "low", "close"):
        fake.loc[fut, col] = 50 + rng.random(fut.sum()) * 40
    fake.loc[fut, "tick_volume"] = rng.integers(1, 500, fut.sum())
    fake.loc[fut, "spread"] = rng.integers(1, 30, fut.sum())
    fake.loc[fut, "upticks"] = rng.integers(0, 20, fut.sum())
    assert {x for x in s.signals(fake, {}) if x.time <= cut} == ref


def test_orb_cassure_haussiere():
    # mardi 2026-03-03 : 9:00 New York = 14:00 UTC (heure d'hiver)
    start = pd.Timestamp("2026-03-03 12:30", tz="UTC")
    rows = noise_rows(90, amp=0.05)                       # 12:30 -> 14:00 : calme
    rows += [(70.0, 70.2, 69.9, 70.1)] * 15               # 14:00 -> 14:14 : range 69,9-70,2
    rows += [(70.1, 70.15, 70.05, 70.1)] * 5              # 14:15 -> 14:19 : dans le range
    rows += [(70.1, 70.40, 70.1, 70.35), (70.35, 70.5, 70.3, 70.4)] + [(70.4, 70.5, 70.3, 70.4)] * 10
    m1 = bars(rows, start)
    sigs = STRATEGIES["orb"].signals(m1, {"range_min": 15})
    assert [(x.side, x.time) for x in sigs] == [("buy", pd.Timestamp("2026-03-03 14:21", tz="UTC"))]
    assert sigs[0].sl_dist > 0 and sigs[0].tp_dist > sigs[0].sl_dist
    # le range n'est complet qu'à 9:15 NY : aucune entrée avant
    assert all(x.time.tz_convert(NY).hour * 60 + x.time.tz_convert(NY).minute >= 9 * 60 + 15 for x in sigs)


def test_orb_un_seul_trade_par_sens_et_par_jour():
    start = pd.Timestamp("2026-03-03 12:30", tz="UTC")
    rows = noise_rows(90, amp=0.05) + [(70.0, 70.2, 69.9, 70.1)] * 15
    rows += [(70.1, 70.6, 70.1, 70.5)] * 3 + [(70.5, 70.6, 70.4, 70.5)] * 20  # plusieurs clôtures > range
    sigs = STRATEGIES["orb"].signals(bars(rows, start), {"range_min": 15})
    assert len([x for x in sigs if x.side == "buy"]) == 1


def eia_day(day, spread_last=3):
    start = pd.Timestamp(f"{day} 14:00", tz="UTC")           # 9:00 NY, hiver
    rows = noise_rows(90, amp=0.05)                          # 14:00 -> 15:29
    rows += [(70.0, 70.3, 69.95, 70.25)]                     # 15:30 (10:30 NY) : impulsion haussière
    rows += [(70.25, 70.4, 70.2, 70.35)] * 10
    m1 = bars(rows, start)
    m1.iloc[90, m1.columns.get_loc("spread")] = spread_last
    return m1


def test_eia_momentum_suit_la_premiere_impulsion():
    m1 = eia_day("2026-03-04")  # mercredi
    sigs = STRATEGIES["eia_momentum"].signals(m1, {"delay": 1})
    assert [(x.side, x.time) for x in sigs] == [("buy", pd.Timestamp("2026-03-04 15:31", tz="UTC"))]
    s3 = STRATEGIES["eia_momentum"].signals(m1, {"delay": 3})
    assert s3 and s3[0].time == pd.Timestamp("2026-03-04 15:33", tz="UTC")


def test_eia_momentum_filtres_jour_et_spread():
    assert STRATEGIES["eia_momentum"].signals(eia_day("2026-03-05"), {"delay": 1}) == []  # jeudi
    large = eia_day("2026-03-04", spread_last=20)           # spread pas revenu à la normale
    assert STRATEGIES["eia_momentum"].signals(large, {"delay": 1}) == []


def test_eia_momentum_vente_sur_impulsion_baissiere():
    m1 = eia_day("2026-03-04")
    m1.iloc[90, m1.columns.get_loc("close")] = 69.7
    sigs = STRATEGIES["eia_momentum"].signals(m1, {"delay": 1})
    assert sigs and sigs[0].side == "sell"


def test_add_tick_features():
    start = pd.Timestamp("2026-03-03 15:00", tz="UTC")
    m1 = bars([(70, 70.1, 69.9, 70)] * 4, start)
    t0 = int(start.timestamp() * 1000) + 60_000          # ticks sur la 2e minute seulement
    mids = [70.00, 70.01, 70.00, 70.00, 70.02]
    ticks = pd.DataFrame({"time_msc": [t0 + i * 1000 for i in range(5)],
                          "bid": [m - 0.01 for m in mids], "ask": [m + 0.01 for m in mids],
                          "flags": 6})
    out = add_tick_features(m1, ticks)
    assert out["upticks"].iloc[1] == 2 and out["downticks"].iloc[1] == 1
    assert out["upticks"].iloc[[0, 2, 3]].isna().all()   # hors période couverte
    assert tick_imbalance_mode(out) == "mixte" and tick_imbalance_mode(m1) == "proxy"
    assert "upticks" not in m1.columns                   # entrée non modifiée


def imbalance_day(with_ticks):
    start = pd.Timestamp("2026-03-03 14:00", tz="UTC")   # 9:00 NY
    rows = noise_rows(60, amp=0.03)
    # 5 barres haussières, clôture au plus haut, gros volume
    rows += [(70.0 + 0.1 * i, 70.1 + 0.1 * i, 70.0 + 0.1 * i, 70.1 + 0.1 * i) for i in range(5)]
    rows += noise_rows(10, base=70.5, amp=0.03)
    m1 = bars(rows, start, tick_volume=10)
    m1.iloc[60:65, m1.columns.get_loc("tick_volume")] = 100
    if with_ticks:
        m1["upticks"] = 5.0
        m1["downticks"] = 5.0
        m1.iloc[60:65, m1.columns.get_loc("upticks")] = 40.0
        m1.iloc[60:65, m1.columns.get_loc("downticks")] = 2.0
    return m1


@pytest.mark.parametrize("with_ticks", [False, True])
def test_tick_imbalance_achete_la_pression_acheteuse(with_ticks):
    m1 = imbalance_day(with_ticks)
    sigs = STRATEGIES["tick_imbalance"].signals(m1, {"k_bars": 3, "threshold": 0.5})
    assert sigs and all(x.side == "buy" for x in sigs)
    assert sigs[0].time >= pd.Timestamp("2026-03-03 15:01", tz="UTC")  # après 3 barres fortes
    assert tick_imbalance_mode(m1) == ("ticks" if with_ticks else "proxy")


def test_tick_imbalance_hors_heures_liquides():
    m1 = imbalance_day(False)
    m1.index = m1.index + pd.Timedelta(hours=8)           # 17:00 NY : hors 8:00-14:30
    assert STRATEGIES["tick_imbalance"].signals(m1, {"k_bars": 3, "threshold": 0.5}) == []


def test_vwap_mr_respecte_la_tendance_m15(synth_dir):
    m1 = load_bars(synth_dir / "CrudeOIL_M1.csv.gz")
    sigs = STRATEGIES["vwap_mr"].signals(m1, {})
    assert sigs
    from ava_mt5.strategies import _features
    f = _features(m1)
    up, valid = f.trend_up
    pos = {t: i for i, t in enumerate(f.decision_time)}
    for sg in sigs[:300]:
        j = pos[sg.time]
        assert valid[j]
        assert (sg.side == "buy") == bool(up[j])
        assert (f.close[j] < f.vwap[j]) == (sg.side == "buy")
