import json

import numpy as np
import pandas as pd
import pytest

from ava_mt5 import backtest as bt
from ava_mt5.backtest import SymbolSpec, simulate
from ava_mt5.strategies import STRATEGIES, Signal

SPEC = SymbolSpec(point=0.01, contract_size=1000.0, volume_min=0.01, volume_step=0.01)
T0 = pd.Timestamp("2026-03-03 15:00:00", tz="UTC")  # mardi, loin du rollover


def bars(rows, start=T0, spread_pts=2):
    """rows : (open, high, low, close) par minute."""
    idx = pd.date_range(start, periods=len(rows), freq="1min")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)
    df["tick_volume"] = 10
    df["spread"] = spread_pts
    return df


def sig(side="buy", sl=0.5, tp=1.0, minutes=10, t=T0):
    return Signal(t, side, sl, tp, minutes, "test")


def run(rows, signal, spread=0.02, **kw):
    return simulate(bars(rows), [signal], SPEC, spread, capital=10_000, risk_pct=0.005, **kw)


FLAT = (70.00, 70.10, 69.95, 70.05)


def test_achat_objectif_touche():
    r = run([FLAT, (70.05, 71.10, 70.0, 71.0), FLAT], sig())
    (t,) = r.trades
    assert t["reason"] == "tp" and t["lots"] == pytest.approx(0.10)
    # entrée = open + spread (0,02) + glissement (0,01)
    assert t["entry"] == pytest.approx(70.03)
    assert t["exit"] == pytest.approx(71.03)
    assert t["pnl"] == pytest.approx(100.0)


def test_achat_stop_touche():
    r = run([FLAT, (70.05, 70.10, 69.40, 69.50), FLAT], sig())
    (t,) = r.trades
    assert t["reason"] == "sl"
    assert t["exit"] == pytest.approx(69.52)  # stop 69,53 moins 1 tick de glissement
    assert t["pnl"] == pytest.approx(-51.0)


def test_stop_et_objectif_meme_barre_stop_prioritaire():
    r = run([FLAT, (70.05, 71.20, 69.40, 70.0), FLAT], sig())
    assert r.trades[0]["reason"] == "sl"
    assert r.trades[0]["pnl"] < 0


def test_sortie_forcee_au_temps():
    rows = [FLAT, (70.05, 70.20, 70.0, 70.10), (70.10, 70.20, 70.0, 70.10), (70.10, 70.5, 69.9, 70.4)]
    r = run(rows, sig(minutes=3))
    (t,) = r.trades
    assert t["reason"] == "temps"
    assert t["exit"] == pytest.approx(70.09)  # close de la 3e barre moins 1 tick
    assert t["pnl"] == pytest.approx(6.0)


def test_vente_stop_cote_ask_et_couts():
    # high 70,50 + spread 0,02 = ask 70,52 >= stop 70,49 (entrée 69,99 + 0,5)
    r = run([FLAT, (70.0, 70.50, 69.90, 70.0), FLAT], sig("sell"))
    (t,) = r.trades
    assert t["entry"] == pytest.approx(69.99)
    assert t["reason"] == "sl" and t["exit"] == pytest.approx(70.50)
    assert t["pnl"] == pytest.approx(-51.0)


def test_vente_objectif():
    r = run([FLAT, (70.0, 70.1, 68.90, 69.0), FLAT], sig("sell"))
    (t,) = r.trades
    assert t["reason"] == "tp" and t["pnl"] == pytest.approx((69.99 - 68.99) * 100)


def test_couts_spread_et_glissement_comptes():
    r = run([FLAT, (70.05, 71.10, 70.0, 71.0), FLAT], sig())
    # objectif : spread 0,02 + glissement d'entrée 0,01 sur 100 barils
    assert r.trades[0]["cost"] == pytest.approx(0.03 * 100)
    r = run([FLAT, (70.05, 70.10, 69.40, 69.50), FLAT], sig())
    # stop : glissement supplémentaire à la sortie
    assert r.trades[0]["cost"] == pytest.approx(0.04 * 100)
    r0 = simulate(bars([FLAT, (70.05, 71.10, 70.0, 71.0)]), [sig()], SPEC, 0.0, slippage_ticks=0)
    assert r0.trades[0]["entry"] == pytest.approx(70.0) and r0.trades[0]["cost"] == 0.0


def test_une_seule_position_a_la_fois():
    s2 = sig(t=T0 + pd.Timedelta(minutes=1))
    rows = [FLAT, FLAT, (70.05, 71.10, 70.0, 71.0), FLAT, FLAT]
    r = simulate(bars(rows), [sig(), s2], SPEC, 0.02)
    assert len(r.trades) == 1 and r.skipped_busy == 1


def test_taille_de_position():
    assert bt.position_size(10_000, 0.005, 0.5, SPEC) == pytest.approx(0.10)
    assert bt.position_size(10_000, 0.005, 0.3, SPEC) == pytest.approx(0.16)  # arrondi au pas
    assert bt.position_size(10_000, 0.005, 100.0, SPEC) == 0.0              # sous le minimum
    assert bt.position_size(20_000, 0.005, 0.5, SPEC) == pytest.approx(0.20)  # capital courant
    # plafond de levier : 20 * 10000 / (1000 * 70) = 2,85 lots
    assert bt.position_size(10_000, 0.5, 0.05, SPEC, 70.0, 20.0) == pytest.approx(2.85)


def test_trade_saute_sous_volume_minimum():
    r = run([FLAT, FLAT], sig(sl=100.0, tp=200.0))
    assert r.trades == [] and r.skipped_min_lot == 1


def test_swap_rollover_et_triple_mercredi():
    spec = SymbolSpec(swap_long=-20.0, swap_short=5.0)
    # hiver : 17:00 New York = 22:00 UTC. Mardi 2026-01-13 : swap simple.
    mardi = pd.Timestamp("2026-01-13 21:58", tz="UTC")
    rows = [FLAT] * 6
    s = Signal(mardi, "buy", 5.0, 5.0, 5, "t")
    r = simulate(bars(rows, mardi), [s], spec, 0.0, slippage_ticks=0)
    lots = r.trades[0]["lots"]
    assert r.trades[0]["swap"] == pytest.approx(-20 * 0.01 * 1000 * lots)
    # mercredi : triple
    mer = pd.Timestamp("2026-01-14 21:58", tz="UTC")
    r = simulate(bars(rows, mer), [Signal(mer, "buy", 5.0, 5.0, 5, "t")], spec, 0.0, slippage_ticks=0)
    assert r.trades[0]["swap"] == pytest.approx(-60 * 0.01 * 1000 * r.trades[0]["lots"])
    # vente : swap_short positif
    r = simulate(bars(rows, mardi), [Signal(mardi, "sell", 5.0, 5.0, 5, "t")], spec, 0.0, slippage_ticks=0)
    assert r.trades[0]["swap"] > 0
    # position qui ne traverse pas le rollover : pas de swap
    r = simulate(bars(rows), [sig(sl=5.0, tp=5.0, minutes=5)], spec, 0.0)
    assert r.trades[0]["swap"] == 0.0


def test_rollover_heure_d_ete():
    hiver, _ = bt.rollover_table(int(pd.Timestamp("2026-01-13", tz="UTC").timestamp()),
                                 int(pd.Timestamp("2026-01-13", tz="UTC").timestamp()))
    ete, _ = bt.rollover_table(int(pd.Timestamp("2026-07-15", tz="UTC").timestamp()),
                               int(pd.Timestamp("2026-07-15", tz="UTC").timestamp()))
    h = {pd.Timestamp(x, unit="s", tz="UTC") for x in hiver}
    e = {pd.Timestamp(x, unit="s", tz="UTC") for x in ete}
    assert pd.Timestamp("2026-01-13 22:00", tz="UTC") in h
    assert pd.Timestamp("2026-07-15 21:00", tz="UTC") in e


def test_decoupe_70_30_chronologique():
    df = bars([FLAT] * 100)
    a, b = bt.split_chronological(df.sample(frac=1, random_state=1), 0.7)  # mélangé en entrée
    assert (len(a), len(b)) == (70, 30)
    assert a.index.max() < b.index.min()
    assert a.index.is_monotonic_increasing and b.index.is_monotonic_increasing
    assert a.index.union(b.index).equals(df.index)


def test_evaluate_strategy_n_utilise_pas_l_echantillon_test_pour_choisir(synth_dir):
    """Les paramètres sont choisis avec les 70 % premiers : modifier la fin ne les change pas."""
    from ava_mt5.strategies import load_bars
    m1 = load_bars(synth_dir / "CrudeOIL_M1.csv.gz")
    prof = json.loads((synth_dir / "spread_profile.json").read_text())["CrudeOIL"]
    spread = bt.spread_array(m1.index, prof)
    strat = STRATEGIES["orb"]
    r1 = bt.evaluate_strategy(strat, m1, spread, SPEC)
    k = int(len(m1) * 0.7)
    m2 = m1.copy()
    cols = ["open", "high", "low", "close"]
    m2.iloc[k + 10:, [m2.columns.get_loc(c) for c in cols]] *= 1.05  # altère l'OOS
    r2 = bt.evaluate_strategy(strat, m2, spread, SPEC)
    assert r1["params"] == r2["params"]
    assert r1["is"] == r2["is"]


def metrics(**kw):
    base = {"trades": 250, "profit_factor": 1.3, "max_drawdown_pct": 6.0, "net_usd": 500.0,
            "loss_win_ratio": 1.5}
    return {**base, **kw}


def test_criteres_acceptation():
    ok, why = bt.evaluate_criteria(metrics())
    assert ok and why == []
    for kw, mot in [({"trades": 199}, "trades"), ({"profit_factor": 1.14}, "profit factor"),
                    ({"max_drawdown_pct": 10.1}, "drawdown"), ({"net_usd": 0.0}, "gain net"),
                    ({"loss_win_ratio": 2.01}, "perte")]:
        ok, why = bt.evaluate_criteria(metrics(**kw))
        assert not ok and len(why) == 1 and mot in why[0]
    ok, why = bt.evaluate_criteria(metrics(trades=200, profit_factor=1.15, max_drawdown_pct=10.0,
                                           loss_win_ratio=2.0))
    assert ok  # bornes incluses
    ok, why = bt.evaluate_criteria(metrics(trades=5, net_usd=-1))
    assert len(why) == 2


def test_metriques():
    trades = [{"pnl": p, "cost": 1.0} for p in (100.0, -50.0, -50.0, 200.0)]
    m = bt.compute_metrics(trades, 10_000, 2)
    assert m["trades"] == 4 and m["trades_per_day"] == 2.0 and m["win_rate"] == 0.5
    assert m["avg_win"] == 150 and m["avg_loss"] == 50
    assert m["loss_win_ratio"] == pytest.approx(1 / 3, abs=1e-3)
    assert m["profit_factor"] == 3.0
    assert m["max_drawdown_pct"] == pytest.approx(100 / 10100 * 100, abs=1e-2)
    assert m["net_usd"] == 200 and m["net_pct"] == 2.0 and m["costs_usd"] == 4.0
    vide = bt.compute_metrics([], 10_000, 5)
    assert vide["trades"] == 0 and vide["profit_factor"] == 0.0


def test_profil_de_spread_horaire_et_evenements():
    prof = {"hourly_median": {str(h): 0.03 for h in range(24)},
            "eia": {"before_5m": 0.05, "minute_0_to_3": [0.20, 0.10, 0.07, 0.04], "normal": 0.03},
            "api": {"before_5m": 0.04, "minute_0_to_3": [0.15, 0.09, 0.06, 0.04], "normal": 0.03}}
    prof["hourly_median"]["15"] = 0.035
    idx = pd.date_range("2026-01-14 15:00", "2026-01-14 16:00", freq="1min", tz="UTC")
    sp = bt.spread_array(idx, prof)
    at = lambda hm: sp[idx.get_loc(pd.Timestamp(f"2026-01-14 {hm}", tz="UTC"))]  # noqa: E731
    # EIA mercredi 10:30 New York = 15:30 UTC en hiver
    assert at("15:30") == 0.20 and at("15:33") == 0.04 and at("15:27") == 0.05
    assert at("15:20") == 0.035 and at("15:40") == 0.035


def test_rapport_complet_format(synth_dir, tmp_path):
    out = tmp_path / "logs" / "rapport.json"
    assert bt.main(["--data", str(synth_dir), "--out", str(out)]) == 0
    rep = json.loads(out.read_text())
    assert set(rep) == {"generated", "data_dir", "synthetic", "criteria", "results"}
    assert rep["synthetic"] is True
    assert set(rep["results"]["CrudeOIL"]) == set(STRATEGIES)
    for r in rep["results"]["CrudeOIL"].values():
        assert set(r) == {"params", "is", "oos", "passed", "fail_reasons"}
        assert {"trades", "trades_per_day", "win_rate", "avg_win", "avg_loss", "loss_win_ratio",
                "profit_factor", "max_drawdown_pct", "net_usd", "net_pct", "costs_usd"} <= set(r["oos"])
        assert r["passed"] == (r["fail_reasons"] == [])
    md = out.with_suffix(".md").read_text()
    assert "Conclusion" in md
