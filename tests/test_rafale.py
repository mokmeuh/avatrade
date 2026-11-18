import numpy as np
import pandas as pd
import pytest

from ava_mt5 import rafale as rf
from ava_mt5.core import Broker, Settings, TradingError
from fake_mt5_rafale import FakeMT5Rafale

SPEC = {"point": 0.01, "digits": 2, "contract_size": 1000.0, "volume_min": 0.01, "volume_step": 0.01,
        "tick_size": 0.01, "tick_value": 10.0, "swap_long": 0.0, "swap_short": 0.0}
T0 = 1_767_657_600_000  # multiple de 1000 ms, lundi, bien avant 17:00 NY


def ticks(bids, spread=0.03, dt_ms=1000):
    b = np.round(np.asarray(bids, dtype=float), 2)
    return pd.DataFrame({"time_msc": T0 + np.arange(len(b)) * dt_ms, "bid": b,
                         "ask": np.round(b + spread, 2), "flags": 6})


def bt(bids, capital=10000.0, detail=True, **p):
    base = dict(mode="pairs", interval_s=1, spread_filter=False, target=0.05, max_life_min=5)
    base.update(p)
    return rf.backtest_rafale(ticks(bids), SPEC, base, capital, return_detail=detail)


# ------------------------------------------------------------------ hedging
def test_check_hedging_vrai_faux():
    ok, msg = rf.check_hedging(FakeMT5Rafale(margin_mode=2))
    assert ok and "HEDGING" in msg
    ok, msg = rf.check_hedging(FakeMT5Rafale(margin_mode=0))
    assert not ok and "paires" in msg


def test_mode_pairs_refuse_sans_hedging(tmp_path):
    assert rf.check_mode_allowed("pairs", False)[0] is False
    assert rf.check_mode_allowed("pairs", True)[0] is True
    assert rf.check_mode_allowed("momentum", False)[0] is True
    with pytest.raises(ValueError, match="hedging"):
        rf.normalize_params({"mode": "pairs", "hedging": False})
    mt5 = FakeMT5Rafale(margin_mode=0)
    with pytest.raises(ValueError):
        rf.RafaleLive(mt5, Broker(mt5, Settings()), "CrudeOIL", {"mode": "pairs"}, orders=False,
                      logs_dir=str(tmp_path))


def test_sans_hedging_une_seule_position_nette():
    p = rf.normalize_params({"mode": "momentum", "hedging": False})
    assert rf.effective_max_positions(p) == 1


# ------------------------------------------------------------------ règles pures
def test_signal_momentum_scalaire_egal_vectorise():
    rng = np.random.default_rng(1)
    mid = 70 + np.round(np.cumsum(rng.normal(0, 0.01, 300)), 3)
    v = rf.momentum_signals(mid, 5, 0.01, 2)
    for s in range(5, 300):
        sc = rf.momentum_signal(mid[s], mid[s - 5], 0.01, 2)
        assert {"buy": 1, "sell": -1, None: 0}[sc] == v[s]


def test_exit_reason_prix_de_sortie_reel():
    p = rf.normalize_params({"target": 0.05, "max_life_min": 5})
    slip = 0.01
    # achat a 70.04 : sortie au bid moins glissement ; 70.10 - 0.01 - 70.04 = 0.05
    assert rf.exit_reason("buy", 70.04, 70.10, 70.13, 10, p, slip) == "objectif"
    assert rf.exit_reason("buy", 70.04, 70.09, 70.12, 10, p, slip) is None
    # vente a 69.99 : sortie au ask plus glissement ; 69.99 - (69.93 + 0.01) = 0.05
    assert rf.exit_reason("sell", 69.99, 69.90, 69.93, 10, p, slip) == "objectif"
    assert rf.exit_reason("sell", 69.99, 69.91, 69.94, 10, p, slip) is None
    assert rf.exit_reason("buy", 70.04, 70.0, 70.03, 301, p, slip) == "duree_max"
    assert rf.exit_reason("buy", 70.04, 69.89, 69.92, 1, p, slip) == "stop"


def test_volume_toujours_minimum_meme_apres_pertes():
    assert rf.fixed_volume(SPEC) == 0.01
    bids = 70 + np.cumsum(np.r_[0.0, -0.02 * np.ones(40), 0.03 * np.ones(40), -0.05 * np.ones(40)])
    r = bt(bids, mode="pairs", target=0.05, sl_mult=1.0)
    df = r["trades_df"]
    assert (df["pnl"] < 0).sum() > 3
    assert set(df["volume"]) == {SPEC["volume_min"]}
    assert r["volume"] == SPEC["volume_min"]


def test_max_10_positions():
    r = bt(np.full(200, 70.0), target=0.2, max_life_min=15)
    assert r["max_open"] == 10
    df = r["trades_df"]
    # recompte indépendant des positions ouvertes à chaque instant
    ev = sorted([(a, 1) for a in df["e"]] + [(b, -1) for b in df["x"]], key=lambda z: (z[0], z[1]))
    cur = mx = 0
    for _, dlt in ev:
        cur += dlt
        mx = max(mx, cur)
    assert mx <= 10


# ------------------------------------------------------------------ backtest : sorties
def test_execution_au_tick_suivant_et_sortie_a_x_net():
    bids = [70.00, 70.00, 70.02, 70.04, 70.06, 70.08, 70.10, 70.12, 70.14, 70.16, 70.18, 70.20]
    r = bt(bids, max_positions=2)
    df = r["trades_df"]
    achat = df[df["side"] == "buy"].iloc[0]
    # décision sur le tick 0, exécution sur le tick 1 : ask 70.03 + glissement 0.01
    assert achat["e"] == 1 and achat["entry"] == pytest.approx(70.04)
    # objectif atteint au tick 6 (bid 70.10 - 0.01 - 70.04 = 0.05), exécuté au tick 7 (bid 70.12)
    assert achat["x"] == 7 and achat["reason"] == "objectif"
    assert achat["exit"] == pytest.approx(70.11)
    k = 1000.0 * 0.01
    assert achat["pnl"] == pytest.approx((70.11 - 70.04) * k)
    assert achat["pnl"] / k >= 0.05


def test_expiration_duree_max():
    r = bt(np.full(200, 70.0), max_life_min=1, max_positions=2)
    ach = r["trades_df"].query("side == 'buy'").iloc[0]
    assert ach["reason"] == "duree_max"
    # ouverte à t = 1 s, expire au premier tick >= 61 s, exécution au tick suivant
    assert ach["e"] == 1 and ach["x"] == 62
    assert r["exits"]["duree_max"] >= 1


def test_stop_de_securite():
    bids = 70.0 - 0.02 * np.arange(60)
    r = bt(bids, max_positions=2, max_life_min=15, target=0.05, sl_mult=3.0)
    ach = r["trades_df"].query("side == 'buy'").iloc[0]
    assert ach["reason"] == "stop"
    perte = (ach["entry"] - ach["exit"])
    assert 0.15 <= perte <= 0.15 + 0.04 + 0.011 + 1e-9   # stop + dépassement + tick d'exécution + glissement


def test_coupe_circuit_perte_latente_2pct():
    bids = np.r_[70.0 - 0.02 * np.arange(25), 69.5 + 0.04 * np.arange(60)]
    r = bt(bids, capital=100.0, mode="momentum", lookback_s=1, thr_ticks=1, target=0.5,
           max_life_min=15, sl_mult=3.0)
    assert r["breaker_cuts"] >= 1
    assert r["exits"]["coupe_circuit"] >= 1


def test_equite_differente_du_solde_et_drawdown_equite_superieur():
    r = bt(np.full(300, 70.0), max_life_min=1)
    eq, bal = r["equity"], r["balance"]
    assert np.any(eq < bal - 1e-9)                 # latent négatif tant que les positions sont ouvertes
    assert r["dd_equity_pct"] >= r["dd_balance_pct"]
    assert r["dd_equity_pct"] > 0
    assert len(eq) >= 299                          # une valeur par seconde


def test_swap_applique_seulement_si_17h_ny_traversee():
    r = bt(np.full(100, 70.0))
    assert r["swap_applied"] is False and r["swap_usd"] == 0.0


def test_synthetic_ticks_et_backtest_rapide():
    t = rf.synthetic_ticks(1, seed=3)
    assert {"time_msc", "bid", "ask", "flags"} <= set(t.columns)
    assert (t["ask"] > t["bid"]).all() and t["time_msc"].is_monotonic_increasing
    r = rf.backtest_rafale(t, rf.SYNTH_META, {"mode": "momentum"})
    assert r["trades"] > 0 and r["max_open"] <= 10


def test_criteres_oos():
    ok, why = rf.evaluate_oos({"trades": 500, "profit_factor": 1.3, "dd_equity_pct": 4.0,
                               "net_usd": 50.0, "loss_gain_ratio": 1.2})
    assert ok and not why
    ok, why = rf.evaluate_oos({"trades": 10, "profit_factor": 0.8, "dd_equity_pct": 20.0,
                               "net_usd": -5.0, "loss_gain_ratio": 3.0})
    assert not ok and len(why) == 5


# ------------------------------------------------------------------ live
def make_live(tmp_path, orders=True, trade_mode=0, margin_mode=2, params=None, stops_level=0):
    mt5 = FakeMT5Rafale(trade_mode=trade_mode, margin_mode=margin_mode)
    mt5.symbols["CrudeOIL"] = mt5.symbols["CrudeOIL"]._replace(trade_stops_level=stops_level)
    broker = Broker(mt5, Settings())
    broker.connect() if trade_mode == 0 else None
    p = {"mode": "pairs", "target": 0.05, "max_life_min": 5, "interval_s": 1, "spread_filter": False}
    p.update(params or {})
    live = rf.RafaleLive(mt5, broker, "CrudeOIL", p, orders=orders, logs_dir=tmp_path / "logs",
                         data_dir=tmp_path / "data", config_dir=tmp_path / "cfg")
    return mt5, live


def test_refus_ordre_sur_compte_reel(tmp_path):
    mt5 = FakeMT5Rafale(trade_mode=FakeMT5Rafale.ACCOUNT_TRADE_MODE_REAL)
    broker = Broker(mt5, Settings())
    broker.connected = True            # même connexion forcée : l'exécuteur revérifie avant l'ordre
    ex = rf.RealExecutor(broker, "CrudeOIL")
    with pytest.raises(rf.CompteReelRefuse):
        ex.open("buy", None, 0.01, 69.0, 71.0, 0.0)
    assert mt5.sent == []
    with pytest.raises(TradingError):
        Broker(FakeMT5Rafale(trade_mode=2), Settings()).connect()


def test_live_ouvre_avec_magic_volume_min_et_niveaux(tmp_path):
    mt5, live = make_live(tmp_path)
    info = live.step(1_700_000_000.0)
    assert info["opened"] == 2
    assert len(mt5.sent) == 2
    for req in mt5.sent:
        assert req["magic"] == rf.MAGIC_RAFALE and req["volume"] == 0.01
    buy = next(r for r in mt5.sent if r["type"] == mt5.ORDER_TYPE_BUY)
    sell = next(r for r in mt5.sent if r["type"] == mt5.ORDER_TYPE_SELL)
    assert buy["tp"] == pytest.approx(70.03 + 0.05 + 0.01) and buy["sl"] == pytest.approx(70.03 - 0.15)
    assert sell["tp"] == pytest.approx(70.00 - 0.05 - 0.01) and sell["sl"] == pytest.approx(70.00 + 0.15)


def test_live_max_10_positions(tmp_path):
    mt5, live = make_live(tmp_path, params={"target": 0.2, "max_positions": 10})
    for s in range(30):
        live.step(1_700_000_000.0 + s)
    assert len([p for p in mt5.pos if p.magic == rf.MAGIC_RAFALE]) == 10


def test_live_ferme_a_x_net_et_ignore_autre_magic(tmp_path):
    mt5, live = make_live(tmp_path, params={"mode": "momentum", "thr_ticks": 100, "max_life_min": 1})
    autre = mt5.add_position("buy", 80.0, rf.MAGIC_AUTRE_BOT, volume=1.0)     # perte énorme, autre bot
    mien = mt5.add_position("buy", 70.00, rf.MAGIC_RAFALE)
    mt5.set_tick(70.04, 70.07)
    live.step(1_700_000_000.0)                  # 70.04 - 0.01 - 70.00 = 0.03 < 0.05 : on garde
    assert any(p.ticket == mien for p in mt5.pos)
    mt5.set_tick(70.06, 70.09)
    live.step(1_700_000_001.0)                  # 0.05 net : fermeture
    assert not any(p.ticket == mien for p in mt5.pos)
    closes = [r for r in mt5.sent if "position" in r]
    assert [r["position"] for r in closes] == [mien]
    assert all(r["magic"] == rf.MAGIC_RAFALE for r in closes)
    assert any(p.ticket == autre for p in mt5.pos)   # jamais touché, même par le coupe-circuit


def test_live_coupe_circuit_ignore_autre_magic(tmp_path):
    mt5, live = make_live(tmp_path, params={"mode": "momentum", "thr_ticks": 100, "target": 5.0,
                                            "sl_usd": 50.0, "max_life_min": 15})
    mt5.balance = 1000.0                          # 2 % = 20 $
    autre = mt5.add_position("buy", 90.0, rf.MAGIC_AUTRE_BOT, volume=1.0)
    for _ in range(3):
        mt5.add_position("buy", 70.5, rf.MAGIC_RAFALE, volume=0.5)   # 3 x 0.5 lot x 1000 x -0.5 = -750
    live.step(1_700_000_000.0)
    assert [p for p in mt5.pos if p.magic == rf.MAGIC_RAFALE] == []
    assert any(p.ticket == autre for p in mt5.pos)
    assert live.cooldown_until > 1_700_000_000.0


def test_live_observation_ne_passe_aucun_ordre(tmp_path):
    mt5, live = make_live(tmp_path, orders=False)
    live.step(1_700_000_000.0)
    assert mt5.sent == []
    assert len(live.executor.book) == 2
    assert (tmp_path / "logs" / "status_rafale.json").exists()
    assert (tmp_path / "logs" / "trades_rafale.csv").exists()


def test_live_volume_inchange_apres_pertes(tmp_path):
    mt5, live = make_live(tmp_path, orders=False)
    live.state["consec_losses"] = 4
    live.state["day_pnl"] = -50.0
    assert live._volume() == 0.01


def test_live_tp_serveur_omis_si_trop_proche_du_niveau_d_arret(tmp_path):
    mt5, live = make_live(tmp_path, stops_level=10)
    live.step(1_700_000_000.0)
    assert all("tp" not in r for r in mt5.sent)
    assert all("sl" in r for r in mt5.sent)
