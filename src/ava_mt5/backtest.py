"""Backtest barre par barre (M1) des stratégies de `strategies.py`.

    python -m ava_mt5.backtest --data data --out logs/backtest_report.json

Hypothèses du moteur (toutes volontairement prudentes) :
  * Prix des barres = côté bid. Achat à l'open + spread + 1 tick ; vente au bid - 1 tick.
  * Spread = profil horaire (heure UTC) de spread_profile.json, remplacé dans les minutes
    EIA/API par le profil de l'événement (5 minutes avant, minutes 0 à 3 après).
  * SL/TP évalués sur high/low (bid pour les achats, ask = bid + spread pour les ventes).
    SL et TP touchés dans la même barre : le SL est retenu. Glissement d'1 tick sur les
    sorties au stop et les sorties forcées (pas sur l'objectif, ordre limite).
  * SL/TP sont mesurés depuis le prix d'entrée effectif (spread et glissement inclus).
  * Swap : si la position traverse 17:00 New York, on applique swap_long ou swap_short
    de meta.json, interprété EN POINTS : swap_usd = swap * point * contract_size * lots
    (devise de cotation supposée USD). Triplé lors du rollover du mercredi (jour de triple
    swap par défaut ; paramétrable dans SymbolSpec).
  * Une seule position à la fois par symbole et par stratégie ; les signaux émis pendant
    qu'une position est ouverte sont ignorés (comme le fera le bot).
  * Risque 0,5 % du capital courant par trade : lots = risque / (sl_dist * contract_size),
    arrondi au pas de volume, plafonné par un levier maximal (20 par défaut). Si le volume
    minimum dépasse le risque, le trade est sauté et compté.
  * Le drawdown est mesuré sur la courbe de capital aux clôtures de trades.
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import math
import sys
import time as _time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ava_mt5.data import (build_profile, event_times_utc, list_symbols, read_meta, to_epoch_s)
from ava_mt5.strategies import (STRATEGIES, Signal, Strategy, add_tick_features, load_bars,
                                load_ticks, tick_imbalance_mode)

NY = ZoneInfo("America/New_York")

CRITERIA: dict[str, float] = {
    "min_trades": 200,
    "min_profit_factor": 1.15,
    "max_drawdown_pct": 10.0,
    "min_net_profit": 0.0,        # gain net strictement supérieur
    "max_loss_win_ratio": 2.0,
}
DEFAULT_CAPITAL = 10_000.0
RISK_PCT = 0.005
SPLIT = 0.7
MAX_LEVERAGE = 20.0
MIN_IS_TRADES = (30, 10)          # seuils successifs de trades minimaux en échantillon
PF_CAP = 999.0


# --------------------------------------------------------------------- modèle
@dataclass(frozen=True)
class SymbolSpec:
    point: float = 0.01
    contract_size: float = 1000.0
    volume_min: float = 0.01
    volume_step: float = 0.01
    volume_max: float = 100.0
    swap_long: float = 0.0
    swap_short: float = 0.0
    swap_triple_weekday: int = 2   # 0 = lundi ... 2 = mercredi

    @classmethod
    def from_meta(cls, d: dict[str, Any] | None) -> "SymbolSpec":
        d = d or {}
        keys = ("point", "contract_size", "volume_min", "volume_step", "volume_max",
                "swap_long", "swap_short")
        return cls(**{k: float(d[k]) for k in keys if d.get(k) is not None})


@dataclass
class SimResult:
    trades: list[dict[str, Any]]
    skipped_min_lot: int = 0
    skipped_busy: int = 0
    skipped_no_bar: int = 0


def _round_step(value: float, step: float) -> float:
    n = math.floor(value / step + 1e-9)
    decimals = max(0, -int(math.floor(math.log10(step)))) if step < 1 else 0
    return round(n * step, decimals)


def position_size(equity: float, risk_pct: float, sl_dist: float, spec: SymbolSpec,
                  price: float | None = None, max_leverage: float | None = None) -> float:
    """Lots pour risquer `risk_pct` du capital sur `sl_dist` ; 0 si sous le volume minimum."""
    if sl_dist <= 0 or equity <= 0:
        return 0.0
    lots = (equity * risk_pct) / (sl_dist * spec.contract_size)
    if max_leverage and price:
        lots = min(lots, equity * max_leverage / (spec.contract_size * price))
    lots = min(_round_step(lots, spec.volume_step), spec.volume_max)
    return lots if lots >= spec.volume_min - 1e-12 else 0.0


def split_chronological(df: pd.DataFrame, frac: float = SPLIT) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Découpe chronologique (échantillon, hors échantillon), sans mélange ni recouvrement."""
    df = df.sort_index()
    k = int(round(len(df) * frac))
    return df.iloc[:k], df.iloc[k:]


def spread_array(index: pd.DatetimeIndex, profile: dict[str, Any]) -> np.ndarray:
    """Spread (prix) à l'ouverture de chaque barre : profil horaire + profil EIA/API."""
    hourly = np.array([profile["hourly_median"][str(h)] for h in range(24)], dtype=float)
    sp = hourly[index.hour.to_numpy()]
    if len(index) == 0:
        return sp
    t = to_epoch_s(index)
    for kind in ("eia", "api"):
        ev = profile.get(kind)
        if not ev:
            continue
        for ts in event_times_utc(index[0], index[-1], kind):
            e = int(ts.timestamp())
            for k in range(-5, 4):
                i = int(np.searchsorted(t, e + 60 * k))
                if i < len(t) and t[i] == e + 60 * k:
                    sp[i] = ev["before_5m"] if k < 0 else ev["minute_0_to_3"][k]
    return sp


def rollover_table(t0: int, t1: int, triple_weekday: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Instants (epoch s) des rollovers 17:00 New York et leur multiplicateur de swap."""
    d0 = dt.datetime.fromtimestamp(t0, dt.timezone.utc).astimezone(NY).date() - dt.timedelta(days=1)
    d1 = dt.datetime.fromtimestamp(t1, dt.timezone.utc).astimezone(NY).date() + dt.timedelta(days=1)
    ts, mult = [], []
    d = d0
    while d <= d1:
        local = dt.datetime(d.year, d.month, d.day, 17, 0, tzinfo=NY)
        ts.append(int(local.timestamp()))
        mult.append(3 if d.weekday() == triple_weekday else 1)
        d += dt.timedelta(days=1)
    return np.asarray(ts, dtype=np.int64), np.asarray(mult, dtype=np.int64)


# --------------------------------------------------------------------- moteur
def simulate(m1: pd.DataFrame, signals: list[Signal], spec: SymbolSpec,
             spread: float | np.ndarray = 0.0, capital: float = DEFAULT_CAPITAL,
             risk_pct: float = RISK_PCT, slippage_ticks: float = 1.0,
             max_leverage: float | None = None) -> SimResult:
    """Rejoue les signaux sur les barres M1 (voir les hypothèses en tête de module)."""
    n = len(m1)
    res = SimResult(trades=[])
    if n == 0 or not signals:
        return res
    t = to_epoch_s(m1.index)
    o, h, lo, c = (m1[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    sp = np.full(n, float(spread)) if np.isscalar(spread) else np.asarray(spread, dtype=float)
    slip = slippage_ticks * spec.point
    cs = spec.contract_size
    roll_ts, roll_mult = rollover_table(int(t[0]), int(t[-1]), spec.swap_triple_weekday)
    order = sorted(range(len(signals)), key=lambda i: signals[i].time)
    equity = capital
    busy_until = -1
    for oi in order:
        sg = signals[oi]
        st = int(round(sg.time.timestamp()))
        if st < busy_until:
            res.skipped_busy += 1
            continue
        i0 = int(np.searchsorted(t, st))
        if i0 >= n or t[i0] != st or sg.sl_dist <= 0 or sg.tp_dist <= 0:
            res.skipped_no_bar += 1
            continue
        lots = position_size(equity, risk_pct, sg.sl_dist, spec, o[i0], max_leverage)
        if lots <= 0:
            res.skipped_min_lot += 1
            continue
        end = max(int(np.searchsorted(t, st + sg.max_minutes * 60)), i0 + 1)
        hi, lw, sw = h[i0:end], lo[i0:end], sp[i0:end]
        buy = sg.side == "buy"
        if buy:
            entry = o[i0] + sp[i0] + slip
            sl, tp = entry - sg.sl_dist, entry + sg.tp_dist
            hit_sl, hit_tp = lw <= sl, hi >= tp
        else:
            entry = o[i0] - slip
            sl, tp = entry + sg.sl_dist, entry - sg.tp_dist
            hit_sl, hit_tp = hi + sw >= sl, lw + sw <= tp
        k_sl = int(hit_sl.argmax()) if hit_sl.any() else None
        k_tp = int(hit_tp.argmax()) if hit_tp.any() else None
        if k_sl is not None and (k_tp is None or k_sl <= k_tp):   # SL d'abord si même barre
            k, reason = k_sl, "sl"
            if buy:
                exit_px = min(sl, o[i0 + k]) - slip
            else:
                exit_px = max(sl, o[i0 + k] + sp[i0 + k]) + slip
        elif k_tp is not None:
            k, reason = k_tp, "tp"
            exit_px = tp
        else:
            k, reason = len(hi) - 1, "temps"
            exit_px = c[i0 + k] - slip if buy else c[i0 + k] + sp[i0 + k] + slip
        exit_ts = int(t[i0 + k]) + 60
        busy_until = exit_ts
        gross = (exit_px - entry) if buy else (entry - exit_px)
        # coûts payés (en prix) : spread, glissement d'entrée, glissement de sortie éventuel
        cost_px = (sp[i0] if buy else sp[i0 + k]) + slip + (slip if reason != "tp" else 0.0)
        # swap : rollovers traversés entre l'entrée et la sortie
        a = int(np.searchsorted(roll_ts, st, side="right"))
        b = int(np.searchsorted(roll_ts, exit_ts, side="right"))
        nroll = int(roll_mult[a:b].sum())
        swap_pts = spec.swap_long if buy else spec.swap_short
        swap_usd = swap_pts * spec.point * cs * lots * nroll
        pnl = gross * lots * cs + swap_usd
        equity += pnl
        res.trades.append({
            "time": int(st), "exit_time": exit_ts, "side": sg.side, "lots": lots,
            "entry": float(entry), "exit": float(exit_px), "reason": reason,
            "pnl": float(pnl), "cost": float(cost_px * lots * cs + max(0.0, -swap_usd)),
            "swap": float(swap_usd), "equity": float(equity),
        })
        if equity <= 0:
            break
    return res


# --------------------------------------------------------------------- métriques
def trading_days(m1: pd.DataFrame) -> int:
    return int(len(np.unique(to_epoch_s(m1.index) // 86400))) if len(m1) else 0


def compute_metrics(trades: list[dict[str, Any]], capital: float, n_days: int,
                    skipped_min_lot: int = 0) -> dict[str, Any]:
    pnl = np.array([t["pnl"] for t in trades], dtype=float)
    n = len(pnl)
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    gross_win, gross_loss = float(wins.sum()), float(-losses.sum())
    avg_win = float(wins.mean()) if wins.size else 0.0
    avg_loss = float(-losses.mean()) if losses.size else 0.0
    if gross_loss > 0:
        pf = gross_win / gross_loss
    else:
        pf = PF_CAP if gross_win > 0 else 0.0
    if avg_win > 0:
        ratio = avg_loss / avg_win
    else:
        ratio = PF_CAP if avg_loss > 0 else 0.0
    eq = np.concatenate([[capital], capital + np.cumsum(pnl)]) if n else np.array([capital])
    peak = np.maximum.accumulate(eq)
    dd = float(((peak - eq) / peak).max() * 100.0)
    net = float(pnl.sum())
    return {
        "trades": int(n),
        "days": int(n_days),
        "trades_per_day": round(n / n_days, 3) if n_days else 0.0,
        "win_rate": round(float((pnl > 0).mean()), 4) if n else 0.0,
        "avg_win": round(avg_win, 4),
        "avg_loss": round(avg_loss, 4),
        "loss_win_ratio": round(min(ratio, PF_CAP), 4),
        "profit_factor": round(min(pf, PF_CAP), 4),
        "max_drawdown_pct": round(dd, 4),
        "net_usd": round(net, 2),
        "net_pct": round(net / capital * 100.0, 4),
        "costs_usd": round(float(sum(t["cost"] for t in trades)), 2),
        "skipped_min_lot": int(skipped_min_lot),
    }


def evaluate_criteria(oos: dict[str, Any], criteria: dict[str, float] = CRITERIA) -> tuple[bool, list[str]]:
    """Critères d'acceptation hors échantillon : (passed, raisons d'échec en français)."""
    why = []
    if oos["trades"] < criteria["min_trades"]:
        why.append(f"trades {oos['trades']} < {criteria['min_trades']:g} requis")
    if oos["profit_factor"] < criteria["min_profit_factor"]:
        why.append(f"profit factor {oos['profit_factor']:.2f} < {criteria['min_profit_factor']:g}")
    if oos["max_drawdown_pct"] > criteria["max_drawdown_pct"]:
        why.append(f"drawdown {oos['max_drawdown_pct']:.1f} % > {criteria['max_drawdown_pct']:g} %")
    if not oos["net_usd"] > criteria["min_net_profit"]:
        why.append(f"gain net {oos['net_usd']:.2f} USD non positif")
    if oos["loss_win_ratio"] > criteria["max_loss_win_ratio"]:
        why.append(f"perte moy./gain moy. {oos['loss_win_ratio']:.2f} > {criteria['max_loss_win_ratio']:g}")
    return (not why), why


def is_score(m: dict[str, Any], min_trades: int) -> float:
    """Critère d'optimisation : profit factor pénalisé par le faible nombre de trades."""
    if m["trades"] < min_trades:
        return -math.inf
    return min(m["profit_factor"], 5.0) - 1.0 / math.sqrt(m["trades"])


# --------------------------------------------------------------------- optimisation
def grid_combos(strategy: Strategy) -> list[dict[str, Any]]:
    keys = list(strategy.grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*strategy.grid.values())]


def _slice_signals(signals: list[Signal], cut: pd.Timestamp) -> tuple[list[Signal], list[Signal]]:
    a = [s for s in signals if s.time < cut]
    b = [s for s in signals if s.time >= cut]
    return a, b


def evaluate_strategy(strategy: Strategy, m1: pd.DataFrame, spread: np.ndarray, spec: SymbolSpec,
                      capital: float = DEFAULT_CAPITAL, split: float = SPLIT,
                      risk_pct: float = RISK_PCT, criteria: dict[str, float] = CRITERIA,
                      max_leverage: float | None = MAX_LEVERAGE) -> dict[str, Any]:
    """Optimise sur les 70 % premiers, évalue le meilleur jeu sur les 30 % restants."""
    k = int(round(len(m1) * split))
    m1_is, m1_oos = m1.iloc[:k], m1.iloc[k:]
    sp_is, sp_oos = spread[:k], spread[k:]
    cut = m1.index[k] if k < len(m1) else m1.index[-1] + pd.Timedelta(minutes=1)
    days_is, days_oos = trading_days(m1_is), trading_days(m1_oos)
    tried = []
    for params in grid_combos(strategy):
        # Les signaux ne dépendent que du passé : on les calcule sur toute la série puis on
        # coupe par date, ce qui évite l'effet de bord du démarrage à froid des indicateurs.
        sig_is, _ = _slice_signals(strategy.signals(m1, params), cut)
        r = simulate(m1_is, sig_is, spec, sp_is, capital, risk_pct, max_leverage=max_leverage)
        tried.append((params, compute_metrics(r.trades, capital, days_is, r.skipped_min_lot)))
    best, note = None, ""
    for min_trades in MIN_IS_TRADES:
        ok = [(is_score(m, min_trades), i) for i, (_, m) in enumerate(tried)]
        ok = [x for x in ok if x[0] > -math.inf]
        if ok:
            best = tried[max(ok)[1]]
            break
    if best is None:
        best = (dict(strategy.default_params), None)
        note = "trop peu de trades en échantillon : paramètres par défaut"
    params = best[0]
    if best[1] is None:
        sig_is, _ = _slice_signals(strategy.signals(m1, params), cut)
        r = simulate(m1_is, sig_is, spec, sp_is, capital, risk_pct, max_leverage=max_leverage)
        m_is = compute_metrics(r.trades, capital, days_is, r.skipped_min_lot)
    else:
        m_is = best[1]
    _, sig_oos = _slice_signals(strategy.signals(m1, params), cut)
    r = simulate(m1_oos, sig_oos, spec, sp_oos, capital, risk_pct, max_leverage=max_leverage)
    m_oos = compute_metrics(r.trades, capital, days_oos, r.skipped_min_lot)
    passed, why = evaluate_criteria(m_oos, criteria)
    return {"params": {k_: _py(v) for k_, v in params.items()}, "is": m_is, "oos": m_oos,
            "passed": passed, "fail_reasons": why,
            "_note": note, "_combos": len(tried)}


def _py(v: Any) -> Any:
    return v.item() if isinstance(v, np.generic) else v


# --------------------------------------------------------------------- orchestration
def load_inputs(data_dir: Path, symbols: list[str] | None):
    meta = read_meta(data_dir)
    prof_path = data_dir / "spread_profile.json"
    if prof_path.exists():
        profile = json.loads(prof_path.read_text(encoding="utf-8"))
    else:
        print("spread_profile.json absent : profil reconstruit à la volée.", file=sys.stderr)
        profile = build_profile(data_dir)
    syms = symbols or list_symbols(data_dir)
    return meta, profile, syms


def run_backtest(data_dir: Path | str, symbols: list[str] | None = None,
                 capital: float = DEFAULT_CAPITAL, strategies: dict[str, Strategy] | None = None,
                 split: float = SPLIT, verbose: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
    """Exécute tout le backtest ; renvoie (rapport au format JSON, informations annexes)."""
    data_dir = Path(data_dir)
    strategies = strategies or STRATEGIES
    meta, profile, syms = load_inputs(data_dir, symbols)
    results: dict[str, Any] = {}
    extras: dict[str, Any] = {"modes": {}, "notes": [], "capital": capital, "split": split}
    for sym in syms:
        path = data_dir / f"{sym}_M1.csv.gz"
        if not path.exists() or sym not in profile:
            print(f"AVERTISSEMENT : données ou profil absents pour {sym}, ignoré.", file=sys.stderr)
            continue
        m1 = load_bars(path)
        tick_path = data_dir / f"{sym}_ticks.csv.gz"
        if tick_path.exists():
            m1 = add_tick_features(m1, load_ticks(tick_path))  # NaN hors période couverte
        extras["modes"][sym] = tick_imbalance_mode(m1)
        spec = SymbolSpec.from_meta(meta.get("symbols", {}).get(sym))
        spread = spread_array(m1.index, profile[sym])
        results[sym] = {}
        for name, strat in strategies.items():
            t0 = _time.time()
            res = evaluate_strategy(strat, m1, spread, spec, capital, split)
            if res.pop("_note"):
                extras["notes"].append(f"{sym}/{name} : paramètres par défaut (peu de trades en échantillon)")
            res.pop("_combos")
            results[sym][name] = res
            if verbose:
                print(f"  {sym:10s} {name:15s} {_time.time() - t0:5.1f} s", file=sys.stderr)
    report = {
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "data_dir": str(data_dir),
        "synthetic": bool(meta.get("synthetic", False)),
        "criteria": dict(CRITERIA),
        "results": results,
    }
    return report, extras


# --------------------------------------------------------------------- affichage
_HEAD = ("Symbole", "Stratégie", "Éch.", "Trades", "Tr/j", "Réuss.%", "PF", "Perte/Gain",
         "DD %", "Net USD", "Net %", "Coûts USD", "OK")


def _rows(report: dict[str, Any]) -> list[tuple[str, ...]]:
    rows = []
    for sym, strats in report["results"].items():
        for name, r in strats.items():
            for label in ("is", "oos"):
                m = r[label]
                rows.append((sym, name, "IS" if label == "is" else "OOS", str(m["trades"]),
                             f"{m['trades_per_day']:.2f}", f"{m['win_rate'] * 100:.1f}",
                             f"{m['profit_factor']:.2f}", f"{m['loss_win_ratio']:.2f}",
                             f"{m['max_drawdown_pct']:.1f}", f"{m['net_usd']:.0f}",
                             f"{m['net_pct']:.1f}", f"{m['costs_usd']:.0f}",
                             ("oui" if r["passed"] else "non") if label == "oos" else ""))
    return rows


def format_table(report: dict[str, Any]) -> str:
    rows = _rows(report)
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(_HEAD)]
    line = "  ".join(h.ljust(w) for h, w in zip(_HEAD, widths))
    out = [line, "-" * len(line)]
    out += ["  ".join(c.ljust(w) for c, w in zip(r, widths)) for r in rows]
    return "\n".join(out)


def _best_candidate(report: dict[str, Any]) -> tuple[str, str, dict[str, Any]] | None:
    cands = [(s, n, r) for s, d in report["results"].items() for n, r in d.items()
             if r["oos"]["trades"] >= 10]
    if not cands:
        return None
    return max(cands, key=lambda x: (x[2]["oos"]["profit_factor"], x[2]["oos"]["net_usd"]))


def render_markdown(report: dict[str, Any], extras: dict[str, Any]) -> str:
    c = report["criteria"]
    L = ["# Rapport de backtest", "",
         f"Généré : {report['generated']} | données : `{report['data_dir']}` | "
         f"capital initial : {extras['capital']:.0f} USD | risque par trade : {RISK_PCT * 100:.1f} % | "
         f"découpe {extras['split'] * 100:.0f} % / {100 - extras['split'] * 100:.0f} %", ""]
    if report["synthetic"]:
        L += ["> **Données SYNTHÉTIQUES** (marche aléatoire) : ce rapport ne valide que la chaîne "
              "technique, jamais une stratégie. Aucune stratégie ne devrait passer ; si l'une passait, "
              "ce serait un signe de biais à chercher dans le code.", ""]
    L += ["## Critères d'acceptation (hors échantillon)", "",
          f"trades ≥ {c['min_trades']:g} ; profit factor ≥ {c['min_profit_factor']:g} ; "
          f"drawdown ≤ {c['max_drawdown_pct']:g} % ; gain net > {c['min_net_profit']:g} ; "
          f"perte moyenne / gain moyen ≤ {c['max_loss_win_ratio']:g}.", ""]
    for sym, strats in report["results"].items():
        L += [f"## {sym}", "",
              "| Stratégie | Paramètres retenus | Éch. | Trades | Tr/j | Réussite | PF | Perte/Gain "
              "| DD % | Net USD | Net % | Coûts USD | Validée |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for name, r in strats.items():
            params = ", ".join(f"{k}={v}" for k, v in r["params"].items())
            for label in ("is", "oos"):
                m = r[label]
                L.append(f"| {name} | {params if label == 'is' else ''} | {label.upper()} | {m['trades']} "
                         f"| {m['trades_per_day']:.2f} | {m['win_rate'] * 100:.1f} % | {m['profit_factor']:.2f} "
                         f"| {m['loss_win_ratio']:.2f} | {m['max_drawdown_pct']:.1f} | {m['net_usd']:.0f} "
                         f"| {m['net_pct']:.1f} | {m['costs_usd']:.0f} "
                         f"| {('oui' if r['passed'] else 'non') if label == 'oos' else ''} |")
        L.append("")
        L += ["Raisons d'échec (OOS) :", ""]
        for name, r in strats.items():
            L.append(f"- **{name}** : " + ("aucune, critères satisfaits" if r["passed"]
                                           else "; ".join(r["fail_reasons"])))
        L.append("")
    L += ["## Conclusion", ""]
    passed = [(s, n) for s, d in report["results"].items() for n, r in d.items() if r["passed"]]
    if passed:
        L.append("Stratégies satisfaisant tous les critères hors échantillon : "
                 + ", ".join(f"{s}/{n}" for s, n in passed) + ". À confirmer en démo avant toute "
                 "conclusion : un seul découpage 70/30 ne suffit pas à écarter la chance.")
    else:
        L.append("**Aucune stratégie ne passe les critères d'acceptation hors échantillon.** "
                 "Il ne faut donc pas la confier au bot en l'état.")
        best = _best_candidate(report)
        if best:
            s, n, r = best
            o = r["oos"]
            L.append("")
            L.append(f"Piste la plus prometteuse d'après les chiffres : **{s}/{n}** (OOS : {o['trades']} trades, "
                     f"PF {o['profit_factor']:.2f}, net {o['net_usd']:.0f} USD, DD {o['max_drawdown_pct']:.1f} %, "
                     f"coûts {o['costs_usd']:.0f} USD). Elle échoue sur : {'; '.join(r['fail_reasons'])}. "
                     "Le point faible commun reste le coût d'exécution (spread + glissement) comparé à "
                     "l'amplitude des mouvements visés ; allonger les objectifs et ne trader que les heures "
                     "au spread le plus bas est la première piste à tester, sur davantage d'historique.")
    L += ["", "## Limites et remarques", "",
          "- `eia_momentum` ne produit qu'environ un trade par semaine : atteindre 200 trades hors "
          "échantillon est structurellement impossible (le critère de nombre de trades échouera toujours).",
          "- La pertinence statistique est faible quand le nombre de trades hors échantillon est petit.",
          "- `tick_imbalance` : mode des données par symbole — "
          + ", ".join(f"{s} = {m}" for s, m in extras["modes"].items())
          + " (« ticks » : upticks/downticks réels ; « proxy » : position de la clôture dans la barre ; "
          "« mixte » : ticks sur la période couverte, proxy ailleurs).",
          "- Le drawdown est mesuré aux clôtures de trades ; le swap est supposé exprimé en points.",
          "- Le spread est celui du profil horaire (médiane), donc optimiste lors des pics ponctuels."]
    for note in extras["notes"]:
        L.append(f"- {note}")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ava_mt5.backtest", description="Backtest des stratégies de scalping pétrole")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="logs/backtest_report.json")
    ap.add_argument("--symbols", default=None, help="liste séparée par des virgules (défaut : tous)")
    ap.add_argument("--capital", type=float, default=DEFAULT_CAPITAL)
    args = ap.parse_args(argv)
    syms = [s.strip() for s in args.symbols.split(",") if s.strip()] if args.symbols else None
    t0 = _time.time()
    report, extras = run_backtest(args.data, syms, args.capital)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md = out.with_suffix(".md")
    md.write_text(render_markdown(report, extras), encoding="utf-8")
    print(format_table(report))
    ok = [f"{s}/{n}" for s, d in report["results"].items() for n, r in d.items() if r["passed"]]
    print(f"\nStratégies validées : {', '.join(ok) if ok else 'aucune'}")
    print(f"Rapports : {out} et {md} ({_time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
