"""Stratégie « rafale » : petites positions à volume minimum ouvertes toutes les 1 à 5 s,
fermées dès que le gain net (spread et glissement inclus) atteint X.

Le module contient :
  - des fonctions pures partagées par le backtest et l'exécution live (règles d'entrée, de sortie,
    garde-fous) ;
  - un backtest sur ticks (achat au ask, vente au bid, glissement d'un tick, décision sur le tick
    courant et exécution au tick suivant), avec courbe d'équité à la seconde ;
  - un générateur de ticks synthétiques (marche aléatoire) pour développer sans données réelles ;
  - l'exécution live, réservée aux comptes DÉMO, avec journaux et coupe-circuits.

Le module MetaTrader5 n'est importé que dans main().
Commandes : check | synth | backtest | live (voir `python -m ava_mt5.rafale --help`).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import heapq
import itertools
import json
import logging
import math
import os
import signal
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ava_mt5.core import Broker, Settings, TradingError

try:  # filtre news facultatif
    from ava_mt5.news import allows, compute_bias, read_bias
    from ava_mt5.news import load_json as _news_load_json
except Exception:  # pragma: no cover - news.py absent
    allows = compute_bias = read_bias = _news_load_json = None

MAGIC_RAFALE = 20261120
MAGIC_AUTRE_BOT = 20261119          # jamais touché
MAX_POSITIONS = 10
NY = ZoneInfo("America/New_York")
UTC = dt.timezone.utc

OBJECTIFS = (0.03, 0.05, 0.10, 0.20)
DUREES_MIN = (2, 5, 15)
INTERVALLES_S = (1, 5)
MODES = ("momentum", "pairs")

SORTIES = ("objectif", "duree_max", "stop", "coupe_circuit", "fin_donnees")

# critères hors échantillon (identiques aux autres stratégies)
CRIT_TRADES_MIN = 200
CRIT_PF_MIN = 1.15
CRIT_DD_MAX_PCT = 10.0
CRIT_RATIO_MAX = 2.0

DEFAULT_PARAMS: dict[str, Any] = {
    "mode": "momentum",
    "interval_s": 1,          # 1 à 5 s entre deux ouvertures
    "lookback_s": 5,          # fenêtre du micro-momentum
    "thr_ticks": 2,           # variation minimale du mid (en ticks) pour entrer
    "target": 0.05,           # X : gain net visé, en $ de prix par baril
    "max_life_min": 5,        # durée de vie maximale d'une position
    "sl_mult": 3.0,           # stop de sécurité = sl_mult x X ...
    "sl_usd": None,           # ... ou distance fixe en $ de prix (prioritaire si renseignée)
    "max_positions": MAX_POSITIONS,
    "slip_points": 1.0,       # glissement défavorable à l'entrée et à la sortie
    "breaker_pct": 0.02,      # fermeture totale si perte latente >= 2 % du capital
    "spread_filter": True,    # pas d'entrée si spread > spread_k x médiane horaire
    "spread_k": 1.5,
    "cooldown_s": 60,         # pause après une fermeture totale
    "hedging": None,          # None : inconnu (backtest) ; False : compte netting
}

# Symbole synthétique (valeurs proches d'un CFD pétrole)
SYNTH_SYMBOL = "CrudeOIL"
SYNTH_META = {
    "point": 0.01, "digits": 2, "contract_size": 1000.0, "volume_min": 0.01,
    "volume_step": 0.01, "tick_size": 0.01, "tick_value": 10.0,
    "swap_long": -25.0, "swap_short": 8.0, "server_offset_h": 2, "synthetic": True,
}


class CompteReelRefuse(TradingError):
    """Le compte n'est pas un compte démo : aucun ordre n'est envoyé."""


# =====================================================================================
# Règles pures (partagées par le backtest et le live)
# =====================================================================================
def check_hedging(mt5: Any) -> tuple[bool, str]:
    """Vrai si le compte est en mode hedging (positions opposées indépendantes)."""
    info = mt5.account_info()
    if info is None:
        return False, "account_info indisponible : mode de marge inconnu, hedging non confirmé."
    mode = getattr(info, "margin_mode", None)
    if mode == mt5.ACCOUNT_MARGIN_MODE_RETAIL_HEDGING:
        return True, "Compte en mode HEDGING : positions achat et vente simultanées possibles."
    return False, (f"Compte NON hedging (margin_mode={mode}) : le mode « paires » est interdit ; "
                   "seul le mode momentum reste possible, avec une seule position nette.")


def check_mode_allowed(mode: str, hedging: bool | None) -> tuple[bool, str]:
    """hedging None (backtest, mode inconnu) : tout est autorisé."""
    if mode not in MODES:
        return False, f"Mode inconnu : {mode}"
    if mode == "pairs" and hedging is False:
        return False, "Mode « paires » interdit : le compte n'est pas en hedging."
    return True, "ok"


def normalize_params(params: dict | None) -> dict:
    p = dict(DEFAULT_PARAMS)
    p.update({k: v for k, v in (params or {}).items() if k in DEFAULT_PARAMS})
    p["interval_s"] = int(p["interval_s"])
    if not 1 <= p["interval_s"] <= 5:
        raise ValueError("interval_s doit être compris entre 1 et 5 secondes.")
    if p["target"] <= 0 or p["max_life_min"] <= 0 or p["max_positions"] < 1:
        raise ValueError("target, max_life_min et max_positions doivent être positifs.")
    if p["max_positions"] > MAX_POSITIONS:
        raise ValueError(f"max_positions ne peut pas dépasser {MAX_POSITIONS}.")
    ok, msg = check_mode_allowed(p["mode"], p["hedging"])
    if not ok:
        raise ValueError(msg)
    return p


def effective_max_positions(params: dict) -> int:
    """Sans hedging : une seule position nette."""
    return 1 if params.get("hedging") is False else int(params["max_positions"])


def stop_distance(params: dict) -> float:
    return float(params["sl_usd"]) if params.get("sl_usd") else float(params["sl_mult"]) * float(params["target"])


def fixed_volume(spec: dict) -> float:
    """Volume toujours égal au minimum du symbole : jamais augmenté, même après une perte."""
    return float(spec["volume_min"])


def momentum_signal(mid_now: float, mid_past: float, tick_size: float, thr_ticks: float) -> str | None:
    d = mid_now - mid_past
    seuil = thr_ticks * tick_size - 1e-9
    if d >= seuil:
        return "buy"
    if d <= -seuil:
        return "sell"
    return None


def momentum_signals(mid_sec: np.ndarray, lookback_s: int, tick_size: float, thr_ticks: float) -> np.ndarray:
    """Version vectorisée de momentum_signal sur une série de mids à la seconde (+1, -1, 0)."""
    sig = np.zeros(len(mid_sec), dtype=np.int8)
    lb = int(lookback_s)
    if len(mid_sec) <= lb:
        return sig
    d = mid_sec[lb:] - mid_sec[:-lb]
    seuil = thr_ticks * tick_size - 1e-9
    vue = sig[lb:]
    vue[d >= seuil] = 1
    vue[d <= -seuil] = -1
    return sig


def spread_ok(spread: float, median: float | None, k: float = 1.5) -> bool:
    if median is None or not median > 0:
        return True
    return spread <= k * median + 1e-9


def plan_entries(mode: str, signal: str | None, n_open: int, max_positions: int) -> list[str]:
    """Côtés à ouvrir maintenant. Une paire est ouverte en entier ou pas du tout."""
    if mode == "pairs":
        return ["buy", "sell"] if n_open + 2 <= max_positions else []
    if signal is None or n_open + 1 > max_positions:
        return []
    return [signal]


def entry_fill(side: str, bid: float, ask: float, slip: float) -> float:
    return ask + slip if side == "buy" else bid - slip


def net_gain(side: str, entry: float, bid: float, ask: float, slip: float) -> float:
    """Gain net en $ de prix par baril si on sort maintenant : prix de sortie réel
    (bid pour un achat, ask pour une vente) moins le glissement."""
    return (bid - slip - entry) if side == "buy" else (entry - (ask + slip))


def exit_reason(side: str, entry: float, bid: float, ask: float, age_s: float,
                params: dict, slip: float) -> str | None:
    """Raison de fermeture d'une position (ou None) : stop, objectif, durée maximale."""
    if side == "buy":
        perte = entry - bid
    else:
        perte = ask - entry
    if perte >= stop_distance(params) - 1e-9:
        return "stop"
    if net_gain(side, entry, bid, ask, slip) >= float(params["target"]) - 1e-9:
        return "objectif"
    if age_s >= float(params["max_life_min"]) * 60.0 - 1e-9:
        return "duree_max"
    return None


def breaker_hit(floating_usd: float, capital: float, pct: float = 0.02) -> bool:
    """Fermeture de tout si la perte latente totale atteint pct du capital."""
    return floating_usd <= -pct * capital + 1e-9


# =====================================================================================
# Données : ticks synthétiques, chargement
# =====================================================================================
def synthetic_ticks(days: int = 5, seed: int = 0, start: str = "2026-01-05",
                    base_price: float = 70.0) -> pd.DataFrame:
    """Marche aléatoire de ticks pétrole : 1 à 3 ticks/s, spread variable par heure
    (≈ 0,03 $ en séance US, plus large la nuit). `days` jours ouvrés, pause 22h-23h UTC."""
    rng = np.random.default_rng(seed)
    jour = dt.datetime.fromisoformat(start).replace(tzinfo=UTC)
    # (taux ticks/s, spread de base en cents, volatilité relative)
    def profil(h: int) -> tuple[float, int, float]:
        if 13 <= h < 20:
            return 3.0, 3, 1.6
        if 7 <= h < 13 or h == 20 or h == 21:
            return 1.8, 4, 1.0
        return 1.0, 8, 0.6

    ts_all, bid_all, ask_all = [], [], []
    mid_c = base_price * 100.0
    fait = 0
    while fait < days:
        if jour.weekday() < 5:
            for h in range(24):
                if h == 22:
                    continue
                taux, spr, vol = profil(h)
                n = int(rng.poisson(taux * 3600))
                if n == 0:
                    continue
                t0 = int((jour + dt.timedelta(hours=h)).timestamp() * 1000)
                ts = t0 + np.sort(rng.integers(0, 3_600_000, n))
                # 0,008 $ de volatilité par racine de seconde, modulée par heure
                sig_tick_c = 100.0 * 0.008 * vol / math.sqrt(taux)
                mid = mid_c + np.cumsum(rng.normal(0.0, sig_tick_c, n))
                mid_c = float(mid[-1])
                if spr == 3:
                    sp = spr + rng.choice([0, 0, 0, 0, 1, 2], n)
                elif spr == 4:
                    sp = spr + rng.choice([0, 0, 1, 1, 2, 4], n)
                else:
                    sp = spr + rng.choice([-2, 0, 0, 2, 4, 8], n)
                bid_c = np.rint(mid - sp / 2.0)
                ts_all.append(ts)
                bid_all.append(bid_c / 100.0)
                ask_all.append((bid_c + sp) / 100.0)
            fait += 1
        jour += dt.timedelta(days=1)
    t = np.concatenate(ts_all)
    return pd.DataFrame({"time_msc": t, "bid": np.round(np.concatenate(bid_all), 2),
                         "ask": np.round(np.concatenate(ask_all), 2),
                         "flags": np.full(len(t), 6, dtype=np.int64)})


def write_synthetic(out_dir: str | os.PathLike, days: int = 5, seed: int = 0,
                    symbol: str = SYNTH_SYMBOL) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = synthetic_ticks(days, seed)
    df.to_csv(out / f"{symbol}_ticks.csv.gz", index=False)
    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump({symbol: SYNTH_META}, f, indent=2)
    return out


def load_ticks(data_dir: str | os.PathLike, symbol: str) -> tuple[pd.DataFrame, dict]:
    d = Path(data_dir)
    ticks = pd.read_csv(d / f"{symbol}_ticks.csv.gz",
                        dtype={"time_msc": "int64", "bid": "float64", "ask": "float64"})
    meta_all = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    return ticks, meta_spec(meta_all, symbol)


def meta_spec(meta: dict, symbol: str | None = None) -> dict:
    """Accepte la spec d'un symbole ou le meta.json complet (par symbole)."""
    if "point" in meta:
        return meta
    if symbol and symbol in meta:
        return meta[symbol]
    if len(meta) == 1:
        return next(iter(meta.values()))
    raise ValueError("meta : symbole introuvable (préciser params['symbol']).")


# =====================================================================================
# Backtest sur ticks
# =====================================================================================
def _rollovers(t0_ms: int, t1_ms: int) -> tuple[np.ndarray, np.ndarray]:
    """Instants (ms UTC) de 17:00 New York en semaine, et poids cumulés (triple le mercredi)."""
    d0 = dt.datetime.fromtimestamp(t0_ms / 1000, UTC).date() - dt.timedelta(days=1)
    d1 = dt.datetime.fromtimestamp(t1_ms / 1000, UTC).date() + dt.timedelta(days=1)
    inst, poids = [], []
    d = d0
    while d <= d1:
        if d.weekday() < 5:
            ny = dt.datetime(d.year, d.month, d.day, 17, tzinfo=NY)
            inst.append(int(ny.timestamp() * 1000))
            poids.append(3 if d.weekday() == 2 else 1)
        d += dt.timedelta(days=1)
    return np.array(inst, dtype=np.int64), np.concatenate([[0], np.cumsum(poids)]).astype(np.int64)


class DonneesTicks:
    """Ticks préparés une fois (grille à la seconde, médiane de spread horaire) pour tout
    un lot de combinaisons."""

    def __init__(self, ticks: pd.DataFrame, spec: dict, hourly_median: np.ndarray | None = None):
        self.spec = spec
        self.t = ticks["time_msc"].to_numpy(np.int64)
        self.bid = ticks["bid"].to_numpy(np.float64)
        self.ask = ticks["ask"].to_numpy(np.float64)
        self.n = len(self.t)
        if self.n < 3:
            raise ValueError("Pas assez de ticks pour un backtest.")
        spread = self.ask - self.bid
        h_t = (self.t // 3_600_000) % 24
        if hourly_median is None:
            glob = float(np.median(spread))
            hourly_median = np.array([float(np.median(spread[h_t == h])) if (h_t == h).any() else glob
                                      for h in range(24)])
        self.hmed = np.asarray(hourly_median, dtype=float)
        t0s = -(-int(self.t[0]) // 1000) * 1000
        self.n_sec = int((self.t[-1] - t0s) // 1000) + 1
        self.times = t0s + np.arange(self.n_sec, dtype=np.int64) * 1000
        self.idx_sec = np.searchsorted(self.t, self.times, side="right") - 1
        self.active = (self.times - self.t[self.idx_sec]) <= 10_000   # marché actif (tick < 10 s)
        self.mid_sec = ((self.bid + self.ask) / 2.0)[self.idx_sec]
        self.spread_sec = spread[self.idx_sec]
        self.hour_sec = (self.times // 3_600_000) % 24
        self.days = int(len(np.unique(self.t // 86_400_000)))
        self.roll_ms, self.roll_cum = _rollovers(int(self.t[0]), int(self.t[-1]))
        self._plans: dict = {}

    def plan(self, p: dict) -> np.ndarray:
        """Par seconde : 0 rien, +1 achat, -1 vente, 2 paire."""
        spec = self.spec
        cle = (p["mode"], p["interval_s"], p["lookback_s"], p["thr_ticks"],
               p["spread_filter"], p["spread_k"], spec["tick_size"])
        if cle in self._plans:
            return self._plans[cle]
        s = np.arange(self.n_sec)
        base = self.active & (s % p["interval_s"] == 0)
        if p["spread_filter"]:
            base &= self.spread_sec <= p["spread_k"] * self.hmed[self.hour_sec] + 1e-9
        if p["mode"] == "pairs":
            plan = np.where(base, 2, 0).astype(np.int8)
        else:
            lb = int(p["lookback_s"])
            sig = momentum_signals(self.mid_sec, lb, spec["tick_size"], p["thr_ticks"])
            passe = np.zeros(self.n_sec, dtype=bool)
            passe[lb:] = self.active[:-lb]
            plan = np.where(base & passe, sig, 0).astype(np.int8)
        self._plans[cle] = plan
        return plan


def _max_dd_pct(serie: np.ndarray) -> float:
    if len(serie) == 0:
        return 0.0
    pic = np.maximum.accumulate(serie)
    # plafonné à 100 % : au-delà, le compte est ruiné
    return float(min(np.max((pic - serie) / np.maximum(pic, 1e-9)) * 100.0, 100.0))


def _metrics_vide(capital: float, d: DonneesTicks, vol: float) -> dict:
    return {"trades": 0, "trades_per_day": 0.0, "win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
            "loss_gain_ratio": None, "profit_factor": None, "dd_equity_pct": 0.0,
            "dd_balance_pct": 0.0, "net_usd": 0.0, "net_pct": 0.0, "spread_cost_usd": 0.0,
            "slippage_cost_usd": 0.0, "swap_usd": 0.0, "swap_applied": False, "total_costs_usd": 0.0,
            "breaker_cuts": 0, "exits": {k: 0 for k in SORTIES}, "max_open": 0, "volume": vol,
            "days": d.days, "final_equity": capital}


def backtest_rafale(ticks: pd.DataFrame | DonneesTicks, meta: dict, params: dict | None = None,
                    capital: float = 10000.0, *, return_detail: bool = False) -> dict:
    """Backtest sur ticks. Décision sur le tick courant (grille d'une seconde), exécution au
    tick SUIVANT : achat au ask, vente au bid, glissement défavorable à l'entrée et à la sortie.
    Équité = solde + latent, valorisée à chaque seconde."""
    p = normalize_params(params)
    d = ticks if isinstance(ticks, DonneesTicks) else DonneesTicks(ticks, meta_spec(meta, (params or {}).get("symbol")))
    spec = d.spec
    t, bid, ask, N = d.t, d.bid, d.ask, d.n
    point = float(spec["point"])
    slip = float(p["slip_points"]) * point
    vol = fixed_volume(spec)
    k = float(spec["contract_size"]) * vol            # $ par $ de mouvement de prix
    target = float(p["target"])
    sl_dist = stop_distance(p)
    life_ms = int(float(p["max_life_min"]) * 60_000)
    max_pos = effective_max_positions(p)
    limit = float(p["breaker_pct"]) * capital
    # le stop borne la perte latente de chaque position : sans risque d'atteindre le seuil, on
    # évite de revaloriser chaque seconde
    need_breaker = max_pos * sl_dist * k >= limit
    plan = d.plan(p)
    idx_sec = d.idx_sec
    visite = np.flatnonzero(d.active) if need_breaker else np.flatnonzero(plan)

    def find_exit(sgn: int, entry: float, e: int) -> tuple[int, int]:
        """Premier tick de décision touchant stop / objectif / expiration ; renvoie
        (index d'exécution = tick suivant, code de sortie)."""
        n_e = int(np.searchsorted(t, t[e] + life_ms, side="left"))
        hi = min(n_e, N - 1)
        expire = n_e <= N - 1
        if sgn > 0:
            arr, tp, sl = bid, entry + target + slip - 1e-9, entry - sl_dist + 1e-9
        else:
            arr, tp, sl = ask, entry - target - slip + 1e-9, entry + sl_dist - 1e-9
        a, pas = e + 1, 64
        while a <= hi:
            b = min(a + pas, hi + 1)
            seg = arr[a:b]
            h = (seg >= tp) | (seg <= sl) if sgn > 0 else (seg <= tp) | (seg >= sl)
            j = int(h.argmax())
            if h[j]:
                i = a + j
                stop = seg[j] <= sl if sgn > 0 else seg[j] >= sl
                return min(i + 1, N - 1), (2 if stop else 0)
            a, pas = b, min(pas * 4, 4096)
        return min(hi + 1, N - 1), (1 if expire else 4)

    pos: dict[int, list] = {}       # uid -> [x, e, sgn, entry, code]
    heap: list[tuple[int, int]] = []
    fait: list[tuple] = []
    uid = 0
    max_open = 0
    cuts = 0
    froid_jusqua = -1
    for s in visite.tolist():
        i = int(idx_sec[s])
        while heap and heap[0][0] <= i:
            _, u = heapq.heappop(heap)
            r = pos.pop(u)
            fait.append((r[1], r[0], r[2], r[3], r[4]))
        if need_breaker and pos and s >= froid_jusqua:
            fl = 0.0
            for r in pos.values():
                if r[1] <= i:
                    fl += (bid[i] - r[3]) if r[2] > 0 else (r[3] - ask[i])
            if breaker_hit(fl * k, capital, p["breaker_pct"]):
                cuts += 1
                heap = []
                for u, r in pos.items():
                    nx = min(max(i + 1, r[1] + 1), N - 1)
                    if nx < r[0]:
                        r[0], r[4] = nx, 3
                    heap.append((r[0], u))
                heapq.heapify(heap)
                froid_jusqua = s + int(p["cooldown_s"])
        code = int(plan[s])
        if code == 0 or s < froid_jusqua:
            continue
        e = i + 1
        if e >= N:
            continue
        cotes = plan_entries(p["mode"], {1: "buy", -1: "sell"}.get(code), len(pos), max_pos)
        for side in cotes:
            sgn = 1 if side == "buy" else -1
            entry = entry_fill(side, bid[e], ask[e], slip)
            x, c = find_exit(sgn, entry, e)
            uid += 1
            pos[uid] = [x, e, sgn, entry, c]
            heapq.heappush(heap, (x, uid))
        if len(pos) > max_open:
            max_open = len(pos)
    for r in pos.values():
        fait.append((r[1], r[0], r[2], r[3], r[4]))

    if not fait:
        res = _metrics_vide(capital, d, vol)
        res["breaker_cuts"] = cuts
        if return_detail:
            res["trades_df"] = pd.DataFrame()
            res["equity"] = np.full(d.n_sec, capital)
            res["balance"] = np.full(d.n_sec, capital)
            res["times"] = d.times
        return res

    E = np.array([f[0] for f in fait], dtype=np.int64)
    X = np.array([f[1] for f in fait], dtype=np.int64)
    SG = np.array([f[2] for f in fait], dtype=np.int64)
    EN = np.array([f[3] for f in fait], dtype=np.float64)
    CD = np.array([f[4] for f in fait], dtype=np.int64)
    ex_prix = np.where(SG > 0, bid[X] - slip, ask[X] + slip)
    pnl_prix = SG * (ex_prix - EN)
    # swap : appliqué seulement aux positions qui traversent 17:00 New York
    n_roll = d.roll_cum[np.searchsorted(d.roll_ms, t[X], side="right")] \
        - d.roll_cum[np.searchsorted(d.roll_ms, t[E], side="right")]
    swap_pts = np.where(SG > 0, float(spec.get("swap_long", 0.0)), float(spec.get("swap_short", 0.0)))
    swap_usd = n_roll * swap_pts * point * k
    pnl = pnl_prix * k + swap_usd
    spread_cost = (((ask - bid)[E] + (ask - bid)[X]) / 2.0) * k
    slip_cost = np.full(len(E), 2.0 * slip * k)

    # courbe d'équité à la seconde : solde + latent (positions ouvertes valorisées au marché)
    ns = d.n_sec
    s0 = np.searchsorted(idx_sec, E, side="left")
    s1 = np.searchsorted(idx_sec, X, side="left")
    achat = SG > 0

    def delta(mask: np.ndarray, poids: np.ndarray | None = None) -> np.ndarray:
        a = np.bincount(s0[mask], weights=None if poids is None else poids[mask], minlength=ns + 1)
        b = np.bincount(s1[mask], weights=None if poids is None else poids[mask], minlength=ns + 1)
        return np.cumsum(a - b)[:ns]

    nb, ns_ = delta(achat), delta(~achat)
    sb, ss = delta(achat, EN), delta(~achat, EN)
    bid_s, ask_s = bid[idx_sec], ask[idx_sec]
    latent = k * (nb * bid_s - sb + ss - ns_ * ask_s)
    realise = np.bincount(np.minimum(s1, ns - 1), weights=pnl, minlength=ns)[:ns]
    solde = capital + np.cumsum(realise)
    equite = solde + latent

    gains, pertes = pnl[pnl > 0], pnl[pnl < 0]
    gp, gl = float(gains.sum()), float(-pertes.sum())
    n = len(pnl)
    avg_win = float(gains.mean()) if len(gains) else 0.0
    avg_loss = float(-pertes.mean()) if len(pertes) else 0.0
    res = {
        "trades": n,
        "trades_per_day": n / max(d.days, 1),
        "win_rate": float((pnl > 0).mean()),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "loss_gain_ratio": (avg_loss / avg_win) if avg_win > 0 else None,
        "profit_factor": (gp / gl) if gl > 0 else (999.0 if gp > 0 else None),
        "dd_equity_pct": _max_dd_pct(equite),
        "dd_balance_pct": _max_dd_pct(solde),
        "net_usd": float(pnl.sum()),
        "net_pct": float(pnl.sum()) / capital * 100.0,
        "spread_cost_usd": float(spread_cost.sum()),
        "slippage_cost_usd": float(slip_cost.sum()),
        "swap_usd": float(swap_usd.sum()),
        "swap_applied": bool((n_roll > 0).any()),
        "breaker_cuts": cuts,
        "exits": {name: int((CD == c).sum()) for c, name in enumerate(SORTIES)},
        "max_open": max_open,
        "volume": vol,
        "days": d.days,
        "final_equity": float(capital + pnl.sum()),
    }
    res["total_costs_usd"] = res["spread_cost_usd"] + res["slippage_cost_usd"]
    if return_detail:
        res["trades_df"] = pd.DataFrame({
            "e": E, "x": X, "side": np.where(achat, "buy", "sell"), "volume": vol,
            "entry": EN, "exit": ex_prix, "pnl": pnl, "swap": swap_usd,
            "reason": [SORTIES[c] for c in CD]}).sort_values("e", kind="stable").reset_index(drop=True)
        res["equity"], res["balance"], res["times"] = equite, solde, d.times
    return res


# =====================================================================================
# Grille, découpe IS / OOS, rapport
# =====================================================================================
def evaluate_oos(m: dict) -> tuple[bool, list[str]]:
    raisons = []
    if m["trades"] < CRIT_TRADES_MIN:
        raisons.append(f"trades {m['trades']} < {CRIT_TRADES_MIN}")
    pf = m["profit_factor"]
    if pf is None or pf < CRIT_PF_MIN:
        raisons.append(f"profit factor {('n/d' if pf is None else f'{pf:.2f}')} < {CRIT_PF_MIN}")
    if m["dd_equity_pct"] > CRIT_DD_MAX_PCT:
        raisons.append(f"drawdown d'équité {m['dd_equity_pct']:.2f} % > {CRIT_DD_MAX_PCT} %")
    if not m["net_usd"] > 0:
        raisons.append(f"résultat net {m['net_usd']:.2f} $ <= 0")
    r = m["loss_gain_ratio"]
    if r is None or r > CRIT_RATIO_MAX:
        raisons.append(f"perte moy./gain moy. {('n/d' if r is None else f'{r:.2f}')} > {CRIT_RATIO_MAX}")
    return (not raisons), raisons


def build_grid(quick: bool = False, max_combos: int | None = None) -> list[dict]:
    grille = [{"mode": m, "target": x, "max_life_min": dur, "interval_s": it}
              for m, x, dur, it in itertools.product(MODES, OBJECTIFS, DUREES_MIN, INTERVALLES_S)]
    if quick:
        grille = [g for g in grille if g["target"] in (0.05, 0.10) and g["max_life_min"] == 5]
    if max_combos and max_combos < len(grille):
        idx = np.unique(np.linspace(0, len(grille) - 1, max_combos).round().astype(int))
        grille = [grille[i] for i in idx]
    return grille


_CTX: dict[str, Any] = {}


def _run_combo(params: dict) -> tuple[dict, dict, dict]:
    c = _CTX
    t0 = time.time()
    m_is = backtest_rafale(c["is"], c["spec"], params, c["capital"])
    m_oos = backtest_rafale(c["oos"], c["spec"], params, c["capital"])
    m_is["seconds"], m_oos["seconds"] = round(time.time() - t0, 2), 0.0
    return params, m_is, m_oos


def run_grid(ticks: pd.DataFrame, spec: dict, grille: list[dict], capital: float = 10000.0,
             split: float = 0.7, jobs: int = 1, progress: bool = False) -> list[dict]:
    cut = int(len(ticks) * split)
    ticks_is, ticks_oos = ticks.iloc[:cut], ticks.iloc[cut:]
    d_is = DonneesTicks(ticks_is, spec)
    d_oos = DonneesTicks(ticks_oos, spec, hourly_median=d_is.hmed)   # médianes tirées de l'IS
    _CTX.update({"is": d_is, "oos": d_oos, "spec": spec, "capital": capital})
    sorties = []
    if jobs > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(jobs) as pool:
            it = pool.imap(_run_combo, grille)
            for n, r in enumerate(it, 1):
                sorties.append(r)
                if progress:
                    print(f"  {n}/{len(grille)}", file=sys.stderr, flush=True)
    else:
        for n, g in enumerate(grille, 1):
            sorties.append(_run_combo(g))
            if progress:
                print(f"  {n}/{len(grille)}", file=sys.stderr, flush=True)
    results = []
    for params, m_is, m_oos in sorties:
        ok, why = evaluate_oos(m_oos)
        results.append({"params": params, "is": m_is, "oos": m_oos, "passed": ok, "fail_reasons": why})
    return results


def choose_best(results: list[dict]) -> dict | None:
    """Choix sur l'IS uniquement (aucune information OOS) : meilleur résultat net IS parmi les
    combinaisons ayant assez de trades et un drawdown acceptable, sinon meilleur net IS."""
    if not results:
        return None
    ok = [r for r in results if r["is"]["trades"] >= CRIT_TRADES_MIN
          and r["is"]["dd_equity_pct"] <= CRIT_DD_MAX_PCT]
    return max(ok or results, key=lambda r: r["is"]["net_usd"])


def _f(x, nd=2, suf=""):
    return "n/d" if x is None else f"{x:.{nd}f}{suf}"


def conclusion_text(results: list[dict], best: dict | None, synthetic: bool) -> str:
    n = len(results)
    n_pass = sum(1 for r in results if r["passed"])
    nets = [r["oos"]["net_usd"] for r in results]
    pos = sum(1 for v in nets if v > 0)
    lignes = []
    if synthetic:
        lignes.append("ATTENTION : ces résultats viennent de ticks SYNTHÉTIQUES (marche aléatoire sans "
                      "structure exploitable). Ils servent à vérifier la mécanique, pas à juger la stratégie.")
    lignes.append(f"{n_pass} combinaison(s) sur {n} satisfont tous les critères hors échantillon "
                  f"(>= {CRIT_TRADES_MIN} trades, PF >= {CRIT_PF_MIN}, DD d'équité <= {CRIT_DD_MAX_PCT:g} %, "
                  f"net > 0, perte moy./gain moy. <= {CRIT_RATIO_MAX:g}) ; {pos} sur {n} ont un net OOS positif.")
    if n > 1:
        lignes.append(f"Avec {n} combinaisons testées, quelques passages par pur hasard sont attendus : "
                      "un résultat isolé ne prouve rien, il doit être confirmé sur une autre période.")
    if best:
        o = best["oos"]
        lignes.append(f"Combinaison retenue sur l'IS seulement : {best['params']}. "
                      f"OOS : {o['trades']} trades, net {o['net_usd']:.2f} $ ({o['net_pct']:.2f} %), "
                      f"PF {_f(o['profit_factor'])}, DD d'équité {o['dd_equity_pct']:.2f} % "
                      f"(DD de solde {o['dd_balance_pct']:.2f} %), passed = {best['passed']}.")
    if best:
        o = best["oos"]
        brut = o["net_usd"] + o["total_costs_usd"]
        lignes.append(f"Coûts de la combinaison retenue (OOS) : spread {o['spread_cost_usd']:.0f} $ + "
                      f"glissement {o['slippage_cost_usd']:.0f} $ = {o['total_costs_usd']:.0f} $ pour un "
                      f"résultat avant coûts de {brut:.0f} $. À chaque aller-retour, le marché prélève le "
                      "spread et deux glissements ; avec un objectif de quelques cents, ce péage est "
                      "du même ordre que le gain visé. Un drawdown affiché à 100 % signifie compte ruiné.")
    lignes.append("Mode « paires » : un achat et une vente simultanés donnent une exposition nette nulle, "
                  "et chaque paire paie deux spreads (plus quatre glissements). Seul l'écart entre les "
                  "deux fermetures crée du résultat, ce qui revient à du momentum désordonné : on garde "
                  "le côté qui va dans le bon sens, l'autre est coupé par le stop ou la durée maximale. "
                  "Sans avantage directionnel, l'espérance reste inférieure à zéro de tous les coûts. "
                  "Les chiffres ci-dessus permettent de le vérifier.")
    pairs = [r for r in results if r["params"]["mode"] == "pairs"]
    if pairs:
        pnet = sum(r["oos"]["net_usd"] for r in pairs)
        pp = sum(1 for r in pairs if r["passed"])
        lignes.append(f"Mode paires : {len(pairs)} combinaisons, net OOS cumulé {pnet:.0f} $, "
                      f"{pp} validée(s).")
    lignes.append("Limites : swap supposé exprimé en points ; exécution pessimiste (les sorties sont "
                  "décidées sur un tick et exécutées au suivant, sans le TP serveur qui, en live, "
                  "exécute au niveau) ; les coupe-circuits de journée du live (−2 % jour, 5 pertes "
                  "consécutives, DD 8 %) ne sont pas simulés ; pas de rejet d'ordre ni de latence variable.")
    if n_pass == 0:
        lignes.append("Conclusion : aucune combinaison ne passe. Ne pas passer en réel ; la stratégie ne "
                      "démontre pas d'avantage statistique net de coûts sur ces données.")
    else:
        lignes.append("Conclusion : au moins une combinaison passe sur ces données. Étape suivante : "
                      "observation puis démo uniquement, sur une période distincte, avant toute autre "
                      "décision.")
    return "\n\n".join(lignes)


def report_markdown(report: dict) -> str:
    out = ["# Rapport stratégie rafale", "",
           f"Généré : {report['generated']} — données synthétiques : {report['synthetic']} — "
           f"hedging : {report['hedging']}", "",
           "| mode | X $ | durée min | intervalle s | IS trades | IS net % | IS PF | OOS trades | "
           "OOS net % | OOS PF | OOS DD éq. % | OOS DD solde % | OOS coûts $ | passed |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(report["results"], key=lambda r: (r["params"]["mode"], r["params"]["target"],
                                                       r["params"]["max_life_min"], r["params"]["interval_s"])):
        p, i, o = r["params"], r["is"], r["oos"]
        out.append(f"| {p['mode']} | {p['target']:.2f} | {p['max_life_min']} | {p['interval_s']} | "
                   f"{i['trades']} | {i['net_pct']:.2f} | {_f(i['profit_factor'])} | {o['trades']} | "
                   f"{o['net_pct']:.2f} | {_f(o['profit_factor'])} | {o['dd_equity_pct']:.2f} | "
                   f"{o['dd_balance_pct']:.2f} | {o['total_costs_usd']:.0f} | "
                   f"{'oui' if r['passed'] else 'non'} |")
    out += ["", "## Conclusion", "", report["conclusion"], ""]
    return "\n".join(out)


def _jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if (isinstance(o, float) and not math.isfinite(o)) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def cmd_backtest(args: argparse.Namespace) -> int:
    d = Path(args.data)
    if args.synthetic or not (d / f"{args.symbol}_ticks.csv.gz").exists():
        if not args.synthetic:
            print(f"Ticks absents dans {d} : génération de ticks SYNTHÉTIQUES ({args.days} jours).",
                  file=sys.stderr)
        write_synthetic(d, args.days, args.seed, args.symbol)
    ticks, spec = load_ticks(d, args.symbol)
    synthetic = bool(spec.get("synthetic"))
    grille = build_grid(args.quick, args.max_combos)
    print(f"{len(ticks):,} ticks, {len(grille)} combinaisons, {args.jobs} processus.", file=sys.stderr)
    t0 = time.time()
    results = run_grid(ticks, spec, grille, args.capital, jobs=args.jobs, progress=True)
    best = choose_best(results)
    report = {"generated": dt.datetime.now(UTC).isoformat(timespec="seconds"), "synthetic": synthetic,
              "hedging": None, "symbol": args.symbol, "capital": args.capital,
              "ticks": int(len(ticks)), "elapsed_s": round(time.time() - t0, 1),
              "results": results, "best": best}
    report["conclusion"] = conclusion_text(results, best, synthetic)
    report = _jsonable(report)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    out.with_suffix(".md").write_text(report_markdown(report), encoding="utf-8")
    print(f"Rapport : {out} et {out.with_suffix('.md')} ({report['elapsed_s']} s)", file=sys.stderr)
    return 0


# =====================================================================================
# Exécution live (démo uniquement)
# =====================================================================================
@dataclass
class Pos:
    ticket: int
    side: str
    volume: float
    price_open: float
    sl: float
    tp: float
    profit: float
    swap: float
    time: float
    magic: int = MAGIC_RAFALE
    virtual: bool = False


class PaperExecutor:
    """Exécution virtuelle (observation) : mêmes règles, aucun ordre envoyé."""

    def __init__(self, spec: dict, slip: float):
        self.spec, self.slip = spec, slip
        self.book: dict[int, Pos] = {}
        self.realised = 0.0
        self._n = 0
        self._sortis: dict[int, tuple[float, str]] = {}

    def _k(self, vol: float) -> float:
        return float(self.spec["contract_size"]) * vol

    def positions(self, tick: Any) -> list[Pos]:
        for p in list(self.book.values()):
            sortie = tick.bid if p.side == "buy" else tick.ask
            p.profit = (sortie - p.price_open if p.side == "buy" else p.price_open - sortie) * self._k(p.volume)
            hit = None
            if p.side == "buy":
                if p.sl and tick.bid <= p.sl:
                    hit = ("sl_serveur", p.sl)
                elif p.tp and tick.bid >= p.tp:
                    hit = ("tp_serveur", p.tp)
            else:
                if p.sl and tick.ask >= p.sl:
                    hit = ("sl_serveur", p.sl)
                elif p.tp and tick.ask <= p.tp:
                    hit = ("tp_serveur", p.tp)
            if hit:
                px = hit[1]
                pnl = (px - p.price_open if p.side == "buy" else p.price_open - px) * self._k(p.volume)
                self.realised += pnl
                self._sortis[p.ticket] = (pnl, hit[0])
                del self.book[p.ticket]
        return list(self.book.values())

    def open(self, side: str, tick: Any, volume: float, sl: float, tp: float, now: float) -> dict:
        self._n += 1
        px = entry_fill(side, tick.bid, tick.ask, self.slip)
        t = -self._n
        self.book[t] = Pos(t, side, volume, px, sl, tp, 0.0, 0.0, now, MAGIC_RAFALE, True)
        return {"price": px, "order": t}

    def close(self, pos: Pos, tick: Any) -> tuple[float, float]:
        px = (tick.bid - self.slip) if pos.side == "buy" else (tick.ask + self.slip)
        pnl = (px - pos.price_open if pos.side == "buy" else pos.price_open - px) * self._k(pos.volume)
        self.realised += pnl
        self.book.pop(pos.ticket, None)
        return px, pnl

    def exit_info(self, ticket: int, dernier_profit: float) -> tuple[float, str]:
        return self._sortis.pop(ticket, (dernier_profit, "sortie_serveur"))


class RealExecutor:
    """Ordres réels, uniquement sur compte démo, uniquement avec le magic MAGIC_RAFALE."""

    def __init__(self, broker: Broker, symbol: str):
        self.broker, self.mt5, self.symbol = broker, broker.mt5, symbol

    def assert_demo(self) -> None:
        info = self.mt5.account_info()
        if info is None or info.trade_mode != self.mt5.ACCOUNT_TRADE_MODE_DEMO:
            raise CompteReelRefuse("Compte non démo ou inconnu : aucun ordre envoyé.")

    @staticmethod
    def _pos(p: Any, mt5: Any) -> Pos:
        return Pos(int(p.ticket), "buy" if p.type == mt5.POSITION_TYPE_BUY else "sell", float(p.volume),
                   float(p.price_open), float(p.sl), float(p.tp), float(p.profit),
                   float(getattr(p, "swap", 0.0)), float(p.time), int(p.magic))

    def positions(self, tick: Any = None) -> list[Pos]:
        raw = self.mt5.positions_get(symbol=self.symbol)
        # on ne garde que SES positions : les autres magic ne sont jamais lus plus loin
        return [self._pos(p, self.mt5) for p in (raw or ())
                if getattr(p, "magic", None) == MAGIC_RAFALE]

    def open(self, side: str, tick: Any, volume: float, sl: float, tp: float, now: float) -> dict:
        self.assert_demo()
        b = self.broker
        info = b._symbol(self.symbol)
        volume = b._check_volume(info, volume)
        t = self.mt5.symbol_info_tick(self.symbol)
        if t is None:
            raise TradingError(f"Pas de cotation pour {self.symbol}.")
        mt5 = self.mt5
        req = {
            "action": mt5.TRADE_ACTION_DEAL, "symbol": self.symbol, "volume": volume,
            "type": mt5.ORDER_TYPE_BUY if side == "buy" else mt5.ORDER_TYPE_SELL,
            "price": t.ask if side == "buy" else t.bid, "deviation": 20, "magic": MAGIC_RAFALE,
            "comment": "rafale", "type_time": mt5.ORDER_TIME_GTC, "type_filling": b._filling(info),
        }
        if sl:
            req["sl"] = round(float(sl), info.digits)
        if tp:
            req["tp"] = round(float(tp), info.digits)
        return b._send(req)

    def close(self, pos: Pos, tick: Any = None) -> tuple[float, float]:
        self.assert_demo()
        mt5, b = self.mt5, self.broker
        found = mt5.positions_get(ticket=int(pos.ticket))
        if not found or getattr(found[0], "magic", None) != MAGIC_RAFALE:
            raise TradingError(f"Position {pos.ticket} absente ou d'un autre magic : fermeture refusée.")
        p = found[0]
        info = b._symbol(p.symbol)
        t = mt5.symbol_info_tick(p.symbol)
        if t is None:
            raise TradingError(f"Pas de cotation pour {p.symbol}.")
        buy = p.type == mt5.POSITION_TYPE_BUY
        req = {
            "action": mt5.TRADE_ACTION_DEAL, "symbol": p.symbol, "position": p.ticket,
            "volume": p.volume, "type": mt5.ORDER_TYPE_SELL if buy else mt5.ORDER_TYPE_BUY,
            "price": t.bid if buy else t.ask, "deviation": 20, "magic": MAGIC_RAFALE,
            "comment": "rafale close", "type_time": mt5.ORDER_TIME_GTC, "type_filling": b._filling(info),
        }
        out = b._send(req)
        pnl, _ = self.exit_info(pos.ticket, pos.profit)
        return float(out.get("price") or req["price"]), pnl

    def exit_info(self, ticket: int, dernier_profit: float) -> tuple[float, str]:
        mt5 = self.mt5
        try:
            deals = mt5.history_deals_get(position=int(ticket))
            sorties = [d for d in (deals or ()) if getattr(d, "entry", None) == mt5.DEAL_ENTRY_OUT]
            if sorties:
                pnl = sum(d.profit + getattr(d, "commission", 0.0) + getattr(d, "swap", 0.0)
                          + getattr(d, "fee", 0.0) for d in sorties)
                raison = {getattr(mt5, "DEAL_REASON_SL", -1): "sl_serveur",
                          getattr(mt5, "DEAL_REASON_TP", -2): "tp_serveur"}.get(
                    getattr(sorties[-1], "reason", None), "sortie_serveur")
                return float(pnl), raison
        except Exception:
            pass
        return dernier_profit, "sortie_serveur"


TRADE_COLS = ["time", "symbol", "side", "volume", "price", "sl", "tp", "ticket", "event", "pnl", "reason"]


class RafaleLive:
    """Boucle à la seconde. `step(now)` fait un tour complet (testable avec un faux MT5)."""

    def __init__(self, mt5: Any, broker: Broker, symbol: str, params: dict, *, orders: bool,
                 logs_dir: str | os.PathLike = "logs", data_dir: str | os.PathLike = "data",
                 config_dir: str | os.PathLike = "config", log: logging.Logger | None = None,
                 hedging: bool | None = None):
        self.mt5, self.broker, self.symbol = mt5, broker, symbol
        self.hedging = check_hedging(mt5)[0] if hedging is None else hedging
        self.params = normalize_params({**params, "hedging": self.hedging})
        self.orders = orders
        self.logs, self.data, self.config = Path(logs_dir), Path(data_dir), Path(config_dir)
        self.logs.mkdir(parents=True, exist_ok=True)
        self.log = log or logging.getLogger("rafale")
        self.spec = broker.symbol_spec(symbol)
        self.spec["tick_size"] = self.spec["point"]
        self.slip = float(self.params["slip_points"]) * float(self.spec["point"])
        self.executor: Any = RealExecutor(broker, symbol) if orders else PaperExecutor(self.spec, self.slip)
        self.mids: deque = deque()
        self.known: dict[int, Pos] = {}
        self.opened_at: dict[int, float] = {}
        self.cooldown_until = 0.0
        self.last_tick_key: tuple | None = None
        self.last_tick_change = time.time()
        self.state_path = self.logs / "state_rafale.json"
        self.state = self._load_state()
        self._news_cache: tuple[float, dict] = (0.0, {"bias": "neutral"})
        self._profile = self._load_profile()
        self._avert_tp = False
        self.errors = 0
        self.connected = True
        self.paper_balance = None

    # ---------------------------------------------------------------- état persistant
    def _load_state(self) -> dict:
        try:
            s = json.loads(self.state_path.read_text(encoding="utf-8"))
            if isinstance(s, dict):
                return s
        except (OSError, ValueError):
            pass
        return {}

    def _save_state(self) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2), encoding="utf-8")
        os.replace(tmp, self.state_path)

    def reset_halt(self) -> None:
        for c in ("halted_full", "halted_day", "halt_reason"):
            self.state.pop(c, None)
        self.state["consec_losses"] = 0
        self.state["peak_equity"] = None
        self._save_state()

    def _load_profile(self) -> dict | None:
        try:
            prof = json.loads((self.data / "spread_profile.json").read_text(encoding="utf-8"))
            return prof.get(self.symbol, {}).get("hourly_median")
        except (OSError, ValueError, AttributeError):
            return None

    # ---------------------------------------------------------------- journaux
    def _trade_row(self, now: float, side: str, volume: float, price: float, sl: float, tp: float,
                   ticket: int, event: str, pnl: float | None, reason: str) -> None:
        path = self.logs / "trades_rafale.csv"
        neuf = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if neuf:
                w.writerow(TRADE_COLS)
            w.writerow([dt.datetime.fromtimestamp(now, UTC).isoformat(timespec="seconds"), self.symbol,
                        side, volume, price, sl, tp, ticket, event,
                        "" if pnl is None else round(pnl, 2), reason])

    # ---------------------------------------------------------------- comptes
    def _account(self) -> tuple[float, float]:
        if self.orders:
            info = self.mt5.account_info()
            if info is None:
                raise TradingError("account_info indisponible.")
            return float(info.balance), float(info.equity)
        if self.paper_balance is None:
            info = self.mt5.account_info()
            self.paper_balance = float(info.balance) if info else 10000.0
        floating = sum(p.profit for p in self.executor.book.values())
        bal = self.paper_balance + self.executor.realised
        return bal, bal + floating

    def _bias(self, now: float) -> dict:
        if allows is None:
            return {"bias": "neutral"}
        if now - self._news_cache[0] < 30:
            return self._news_cache[1]
        now_utc = dt.datetime.fromtimestamp(now, UTC)
        try:
            chemin = self.logs / "news_bias.json"
            if chemin.exists():
                b = read_bias(chemin, now_utc)
            else:
                b = compute_bias(now_utc, _news_load_json(self.config / "events.json") or {},
                                 _news_load_json(self.config / "bias_override.json"))
        except Exception as exc:
            self.log.warning("Filtre news indisponible (%s) : biais neutre.", exc)
            b = {"bias": "neutral"}
        self._news_cache = (now, b)
        return b

    # ---------------------------------------------------------------- règles
    def _volume(self) -> float:
        return fixed_volume(self.spec)    # jamais plus gros, quelle que soit la série de pertes

    def _levels(self, side: str, tick: Any) -> tuple[float, float]:
        """Stop de sécurité et TP serveur au niveau du prix correspondant à X net."""
        pt, dg = float(self.spec["point"]), int(self.spec["digits"])
        mind = (float(self.spec.get("stops_level_points", 0)) + 1) * pt
        sl_d = max(stop_distance(self.params), mind)
        tp_d = float(self.params["target"]) + self.slip
        if side == "buy":
            ref = entry_fill("buy", tick.bid, tick.ask, 0.0)
            sl, tp = ref - sl_d, ref + tp_d
        else:
            ref = entry_fill("sell", tick.bid, tick.ask, 0.0)
            sl, tp = ref + sl_d, ref - tp_d
        if tp_d < mind:
            if not self._avert_tp:
                self.log.warning("TP serveur impossible (distance %.3f < niveau d'arrêt minimal %.3f) : "
                                 "sortie gérée par la boucle seulement.", tp_d, mind)
                self._avert_tp = True
            tp = 0.0
        return round(sl, dg), round(tp, dg) if tp else 0.0

    def _record_close(self, now: float, pos: Pos, price: float, pnl: float, reason: str) -> None:
        self._trade_row(now, pos.side, pos.volume, price, pos.sl, pos.tp, pos.ticket, "close", pnl, reason)
        st = self.state
        st["day_pnl"] = st.get("day_pnl", 0.0) + pnl
        st["trades_today"] = st.get("trades_today", 0) + 1
        st["consec_losses"] = st.get("consec_losses", 0) + 1 if pnl < 0 else 0
        self.log.info("FERMETURE %s #%s pnl %.2f (%s)", pos.side, pos.ticket, pnl, reason)

    def _roll_day(self, now: float, balance: float) -> None:
        jour = dt.datetime.fromtimestamp(now, UTC).strftime("%Y-%m-%d")
        if self.state.get("day") != jour:
            self.state.update({"day": jour, "day_start_balance": balance, "day_pnl": 0.0,
                               "trades_today": 0, "consec_losses": 0, "halted_day": False})
            if self.state.get("halt_reason", "").startswith("jour"):
                self.state.pop("halt_reason", None)

    def _close(self, now: float, pos: Pos, tick: Any, reason: str) -> None:
        try:
            price, pnl = self.executor.close(pos, tick)
        except CompteReelRefuse:
            raise
        except TradingError as exc:
            self.log.error("Fermeture #%s impossible : %s", pos.ticket, exc)
            self.errors += 1
            return
        self.known.pop(pos.ticket, None)
        self._record_close(now, pos, price, pnl, reason)

    def _signal(self, now: float, mid: float) -> str | None:
        cible = int(now) - int(self.params["lookback_s"])
        passe = None
        for sec, m in self.mids:
            if sec <= cible:
                passe = m
            else:
                break
        if passe is None:
            return None
        return momentum_signal(mid, passe, self.spec["tick_size"], self.params["thr_ticks"])

    # ---------------------------------------------------------------- un tour de boucle
    def step(self, now: float) -> dict:
        p, st = self.params, self.state
        info: dict[str, Any] = {"opened": 0, "closed": 0, "skipped": None}
        tick = self.mt5.symbol_info_tick(self.symbol)
        if tick is None:
            info["skipped"] = "pas de cotation"
            return info
        key = (tick.bid, tick.ask, tick.time)
        if key != self.last_tick_key:
            self.last_tick_key, self.last_tick_change = key, now
        frais = (now - self.last_tick_change) < 60
        mid = (tick.bid + tick.ask) / 2.0
        self.mids.append((int(now), mid))
        while self.mids and self.mids[0][0] < int(now) - int(p["lookback_s"]) - 10:
            self.mids.popleft()

        balance, equity = self._account()
        self._roll_day(now, balance)
        if not st.get("peak_equity") or equity > st["peak_equity"]:
            st["peak_equity"] = equity

        # positions : détecte les sorties serveur (TP / SL) et les journalise
        cur = self.executor.positions(tick)
        ids = {q.ticket for q in cur}
        for tkt in [t for t in self.known if t not in ids]:
            last = self.known.pop(tkt)
            pnl, raison = self.executor.exit_info(tkt, last.profit)
            self._record_close(now, last, last.price_open, pnl, raison)
            info["closed"] += 1
        for q in cur:
            self.known[q.ticket] = q
            self.opened_at.setdefault(q.ticket, now)

        # sorties : objectif net / durée max / stop de sécurité
        for q in cur:
            age = now - self.opened_at.get(q.ticket, now)
            r = exit_reason(q.side, q.price_open, tick.bid, tick.ask, age, p, self.slip)
            if r:
                self._close(now, q, tick, r)
                info["closed"] += 1
        restant = [q for q in cur if q.ticket in self.known]

        # fermeture totale à -2 % de perte latente (SES positions uniquement)
        flottant = sum(q.profit + q.swap for q in restant)
        if restant and breaker_hit(flottant, balance, p["breaker_pct"]):
            self.log.warning("COUPE-CIRCUIT : perte latente %.2f $ >= %.0f %% du capital, fermeture totale.",
                             flottant, p["breaker_pct"] * 100)
            for q in restant:
                self._close(now, q, tick, "coupe_circuit")
                info["closed"] += 1
            self.cooldown_until = now + float(p["cooldown_s"])
            restant = []

        # coupe-circuits de journée et de drawdown
        if not st.get("halted_full") and st["peak_equity"] and \
                (st["peak_equity"] - equity) / st["peak_equity"] >= 0.08:
            st.update({"halted_full": True, "halt_reason": "drawdown 8 % depuis le pic"})
            self.log.error("ARRÊT COMPLET : drawdown 8 %% depuis le pic (équité %.2f).", equity)
            for q in restant:
                self._close(now, q, tick, "arret_drawdown")
            restant = []
        if not st.get("halted_day") and not st.get("halted_full"):
            if st.get("day_pnl", 0.0) <= -0.02 * st.get("day_start_balance", balance):
                st.update({"halted_day": True, "halt_reason": "jour : -2 % du jour"})
            elif st.get("consec_losses", 0) >= 5:
                st.update({"halted_day": True, "halt_reason": "jour : 5 pertes consécutives"})
            if st.get("halted_day"):
                self.log.error("ARRÊT POUR LA JOURNÉE : %s", st["halt_reason"])

        # ouvertures
        raison_skip = None
        if st.get("halted_full") or st.get("halted_day"):
            raison_skip = st.get("halt_reason", "arrêt")
        elif now < self.cooldown_until:
            raison_skip = "pause après coupe-circuit"
        elif not frais:
            raison_skip = "marché figé"
        elif int(now) % int(p["interval_s"]) != 0:
            raison_skip = "hors intervalle"
        else:
            spread = tick.ask - tick.bid
            med = None
            if self._profile:
                med = self._profile.get(str(dt.datetime.fromtimestamp(now, UTC).hour))
            sig = self._signal(now, mid) if p["mode"] == "momentum" else None
            cotes = plan_entries(p["mode"], sig, len(restant), effective_max_positions(p))
            bias = self._bias(now)
            if not cotes:
                raison_skip = "pas de signal ou limite de positions"
            elif not spread_ok(spread, med, p["spread_k"]):
                raison_skip = f"spread {spread:.3f} > {p['spread_k']} x médiane"
            elif allows is not None and not all(allows(bias, c) for c in cotes):
                raison_skip = f"filtre news : {bias.get('bias')}"
            else:
                for side in cotes:
                    sl, tp = self._levels(side, tick)
                    try:
                        out = self.executor.open(side, tick, self._volume(), sl, tp, now)
                    except CompteReelRefuse:
                        raise
                    except TradingError as exc:
                        self.log.error("Ouverture %s refusée : %s", side, exc)
                        self.errors += 1
                        continue
                    px = float(out.get("price") or entry_fill(side, tick.bid, tick.ask, 0.0))
                    tkt = int(out.get("order") or 0)
                    self.opened_at[tkt] = now
                    self._trade_row(now, side, self._volume(), px, sl, tp, tkt, "open", None, "signal")
                    self.log.info("OUVERTURE%s %s %.2f lot @ %.2f sl %.2f tp %.2f",
                                  "" if self.orders else " (virtuelle)", side, self._volume(), px, sl, tp)
                    info["opened"] += 1
        info["skipped"] = raison_skip
        self._status(now, balance, equity, restant, info)
        self._save_state()
        return info

    def _status(self, now: float, balance: float, equity: float, pos: list[Pos], info: dict) -> None:
        st = self.state
        data = {
            "time": dt.datetime.fromtimestamp(now, UTC).isoformat(timespec="seconds"),
            "mode": self.params["mode"], "orders": "reels (demo)" if self.orders else "observation",
            "connected": self.connected, "demo": True, "hedging": self.hedging,
            "equity": round(equity, 2), "balance": round(balance, 2),
            "open_positions": len(pos) + info.get("opened", 0), "floating_pnl": round(sum(q.profit + q.swap for q in pos), 2),
            "trades_today": st.get("trades_today", 0), "day_pnl": round(st.get("day_pnl", 0.0), 2),
            "consec_losses": st.get("consec_losses", 0),
            "halted": bool(st.get("halted_full") or st.get("halted_day")),
            "halt_reason": st.get("halt_reason"), "last_skip": info.get("skipped"),
            "errors": self.errors, "params": self.params,
        }
        tmp = self.logs / "status_rafale.tmp"
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, self.logs / "status_rafale.json")

    def close_all(self, now: float, raison: str = "arret") -> None:
        tick = self.mt5.symbol_info_tick(self.symbol)
        try:
            for q in self.executor.positions(tick):
                self._close(now, q, tick, raison)
        except TradingError as exc:
            self.log.error("Fermeture à l'arrêt impossible : %s", exc)


def setup_logging(logs_dir: Path) -> logging.Logger:
    logs_dir.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("rafale")
    log.setLevel(logging.INFO)
    if not log.handlers:
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        for h in (logging.FileHandler(logs_dir / "rafale.log", encoding="utf-8"), logging.StreamHandler()):
            h.setFormatter(fmt)
            log.addHandler(h)
    return log


def _settings_from_env() -> Settings:
    login = os.environ.get("MT5_LOGIN") or None
    return Settings(login=int(login) if login else None, password=os.environ.get("MT5_PASSWORD") or None,
                    server=os.environ.get("MT5_SERVER") or None, path=os.environ.get("MT5_PATH") or None,
                    max_volume=float(os.environ.get("MT5_MAX_VOLUME") or 1.0))


def _import_mt5() -> Any:
    try:
        import MetaTrader5 as mt5  # type: ignore[import-not-found]
    except ImportError:
        print("Le paquet MetaTrader5 est absent (Windows uniquement).", file=sys.stderr)
        raise SystemExit(2)
    return mt5


def cmd_check(args: argparse.Namespace, mt5: Any | None = None) -> int:
    mt5 = mt5 or _import_mt5()
    broker = Broker(mt5, _settings_from_env())
    try:
        acc = broker.connect()
        ok, msg = check_hedging(mt5)
        info = mt5.account_info()
        print(f"Compte {acc['login']} ({acc['server']}) démo={acc['demo']}")
        print(f"Mode de marge : {getattr(info, 'margin_mode', '?')} — hedging = {ok}")
        print(msg)
        print(json.dumps(broker.symbol_spec(args.symbol), indent=2, ensure_ascii=False))
        return 0
    except TradingError as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1
    finally:
        broker.shutdown()


def load_live_params(args: argparse.Namespace) -> tuple[dict, bool, bool, str]:
    """Renvoie (params, rapport passed, synthétique, description)."""
    params = dict(DEFAULT_PARAMS)
    passed, synth, desc = False, False, "paramètres par défaut, aucun rapport"
    if args.params_from:
        rep = json.loads(Path(args.params_from).read_text(encoding="utf-8"))
        best = rep.get("best") or {}
        params.update(best.get("params") or {})
        synth = bool(rep.get("synthetic"))
        passed = bool(best.get("passed")) and not synth
        desc = f"rapport {args.params_from} : passed={best.get('passed')} synthétique={synth}"
    for cle, attr in (("mode", "mode"), ("target", "target"), ("max_life_min", "max_life"),
                      ("interval_s", "interval_s"), ("lookback_s", "lookback_s")):
        v = getattr(args, attr, None)
        if v is not None:
            params[cle] = v
    return params, passed, synth, desc


def cmd_live(args: argparse.Namespace, mt5: Any | None = None, sleeper=None) -> int:
    logs = Path(args.logs)
    log = setup_logging(logs)
    params, passed, synth, desc = load_live_params(args)
    mt5 = mt5 or _import_mt5()
    broker = Broker(mt5, _settings_from_env())
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, lambda *_: stop.set())
        except ValueError:
            pass
    orders = (not args.dry_run) and (passed or args.force)
    if args.dry_run:
        log.info("Mode --dry-run : observation, aucun ordre.")
    elif not passed and args.force:
        log.warning("*** --force : le rapport n'indique pas passed=true (%s). Ordres DÉMO autorisés "
                    "malgré tout. Aucun avantage statistique démontré. ***", desc)
    elif not passed:
        log.warning("Rapport non validé (%s) : observation seulement, aucun ordre (--force pour passer "
                    "des ordres démo).", desc)
    backoff = 2.0
    live: RafaleLive | None = None
    while not stop.is_set():
        try:
            acc = broker.connect()
            if not acc["demo"]:
                raise CompteReelRefuse("compte non démo")
            if live is None:
                ok, msg = check_hedging(mt5)
                log.info("Hedging : %s", msg)
                allowed, why = check_mode_allowed(params["mode"], ok)
                if not allowed:
                    log.error("%s Arrêt.", why)
                    return 2
                live = RafaleLive(mt5, broker, args.symbol, params, orders=orders, logs_dir=logs,
                                  data_dir=args.data, config_dir=args.config, log=log, hedging=ok)
                if args.reset_halt:
                    live.reset_halt()
                    log.info("Arrêts remis à zéro (--reset-halt).")
                log.info("Démarrage rafale %s, paramètres %s, ordres %s.", args.symbol, live.params,
                         "réels (démo)" if orders else "virtuels")
            live.connected = True
            backoff = 2.0
            while not stop.is_set():
                maintenant = time.time()
                prochain = math.floor(maintenant) + 1
                if sleeper:
                    sleeper(prochain - maintenant)
                else:
                    stop.wait(max(0.0, prochain - maintenant))
                if stop.is_set():
                    break
                live.step(time.time())
        except CompteReelRefuse as exc:
            log.error("COMPTE RÉEL REFUSÉ : %s", exc)
            return 3
        except TradingError as exc:
            if "pas un compte démo" in str(exc):
                log.error("COMPTE RÉEL REFUSÉ : %s", exc)
                return 3
            log.error("Erreur de trading : %s", exc)
        except Exception as exc:  # reconnexion automatique
            log.error("Erreur : %s", exc)
        else:
            break
        if live:
            live.connected = False
        try:
            broker.shutdown()
        except Exception:
            pass
        broker.connected = False
        log.info("Reconnexion dans %.0f s.", backoff)
        stop.wait(backoff)
        backoff = min(backoff * 2, 60.0)
    if live:
        live.close_all(time.time(), "arret")
    try:
        broker.shutdown()
    except Exception:
        pass
    log.info("Arrêt propre.")
    return 0


# =====================================================================================
# CLI
# =====================================================================================
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m ava_mt5.rafale", description="Stratégie rafale")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="connexion, mode de marge et spécification du symbole")
    c.add_argument("--symbol", default="CrudeOIL")
    s = sub.add_parser("synth", help="écrit des ticks synthétiques")
    s.add_argument("--out", default="data")
    s.add_argument("--days", type=int, default=5)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--symbol", default=SYNTH_SYMBOL)
    b = sub.add_parser("backtest", help="backtest sur ticks, grille, IS/OOS")
    b.add_argument("--data", default="data")
    b.add_argument("--symbol", default="CrudeOIL")
    b.add_argument("--out", default="logs/rafale_report.json")
    b.add_argument("--capital", type=float, default=10000.0)
    b.add_argument("--quick", action="store_true", help="grille réduite")
    b.add_argument("--max-combos", type=int, default=None)
    b.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    b.add_argument("--synthetic", action="store_true", help="génère des ticks synthétiques dans --data")
    b.add_argument("--days", type=int, default=5)
    b.add_argument("--seed", type=int, default=0)
    lv = sub.add_parser("live", help="exécution sur compte démo")
    lv.add_argument("--symbol", default="CrudeOIL")
    lv.add_argument("--params-from", default=None)
    lv.add_argument("--dry-run", action="store_true")
    lv.add_argument("--force", action="store_true")
    lv.add_argument("--reset-halt", action="store_true")
    lv.add_argument("--logs", default="logs")
    lv.add_argument("--data", default="data")
    lv.add_argument("--config", default="config")
    lv.add_argument("--mode", choices=MODES, default=None)
    lv.add_argument("--target", type=float, default=None)
    lv.add_argument("--max-life", dest="max_life", type=float, default=None)
    lv.add_argument("--interval-s", dest="interval_s", type=int, default=None)
    lv.add_argument("--lookback-s", dest="lookback_s", type=int, default=None)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "check":
        return cmd_check(args)
    if args.cmd == "synth":
        out = write_synthetic(args.out, args.days, args.seed, args.symbol)
        print(f"Ticks synthétiques écrits dans {out}")
        return 0
    if args.cmd == "backtest":
        return cmd_backtest(args)
    return cmd_live(args)


if __name__ == "__main__":
    sys.exit(main())
