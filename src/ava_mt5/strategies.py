"""Stratégies à règles fixes, PARTAGÉES par le backtest et le bot.

Contrat : une stratégie reçoit des barres M1 (index = heure UTC d'ouverture) et renvoie
une liste de `Signal`. La décision est prise à la clôture de la barre t-1 et l'entrée a
lieu à l'open de la barre t (`Signal.time`). Aucune information de la barre t ou
postérieure n'est utilisée : les signaux sont déterministes et ne dépendent que du passé.

Les heures de session sont en America/New_York (zoneinfo, heure d'été gérée).
Pas de martingale, de grille ni de moyenne à la baisse : chaque signal porte un stop.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

NY = "America/New_York"
ONE_MIN = pd.Timedelta(minutes=1)


@dataclass(frozen=True)
class Signal:
    time: pd.Timestamp   # ouverture (UTC) de la barre M1 où l'entrée a lieu (entrée à l'open)
    side: str            # "buy" | "sell"
    sl_dist: float       # distance du stop en prix (>0)
    tp_dist: float       # distance de l'objectif en prix (>0)
    max_minutes: int     # sortie forcée après N minutes
    strategy: str


class Strategy(Protocol):
    name: str
    default_params: dict
    grid: dict[str, list]

    def signals(self, m1: pd.DataFrame, params: dict) -> list[Signal]: ...


# ------------------------------------------------------------------------- lecture
def load_bars(path: str | Path) -> pd.DataFrame:
    """Lit un csv.gz de barres en DataFrame indexé par `time` (UTC, tz-aware)."""
    df = pd.read_csv(path)
    if pd.api.types.is_numeric_dtype(df["time"]):
        idx = pd.to_datetime(df["time"], unit="s", utc=True)
    else:
        idx = pd.to_datetime(df["time"], utc=True)
    df = df.drop(columns="time")
    df.index = pd.DatetimeIndex(idx, name="time")
    return df.sort_index()


def load_ticks(path: str | Path) -> pd.DataFrame:
    """Lit un csv.gz de ticks (time_msc en millisecondes epoch UTC, bid, ask, flags)."""
    return pd.read_csv(path)


def add_tick_features(m1: pd.DataFrame, ticks: pd.DataFrame) -> pd.DataFrame:
    """Ajoute `upticks` et `downticks` (nombre de ticks haussiers/baissiers par minute).

    Un tick est haussier si le milieu (bid+ask)/2 monte par rapport au tick précédent.
    Les minutes situées hors de la période couverte par les ticks restent à NaN ; à
    l'intérieur de la période, une minute sans tick vaut 0.
    """
    out = m1.copy()
    out["upticks"] = np.nan
    out["downticks"] = np.nan
    if ticks is None or len(ticks) == 0:
        return out
    t = ticks.sort_values("time_msc", kind="stable")
    mid = ((t["bid"] + t["ask"]) / 2.0).to_numpy()
    d = np.diff(mid, prepend=mid[0])
    minute = (t["time_msc"].to_numpy(np.int64) // 60_000)
    df = pd.DataFrame({"m": minute, "up": (d > 0).astype(np.int64), "dn": (d < 0).astype(np.int64)})
    agg = df.groupby("m").sum()
    idx_min = (m1.index.as_unit("s").asi8 // 60)
    first, last = minute.min(), minute.max()
    covered = (idx_min >= first) & (idx_min <= last)
    up = agg["up"].reindex(idx_min).to_numpy(dtype=float, copy=True)
    dn = agg["dn"].reindex(idx_min).to_numpy(dtype=float, copy=True)
    up[covered & np.isnan(up)] = 0.0
    dn[covered & np.isnan(dn)] = 0.0
    out["upticks"] = np.where(covered, up, np.nan)
    out["downticks"] = np.where(covered, dn, np.nan)
    return out


# ------------------------------------------------------------------------- indicateurs
_CACHE: dict[int, "_Features"] = {}


def _data_key(m1: pd.DataFrame) -> int:
    """Empreinte rapide du contenu (le cache ne doit jamais servir des données périmées)."""
    h = zlib.crc32(m1.index.as_unit("s").asi8.tobytes())
    for col in ("open", "high", "low", "close", "tick_volume", "spread", "upticks", "downticks"):
        if col in m1:
            h = zlib.crc32(np.ascontiguousarray(m1[col].to_numpy(float)).tobytes(), h)
    return h ^ (len(m1) << 32)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def _resample(m1: pd.DataFrame, rule: str) -> pd.DataFrame:
    return m1.resample(rule, closed="left", label="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()


def _available(series: pd.Series, width: pd.Timedelta, target: pd.DatetimeIndex) -> np.ndarray:
    """Valeur d'une série de barres agrégées, connue à chaque instant `target`.

    Une barre débutant en s n'est connue qu'à s+width : pas d'information du futur.
    """
    s = series.copy()
    s.index = s.index + width
    return s.reindex(target, method="ffill").to_numpy(float)


class _Features:
    """Indicateurs calculés une fois par jeu de barres ; valeur d'indice j = barres <= j."""

    def __init__(self, m1: pd.DataFrame) -> None:
        self.m1 = m1
        self.idx = m1.index
        self.open = m1["open"].to_numpy(float)
        self.high = m1["high"].to_numpy(float)
        self.low = m1["low"].to_numpy(float)
        self.close = m1["close"].to_numpy(float)
        self.tv = m1["tick_volume"].to_numpy(float) if "tick_volume" in m1 else np.ones(len(m1))
        self.spread = m1["spread"].to_numpy(float) if "spread" in m1 else np.zeros(len(m1))
        ny = self.idx.tz_convert(NY).tz_localize(None)
        self.ny_min = (ny.hour * 60 + ny.minute).to_numpy()
        self.ny_dow = ny.dayofweek.to_numpy()
        self.ny_day = ny.normalize().as_unit("s").asi8 // 86400
        self.sess = (ny + pd.Timedelta(hours=6)).normalize().as_unit("s").asi8 // 86400
        self._memo: dict[str, Any] = {}
        # instant où une décision prise à la clôture de la barre j devient exécutable
        self.decision_time = self.idx + ONE_MIN

    def memo(self, key: str, fn):
        if key not in self._memo:
            self._memo[key] = fn()
        return self._memo[key]

    @property
    def atr1(self) -> np.ndarray:
        return self.memo("atr1", lambda: _atr(self.m1["high"], self.m1["low"], self.m1["close"]).to_numpy())

    @property
    def atr5(self) -> np.ndarray:
        def build():
            b = _resample(self.m1, "5min")
            a = _atr(b["high"], b["low"], b["close"])
            return _available(a, pd.Timedelta(minutes=5), self.decision_time)
        return self.memo("atr5", build)

    @property
    def trend_up(self) -> tuple[np.ndarray, np.ndarray]:
        """(EMA20 M15 > EMA50 M15, validité) : barres M15 clôturées uniquement."""
        def build():
            b = _resample(self.m1, "15min")["close"]
            e20 = b.ewm(span=20, adjust=False, min_periods=20).mean()
            e50 = b.ewm(span=50, adjust=False, min_periods=50).mean()
            diff = _available(e20 - e50, pd.Timedelta(minutes=15), self.decision_time)
            return diff > 0, np.isfinite(diff)
        return self.memo("trend", build)

    @property
    def vwap(self) -> np.ndarray:
        def build():
            typ = (self.m1["high"] + self.m1["low"] + self.m1["close"]) / 3.0
            w = pd.Series(np.maximum(self.tv, 1.0), index=self.idx)
            key = self.sess
            num = (typ * w).groupby(key).cumsum()
            den = w.groupby(key).cumsum()
            return (num / den).to_numpy()
        return self.memo("vwap", build)

    @property
    def sess_count(self) -> np.ndarray:
        return self.memo("sess_count", lambda: pd.Series(1, index=self.idx).groupby(self.sess).cumsum().to_numpy())

    def atr1_rank(self, window: int) -> np.ndarray:
        return self.memo(f"rank{window}", lambda: pd.Series(self.atr1).rolling(
            window, min_periods=window // 2).rank(pct=True).to_numpy())

    def vol_median(self, window: int) -> np.ndarray:
        return self.memo(f"volmed{window}", lambda: pd.Series(self.tv).rolling(
            window, min_periods=window // 2).median().to_numpy())


def _features(m1: pd.DataFrame) -> _Features:
    key = _data_key(m1)
    f = _CACHE.get(key)
    if f is None:
        if len(_CACHE) >= 4:
            _CACHE.pop(next(iter(_CACHE)))
        f = _CACHE[key] = _Features(m1)
    return f


def _make(f: _Features, js: np.ndarray, side: str | np.ndarray, sl: np.ndarray, tp: np.ndarray,
          max_minutes: int, name: str, p: dict) -> list[Signal]:
    """Construit les Signal de décisions prises à la clôture des barres d'indices `js`.

    Filtre de coût : on écarte les signaux dont le stop (resp. l'objectif) est inférieur à
    min_sl_spread_mult (resp. min_tp_spread_mult) fois le spread de la dernière barre
    connue, exprimé en prix grâce au paramètre `point`.
    """
    sp_price = f.spread[js] * p["point"]
    keep = (sl >= p["min_sl_spread_mult"] * sp_price) & (tp >= p["min_tp_spread_mult"] * sp_price)
    js, sl, tp = js[keep], sl[keep], tp[keep]
    side = np.broadcast_to(np.asarray(side), keep.shape)[keep]
    times = f.decision_time[js]
    sides = np.broadcast_to(np.asarray(side), js.shape)
    return [Signal(times[i], str(sides[i]), float(sl[i]), float(tp[i]), int(max_minutes), name)
            for i in range(len(js))]


def _day_bounds(day: np.ndarray) -> list[tuple[int, int]]:
    if len(day) == 0:
        return []
    cuts = np.flatnonzero(np.diff(day)) + 1
    starts = np.concatenate([[0], cuts])
    ends = np.concatenate([cuts, [len(day)]])
    return list(zip(starts.tolist(), ends.tolist()))


# Paramètres communs (hors grille) : taille du point et garde-fou de coût.
COMMON_PARAMS = {"point": 0.01, "min_sl_spread_mult": 4.0, "min_tp_spread_mult": 2.0}


def _merged(strategy: Any, params: dict | None) -> dict:
    return {**COMMON_PARAMS, **strategy.default_params, **(params or {})}


# ------------------------------------------------------------------------- a) ORB
class OrbStrategy:
    """Cassure du range d'ouverture NY (9:00 -> 9:00 + N min), clôture M1 au-delà du range."""

    name = "orb"
    default_params = {"range_min": 15, "sl_atr": 1.5, "tp_atr": 2.5, "max_minutes": 60,
                      "entry_end_min": 11 * 60 + 30}
    grid = {"range_min": [5, 15, 30], "sl_atr": [1.0, 1.5], "tp_atr": [1.5, 2.5],
            "max_minutes": [30, 60]}

    def signals(self, m1: pd.DataFrame, params: dict) -> list[Signal]:
        p = _merged(self, params)
        f = _features(m1)
        n_rng, start = int(p["range_min"]), 9 * 60
        atr5 = f.atr5
        js, sides = [], []
        for s, e in _day_bounds(f.ny_day):
            if f.ny_dow[s] > 4:
                continue
            mn = f.ny_min[s:e]
            in_rng = (mn >= start) & (mn < start + n_rng)
            if in_rng.sum() < max(1, n_rng // 2):
                continue
            hi = f.high[s:e][in_rng].max()
            lo = f.low[s:e][in_rng].min()
            # décision à la clôture de la barre j, entrée à j+1 : fenêtre jusqu'à entry_end_min
            ok = (mn >= start + n_rng) & (mn + 1 <= p["entry_end_min"])
            c = f.close[s:e]
            for side, hit in (("buy", ok & (c > hi)), ("sell", ok & (c < lo))):
                k = np.flatnonzero(hit)
                if k.size:  # au plus un trade par sens et par jour
                    js.append(s + int(k[0]))
                    sides.append(side)
        if not js:
            return []
        order = np.argsort(js)
        js_a = np.asarray(js)[order]
        sides_a = np.asarray(sides)[order]
        a = atr5[js_a]
        good = np.isfinite(a) & (a > 0)
        js_a, sides_a, a = js_a[good], sides_a[good], a[good]
        return _make(f, js_a, sides_a, p["sl_atr"] * a, p["tp_atr"] * a, p["max_minutes"], self.name, p)


# ------------------------------------------------------------------------- b) VWAP
class VwapMeanReversion:
    """Retour au VWAP de session (depuis 18:00 NY), en régime calme et dans le sens M15."""

    name = "vwap_mr"
    default_params = {"k": 2.0, "calm_pct": 0.6, "rank_window": 500, "sl_atr": 2.0,
                      "tp_frac": 0.7, "max_minutes": 45, "min_session_bars": 60}
    grid = {"k": [1.5, 2.5], "calm_pct": [0.5, 0.7], "sl_atr": [1.5, 2.5],
            "tp_frac": [0.5, 1.0], "max_minutes": [30, 60]}

    def signals(self, m1: pd.DataFrame, params: dict) -> list[Signal]:
        p = _merged(self, params)
        f = _features(m1)
        atr = f.atr1
        dev = f.close - f.vwap
        up, valid = f.trend_up
        calm = f.atr1_rank(int(p["rank_window"])) <= p["calm_pct"]
        base = (np.isfinite(atr) & (atr > 0) & calm & valid
                & (f.sess_count >= p["min_session_bars"]))
        with np.errstate(invalid="ignore"):
            far = np.abs(dev) > p["k"] * atr
            buy = base & far & (dev < 0) & up       # sous le VWAP, tendance haussière
            sell = base & far & (dev > 0) & ~up     # au-dessus du VWAP, tendance baissière
        js = np.flatnonzero(buy | sell)
        if js.size == 0:
            return []
        sl = p["sl_atr"] * atr[js]
        tp = np.maximum(0.5 * sl, p["tp_frac"] * np.abs(dev[js]))
        return _make(f, js, np.where(buy[js], "buy", "sell"), sl, tp, p["max_minutes"], self.name, p)


# ------------------------------------------------------------------------- c) EIA
class EiaMomentum:
    """Mercredi, publication EIA 10:30 NY : suit la première impulsion si le spread s'est resserré.

    Environ un trade par semaine : le critère des 200 trades hors échantillon est
    structurellement inatteignable avec cette stratégie.
    """

    name = "eia_momentum"
    default_params = {"delay": 2, "spread_factor": 2.0, "sl_atr": 1.5, "tp_atr": 2.5,
                      "max_minutes": 20, "impulse": "cum", "min_move_atr": 0.0,
                      "min_sl_spread_mult": 2.0, "min_tp_spread_mult": 1.5}
    grid = {"delay": [1, 2, 3], "spread_factor": [1.5, 3.0], "sl_atr": [1.5, 2.5],
            "tp_atr": [2.0, 4.0]}

    def signals(self, m1: pd.DataFrame, params: dict) -> list[Signal]:
        p = _merged(self, params)
        f = _features(m1)
        rel = 10 * 60 + 30
        entry_min = rel + int(p["delay"])
        atr5 = f.atr5
        js, sides, sl, tp = [], [], [], []
        for s, e in _day_bounds(f.ny_day):
            if f.ny_dow[s] != 2:
                continue
            mn = f.ny_min[s:e]
            i0 = np.flatnonzero(mn == rel)
            ij = np.flatnonzero(mn == entry_min - 1)
            if not i0.size or not ij.size:
                continue
            a, j = s + int(i0[0]), s + int(ij[0])
            # le premier mouvement ne doit pas contenir de trou de données
            if j - a != entry_min - 1 - rel:
                continue
            pre = (mn >= rel - 60) & (mn < rel - 5)
            if pre.sum() < 5:
                continue
            med = float(np.median(f.spread[s:e][pre]))
            if not f.spread[j] <= p["spread_factor"] * med:
                continue
            ref_close = f.close[a] if p["impulse"] == "bar" else f.close[j]
            move = ref_close - f.open[a]
            atr = atr5[j]
            if not (np.isfinite(atr) and atr > 0) or move == 0 or abs(move) < p["min_move_atr"] * atr:
                continue
            js.append(j)
            sides.append("buy" if move > 0 else "sell")
            sl.append(p["sl_atr"] * atr)
            tp.append(p["tp_atr"] * atr)
        if not js:
            return []
        return _make(f, np.asarray(js), np.asarray(sides), np.asarray(sl), np.asarray(tp),
                     p["max_minutes"], self.name, p)


# ------------------------------------------------------------------------- d) microstructure
def tick_imbalance_mode(m1: pd.DataFrame) -> str:
    """'ticks' si les upticks/downticks couvrent toutes les barres, 'proxy' sans aucun, sinon 'mixte'."""
    if "upticks" not in m1 or m1["upticks"].notna().sum() == 0:
        return "proxy"
    return "ticks" if m1["upticks"].notna().all() else "mixte"


class TickImbalance:
    """Déséquilibre acheteur/vendeur sur K barres, volume supérieur à la médiane, heures liquides.

    Avec les colonnes upticks/downticks : ratio (up-down)/(up+down) sur les K dernières
    barres. Sur les barres sans ticks (ou si les colonnes sont absentes) : PROXY barre =
    position de la clôture dans la barre ((c-l)-(h-c))/(h-l), pondérée par le volume.
    """

    name = "tick_imbalance"
    default_params = {"k_bars": 5, "threshold": 0.4, "sl_atr": 2.0, "tp_atr": 3.0,
                      "max_minutes": 15, "vol_window": 60,
                      "start_min": 8 * 60, "end_min": 14 * 60 + 30}
    grid = {"k_bars": [3, 5, 10], "threshold": [0.3, 0.5], "sl_atr": [2.0, 3.0],
            "tp_atr": [3.0, 5.0]}

    @staticmethod
    def _ratio(f: _Features, k: int) -> np.ndarray:
        def build():
            tv = pd.Series(np.maximum(f.tv, 1.0))
            rng = f.high - f.low
            with np.errstate(invalid="ignore", divide="ignore"):
                clv = np.where(rng > 0, ((f.close - f.low) - (f.high - f.close)) / rng, 0.0)
            proxy = (pd.Series(clv * tv).rolling(k, min_periods=k).sum()
                     / tv.rolling(k, min_periods=k).sum()).to_numpy()
            if "upticks" in f.m1:
                up = pd.Series(f.m1["upticks"].to_numpy(float)).rolling(k, min_periods=k).sum()
                dn = pd.Series(f.m1["downticks"].to_numpy(float)).rolling(k, min_periods=k).sum()
                tot = up + dn
                with np.errstate(invalid="ignore", divide="ignore"):
                    real = ((up - dn) / tot.where(tot > 0)).to_numpy()
                return np.where(np.isfinite(real), real, proxy)
            return proxy
        return f.memo(f"imb{k}", build)

    def signals(self, m1: pd.DataFrame, params: dict) -> list[Signal]:
        p = _merged(self, params)
        f = _features(m1)
        ratio = self._ratio(f, int(p["k_bars"]))
        atr = f.atr1
        vol_ok = f.tv > f.vol_median(int(p["vol_window"]))
        entry_min = f.ny_min + 1
        hours = (entry_min >= p["start_min"]) & (entry_min <= p["end_min"]) & (f.ny_dow <= 4)
        base = vol_ok & hours & np.isfinite(atr) & (atr > 0)
        with np.errstate(invalid="ignore"):
            buy = base & (ratio > p["threshold"])
            sell = base & (ratio < -p["threshold"])
        js = np.flatnonzero(buy | sell)
        if js.size == 0:
            return []
        a = atr[js]
        return _make(f, js, np.where(buy[js], "buy", "sell"), p["sl_atr"] * a, p["tp_atr"] * a,
                     p["max_minutes"], self.name, p)


STRATEGIES: dict[str, Strategy] = {
    s.name: s for s in (OrbStrategy(), VwapMeanReversion(), EiaMomentum(), TickImbalance())
}
