"""Collecte des historiques MT5, profil de coûts et données synthétiques.

Utilisation (Windows pour la collecte, tout système pour le reste) :

    python -m ava_mt5.data collect --symbols CrudeOIL,BrentOIL --out data
    python -m ava_mt5.data profile --data data
    python -m ava_mt5.data synthetic --out data_synth --days 120

Fuseau horaire : MT5 fournit des heures en heure du SERVEUR du courtier. La collecte
estime le décalage serveur-UTC (en heures entières) à partir du dernier tick, puis tout
est converti et stocké en UTC. LIMITE : un décalage unique est appliqué à tout
l'historique ; or le serveur suit en général l'heure d'été américaine (UTC+2 l'hiver,
UTC+3 l'été). Sur un historique qui traverse un changement d'heure, les barres situées de
l'autre côté du changement sont donc décalées d'une heure. Le décalage utilisé est
mémorisé dans meta.json (server_offset_h) ; --server-offset permet de le forcer.

Format des fichiers (colonne `time` : secondes epoch UTC pour les barres, `time_msc` :
millisecondes epoch UTC pour les ticks) :
    {SYM}_M1.csv.gz, {SYM}_M5.csv.gz : time, open, high, low, close, tick_volume, spread
                                         (spread en points)
    {SYM}_ticks.csv.gz               : time_msc, bid, ask, flags
    meta.json                        : paramètres des symboles et décalage horaire
    spread_profile.json              : profil de spread (voir build_profile)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time as _time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

NY = ZoneInfo("America/New_York")
UTC = dt.timezone.utc

BAR_COLUMNS = ["time", "open", "high", "low", "close", "tick_volume", "spread"]
TICK_COLUMNS = ["time_msc", "bid", "ask", "flags"]

MAX_BARS = 200_000
BLOCK_BARS = 50_000

# Événements hebdomadaires : (jour de semaine ISO python lundi=0, heure, minute) en heure de New York.
EVENTS = {
    "eia": (2, 10, 30),   # mercredi 10:30
    "api": (1, 16, 30),   # mardi 16:30
}


# --------------------------------------------------------------------- utilitaires
def to_epoch_s(index: pd.DatetimeIndex) -> np.ndarray:
    """Secondes epoch d'un index datetime, quelle que soit sa résolution interne."""
    return index.as_unit("s").asi8


def event_times_utc(start: pd.Timestamp, end: pd.Timestamp, kind: str) -> pd.DatetimeIndex:
    """Instants UTC des publications `kind` ('eia' ou 'api') entre start et end inclus.

    La conversion passe par zoneinfo : l'heure d'été de New York est donc gérée.
    """
    weekday, hour, minute = EVENTS[kind]
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    day = (start - pd.Timedelta(days=1)).date()
    last = (end + pd.Timedelta(days=1)).date()
    out = []
    while day <= last:
        if day.weekday() == weekday:
            local = dt.datetime(day.year, day.month, day.day, hour, minute, tzinfo=NY)
            ts = pd.Timestamp(local.astimezone(UTC))
            if start <= ts <= end:
                out.append(ts)
        day += dt.timedelta(days=1)
    return pd.DatetimeIndex(out, tz="UTC") if out else pd.DatetimeIndex([], tz="UTC")


def write_bars(df: pd.DataFrame, path: Path) -> None:
    """Écrit des barres (index temps UTC) au format csv.gz, temps en secondes epoch."""
    out = df.copy()
    out.insert(0, "time", to_epoch_s(out.index))
    out = out[BAR_COLUMNS].astype({"time": "int64", "tick_volume": "int64", "spread": "int64"})
    out.to_csv(path, index=False)


def read_meta(data_dir: Path | str) -> dict[str, Any]:
    p = Path(data_dir) / "meta.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def write_meta(data_dir: Path | str, meta: dict[str, Any]) -> None:
    (Path(data_dir) / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def list_symbols(data_dir: Path | str) -> list[str]:
    """Symboles ayant un fichier M1 dans le dossier de données."""
    return sorted(p.name[: -len("_M1.csv.gz")] for p in Path(data_dir).glob("*_M1.csv.gz"))


# --------------------------------------------------------------------- collecte
def estimate_server_offset(mt5: Any, symbol: str, now: float | None = None) -> int:
    """Décalage serveur-UTC en heures entières, d'après le dernier tick du symbole.

    Exige un marché ouvert : sur un tick périmé l'écart ne ressemble pas à un nombre
    entier d'heures et l'on lève une erreur (utiliser alors --server-offset).
    """
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError(f"Pas de tick pour {symbol} : impossible d'estimer le fuseau.")
    now = _time.time() if now is None else now
    diff = tick.time - now
    offset = round(diff / 3600)
    if abs(diff - offset * 3600) > 900 or not -12 <= offset <= 14:
        raise RuntimeError(
            f"Décalage horaire incohérent pour {symbol} (écart {diff:.0f} s) : marché fermé ? "
            "Relancer marché ouvert ou utiliser --server-offset.")
    return int(offset)


def fetch_bars(mt5: Any, symbol: str, timeframe: int, offset_h: int,
               max_bars: int = MAX_BARS, block: int = BLOCK_BARS) -> pd.DataFrame:
    """Récupère le maximum de barres disponibles, par blocs, et les convertit en UTC."""
    chunks: list[pd.DataFrame] = []
    total = 0
    pos = 0
    while total < max_bars:
        count = min(block, max_bars - total)
        data = mt5.copy_rates_from_pos(symbol, timeframe, pos, count)
        if data is None or len(data) == 0:
            break
        df = pd.DataFrame(data)
        chunks.insert(0, df)  # on remonte dans le temps : les blocs lus plus tard sont plus anciens
        total += len(df)
        pos += len(df)
        if len(df) < count:
            break
    if not chunks:
        return pd.DataFrame(columns=BAR_COLUMNS[1:], index=pd.DatetimeIndex([], tz="UTC"))
    df = pd.concat(chunks, ignore_index=True)
    if "spread" not in df:
        df["spread"] = 0
    df["time"] = pd.to_datetime(df["time"].astype("int64") - offset_h * 3600, unit="s", utc=True)
    df = df.drop_duplicates("time").sort_values("time").set_index("time")
    return df[BAR_COLUMNS[1:]]


def fetch_ticks(mt5: Any, symbol: str, weeks: int, offset_h: int,
                now_utc: dt.datetime | None = None) -> pd.DataFrame:
    """Ticks (COPY_TICKS_ALL) des `weeks` dernières semaines, tranches d'un jour.

    Particularité de l'API Python MT5 : les datetime passés à copy_ticks_range sont
    interprétés comme de l'heure serveur « déguisée » en UTC. On construit donc les bornes
    en heure serveur (UTC + décalage) avec un tzinfo UTC, et l'on corrige time_msc au retour.
    """
    now_utc = now_utc or dt.datetime.now(UTC)
    end_srv = now_utc + dt.timedelta(hours=offset_h)
    start_srv = end_srv - dt.timedelta(weeks=weeks)
    chunks = []
    cur = start_srv
    while cur < end_srv:
        nxt = min(cur + dt.timedelta(days=1), end_srv)
        data = mt5.copy_ticks_range(symbol, cur, nxt, mt5.COPY_TICKS_ALL)
        if data is not None and len(data) > 0:
            df = pd.DataFrame(data)
            chunks.append(df[["time_msc", "bid", "ask", "flags"]])
        cur = nxt
    if not chunks:
        return pd.DataFrame(columns=TICK_COLUMNS)
    df = pd.concat(chunks, ignore_index=True)
    df["time_msc"] = df["time_msc"].astype("int64") - offset_h * 3_600_000
    return df.drop_duplicates().sort_values("time_msc", kind="stable").reset_index(drop=True)


def symbol_meta(mt5: Any, symbol: str, offset_h: int) -> dict[str, Any]:
    i = mt5.symbol_info(symbol)
    g = lambda name, default=0.0: getattr(i, name, default)  # noqa: E731
    return {
        "point": g("point"), "digits": g("digits", 2),
        "contract_size": g("trade_contract_size", 1.0),
        "volume_min": g("volume_min"), "volume_step": g("volume_step"),
        "volume_max": g("volume_max", 100.0),
        "swap_long": g("swap_long"), "swap_short": g("swap_short"),
        "swap_mode": g("swap_mode", 1),
        "tick_size": g("trade_tick_size", g("point")),
        "tick_value": g("trade_tick_value"),
        "server_offset_h": offset_h,
    }


def collect(broker: Any, symbols: list[str], out: Path | str, weeks: int = 4,
            server_offset: int | None = None, max_bars: int = MAX_BARS,
            block: int = BLOCK_BARS) -> dict[str, Any]:
    """Collecte M1, M5 et ticks de chaque symbole présent et écrit meta.json."""
    mt5 = broker.mt5
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    meta = read_meta(out) or {}
    meta.setdefault("symbols", {})
    meta["synthetic"] = False
    offset = server_offset
    done = []
    for sym in symbols:
        if mt5.symbol_info(sym) is None:
            hints = [h["name"] for key in ("oil", "brent", "wti") for h in broker.search_symbols(key)]
            hints = sorted(set(hints))
            print(f"AVERTISSEMENT : symbole introuvable : {sym}. "
                  f"Candidats : {', '.join(hints) if hints else 'aucun'}", file=sys.stderr)
            continue
        broker._symbol(sym)  # active le symbole dans le Market Watch
        if offset is None:
            offset = estimate_server_offset(mt5, sym)
            print(f"Décalage serveur-UTC estimé : {offset:+d} h")
        for tf_name in ("M1", "M5"):
            tf = getattr(mt5, f"TIMEFRAME_{tf_name}")
            bars = fetch_bars(mt5, sym, tf, offset, max_bars=max_bars, block=block)
            if bars.empty:
                print(f"AVERTISSEMENT : aucune barre {tf_name} pour {sym}", file=sys.stderr)
                continue
            write_bars(bars, out / f"{sym}_{tf_name}.csv.gz")
            print(f"{sym} {tf_name} : {len(bars)} barres, {bars.index[0]} -> {bars.index[-1]}")
        ticks = fetch_ticks(mt5, sym, weeks, offset)
        if len(ticks):
            ticks.to_csv(out / f"{sym}_ticks.csv.gz", index=False)
        print(f"{sym} ticks : {len(ticks)}")
        meta["symbols"][sym] = symbol_meta(mt5, sym, offset)
        done.append(sym)
    if offset is not None:
        meta["server_offset_h"] = offset
    meta["collected_at"] = dt.datetime.now(UTC).isoformat(timespec="seconds")
    meta["weeks"] = weeks
    write_meta(out, meta)
    return meta


# --------------------------------------------------------------------- profil de coûts
def _load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _minute_spread(data_dir: Path, sym: str, point: float) -> tuple[pd.Series, pd.Series | None, str]:
    """Spread (unités de prix) : série par minute combinée, ticks bruts, source.

    Les barres M1 donnent la couverture longue ; les minutes couvertes par des ticks sont
    remplacées par la médiane tick à tick (plus fine).
    """
    m1_path = data_dir / f"{sym}_M1.csv.gz"
    bars = None
    if m1_path.exists():
        df = _load_csv(m1_path)
        idx = pd.to_datetime(df["time"], unit="s", utc=True)
        bars = pd.Series(df["spread"].to_numpy(float) * point, index=idx)
    spread_ticks = None
    minute = None
    tick_path = data_dir / f"{sym}_ticks.csv.gz"
    if tick_path.exists():
        t = _load_csv(tick_path)
        if len(t):
            idx = pd.to_datetime(t["time_msc"], unit="ms", utc=True)
            spread_ticks = pd.Series((t["ask"] - t["bid"]).to_numpy(float), index=idx)
            minute = spread_ticks.groupby(spread_ticks.index.floor("min")).median()
    if bars is None and minute is None:
        raise FileNotFoundError(f"Aucune donnée de spread pour {sym} dans {data_dir}")
    if bars is not None and minute is not None:
        combined = bars.copy()
        combined = combined[~combined.index.isin(minute.index)]
        combined = pd.concat([combined, minute]).sort_index()
    else:
        combined = minute if minute is not None else bars
    return combined, spread_ticks, "ticks" if spread_ticks is not None else "bars"


def _event_stats(minute: pd.Series, kind: str, fallback: float) -> dict[str, Any]:
    if minute.empty:
        return {"before_5m": fallback, "minute_0_to_3": [fallback] * 4, "normal": fallback}
    events = event_times_utc(minute.index[0], minute.index[-1], kind)
    before, normal = [], []
    mins: list[list[float]] = [[], [], [], []]
    for ev in events:
        b = minute[(minute.index >= ev - pd.Timedelta(minutes=5)) & (minute.index < ev)]
        n = minute[(minute.index >= ev - pd.Timedelta(minutes=60))
                   & (minute.index < ev - pd.Timedelta(minutes=10))]
        if b.empty or n.empty:
            continue
        before.append(float(b.median()))
        normal.append(float(n.median()))
        for k in range(4):
            v = minute.get(ev + pd.Timedelta(minutes=k))
            if v is not None and np.isfinite(v):
                mins[k].append(float(v))
    if not normal:
        return {"before_5m": fallback, "minute_0_to_3": [fallback] * 4, "normal": fallback}
    norm = float(np.median(normal))
    return {
        "before_5m": round(float(np.median(before)), 6),
        "minute_0_to_3": [round(float(np.median(m)), 6) if m else norm for m in mins],
        "normal": round(norm, 6),
    }


def build_profile(data_dir: Path | str) -> dict[str, Any]:
    """Construit le profil de spread de chaque symbole (unités de prix, heures UTC).

    Format : {"SYM": {"point", "hourly_median": {"0".."23"}, "hourly_p90": {...},
              "eia": {"before_5m", "minute_0_to_3": [4 valeurs], "normal"}, "api": {...},
              "source": "ticks"|"bars"}}
    """
    data_dir = Path(data_dir)
    meta = read_meta(data_dir).get("symbols", {})
    symbols = list(meta) or list_symbols(data_dir)
    profile: dict[str, Any] = {}
    for sym in symbols:
        point = float(meta.get(sym, {}).get("point") or 0.01)
        minute, ticks, source = _minute_spread(data_dir, sym, point)
        raw = ticks if ticks is not None else minute
        grp = raw.groupby(raw.index.hour)
        med = grp.median()
        p90 = grp.quantile(0.9)
        overall_med, overall_p90 = float(raw.median()), float(raw.quantile(0.9))
        hourly_median = {str(h): round(float(med.get(h, overall_med)), 6) for h in range(24)}
        hourly_p90 = {str(h): round(float(p90.get(h, overall_p90)), 6) for h in range(24)}
        profile[sym] = {
            "point": point,
            "hourly_median": hourly_median,
            "hourly_p90": hourly_p90,
            "eia": _event_stats(minute, "eia", overall_med),
            "api": _event_stats(minute, "api", overall_med),
            "source": source,
        }
    return profile


def format_profile(profile: dict[str, Any]) -> str:
    lines = []
    for sym, p in profile.items():
        lines.append(f"\n=== {sym} (point {p['point']}, source : {p['source']}) ===")
        lines.append("Heure UTC | spread médian | spread p90   (en prix)")
        for h in range(24):
            lines.append(f"   {h:02d}h    |    {p['hourly_median'][str(h)]:.4f}     |  "
                         f"{p['hourly_p90'][str(h)]:.4f}")
        for kind in ("eia", "api"):
            e = p[kind]
            m = " ".join(f"{v:.4f}" for v in e["minute_0_to_3"])
            lines.append(f"{kind.upper()} : normal {e['normal']:.4f} | 5 min avant {e['before_5m']:.4f}"
                         f" | minutes 0 à 3 : {m}")
    return "\n".join(lines)


def profile_command(data_dir: Path | str) -> dict[str, Any]:
    data_dir = Path(data_dir)
    profile = build_profile(data_dir)
    (data_dir / "spread_profile.json").write_text(
        json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
    print(format_profile(profile))
    print(f"\nProfil écrit dans {data_dir / 'spread_profile.json'}")
    return profile


# --------------------------------------------------------------------- données synthétiques
# Profils par heure UTC (heure NY = UTC-5/-4 : l'ouverture NY tombe vers 13-15 UTC).
_VOL_BY_HOUR = np.array([0.55, 0.5, 0.5, 0.5, 0.55, 0.6, 0.7, 0.85, 0.9, 0.9, 0.95, 1.0,
                         1.1, 1.3, 1.7, 1.6, 1.2, 1.1, 1.0, 0.8, 0.6, 0.5, 0.4, 0.45])
_SPREAD_BY_HOUR = np.array([0.060, 0.065, 0.065, 0.060, 0.055, 0.050, 0.045, 0.040, 0.035,
                            0.035, 0.035, 0.032, 0.030, 0.028, 0.028, 0.028, 0.030, 0.032,
                            0.035, 0.045, 0.055, 0.070, 0.080, 0.070])
_EIA_SPREAD_MULT = (6.0, 4.0, 2.5, 1.6)
_API_SPREAD_MULT = (4.0, 3.0, 2.0, 1.4)
_JUMP_FRACTIONS = (0.5, 0.25, 0.15, 0.10)


def generate_synthetic(out: Path | str, days: int = 120, seed: int = 20261118,
                       symbols: tuple[str, ...] = ("CrudeOIL", "BrentOIL"),
                       tick_days: int = 28) -> dict[str, Any]:
    """Écrit des données M1/M5/ticks synthétiques (marche aléatoire, graine fixe)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    end = pd.Timestamp("2026-11-13 21:00", tz="UTC")  # date fixe : sorties reproductibles
    start = end - pd.Timedelta(days=days)
    full = pd.date_range(start.floor("min"), end, freq="1min", tz="UTC")
    keep = (full.dayofweek < 5) & (full.hour != 22)  # week-end et pause quotidienne (17:00 NY en hiver) fermés
    idx = full[keep]
    n = len(idx)
    hours = idx.hour.to_numpy()
    sigma = 0.03 * _VOL_BY_HOUR[hours]
    spread_price = _SPREAD_BY_HOUR[hours].copy()
    vol_mult = np.ones(n)
    jump = np.zeros((2, n))  # lignes : EIA, API
    spread_mult = np.ones(n)
    epoch_min = to_epoch_s(idx) // 60
    pos_of = {int(v): i for i, v in enumerate(epoch_min)}
    for kind, mults, amp, row in (("eia", _EIA_SPREAD_MULT, 0.5, 0), ("api", _API_SPREAD_MULT, 0.3, 1)):
        for ev in event_times_utc(idx[0], idx[-1], kind):
            e = int(ev.value // 60_000_000_000)
            shock = rng.normal(0, amp)
            for k in range(-5, 6):
                i = pos_of.get(e + k)
                if i is None:
                    continue
                if k < 0:
                    spread_mult[i] = max(spread_mult[i], 1.5)
                else:
                    vol_mult[i] = 3.0
                    if k < 4:
                        spread_mult[i] = max(spread_mult[i], mults[k])
                        jump[row, i] += shock * _JUMP_FRACTIONS[k]
    common = rng.standard_t(5, n) / np.sqrt(5 / 3)
    base_prices = {"CrudeOIL": 70.0, "BrentOIL": 73.5}
    meta_syms: dict[str, Any] = {}
    for si, sym in enumerate(symbols):
        own = rng.standard_t(5, n) / np.sqrt(5 / 3)
        noise = (0.85 * common + np.sqrt(1 - 0.85 ** 2) * own) if si else common
        step = noise * sigma * vol_mult + jump.sum(axis=0)
        close = np.maximum(base_prices.get(sym, 70.0) + np.cumsum(step), 20.0)
        close = np.round(close, 2)
        open_ = np.concatenate([[close[0]], close[:-1]])
        wick = np.abs(rng.normal(0, sigma * 0.5, (2, n)))
        high = np.round(np.maximum(open_, close) + wick[0], 2)
        low = np.round(np.minimum(open_, close) - wick[1], 2)
        high = np.maximum(high, np.maximum(open_, close))
        low = np.minimum(low, np.minimum(open_, close))
        sp_pts = np.maximum(1, np.round(spread_price * spread_mult
                                        * np.exp(rng.normal(0, 0.2, n)) / 0.01)).astype(int)
        lam = 40 * _VOL_BY_HOUR[hours] * vol_mult
        tick_volume = rng.poisson(lam) + 1
        m1 = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                           "tick_volume": tick_volume, "spread": sp_pts}, index=idx)
        m5 = m1.resample("5min", closed="left", label="left").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last",
             "tick_volume": "sum", "spread": "max"}).dropna()
        write_bars(m1, out / f"{sym}_M1.csv.gz")
        write_bars(m5, out / f"{sym}_M5.csv.gz")
        # ticks sur les dernières semaines seulement
        tmask = idx >= end - pd.Timedelta(days=tick_days)
        ti = np.flatnonzero(tmask)
        cnt = np.clip(tick_volume[ti] // 5, 2, 30)
        grp = np.repeat(np.arange(len(ti)), cnt)
        total = int(cnt.sum())
        u = rng.random(total)
        order = np.lexsort((u, grp))
        u = u[order]
        bi = ti[grp]
        mid = open_[bi] + (close[bi] - open_[bi]) * u + rng.normal(0, 1, total) * sigma[bi] * 0.3
        mid = np.clip(mid, low[bi], high[bi])
        bid = np.round(mid, 2)
        sp_tick = np.maximum(1, sp_pts[bi] + np.round(rng.normal(0, 0.6, total)).astype(int))
        ask = np.round(bid + sp_tick * 0.01, 2)
        tmsc = epoch_min[bi] * 60_000 + np.floor(u * 59_999).astype(np.int64)
        ticks = pd.DataFrame({"time_msc": tmsc, "bid": bid, "ask": ask,
                              "flags": np.full(total, 6, dtype=np.int64)})
        ticks.to_csv(out / f"{sym}_ticks.csv.gz", index=False)
        meta_syms[sym] = {
            "point": 0.01, "digits": 2, "contract_size": 1000.0,
            "volume_min": 0.01, "volume_step": 0.01, "volume_max": 100.0,
            "swap_long": -25.0, "swap_short": 5.0, "swap_mode": 1,
            "tick_size": 0.01, "tick_value": 10.0, "server_offset_h": 0,
        }
    meta = {"synthetic": True, "collected_at": dt.datetime.now(UTC).isoformat(timespec="seconds"),
            "server_offset_h": 0, "weeks": tick_days // 7, "seed": seed, "symbols": meta_syms}
    write_meta(out, meta)
    print(f"Données synthétiques écrites dans {out} ({n} barres M1 par symbole, "
          f"ticks sur {tick_days} jours).")
    return meta


# --------------------------------------------------------------------- ligne de commande
def _settings_from_env():
    from ava_mt5.core import Settings
    login = os.environ.get("MT5_LOGIN") or None
    return Settings(
        login=int(login) if login else None,
        password=os.environ.get("MT5_PASSWORD") or None,
        server=os.environ.get("MT5_SERVER") or None,
        path=os.environ.get("MT5_PATH") or None,
    )


def _cmd_collect(args: argparse.Namespace) -> int:
    try:
        import MetaTrader5 as mt5  # type: ignore[import-not-found]  # Windows uniquement
    except ImportError:
        print("Le paquet MetaTrader5 est absent : la collecte ne fonctionne que sous Windows.",
              file=sys.stderr)
        return 2
    from ava_mt5.core import Broker, TradingError
    broker = Broker(mt5, _settings_from_env())
    try:
        broker.connect()  # refuse les comptes réels
        collect(broker, [s.strip() for s in args.symbols.split(",") if s.strip()], args.out,
                weeks=args.weeks, server_offset=args.server_offset)
    except (TradingError, RuntimeError) as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1
    finally:
        broker.shutdown()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ava_mt5.data", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect", help="collecter les historiques depuis MT5 (Windows)")
    c.add_argument("--symbols", default="CrudeOIL,BrentOIL")
    c.add_argument("--out", default="data")
    c.add_argument("--weeks", type=int, default=4, help="semaines de ticks (défaut 4)")
    c.add_argument("--server-offset", type=int, default=None,
                   help="décalage serveur-UTC en heures (sinon estimé)")
    p = sub.add_parser("profile", help="construire spread_profile.json")
    p.add_argument("--data", default="data")
    s = sub.add_parser("synthetic", help="générer des données synthétiques")
    s.add_argument("--out", default="data_synth")
    s.add_argument("--days", type=int, default=120)
    args = parser.parse_args(argv)
    if args.cmd == "collect":
        return _cmd_collect(args)
    if args.cmd == "profile":
        profile_command(args.data)
        return 0
    generate_synthetic(args.out, days=args.days)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
