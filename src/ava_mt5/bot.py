"""Bot de scalp pétrole sur compte DÉMO MetaTrader 5.

Lancement : python -m ava_mt5.bot [--symbols CrudeOIL,BrentOIL] [--dry-run] [--once] ...

Principes de sécurité
  - Jamais d'ordre sur un compte réel : trade_mode est revérifié à chaque connexion,
    à chaque cycle et juste avant chaque ordre.
  - MODE OBSERVATION (aucun ordre) tant qu'aucune stratégie n'a passé le backtest
    (rapport absent, "synthetic": true ou aucun "passed"), sauf --force-strategies.
  - Le bot ne touche et ne compte que les positions / deals de son propre magic
    (MAGIC_SCALP) : un autre module tournant sur le même compte est ignoré.

Dimensionnement : voir position_size (tick_value / tick_size si disponibles, sinon
contract_size x prix, supposant une devise de profit identique à celle du compte).

Journée des coupe-circuits : jour de trading démarrant à 17:00 New York (clôture
FX/futures), indépendant de l'heure d'été.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable, Iterable
from zoneinfo import ZoneInfo

import pandas as pd

from . import news
from .core import Broker, Settings, TradingError

log = logging.getLogger("ava_mt5.bot")

MAGIC_SCALP = 20261119
UTC = timezone.utc
NY = ZoneInfo("America/New_York")

TRADE_COLUMNS = ["time", "symbol", "strategy", "side", "volume", "price", "sl", "tp",
                 "ticket", "event", "pnl", "reason"]


class ConnectionLost(RuntimeError):
    """Terminal MT5 injoignable ou réponse vide : il faut se reconnecter."""


class RealAccountError(TradingError):
    """Le compte connecté n'est pas un compte démo."""


# ====================================================================== logique pure
def position_size(equity: float, risk_frac: float, sl_dist: float, contract_size: float,
                  volume_min: float, volume_step: float, volume_max: float, *,
                  max_volume: float | None = None, tick_value: float | None = None,
                  tick_size: float | None = None) -> float:
    """Lots = équité x risque / perte par lot au stop, arrondis À LA BAISSE au pas.

    Perte par lot : sl_dist / tick_size x tick_value (exact, en devise du compte) si ces
    deux valeurs sont fournies, sinon sl_dist x contract_size (approximation).
    Retourne 0 si le résultat est sous volume_min. Plafonné à volume_max et max_volume.
    """
    if equity <= 0 or risk_frac <= 0 or sl_dist <= 0 or volume_step <= 0:
        return 0.0
    if tick_value and tick_size and tick_value > 0 and tick_size > 0:
        perte_par_lot = sl_dist / tick_size * tick_value
    else:
        perte_par_lot = sl_dist * contract_size
    if perte_par_lot <= 0:
        return 0.0
    lots = equity * risk_frac / perte_par_lot
    plafond = volume_max if max_volume is None else min(volume_max, max_volume)
    vol = Broker._round_step(min(lots, plafond), volume_step)
    return vol if vol >= volume_min - 1e-12 else 0.0


def evaluate_entry(side: str, *, has_position: bool, spread: float | None,
                   median_spread: float | None, bias: dict | str,
                   spread_mult: float = 1.5) -> tuple[bool, str]:
    """Décision d'entrée (pure) : (autorisé, raison)."""
    if has_position:
        return False, "position déjà ouverte sur ce symbole"
    if median_spread is None or median_spread <= 0:
        return False, "profil de spread absent pour cette heure"
    if spread is None:
        return False, "spread inconnu"
    if spread > spread_mult * median_spread + 1e-9:
        return False, f"spread {spread:.5g} > {spread_mult:g} x médiane {median_spread:.5g}"
    if not news.allows(bias, side):
        b = bias.get("bias") if isinstance(bias, dict) else bias
        return False, f"biais {b} interdit {side}"
    return True, "ok"


def combine_bias(a: dict, b: dict) -> dict:
    """La plus restrictive gagne ; long contre short -> no_trade."""
    ba, bb = a["bias"], b["bias"]
    if "no_trade" in (ba, bb) or {ba, bb} == {"long", "short"}:
        res = "no_trade"
    elif ba == "neutral":
        res = bb
    else:
        res = ba
    return {"bias": res, "until": a.get("until") if res == ba else b.get("until"),
            "reasons": list(a.get("reasons", [])) + list(b.get("reasons", []))}


def forced_exit_due(open_time: float, now_server: float, max_minutes: float) -> bool:
    return (now_server - open_time) / 60.0 > max_minutes


def trading_day(now_utc: datetime) -> str:
    """Étiquette du jour de trading (rollover à 17:00 New York)."""
    return (now_utc.astimezone(NY) + timedelta(hours=7)).date().isoformat()


class CircuitBreaker:
    """Coupe-circuits purs ; état sérialisable (dict).

    - arrêt du jour : équité <= -max_daily_loss vs début de journée, ou N pertes
      consécutives (le compteur repart à zéro à chaque nouveau jour de trading) ;
    - arrêt complet : drawdown depuis le pic d'équité >= max_drawdown ; persistant,
      levé uniquement par reset_halt().
    """

    def __init__(self, max_daily_loss: float = 0.02, max_consecutive_losses: int = 5,
                 max_drawdown: float = 0.08, state: dict | None = None) -> None:
        self.max_daily_loss = max_daily_loss
        self.max_consecutive_losses = max_consecutive_losses
        self.max_drawdown = max_drawdown
        s = state or {}
        self.day: str | None = s.get("day")
        self.day_start_equity: float | None = s.get("day_start_equity")
        self.peak_equity: float | None = s.get("peak_equity")
        self.consecutive_losses: int = int(s.get("consecutive_losses", 0))
        self.halted_today: bool = bool(s.get("halted_today", False))
        self.halted_full: bool = bool(s.get("halted_full", False))
        self.reason: str = s.get("reason", "")
        self.equity: float | None = None

    def to_dict(self) -> dict:
        return {"day": self.day, "day_start_equity": self.day_start_equity,
                "peak_equity": self.peak_equity,
                "consecutive_losses": self.consecutive_losses,
                "halted_today": self.halted_today, "halted_full": self.halted_full,
                "reason": self.reason}

    @property
    def day_pnl_pct(self) -> float:
        if not self.day_start_equity or self.equity is None:
            return 0.0
        return (self.equity - self.day_start_equity) / self.day_start_equity * 100.0

    @property
    def drawdown_pct(self) -> float:
        if not self.peak_equity or self.equity is None:
            return 0.0
        return (self.peak_equity - self.equity) / self.peak_equity * 100.0

    def can_trade(self) -> bool:
        return not (self.halted_today or self.halted_full)

    def update(self, now_utc: datetime, equity: float, closed_pnls: Iterable[float] = ()) -> None:
        self.equity = equity
        jour = trading_day(now_utc)
        if jour != self.day:
            self.day, self.day_start_equity = jour, equity
            self.consecutive_losses = 0
            self.halted_today = False
            if not self.halted_full:
                self.reason = ""
        if self.peak_equity is None or equity > self.peak_equity:
            self.peak_equity = equity
        for pnl in closed_pnls:
            self.consecutive_losses = self.consecutive_losses + 1 if pnl < 0 else 0
        if self.consecutive_losses >= self.max_consecutive_losses and not self.halted_today:
            self.halted_today = True
            self.reason = f"{self.consecutive_losses} pertes consécutives"
        if self.day_start_equity and self.day_pnl_pct <= -self.max_daily_loss * 100 + 1e-9 \
                and not self.halted_today:
            self.halted_today = True
            self.reason = f"perte du jour {self.day_pnl_pct:.2f} %"
        if self.drawdown_pct >= self.max_drawdown * 100 - 1e-9 and not self.halted_full:
            self.halted_full = True
            self.reason = f"drawdown {self.drawdown_pct:.2f} % depuis le pic"

    def reset_halt(self) -> None:
        """Reprise manuelle : lève les arrêts, le pic repart de l'équité courante."""
        self.halted_full = False
        self.halted_today = False
        self.peak_equity = None
        self.consecutive_losses = 0
        self.reason = ""


def load_report(path: str | os.PathLike) -> dict | None:
    data = news.load_json(path)
    return data if isinstance(data, dict) else None


def active_strategies(report: dict | None, symbol: str, registry: dict | None = None,
                      force: Iterable[str] = ()) -> dict[str, dict]:
    """{nom: params} des stratégies autorisées à trader ce symbole.

    Rapport absent / synthétique : aucune (observation). Les stratégies forcées
    s'ajoutent quoi qu'il arrive (params du rapport, sinon default_params).
    """
    registry = registry or {}
    res: dict[str, dict] = {}
    synthetic = bool(report and report.get("synthetic"))
    results = ((report or {}).get("results") or {}).get(symbol) or {}
    if report and not synthetic:
        for name, r in results.items():
            if isinstance(r, dict) and r.get("passed") is True:
                res[name] = dict(r.get("params") or {})
    for name in force:
        r = results.get(name)
        if isinstance(r, dict) and r.get("params") is not None:
            res[name] = dict(r["params"])
        else:
            strat = registry.get(name)
            res[name] = dict(getattr(strat, "default_params", {}) or {})
    return {n: p for n, p in res.items() if n in registry}


def load_registry() -> dict:
    try:
        from .strategies import STRATEGIES
        return dict(STRATEGIES)
    except Exception as e:  # module absent ou cassé : observation
        log.warning("Stratégies indisponibles (%s) : mode observation", e)
        return {}


# ====================================================================== configuration
@dataclass
class BotConfig:
    symbols: list[str] = field(default_factory=lambda: ["CrudeOIL", "BrentOIL"])
    report_path: str = "logs/backtest_report.json"
    profile_path: str = "data/spread_profile.json"
    meta_path: str = "data/meta.json"
    config_dir: str = "config"
    logs_dir: str = "logs"
    risk: float = 0.005
    dry_run: bool = False
    once: bool = False
    force_strategies: list[str] = field(default_factory=list)
    spread_mult: float = 1.5
    poll_s: float = 2.0
    n_bars: int = 3000
    max_entry_delay_s: float = 20.0     # au-delà, la barre est trop avancée pour entrer à l'open
    default_max_minutes: float = 60.0   # sortie forcée si ticket inconnu (prudent)
    max_daily_loss: float = 0.02
    max_consecutive_losses: int = 5
    max_drawdown: float = 0.08
    magic: int = MAGIC_SCALP


class _Writer:
    """Écriture JSON atomique."""

    @staticmethod
    def json(path: Path, data: Any) -> None:
        news._write_json_atomic(path, data)


# ====================================================================== bot (E/S)
class Bot:
    def __init__(self, mt5: Any, settings: Settings, cfg: BotConfig, *,
                 clock: Callable[[], float] = time.time,
                 sleep: Callable[[float], Any] | None = None,
                 registry: dict | None = None) -> None:
        self.mt5, self.cfg, self.settings = mt5, cfg, settings
        self.broker = Broker(mt5, settings)
        self.clock = clock
        self._stop = threading.Event()
        self.sleep = sleep or self._stop.wait
        self.registry = load_registry() if registry is None else registry
        self.logs = Path(cfg.logs_dir)
        self.logs.mkdir(parents=True, exist_ok=True)
        self.state_path = self.logs / "state.json"
        state = news.load_json(self.state_path) or {}
        self.breaker = CircuitBreaker(cfg.max_daily_loss, cfg.max_consecutive_losses,
                                      cfg.max_drawdown, state.get("breaker"))
        self.tickets: dict[str, dict] = dict(state.get("tickets") or {})
        self.seen_deals: list[int] = list(state.get("seen_deals") or [])
        self.baselined = "seen_deals" in state
        self.forced_tickets: set[str] = set(state.get("forced") or [])
        self.offsets: dict[str, int] = {}
        self.last_bar: dict[str, int] = {}
        self.active: dict[str, dict] = {}
        self.forced_mode = False
        self.mode = "observation"
        self.connected = False
        self.backoff = 2.0
        self.last_error: str | None = None
        self.snap: dict[str, Any] = {"spreads": {}, "bias": None, "account": {}}
        self._json_cache: dict[str, tuple[float, Any]] = {}
        self._dirty = True
        self._mode_logged = False

    # ---------------------------------------------------------------- horloge / E/S
    def now_utc(self) -> datetime:
        return datetime.fromtimestamp(self.clock(), tz=UTC)

    def stop(self) -> None:
        self._stop.set()

    def _cached(self, path: str) -> Any:
        try:
            mt = os.path.getmtime(path)
        except OSError:
            return None
        hit = self._json_cache.get(path)
        if hit and hit[0] == mt:
            return hit[1]
        data = news.load_json(path)
        self._json_cache[path] = (mt, data)
        return data

    def _log_trade(self, **row: Any) -> None:
        p = self.logs / "trades_scalp.csv"
        new = not p.exists()
        with open(p, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, TRADE_COLUMNS)
            if new:
                w.writeheader()
            w.writerow({c: row.get(c, "") for c in TRADE_COLUMNS})

    def _save_state(self) -> None:
        _Writer.json(self.state_path, {
            "breaker": self.breaker.to_dict(), "tickets": self.tickets,
            "seen_deals": self.seen_deals[-2000:], "forced": sorted(self.forced_tickets)[-500:]})

    def _write_status(self) -> None:
        b, a = self.breaker, self.snap.get("account", {})
        _Writer.json(self.logs / "status.json", {
            "time": self.now_utc().isoformat(timespec="seconds"),
            "mode": self.mode, "dry_run": self.cfg.dry_run, "connected": self.connected,
            "account_login": a.get("login"), "demo": a.get("demo"),
            "equity": a.get("equity"), "balance": a.get("balance"),
            "open_positions": self.snap.get("open_positions", 0),
            "halted_today": b.halted_today, "halted_full": b.halted_full,
            "halt_reason": b.reason, "consecutive_losses": b.consecutive_losses,
            "day_pnl_pct": round(b.day_pnl_pct, 3), "drawdown_pct": round(b.drawdown_pct, 3),
            "last_bar": {s: datetime.fromtimestamp(t - self.offsets.get(s, 0) * 3600, UTC)
                         .isoformat(timespec="seconds") for s, t in self.last_bar.items()},
            "spreads": self.snap.get("spreads", {}), "bias": self.snap.get("bias"),
            "active_strategies": {s: sorted(v) for s, v in self.active.items()},
            "last_error": self.last_error})

    # ---------------------------------------------------------------- connexion
    def _check_account(self) -> Any:
        info = self.mt5.account_info()
        if info is None:
            raise ConnectionLost(f"account_info indisponible : {self.mt5.last_error()}")
        if self.mt5.terminal_info() is None:
            raise ConnectionLost("terminal_info indisponible")
        if info.trade_mode != self.mt5.ACCOUNT_TRADE_MODE_DEMO:
            raise RealAccountError(f"Compte {info.login} non démo : aucun ordre autorisé.")
        return info

    def connect(self) -> None:
        self._drop()
        self.broker.connect()          # refuse déjà les comptes non démo
        self._check_account()
        self.connected = True
        self.offsets.clear()
        log.info("Connecté au compte démo %s", self.broker.account().get("login"))

    def _drop(self) -> None:
        try:
            self.mt5.shutdown()
        except Exception:
            pass
        self.broker.connected = False
        self.connected = False

    def _offset(self, symbol: str) -> int:
        """Décalage heure serveur - UTC (h) : data/meta.json, sinon estimation sur le tick."""
        if symbol in self.offsets:
            return self.offsets[symbol]
        meta = self._cached(self.cfg.meta_path)
        off = None
        if isinstance(meta, dict):
            g = meta.get("server_offset_h")
            if isinstance(g, (int, float)):
                off = g
            elif isinstance(g, dict) and isinstance(g.get(symbol), (int, float)):
                off = g[symbol]
            elif isinstance(meta.get(symbol), dict):
                off = meta[symbol].get("server_offset_h")
        if off is None:
            tick = self.mt5.symbol_info_tick(symbol)
            if tick is None:
                raise ConnectionLost(f"pas de tick pour estimer le décalage de {symbol}")
            off = round((tick.time - self.clock()) / 3600)
            if abs(off) > 14:
                raise ConnectionLost(f"décalage serveur aberrant ({off} h, tick périmé ?) : "
                                     "renseigner server_offset_h dans data/meta.json")
            log.info("Décalage serveur estimé pour %s : %+d h", symbol, off)
        self.offsets[symbol] = int(off)
        return self.offsets[symbol]

    # ---------------------------------------------------------------- positions / deals
    def _my_positions(self, symbol: str | None = None) -> list[Any]:
        raw = self.mt5.positions_get(symbol=symbol) if symbol else self.mt5.positions_get()
        return [p for p in raw or () if getattr(p, "magic", None) == self.cfg.magic]

    def _process_deals(self, now: datetime) -> list[float]:
        off = next(iter(self.offsets.values()), 0)
        srv = now.replace(tzinfo=None) + timedelta(hours=off)
        deals = self.mt5.history_deals_get(srv - timedelta(days=2), srv + timedelta(days=1))
        seen = set(self.seen_deals)
        new = sorted((d for d in deals or ()
                      if getattr(d, "magic", None) == self.cfg.magic and d.ticket not in seen),
                     key=lambda d: d.time)
        sortis = {getattr(self.mt5, "DEAL_ENTRY_OUT", 1), getattr(self.mt5, "DEAL_ENTRY_INOUT", 2),
                  getattr(self.mt5, "DEAL_ENTRY_OUT_BY", 3)}
        pnls: list[float] = []
        for d in new:
            self.seen_deals.append(d.ticket)
            self._dirty = True
            if not self.baselined or d.entry not in sortis:
                continue
            pnl = float(d.profit) + float(getattr(d, "commission", 0) or 0) \
                + float(getattr(d, "swap", 0) or 0) + float(getattr(d, "fee", 0) or 0)
            pid = str(d.position_id)
            reason = {getattr(self.mt5, "DEAL_REASON_SL", 4): "sl",
                      getattr(self.mt5, "DEAL_REASON_TP", 5): "tp"}.get(
                          getattr(d, "reason", None), "time" if pid in self.forced_tickets else "other")
            meta = self.tickets.pop(pid, {})
            self.forced_tickets.discard(pid)
            side = "sell" if d.type == getattr(self.mt5, "DEAL_TYPE_BUY", 0) else "buy"
            self._log_trade(time=now.isoformat(timespec="seconds"), symbol=d.symbol,
                            strategy=meta.get("strategy", ""), side=meta.get("side", side),
                            volume=d.volume, price=d.price, ticket=pid, event="close",
                            pnl=round(pnl, 2), reason=reason)
            log.info("Fermeture %s %s pnl=%.2f (%s)", d.symbol, pid, pnl, reason)
            pnls.append(pnl)
        self.baselined = True
        return pnls

    def _forced_exits(self, now: datetime) -> None:
        for p in self._my_positions():
            tid = str(p.ticket)
            lim = float(self.tickets.get(tid, {}).get("max_minutes", self.cfg.default_max_minutes))
            srv_now = now.timestamp() + self._offset(p.symbol) * 3600
            if not forced_exit_due(p.time, srv_now, lim):
                continue
            if self.cfg.dry_run:
                log.info("DRY fermeture forcée %s ticket %s (> %g min)", p.symbol, tid, lim)
                continue
            self._check_account()
            try:
                self.broker.close_position(p.ticket, magic=self.cfg.magic)
                self.forced_tickets.add(tid)
                self._dirty = True
                log.info("Sortie forcée %s ticket %s (> %g min)", p.symbol, tid, lim)
            except TradingError as e:
                log.error("Sortie forcée impossible (ticket %s) : %s", tid, e)

    # ---------------------------------------------------------------- biais
    def _current_bias(self, now: datetime) -> dict:
        cfg_dir = Path(self.cfg.config_dir)
        cal = news.compute_bias(now, self._cached(str(cfg_dir / "events.json")) or {},
                                self._cached(str(cfg_dir / "bias_override.json")))
        fichier = news.read_bias(self.logs / "news_bias.json", now)
        return combine_bias(cal, fichier)

    # ---------------------------------------------------------------- cycle
    def cycle(self) -> None:
        info = self._check_account()
        now = self.now_utc()
        for s in self.cfg.symbols:
            self._offset(s)
        pnls = self._process_deals(now)
        self.breaker.update(now, float(info.equity), pnls)
        self.snap["account"] = {"login": info.login, "demo": True,
                                "equity": info.equity, "balance": info.balance}
        self.snap["bias"] = bias = self._current_bias(now)

        report = self._cached(self.cfg.report_path)
        self.active = {s: active_strategies(report, s, self.registry, self.cfg.force_strategies)
                       for s in self.cfg.symbols}
        self.forced_mode = bool(self.cfg.force_strategies)
        mode = ("forced" if self.forced_mode and any(self.active.values())
                else "active" if any(self.active.values()) else "observation")
        if mode != self.mode or not self._mode_logged:
            self._mode_logged = True
            if mode == "observation":
                log.warning("MODE OBSERVATION : aucune stratégie validée, AUCUN ordre ne sera envoyé")
            elif mode == "forced":
                log.warning("ATTENTION FORTE : stratégies FORCÉES en démo (%s) malgré le backtest",
                            ",".join(self.cfg.force_strategies))
            else:
                log.info("Mode actif : %s", {s: sorted(v) for s, v in self.active.items()})
        self.mode = mode

        if mode != "observation":
            self._forced_exits(now)
        for sym in self.cfg.symbols:
            self._process_symbol(sym, now, float(info.equity), bias)
        self.snap["open_positions"] = len(self._my_positions())
        self._save_state()
        self._dirty = False
        self._write_status()

    def _median_spread(self, sym: str, hour: int) -> float | None:
        prof = self._cached(self.cfg.profile_path)
        try:
            return float(prof[sym]["hourly_median"][str(hour)])
        except (TypeError, KeyError, ValueError):
            return None

    def _process_symbol(self, sym: str, now: datetime, equity: float, bias: dict) -> None:
        tick = self.mt5.symbol_info_tick(sym)
        if tick is not None:
            med = self._median_spread(sym, now.hour)
            self.snap["spreads"][sym] = {"spread": round(tick.ask - tick.bid, 6),
                                         "median": med,
                                         "limit": None if med is None else round(med * self.cfg.spread_mult, 6)}
        self.broker._symbol(sym)
        rates = self.mt5.copy_rates_from_pos(sym, self.mt5.TIMEFRAME_M1, 0, self.cfg.n_bars)
        if rates is None:
            raise ConnectionLost(f"copy_rates_from_pos a échoué : {self.mt5.last_error()}")
        if len(rates) == 0:
            return
        bar_time = int(rates[-1]["time"])
        if self.last_bar.get(sym) == bar_time:
            return
        self.last_bar[sym] = bar_time
        strats = self.active.get(sym) or {}
        if not strats:
            return
        if not self.breaker.can_trade():
            log.info("%s : entrées suspendues (%s)", sym, self.breaker.reason)
            return
        off = self._offset(sym)
        retard = now.timestamp() + off * 3600 - bar_time
        if retard > self.cfg.max_entry_delay_s:
            log.info("%s : barre en retard de %.0f s, pas d'entrée", sym, retard)
            return
        m1 = self._frame(rates, off)
        bar_utc = pd.Timestamp(bar_time - off * 3600, unit="s", tz="UTC")
        for name, params in strats.items():
            signaux = self.registry[name].signals(m1, params)
            for sig in signaux:
                t = pd.Timestamp(sig.time)
                t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
                if t == bar_utc:
                    self._try_entry(sym, sig, equity, bias)

    @staticmethod
    def _frame(rates: Any, offset_h: int) -> pd.DataFrame:
        df = pd.DataFrame(rates)
        if "spread" not in df:
            df["spread"] = 0
        df.index = pd.to_datetime(df["time"].astype("int64") - offset_h * 3600, unit="s", utc=True)
        df.index.name = "time"
        return df[["open", "high", "low", "close", "tick_volume", "spread"]]

    def _try_entry(self, sym: str, sig: Any, equity: float, bias: dict) -> bool:
        if not self.active.get(sym) or self.mode == "observation":
            return False
        tick = self.mt5.symbol_info_tick(sym)
        if tick is None:
            raise ConnectionLost(f"pas de tick pour {sym}")
        hour = pd.Timestamp(sig.time).hour
        ok, why = evaluate_entry(sig.side, has_position=bool(self._my_positions(sym)),
                                 spread=tick.ask - tick.bid,
                                 median_spread=self._median_spread(sym, hour),
                                 bias=bias, spread_mult=self.cfg.spread_mult)
        if not ok:
            log.info("%s %s %s refusé : %s", sym, sig.strategy, sig.side, why)
            return False
        spec = self.broker.symbol_spec(sym)
        info = self.mt5.symbol_info(sym)
        vol = position_size(equity, self.cfg.risk, sig.sl_dist, spec["contract_size"],
                            spec["volume_min"], spec["volume_step"], spec["volume_max"],
                            max_volume=self.settings.max_volume,
                            tick_value=getattr(info, "trade_tick_value", None),
                            tick_size=getattr(info, "trade_tick_size", None))
        if vol <= 0:
            log.info("%s : volume calculé nul (risque trop faible pour le stop), pas de trade", sym)
            return False
        mini = spec["stops_level_points"] * spec["point"]
        if sig.sl_dist < mini or sig.tp_dist < mini:
            log.info("%s : stops sous le niveau minimal du courtier (%g)", sym, mini)
            return False
        buy = sig.side == "buy"
        prix = tick.ask if buy else tick.bid
        sl = prix - sig.sl_dist if buy else prix + sig.sl_dist
        tp = prix + sig.tp_dist if buy else prix - sig.tp_dist
        comment = str(sig.strategy)[:31]
        now = self.now_utc().isoformat(timespec="seconds")
        if not self.breaker.can_trade():
            return False
        if self.cfg.dry_run:
            log.info("DRY %s %s %s vol=%s prix=%s sl=%s tp=%s", sym, sig.strategy, sig.side,
                     vol, prix, round(sl, spec["digits"]), round(tp, spec["digits"]))
            self._log_trade(time=now, symbol=sym, strategy=sig.strategy, side=sig.side,
                            volume=vol, price=prix, sl=sl, tp=tp, event="dry", reason="DRY")
            return False
        self._check_account()      # JAMAIS d'ordre sur un compte réel
        avant = {p.ticket for p in self._my_positions(sym)}
        res = self.broker.market_order(sym, sig.side, volume=vol, sl=sl, tp=tp,
                                       comment=comment, magic=self.cfg.magic)
        nouveaux = [p.ticket for p in self._my_positions(sym) if p.ticket not in avant]
        ticket = nouveaux[0] if nouveaux else res.get("order")
        self.tickets[str(ticket)] = {"max_minutes": int(sig.max_minutes), "strategy": sig.strategy,
                                     "side": sig.side, "symbol": sym}
        self._dirty = True
        self._log_trade(time=now, symbol=sym, strategy=sig.strategy, side=sig.side, volume=vol,
                        price=res.get("price") or prix, sl=sl, tp=tp, ticket=ticket, event="open")
        log.info("Ordre %s %s vol=%s prix=%s sl=%s tp=%s ticket=%s", sym, sig.side, vol,
                 prix, sl, tp, ticket)
        return True

    # ---------------------------------------------------------------- boucle
    def step(self) -> float:
        """Une itération avec reconnexion ; retourne le délai avant la suivante."""
        try:
            if not self.connected:
                self.connect()
            self.cycle()
            self.backoff = 2.0
            self.last_error = None
            return self.cfg.poll_s
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            if isinstance(e, (ConnectionLost, TradingError)):
                log.error("Erreur : %s", self.last_error)
            else:
                log.exception("Erreur inattendue")
            self._drop()
            wait = self.backoff
            self.backoff = min(60.0, self.backoff * 2)
            log.info("Reconnexion dans %g s", wait)
            try:
                self._write_status()
            except Exception:
                pass
            return wait

    def run(self) -> None:
        try:
            while not self._stop.is_set():
                wait = self.step()
                if self.cfg.once:
                    break
                self.sleep(wait)
        finally:
            self.close()

    def close(self) -> None:
        try:
            self._save_state()
            self._drop()
            self._write_status()
        except Exception:
            log.exception("Erreur à l'arrêt")
        log.info("Arrêt propre du bot")


# ====================================================================== CLI
def _settings_from_env() -> Settings:
    login = os.environ.get("MT5_LOGIN") or None
    return Settings(login=int(login) if login else None,
                    password=os.environ.get("MT5_PASSWORD") or None,
                    server=os.environ.get("MT5_SERVER") or None,
                    path=os.environ.get("MT5_PATH") or None,
                    max_volume=float(os.environ.get("MT5_MAX_VOLUME") or 1.0))


def _setup_logging(logs_dir: str) -> None:
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    root = logging.getLogger("ava_mt5")
    root.setLevel(logging.INFO)
    fh = RotatingFileHandler(Path(logs_dir) / "bot.log", maxBytes=2_000_000, backupCount=5,
                             encoding="utf-8")
    sh = logging.StreamHandler()
    for h in (fh, sh):
        h.setFormatter(fmt)
        root.addHandler(h)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="ava_mt5.bot", description="Bot de scalp pétrole (démo)")
    ap.add_argument("--symbols", default="CrudeOIL,BrentOIL")
    ap.add_argument("--report", default="logs/backtest_report.json")
    ap.add_argument("--profile", default="data/spread_profile.json")
    ap.add_argument("--meta", default="data/meta.json")
    ap.add_argument("--config", default="config", help="dossier events.json / bias_override.json")
    ap.add_argument("--risk", type=float, default=0.005, help="fraction d'équité risquée par trade")
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--dry-run", action="store_true", help="tout sauf order_send")
    ap.add_argument("--once", action="store_true", help="un seul cycle puis arrêt")
    ap.add_argument("--force-strategies", default="",
                    help="ex. orb,vwap_mr : trade en démo même sans validation du backtest")
    ap.add_argument("--reset-halt", action="store_true", help="lève l'arrêt complet et quitte")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None, mt5: Any = None) -> int:
    a = parse_args(argv)
    _setup_logging(a.logs)
    if a.reset_halt:
        sp = Path(a.logs) / "state.json"
        state = news.load_json(sp) or {}
        br = CircuitBreaker(state=state.get("breaker"))
        br.reset_halt()
        state["breaker"] = br.to_dict()
        _Writer.json(sp, state)
        log.info("Arrêt complet levé (state.json mis à jour)")
        return 0
    if mt5 is None:
        try:
            import MetaTrader5 as mt5  # import paresseux : Windows + terminal requis
        except ImportError:
            log.error("Module MetaTrader5 introuvable (Windows + terminal MT5 requis)")
            return 2
    cfg = BotConfig(symbols=[s.strip() for s in a.symbols.split(",") if s.strip()],
                    report_path=a.report, profile_path=a.profile, meta_path=a.meta,
                    config_dir=a.config, logs_dir=a.logs, risk=a.risk, dry_run=a.dry_run,
                    once=a.once,
                    force_strategies=[s.strip() for s in a.force_strategies.split(",") if s.strip()])
    bot = Bot(mt5, _settings_from_env(), cfg)
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, lambda *_: bot.stop())
        except ValueError:
            pass
    log.info("Démarrage du bot (symboles %s, risque %.3f%s)", cfg.symbols, cfg.risk,
             ", DRY-RUN" if cfg.dry_run else "")
    try:
        bot.run()
    except KeyboardInterrupt:
        bot.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
