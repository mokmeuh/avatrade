"""Logique de trading indépendante du transport MCP.

Le module MetaTrader5 est injecté, ce qui permet de tester sans terminal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

MAGIC = 20261118
DEVIATION_POINTS = 20

# Bits de SYMBOL_FILLING_MODE
_FILLING_FOK = 1
_FILLING_IOC = 2

_TIMEFRAMES = ("M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1")


class TradingError(RuntimeError):
    pass


@dataclass
class Settings:
    login: int | None = None
    password: str | None = None
    server: str | None = None
    path: str | None = None
    max_volume: float = 1.0


def _as_dict(obj: Any) -> dict[str, Any]:
    return obj._asdict() if hasattr(obj, "_asdict") else dict(obj)


class Broker:
    def __init__(self, mt5: Any, settings: Settings) -> None:
        self.mt5 = mt5
        self.settings = settings
        self.connected = False

    # ------------------------------------------------------------ connexion
    def connect(self) -> dict[str, Any]:
        if self.connected:
            return self.account()
        s = self.settings
        kwargs: dict[str, Any] = {}
        if s.path:
            kwargs["path"] = s.path
        if s.login:
            kwargs.update(login=int(s.login), password=s.password or "", server=s.server or "")
        if not self.mt5.initialize(**kwargs):
            raise TradingError(f"Échec d'initialisation MT5 : {self.mt5.last_error()}")
        info = self.mt5.account_info()
        if info is None:
            self.mt5.shutdown()
            raise TradingError("Aucun compte connecté dans le terminal MT5.")
        if info.trade_mode != self.mt5.ACCOUNT_TRADE_MODE_DEMO:
            self.mt5.shutdown()
            raise TradingError(
                f"Le compte {info.login} n'est pas un compte démo : connexion refusée."
            )
        self.connected = True
        return self.account()

    def _ensure(self) -> None:
        if not self.connected:
            self.connect()

    def shutdown(self) -> None:
        if self.connected:
            self.mt5.shutdown()
            self.connected = False

    # ------------------------------------------------------------ lecture
    def account(self) -> dict[str, Any]:
        info = self.mt5.account_info()
        if info is None:
            raise TradingError(f"account_info indisponible : {self.mt5.last_error()}")
        d = _as_dict(info)
        keys = ("login", "server", "name", "currency", "balance", "equity",
                "margin", "margin_free", "margin_level", "leverage", "profit")
        out = {k: d.get(k) for k in keys}
        out["demo"] = info.trade_mode == self.mt5.ACCOUNT_TRADE_MODE_DEMO
        return out

    def search_symbols(self, text: str, limit: int = 30) -> list[dict[str, Any]]:
        self._ensure()
        needle = text.lower()
        found = []
        for s in self.mt5.symbols_get() or ():
            desc = getattr(s, "description", "") or ""
            if needle in s.name.lower() or needle in desc.lower():
                found.append({"name": s.name, "description": desc,
                              "path": getattr(s, "path", ""), "visible": s.visible})
                if len(found) >= limit:
                    break
        return found

    def _symbol(self, symbol: str) -> Any:
        info = self.mt5.symbol_info(symbol)
        if info is None:
            raise TradingError(f"Symbole inconnu : {symbol}")
        if not info.visible:
            if not self.mt5.symbol_select(symbol, True):
                raise TradingError(f"Impossible d'activer {symbol} dans le Market Watch.")
            info = self.mt5.symbol_info(symbol)
        return info

    def symbol_spec(self, symbol: str) -> dict[str, Any]:
        self._ensure()
        i = self._symbol(symbol)
        return {
            "name": i.name,
            "description": getattr(i, "description", ""),
            "digits": i.digits,
            "point": i.point,
            "contract_size": i.trade_contract_size,
            "volume_min": i.volume_min,
            "volume_max": i.volume_max,
            "volume_step": i.volume_step,
            "stops_level_points": getattr(i, "trade_stops_level", 0),
            "currency_profit": getattr(i, "currency_profit", ""),
            "trade_allowed": i.trade_mode != self.mt5.SYMBOL_TRADE_MODE_DISABLED,
        }

    def quote(self, symbol: str) -> dict[str, Any]:
        self._ensure()
        info = self._symbol(symbol)
        tick = self.mt5.symbol_info_tick(symbol)
        if tick is None:
            raise TradingError(f"Pas de cotation pour {symbol} (marché fermé ?).")
        return {"symbol": symbol, "bid": tick.bid, "ask": tick.ask,
                "spread": round(tick.ask - tick.bid, info.digits), "time": tick.time}

    def rates(self, symbol: str, timeframe: str = "H1", count: int = 100) -> list[dict[str, Any]]:
        self._ensure()
        tf = timeframe.upper()
        if tf not in _TIMEFRAMES:
            raise TradingError(f"Unité de temps invalide : {timeframe}. Choix : {', '.join(_TIMEFRAMES)}")
        self._symbol(symbol)
        count = max(1, min(int(count), 1000))
        data = self.mt5.copy_rates_from_pos(symbol, getattr(self.mt5, f"TIMEFRAME_{tf}"), 0, count)
        if data is None:
            raise TradingError(f"Historique indisponible : {self.mt5.last_error()}")
        cols = ("time", "open", "high", "low", "close", "tick_volume")
        return [{c: (int(r[c]) if c in ("time", "tick_volume") else float(r[c])) for c in cols}
                for r in data]

    def positions(self, symbol: str | None = None) -> list[dict[str, Any]]:
        self._ensure()
        raw = self.mt5.positions_get(symbol=symbol) if symbol else self.mt5.positions_get()
        out = []
        for p in raw or ():
            out.append({
                "ticket": p.ticket, "symbol": p.symbol,
                "side": "buy" if p.type == self.mt5.POSITION_TYPE_BUY else "sell",
                "volume": p.volume, "price_open": p.price_open,
                "price_current": p.price_current, "sl": p.sl, "tp": p.tp,
                "profit": p.profit, "swap": p.swap, "time": p.time,
            })
        return out

    # ------------------------------------------------------------ calculs
    @staticmethod
    def _round_step(value: float, step: float) -> float:
        n = math.floor(value / step + 1e-9)
        decimals = max(0, -int(math.floor(math.log10(step)))) if step < 1 else 0
        return round(n * step, decimals)

    def barrels_to_volume(self, symbol: str, barrels: float) -> float:
        info = self._symbol(symbol)
        if barrels <= 0:
            raise TradingError("Le nombre de barils doit être positif.")
        vol = self._round_step(barrels / info.trade_contract_size, info.volume_step)
        if vol < info.volume_min:
            raise TradingError(
                f"{barrels} barils = {barrels / info.trade_contract_size:g} lot, "
                f"sous le minimum de {info.volume_min} lot "
                f"({info.volume_min * info.trade_contract_size:g} barils)."
            )
        return vol

    def _filling(self, info: Any) -> int:
        mode = getattr(info, "filling_mode", 0)
        if mode & _FILLING_FOK:
            return self.mt5.ORDER_FILLING_FOK
        if mode & _FILLING_IOC:
            return self.mt5.ORDER_FILLING_IOC
        return self.mt5.ORDER_FILLING_RETURN

    def _check_volume(self, info: Any, volume: float) -> float:
        if volume <= 0:
            raise TradingError("Le volume doit être positif.")
        if volume > self.settings.max_volume:
            raise TradingError(
                f"Volume {volume} supérieur à la limite MT5_MAX_VOLUME={self.settings.max_volume}."
            )
        if volume < info.volume_min or volume > info.volume_max:
            raise TradingError(f"Volume hors bornes [{info.volume_min}, {info.volume_max}].")
        rounded = self._round_step(volume, info.volume_step)
        if abs(rounded - volume) > 1e-9:
            raise TradingError(f"Le volume doit être un multiple de {info.volume_step}.")
        return rounded

    def _send(self, request: dict[str, Any]) -> dict[str, Any]:
        result = self.mt5.order_send(request)
        if result is None:
            raise TradingError(f"order_send a échoué : {self.mt5.last_error()}")
        ok = result.retcode in (self.mt5.TRADE_RETCODE_DONE, self.mt5.TRADE_RETCODE_PLACED)
        d = _as_dict(result)
        out = {"ok": ok, "retcode": result.retcode, "comment": d.get("comment"),
               "order": d.get("order"), "deal": d.get("deal"),
               "volume": d.get("volume"), "price": d.get("price")}
        if not ok:
            raise TradingError(f"Ordre rejeté (retcode {result.retcode}) : {d.get('comment')}")
        return out

    # ------------------------------------------------------------ ordres
    def market_order(self, symbol: str, side: str, volume: float | None = None,
                     barrels: float | None = None, sl: float | None = None,
                     tp: float | None = None, comment: str = "") -> dict[str, Any]:
        self._ensure()
        side = side.lower()
        if side not in ("buy", "sell"):
            raise TradingError("side doit valoir 'buy' ou 'sell'.")
        if (volume is None) == (barrels is None):
            raise TradingError("Indiquer soit volume (lots), soit barrels, pas les deux.")
        info = self._symbol(symbol)
        if barrels is not None:
            volume = self.barrels_to_volume(symbol, barrels)
        volume = self._check_volume(info, float(volume))
        tick = self.mt5.symbol_info_tick(symbol)
        if tick is None:
            raise TradingError(f"Pas de cotation pour {symbol} (marché fermé ?).")
        price = tick.ask if side == "buy" else tick.bid
        if sl is not None and ((side == "buy" and sl >= price) or (side == "sell" and sl <= price)):
            raise TradingError(f"Stop loss {sl} du mauvais côté du prix {price}.")
        if tp is not None and ((side == "buy" and tp <= price) or (side == "sell" and tp >= price)):
            raise TradingError(f"Take profit {tp} du mauvais côté du prix {price}.")
        request = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": self.mt5.ORDER_TYPE_BUY if side == "buy" else self.mt5.ORDER_TYPE_SELL,
            "price": price,
            "deviation": DEVIATION_POINTS,
            "magic": MAGIC,
            "comment": (comment or "ava-mt5")[:31],
            "type_time": self.mt5.ORDER_TIME_GTC,
            "type_filling": self._filling(info),
        }
        if sl is not None:
            request["sl"] = round(float(sl), info.digits)
        if tp is not None:
            request["tp"] = round(float(tp), info.digits)
        out = self._send(request)
        out["barrels"] = round(volume * info.trade_contract_size, 6)
        return out

    def close_position(self, ticket: int, volume: float | None = None) -> dict[str, Any]:
        self._ensure()
        found = self.mt5.positions_get(ticket=int(ticket))
        if not found:
            raise TradingError(f"Position {ticket} introuvable.")
        p = found[0]
        info = self._symbol(p.symbol)
        vol = p.volume if volume is None else self._round_step(float(volume), info.volume_step)
        if vol <= 0 or vol > p.volume:
            raise TradingError(f"Volume de fermeture invalide (position : {p.volume}).")
        tick = self.mt5.symbol_info_tick(p.symbol)
        if tick is None:
            raise TradingError(f"Pas de cotation pour {p.symbol}.")
        is_buy = p.type == self.mt5.POSITION_TYPE_BUY
        request = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "symbol": p.symbol,
            "position": p.ticket,
            "volume": vol,
            "type": self.mt5.ORDER_TYPE_SELL if is_buy else self.mt5.ORDER_TYPE_BUY,
            "price": tick.bid if is_buy else tick.ask,
            "deviation": DEVIATION_POINTS,
            "magic": MAGIC,
            "comment": "ava-mt5 close",
            "type_time": self.mt5.ORDER_TIME_GTC,
            "type_filling": self._filling(info),
        }
        return self._send(request)

    def modify_sltp(self, ticket: int, sl: float | None = None,
                    tp: float | None = None) -> dict[str, Any]:
        self._ensure()
        found = self.mt5.positions_get(ticket=int(ticket))
        if not found:
            raise TradingError(f"Position {ticket} introuvable.")
        p = found[0]
        info = self._symbol(p.symbol)
        request = {
            "action": self.mt5.TRADE_ACTION_SLTP,
            "symbol": p.symbol,
            "position": p.ticket,
            "sl": round(float(sl), info.digits) if sl is not None else p.sl,
            "tp": round(float(tp), info.digits) if tp is not None else p.tp,
            "magic": MAGIC,
        }
        return self._send(request)
