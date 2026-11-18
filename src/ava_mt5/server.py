"""Serveur MCP (stdio) exposant le terminal MetaTrader 5 AvaTrade."""

from __future__ import annotations

import functools
import os
import sys
from typing import Any, Callable, TypeVar

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from ava_mt5.core import Broker, Settings, TradingError

INSTRUCTIONS = """\
Accès à un terminal MetaTrader 5 AvaTrade, limité aux comptes démo.
Démarche conseillée pour le pétrole : search_symbols("oil") pour trouver le symbole
(souvent CrudeOIL ou BrentOIL), symbol_spec pour connaître la taille du contrat
en barils, quote pour le prix, puis place_market_order avec un stop loss.
"""

mcp = MCPServer("ava-mt5", instructions=INSTRUCTIONS)
_broker: Broker | None = None


def _settings_from_env() -> Settings:
    login = os.environ.get("MT5_LOGIN") or None
    return Settings(
        login=int(login) if login else None,
        password=os.environ.get("MT5_PASSWORD") or None,
        server=os.environ.get("MT5_SERVER") or None,
        path=os.environ.get("MT5_PATH") or None,
        max_volume=float(os.environ.get("MT5_MAX_VOLUME") or 1.0),
    )


def broker() -> Broker:
    global _broker
    if _broker is None:
        try:
            import MetaTrader5 as mt5  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ToolError(
                "Le paquet MetaTrader5 est absent : il ne s'installe que sous Windows."
            ) from exc
        _broker = Broker(mt5, _settings_from_env())
    return _broker


F = TypeVar("F", bound=Callable[..., Any])


def guarded(fn: F) -> F:
    """Transmet au client le message des erreurs de trading au lieu d'un message générique."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except TradingError as exc:
            raise ToolError(str(exc)) from exc

    return wrapper  # type: ignore[return-value]


@mcp.tool()
@guarded
def account_info() -> dict[str, Any]:
    """Connecte le terminal si besoin et renvoie solde, équité, marge et devise du compte démo."""
    return broker().connect()


@mcp.tool()
@guarded
def search_symbols(text: str, limit: int = 30) -> list[dict[str, Any]]:
    """Cherche des symboles par nom ou description (ex. "oil", "brent", "crude")."""
    return broker().search_symbols(text, limit)


@mcp.tool()
@guarded
def symbol_spec(symbol: str) -> dict[str, Any]:
    """Spécification du contrat : taille en barils par lot, volume min/max/pas, décimales."""
    return broker().symbol_spec(symbol)


@mcp.tool()
@guarded
def get_quote(symbol: str) -> dict[str, Any]:
    """Prix bid/ask courant et écart."""
    return broker().quote(symbol)


@mcp.tool()
@guarded
def get_rates(symbol: str, timeframe: str = "H1", count: int = 100) -> list[dict[str, Any]]:
    """Bougies OHLC récentes. timeframe : M1, M5, M15, M30, H1, H4, D1, W1, MN1 (max 1000)."""
    return broker().rates(symbol, timeframe, count)


@mcp.tool()
@guarded
def get_positions(symbol: str | None = None) -> list[dict[str, Any]]:
    """Positions ouvertes, toutes ou pour un symbole."""
    return broker().positions(symbol)


@mcp.tool()
@guarded
def place_market_order(
    symbol: str,
    side: str,
    volume: float | None = None,
    barrels: float | None = None,
    sl: float | None = None,
    tp: float | None = None,
    comment: str = "",
) -> dict[str, Any]:
    """Ordre au marché. side = "buy" ou "sell". Donner volume (lots) OU barrels
    (converti selon la taille du contrat). sl/tp sont des prix absolus."""
    return broker().market_order(symbol, side, volume, barrels, sl, tp, comment)


@mcp.tool()
@guarded
def close_position(ticket: int, volume: float | None = None) -> dict[str, Any]:
    """Ferme une position, en entier ou partiellement (volume en lots)."""
    return broker().close_position(ticket, volume)


@mcp.tool()
@guarded
def modify_position(ticket: int, sl: float | None = None, tp: float | None = None) -> dict[str, Any]:
    """Modifie le stop loss et/ou le take profit d'une position ouverte."""
    return broker().modify_sltp(ticket, sl, tp)


def main() -> None:
    if sys.platform != "win32":
        print("Avertissement : MetaTrader5 ne fonctionne que sous Windows.", file=sys.stderr)
    mcp.run("stdio")


if __name__ == "__main__":
    main()
