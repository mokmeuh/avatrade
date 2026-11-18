"""Imitation minimale du module MetaTrader5 pour les tests."""

import calendar
import datetime as _dt
from collections import namedtuple

import numpy as np

from ava_mt5.core import MAGIC

Account = namedtuple("Account", "login server name currency balance equity margin margin_free "
                                "margin_level leverage profit trade_mode")
Symbol = namedtuple("Symbol", "name description path visible digits point trade_contract_size "
                              "volume_min volume_max volume_step trade_stops_level "
                              "currency_profit trade_mode filling_mode trade_tick_value trade_tick_size",
                defaults=(None, None))
Tick = namedtuple("Tick", "bid ask time")
Position = namedtuple("Position", "ticket symbol type volume price_open price_current sl tp "
                                  "profit swap time magic", defaults=(MAGIC,))
Deal = namedtuple("Deal", "ticket order position_id time type entry magic volume price profit "
                          "commission swap fee symbol reason comment", defaults=("",))
Terminal = namedtuple("Terminal", "connected trade_allowed")
Result = namedtuple("Result", "retcode comment order deal volume price")


class FakeMT5:
    ACCOUNT_TRADE_MODE_DEMO = 0
    ACCOUNT_TRADE_MODE_REAL = 2
    SYMBOL_TRADE_MODE_DISABLED = 0
    POSITION_TYPE_BUY = 0
    POSITION_TYPE_SELL = 1
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    TRADE_ACTION_DEAL = 1
    TRADE_ACTION_SLTP = 6
    ORDER_TIME_GTC = 0
    ORDER_FILLING_FOK = 0
    ORDER_FILLING_IOC = 1
    ORDER_FILLING_RETURN = 2
    TRADE_RETCODE_DONE = 10009
    TRADE_RETCODE_PLACED = 10008
    TIMEFRAME_M1 = 1
    TIMEFRAME_H1 = 16385
    DEAL_ENTRY_IN = 0
    DEAL_ENTRY_OUT = 1
    DEAL_ENTRY_INOUT = 2
    DEAL_REASON_SL = 4
    DEAL_REASON_TP = 5

    def __init__(self, trade_mode=0, filling_mode=2):
        self.trade_mode = trade_mode
        self.sent = []
        self.init_kwargs = None
        self.shut = False
        self.init_count = 0
        # --- réglages dynamiques pour les tests du bot
        self.bid = 71.00
        self.ask = 71.03
        self.tick_time = None          # None -> 1700000000
        self.equity = 10100.0
        self.balance = 10000.0
        self.terminal_ok = True
        self.bars = None               # tableau numpy structuré (voir make_bars)
        self.deals = []
        self.simulate_fills = False    # True : order_send modifie positions / deals
        self.fail_next = {}            # nom de fonction -> exception levée une fois
        self._ticket = 1000
        self.symbols = {
            "CrudeOIL": Symbol("CrudeOIL", "Crude Oil WTI", "Commodities\\CrudeOIL", False, 2,
                               0.01, 1000.0, 0.01, 100.0, 0.01, 10, "USD", 4, filling_mode,
                               10.0, 0.01),
            "EURUSD": Symbol("EURUSD", "Euro vs US Dollar", "Forex\\EURUSD", True, 5, 0.00001,
                             100000.0, 0.01, 100.0, 0.01, 0, "USD", 4, filling_mode,
                             1.0, 0.00001),
        }
        self.positions = [Position(555, "CrudeOIL", 0, 0.10, 70.00, 71.00, 0.0, 0.0, 100.0, 0.0, 0)]

    def _maybe_fail(self, name):
        exc = self.fail_next.pop(name, None)
        if exc is not None:
            raise exc

    def initialize(self, **kwargs):
        self.init_kwargs = kwargs
        self.init_count += 1
        self.shut = False
        return True

    def terminal_info(self):
        return Terminal(True, True) if self.terminal_ok else None

    def make_bars(self, n, last_time, start_price=70.0, spread=30):
        """Génère n barres M1 réalistes (marche aléatoire déterministe) finissant à last_time."""
        rng = np.random.default_rng(7)
        dtype = [("time", "i8"), ("open", "f8"), ("high", "f8"), ("low", "f8"), ("close", "f8"),
                 ("tick_volume", "i8"), ("spread", "i4"), ("real_volume", "i8")]
        arr = np.zeros(n, dtype=dtype)
        price = start_price
        for i in range(n):
            o = price
            c = round(o + rng.normal(0, 0.03), 2)
            h = round(max(o, c) + abs(rng.normal(0, 0.01)), 2)
            lo = round(min(o, c) - abs(rng.normal(0, 0.01)), 2)
            arr[i] = (last_time - 60 * (n - 1 - i), o, h, lo, c, int(rng.integers(10, 200)), spread, 0)
            price = c
        self.bars = arr
        return arr

    def shutdown(self):
        self.shut = True

    def last_error(self):
        return (1, "fake")

    def account_info(self):
        self._maybe_fail("account_info")
        return Account(123, "AvaTrade-Demo", "Test", "USD", self.balance, self.equity, 0.0,
                       10000.0, 0.0, 100, 100.0, self.trade_mode)

    def symbols_get(self):
        return list(self.symbols.values())

    def symbol_info(self, name):
        return self.symbols.get(name)

    def symbol_select(self, name, enable):
        self.symbols[name] = self.symbols[name]._replace(visible=enable)
        return True

    def symbol_info_tick(self, name):
        self._maybe_fail("symbol_info_tick")
        t = 1700000000 if self.tick_time is None else self.tick_time
        return Tick(self.bid, self.ask, t)

    def positions_get(self, symbol=None, ticket=None):
        self._maybe_fail("positions_get")
        out = self.positions
        if symbol:
            out = [p for p in out if p.symbol == symbol]
        if ticket:
            out = [p for p in out if p.ticket == ticket]
        return tuple(out)

    def copy_rates_from_pos(self, symbol, tf, start, count):
        self._maybe_fail("copy_rates_from_pos")
        if self.bars is not None:
            end = len(self.bars) - start
            return self.bars[max(0, end - count):end]
        return [{"time": 1, "open": 70.0, "high": 71.0, "low": 69.5, "close": 70.5,
                 "tick_volume": 42}] * count

    @staticmethod
    def _to_epoch(x):
        if isinstance(x, _dt.datetime):
            return calendar.timegm(x.timetuple())
        return int(x)

    def history_deals_get(self, date_from=None, date_to=None, **kwargs):
        self._maybe_fail("history_deals_get")
        lo, hi = self._to_epoch(date_from), self._to_epoch(date_to)
        return tuple(d for d in self.deals if lo <= d.time <= hi)

    def order_send(self, request):
        self._maybe_fail("order_send")
        self.sent.append(request)
        order, deal = 1, 2
        if self.simulate_fills:
            self._ticket += 1
            order, deal = self._ticket, self._ticket + 5000
            now = 1700000000 if self.tick_time is None else self.tick_time
            sym = self.symbols[request["symbol"]]
            if "position" in request:
                pos = next(p for p in self.positions if p.ticket == request["position"])
                sign = 1 if pos.type == self.POSITION_TYPE_BUY else -1
                profit = sign * (request["price"] - pos.price_open) * pos.volume \
                    * sym.trade_contract_size
                self.positions = [p for p in self.positions if p is not pos]
                reason = request.get("fake_reason", 0)
                self.deals.append(Deal(deal, order, pos.ticket, now, request["type"],
                                       self.DEAL_ENTRY_OUT, request.get("magic", 0),
                                       pos.volume, request["price"], profit, 0.0, 0.0, 0.0,
                                       pos.symbol, reason))
            else:
                ptype = (self.POSITION_TYPE_BUY if request["type"] == self.ORDER_TYPE_BUY
                         else self.POSITION_TYPE_SELL)
                self.positions.append(Position(order, request["symbol"], ptype,
                                               request["volume"], request["price"],
                                               request["price"], request.get("sl", 0.0),
                                               request.get("tp", 0.0), 0.0, 0.0, now,
                                               request.get("magic", 0)))
        return Result(self.TRADE_RETCODE_DONE, "Request executed", order, deal,
                      request.get("volume"), request.get("price"))
