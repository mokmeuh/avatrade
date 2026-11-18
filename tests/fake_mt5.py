"""Imitation minimale du module MetaTrader5 pour les tests."""

from collections import namedtuple

Account = namedtuple("Account", "login server name currency balance equity margin margin_free "
                                "margin_level leverage profit trade_mode")
Symbol = namedtuple("Symbol", "name description path visible digits point trade_contract_size "
                              "volume_min volume_max volume_step trade_stops_level "
                              "currency_profit trade_mode filling_mode")
Tick = namedtuple("Tick", "bid ask time")
Position = namedtuple("Position", "ticket symbol type volume price_open price_current sl tp "
                                  "profit swap time")
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
    TIMEFRAME_H1 = 16385

    def __init__(self, trade_mode=0, filling_mode=2):
        self.trade_mode = trade_mode
        self.sent = []
        self.init_kwargs = None
        self.shut = False
        self.symbols = {
            "CrudeOIL": Symbol("CrudeOIL", "Crude Oil WTI", "Commodities\\CrudeOIL", False, 2,
                               0.01, 1000.0, 0.01, 100.0, 0.01, 10, "USD", 4, filling_mode),
            "EURUSD": Symbol("EURUSD", "Euro vs US Dollar", "Forex\\EURUSD", True, 5, 0.00001,
                             100000.0, 0.01, 100.0, 0.01, 0, "USD", 4, filling_mode),
        }
        self.positions = [Position(555, "CrudeOIL", 0, 0.10, 70.00, 71.00, 0.0, 0.0, 100.0, 0.0, 0)]

    def initialize(self, **kwargs):
        self.init_kwargs = kwargs
        return True

    def shutdown(self):
        self.shut = True

    def last_error(self):
        return (1, "fake")

    def account_info(self):
        return Account(123, "AvaTrade-Demo", "Test", "USD", 10000.0, 10100.0, 0.0, 10000.0,
                       0.0, 100, 100.0, self.trade_mode)

    def symbols_get(self):
        return list(self.symbols.values())

    def symbol_info(self, name):
        return self.symbols.get(name)

    def symbol_select(self, name, enable):
        self.symbols[name] = self.symbols[name]._replace(visible=enable)
        return True

    def symbol_info_tick(self, name):
        return Tick(71.00, 71.03, 1700000000)

    def positions_get(self, symbol=None, ticket=None):
        out = self.positions
        if symbol:
            out = [p for p in out if p.symbol == symbol]
        if ticket:
            out = [p for p in out if p.ticket == ticket]
        return tuple(out)

    def copy_rates_from_pos(self, symbol, tf, start, count):
        return [{"time": 1, "open": 70.0, "high": 71.0, "low": 69.5, "close": 70.5,
                 "tick_volume": 42}] * count

    def order_send(self, request):
        self.sent.append(request)
        return Result(self.TRADE_RETCODE_DONE, "Request executed", 1, 2,
                      request.get("volume"), request.get("price"))
