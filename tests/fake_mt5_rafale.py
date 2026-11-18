"""Faux MT5 plus riche pour les tests de la stratégie rafale : mode de marge, positions avec
magic, exécution simulée des ordres au prix courant."""

from collections import namedtuple

from fake_mt5 import FakeMT5, Result

AccountR = namedtuple("AccountR", "login server name currency balance equity margin margin_free "
                                  "margin_level leverage profit trade_mode margin_mode")
PositionR = namedtuple("PositionR", "ticket symbol type volume price_open price_current sl tp "
                                    "profit swap time magic")


class FakeMT5Rafale(FakeMT5):
    ACCOUNT_MARGIN_MODE_RETAIL_NETTING = 0
    ACCOUNT_MARGIN_MODE_EXCHANGE = 1
    ACCOUNT_MARGIN_MODE_RETAIL_HEDGING = 2

    def __init__(self, trade_mode=0, margin_mode=2, filling_mode=2):
        super().__init__(trade_mode=trade_mode, filling_mode=filling_mode)
        self.margin_mode = margin_mode
        self.pos = []                       # PositionR
        self._tk = 5000
        self.bid, self.ask = 70.00, 70.03

    def set_tick(self, bid, ask):
        self.bid, self.ask = bid, ask

    def add_position(self, side, price, magic, volume=0.01, sl=0.0, tp=0.0):
        self._tk += 1
        typ = self.POSITION_TYPE_BUY if side == "buy" else self.POSITION_TYPE_SELL
        self.pos.append(PositionR(self._tk, "CrudeOIL", typ, volume, price, price, sl, tp, 0.0, 0.0,
                                  1700000000, magic))
        return self._tk

    def _profit(self, p):
        px = self.bid if p.type == self.POSITION_TYPE_BUY else self.ask
        sign = 1 if p.type == self.POSITION_TYPE_BUY else -1
        return sign * (px - p.price_open) * p.volume * 1000.0

    def account_info(self):
        return AccountR(123, "Demo", "T", "USD", self.balance, self.balance, 0.0, 1e4, 0.0, 100,
                        0.0, self.trade_mode, self.margin_mode)

    def positions_get(self, symbol=None, ticket=None):
        out = [p._replace(profit=self._profit(p)) for p in self.pos]
        if symbol:
            out = [p for p in out if p.symbol == symbol]
        if ticket:
            out = [p for p in out if p.ticket == ticket]
        return tuple(out)

    def order_send(self, request):
        self.sent.append(request)
        if "position" in request:
            self.pos = [p for p in self.pos if p.ticket != request["position"]]
            return Result(self.TRADE_RETCODE_DONE, "ok", request["position"], 1,
                          request["volume"], request["price"])
        side = "buy" if request["type"] == self.ORDER_TYPE_BUY else "sell"
        t = self.add_position(side, request["price"], request.get("magic", 0), request["volume"],
                              request.get("sl", 0.0), request.get("tp", 0.0))
        return Result(self.TRADE_RETCODE_DONE, "ok", t, 1, request["volume"], request["price"])
