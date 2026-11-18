import pytest

from ava_mt5.core import Broker, Settings, TradingError
from fake_mt5 import FakeMT5


def make(max_volume=1.0, **kw):
    mt5 = FakeMT5(**kw)
    return mt5, Broker(mt5, Settings(max_volume=max_volume))


def test_refuse_compte_reel():
    mt5, b = make(trade_mode=FakeMT5.ACCOUNT_TRADE_MODE_REAL)
    with pytest.raises(TradingError, match="démo"):
        b.connect()
    assert mt5.shut


def test_connexion_demo_identifiants():
    mt5 = FakeMT5()
    b = Broker(mt5, Settings(login=42, password="x", server="AvaTrade-Demo"))
    acc = b.connect()
    assert acc["demo"] is True
    assert mt5.init_kwargs == {"login": 42, "password": "x", "server": "AvaTrade-Demo"}


def test_recherche_petrole():
    _, b = make()
    names = [s["name"] for s in b.search_symbols("oil")]
    assert names == ["CrudeOIL"]


def test_conversion_barils():
    _, b = make()
    b.connect()
    assert b.barrels_to_volume("CrudeOIL", 100) == 0.1
    assert b.barrels_to_volume("CrudeOIL", 155) == 0.15
    with pytest.raises(TradingError, match="minimum"):
        b.barrels_to_volume("CrudeOIL", 5)


def test_achat_en_barils_avec_stop():
    mt5, b = make()
    res = b.market_order("CrudeOIL", "buy", barrels=100, sl=70.0, tp=73.0)
    req = mt5.sent[-1]
    assert req["type"] == mt5.ORDER_TYPE_BUY
    assert req["price"] == 71.03
    assert req["volume"] == 0.1
    assert req["sl"] == 70.0 and req["tp"] == 73.0
    assert req["type_filling"] == mt5.ORDER_FILLING_IOC
    assert res["ok"] and res["barrels"] == 100.0


def test_vente_utilise_bid_et_fok():
    mt5, b = make(filling_mode=1)
    b.market_order("CrudeOIL", "sell", volume=0.05)
    req = mt5.sent[-1]
    assert req["price"] == 71.00
    assert req["type_filling"] == mt5.ORDER_FILLING_FOK


def test_limite_volume():
    _, b = make(max_volume=0.5)
    with pytest.raises(TradingError, match="MT5_MAX_VOLUME"):
        b.market_order("CrudeOIL", "buy", volume=1.0)


def test_stop_du_mauvais_cote():
    _, b = make()
    with pytest.raises(TradingError, match="Stop loss"):
        b.market_order("CrudeOIL", "buy", volume=0.1, sl=72.0)


def test_volume_et_barils_exclusifs():
    _, b = make()
    with pytest.raises(TradingError):
        b.market_order("CrudeOIL", "buy", volume=0.1, barrels=100)


def test_fermeture_position_achat():
    mt5, b = make()
    b.close_position(555)
    req = mt5.sent[-1]
    assert req["position"] == 555
    assert req["type"] == mt5.ORDER_TYPE_SELL
    assert req["price"] == 71.00
    assert req["volume"] == 0.10


def test_modification_sltp():
    mt5, b = make()
    b.modify_sltp(555, sl=69.5)
    req = mt5.sent[-1]
    assert req["action"] == mt5.TRADE_ACTION_SLTP
    assert req["sl"] == 69.5 and req["tp"] == 0.0


def test_ordre_rejete():
    mt5, b = make()
    mt5.order_send = lambda r: __import__("fake_mt5").Result(10019, "No money", 0, 0, 0, 0)
    with pytest.raises(TradingError, match="10019"):
        b.market_order("CrudeOIL", "buy", volume=0.1)
