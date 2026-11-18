import csv
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from ava_mt5.bot import (MAGIC_SCALP, Bot, BotConfig, CircuitBreaker, evaluate_entry,
                         position_size)
from ava_mt5.core import MAGIC, Settings
from fake_mt5 import FakeMT5, Position

UTC = timezone.utc
# lundi 16 novembre 2026 12:00 UTC (07:00 NY) : aucune fenêtre de news
T0 = int(datetime(2026, 11, 16, 12, 0, tzinfo=UTC).timestamp())


# ------------------------------------------------------------ position_size
def test_taille_nominale():
    # 10000 x 0,5 % = 50 ; stop 0,5 x contrat 1000 = 500 par lot -> 0,10
    assert position_size(10000, 0.005, 0.5, 1000, 0.01, 0.01, 100) == 0.10


def test_taille_arrondie_a_la_baisse():
    # 50 / (0,3 x 1000) = 0,1666 -> 0,16
    assert position_size(10000, 0.005, 0.3, 1000, 0.01, 0.01, 100) == 0.16


def test_taille_sous_minimum():
    assert position_size(1000, 0.005, 1.0, 1000, 0.01, 0.01, 100) == 0.0


def test_taille_plafonnee():
    assert position_size(1_000_000, 0.01, 0.1, 1000, 0.01, 0.01, 50) == 50
    assert position_size(1_000_000, 0.01, 0.1, 1000, 0.01, 0.01, 50, max_volume=1.0) == 1.0


def test_taille_tick_value():
    # tick_value 5 par 0,01 : perte par lot = 0,5/0,01 x 5 = 250 -> 0,20
    assert position_size(10000, 0.005, 0.5, 1000, 0.01, 0.01, 100,
                         tick_value=5.0, tick_size=0.01) == 0.20


# ------------------------------------------------------------ CircuitBreaker
def dt(jour, h=12):
    return datetime(2026, 11, jour, h, 0, tzinfo=UTC)


def test_arret_jour_2pct_et_reprise():
    cb = CircuitBreaker()
    cb.update(dt(16), 10000)
    cb.update(dt(16, 13), 9850)
    assert cb.can_trade()
    cb.update(dt(16, 14), 9800)          # -2 %
    assert cb.halted_today and not cb.can_trade()
    cb.update(dt(16, 20), 9800)          # 15:00 NY : même jour de trading
    assert cb.halted_today
    cb.update(dt(17, 12), 9800)          # jour suivant
    assert not cb.halted_today and cb.can_trade()
    assert cb.day_start_equity == 9800


def test_rollover_17h_new_york():
    cb = CircuitBreaker()
    cb.update(datetime(2026, 11, 16, 21, 30, tzinfo=UTC), 10000)   # 16:30 NY
    j1 = cb.day
    cb.update(datetime(2026, 11, 16, 22, 30, tzinfo=UTC), 10000)   # 17:30 NY
    assert cb.day != j1


def test_cinq_pertes_et_gain_remet_a_zero():
    cb = CircuitBreaker()
    cb.update(dt(16), 10000)
    cb.update(dt(16), 10000, [-1, -1, -1, -1])
    assert cb.can_trade() and cb.consecutive_losses == 4
    cb.update(dt(16), 10000, [5.0])
    assert cb.consecutive_losses == 0
    cb.update(dt(16), 10000, [-1] * 5)
    assert cb.halted_today and "consécutives" in cb.reason
    cb.update(dt(17), 10000)
    assert cb.can_trade()


def test_drawdown_8pct_arret_complet_persistant():
    cb = CircuitBreaker()
    cb.update(dt(16), 10000)
    cb.update(dt(17), 11000)                       # nouveau pic
    cb.update(dt(18), 10200)                       # -7,3 %
    assert not cb.halted_full
    cb.update(dt(19), 10120)                       # -8 %
    assert cb.halted_full
    cb2 = CircuitBreaker(state=cb.to_dict())       # redémarrage
    cb2.update(dt(25), 11000)
    assert cb2.halted_full and not cb2.can_trade()
    cb2.reset_halt()
    cb2.update(dt(25, 13), 11000)
    assert cb2.can_trade()


# ------------------------------------------------------------ evaluate_entry
BN = {"bias": "neutral"}


def test_filtre_spread():
    assert evaluate_entry("buy", has_position=False, spread=0.045, median_spread=0.03, bias=BN)[0]
    ok, why = evaluate_entry("buy", has_position=False, spread=0.046, median_spread=0.03, bias=BN)
    assert not ok and "spread" in why
    assert not evaluate_entry("buy", has_position=False, spread=0.01, median_spread=None, bias=BN)[0]


def test_une_position_et_biais():
    assert not evaluate_entry("buy", has_position=True, spread=0.03, median_spread=0.03, bias=BN)[0]
    assert not evaluate_entry("buy", has_position=False, spread=0.03, median_spread=0.03,
                              bias={"bias": "short"})[0]
    assert evaluate_entry("sell", has_position=False, spread=0.03, median_spread=0.03,
                          bias={"bias": "short"})[0]


# ------------------------------------------------------------ banc de test
@dataclass(frozen=True)
class Sig:
    time: pd.Timestamp
    side: str
    sl_dist: float
    tp_dist: float
    max_minutes: int
    strategy: str


class FakeStrat:
    name = "fake"
    default_params = {"x": 1}

    def __init__(self, side="buy"):
        self.side = side
        self.calls = 0

    def signals(self, m1, params):
        self.calls += 1
        assert list(m1.columns) == ["open", "high", "low", "close", "tick_volume", "spread"]
        assert m1.index.tz is not None
        return [Sig(m1.index[-1], self.side, 0.5, 1.0, 15, "fake")]


class Env:
    def __init__(self, tmp_path, report="passed", strat=None, **cfgkw):
        tmp_path.mkdir(parents=True, exist_ok=True)
        self.t = T0
        self.mt5 = FakeMT5()
        self.mt5.positions = []
        self.mt5.simulate_fills = True
        self.mt5.make_bars(300, T0)
        self.mt5.tick_time = T0 + 5
        self.mt5.equity = 10000.0
        self.logs = tmp_path / "logs"
        self.report = tmp_path / "report.json"
        prof = tmp_path / "profile.json"
        prof.write_text(json.dumps({"CrudeOIL": {"hourly_median": {str(h): 0.03 for h in range(24)}}}))
        if report == "passed":
            self.report.write_text(json.dumps({"synthetic": False, "results": {
                "CrudeOIL": {"fake": {"params": {}, "passed": True}}}}))
        elif report == "synthetic":
            self.report.write_text(json.dumps({"synthetic": True, "results": {
                "CrudeOIL": {"fake": {"params": {}, "passed": True}}}}))
        elif report == "nonepassed":
            self.report.write_text(json.dumps({"synthetic": False, "results": {
                "CrudeOIL": {"fake": {"params": {}, "passed": False}}}}))
        self.strat = strat or FakeStrat()
        cfg = BotConfig(symbols=["CrudeOIL"], report_path=str(self.report), profile_path=str(prof),
                        meta_path=str(tmp_path / "nometa.json"), config_dir=str(tmp_path / "cfg"),
                        logs_dir=str(self.logs), **cfgkw)
        self.bot = Bot(self.mt5, Settings(max_volume=1.0), cfg, clock=lambda: self.t,
                       sleep=lambda s: None, registry={"fake": self.strat})

    def next_bar(self, dt_s=60):
        self.t += dt_s
        self.mt5.tick_time = self.t
        self.mt5.make_bars(300, self.t - 5)  # barre ouverte il y a 5 s

    def status(self):
        return json.loads((self.logs / "status.json").read_text())


# ------------------------------------------------------------ cycles
def test_cycle_complet_un_ordre_avec_sl_tp(tmp_path):
    e = Env(tmp_path)
    e.t = T0 + 5
    e.bot.step()
    assert len(e.mt5.sent) == 1
    r = e.mt5.sent[0]
    assert r["magic"] == MAGIC_SCALP and r["comment"] == "fake"
    assert r["sl"] == pytest.approx(r["price"] - 0.5) and r["tp"] == pytest.approx(r["price"] + 1.0)
    assert r["volume"] == 0.10
    assert e.status()["mode"] == "active"
    rows = list(csv.DictReader(open(e.logs / "trades_scalp.csv")))
    assert rows[0]["event"] == "open" and rows[0]["strategy"] == "fake"
    assert e.bot.tickets[rows[0]["ticket"]]["max_minutes"] == 15
    # même barre : pas de second ordre ; barre suivante : position déjà ouverte
    e.bot.step()
    e.next_bar()
    e.bot.step()
    assert len(e.mt5.sent) == 1


def test_signal_hors_barre_courante_ignore(tmp_path):
    class Vieux(FakeStrat):
        def signals(self, m1, params):
            return [Sig(m1.index[-2], "buy", 0.5, 1.0, 15, "fake")]
    e = Env(tmp_path, strat=Vieux())
    e.t = T0 + 5
    e.bot.step()
    assert e.mt5.sent == []


def test_spread_trop_large_refuse(tmp_path):
    e = Env(tmp_path)
    e.t = T0 + 5
    e.mt5.ask = e.mt5.bid + 0.05
    e.bot.step()
    assert e.mt5.sent == []
    e.next_bar()
    e.mt5.ask = e.mt5.bid + 0.04
    e.bot.step()
    assert len(e.mt5.sent) == 1


def test_profil_absent_refuse(tmp_path):
    e = Env(tmp_path)
    e.bot.cfg.profile_path = str(tmp_path / "absent.json")
    e.t = T0 + 5
    e.bot.step()
    assert e.mt5.sent == []


def test_une_seule_position_par_symbole_et_autre_magic(tmp_path):
    e = Env(tmp_path)
    e.t = T0 + 5
    e.mt5.positions = [Position(9, "CrudeOIL", 0, 0.1, 70, 71, 0, 0, 0, 0, T0, MAGIC_SCALP)]
    e.bot.step()
    assert e.mt5.sent == []
    # une position d'un autre magic ne bloque pas l'entrée
    e2 = Env(tmp_path / "b")
    e2.t = T0 + 5
    e2.mt5.positions = [Position(8, "CrudeOIL", 0, 0.1, 70, 71, 0, 0, 0, 0, T0, 20261120)]
    e2.bot.step()
    assert len(e2.mt5.sent) == 1 and e2.mt5.sent[0]["magic"] == MAGIC_SCALP


def test_biais_short_refuse_buy(tmp_path):
    e = Env(tmp_path)
    e.t = T0 + 5
    now = datetime.fromtimestamp(e.t, UTC)
    e.logs.mkdir(exist_ok=True)
    (e.logs / "news_bias.json").write_text(json.dumps({
        "bias": "short", "generated": now.isoformat(), "until": (now + timedelta(hours=1)).isoformat(),
        "reasons": ["test"]}))
    e.bot.step()
    assert e.mt5.sent == []
    assert e.status()["bias"]["bias"] == "short"
    e.strat.side = "sell"
    e.next_bar()
    (e.logs / "news_bias.json").write_text(json.dumps({
        "bias": "short", "generated": datetime.fromtimestamp(e.t, UTC).isoformat(),
        "until": (now + timedelta(hours=1)).isoformat(), "reasons": []}))
    e.bot.step()
    assert len(e.mt5.sent) == 1


def test_calendrier_no_trade_sans_fichier(tmp_path):
    # mercredi 10:30 NY (EIA) = 15:30 UTC en novembre
    t = int(datetime(2026, 11, 18, 15, 30, tzinfo=UTC).timestamp())
    e = Env(tmp_path)
    e.t = t + 5
    e.mt5.tick_time = e.t
    e.mt5.make_bars(300, t)
    e.bot.step()
    assert e.mt5.sent == [] and e.status()["bias"]["bias"] == "no_trade"


@pytest.mark.parametrize("report", ["absent", "synthetic", "nonepassed"])
def test_mode_observation(tmp_path, report, caplog):
    e = Env(tmp_path, report=report)
    e.t = T0 + 5
    with caplog.at_level(logging.WARNING, logger="ava_mt5.bot"):
        e.bot.step()
        e.next_bar()
        e.bot.step()
    assert e.mt5.sent == [] and e.strat.calls == 0
    assert e.status()["mode"] == "observation"
    assert "OBSERVATION" in caplog.text


def test_force_strategies(tmp_path, caplog):
    e = Env(tmp_path, report="synthetic", force_strategies=["fake"])
    e.t = T0 + 5
    with caplog.at_level(logging.WARNING, logger="ava_mt5.bot"):
        e.bot.step()
    assert len(e.mt5.sent) == 1 and e.status()["mode"] == "forced"
    assert "FORCÉES" in caplog.text


def test_dry_run(tmp_path):
    e = Env(tmp_path, dry_run=True)
    e.t = T0 + 5
    e.bot.step()
    assert e.mt5.sent == []
    rows = list(csv.DictReader(open(e.logs / "trades_scalp.csv")))
    assert rows[0]["reason"] == "DRY"


def test_compte_reel_aucun_ordre(tmp_path):
    e = Env(tmp_path)
    e.mt5.trade_mode = FakeMT5.ACCOUNT_TRADE_MODE_REAL
    e.t = T0 + 5
    for _ in range(3):
        e.bot.step()
    assert e.mt5.sent == [] and not e.bot.connected
    assert e.status()["connected"] is False


def test_compte_devient_reel_en_cours_de_route(tmp_path):
    e = Env(tmp_path)
    e.t = T0 + 5
    e.bot.step()
    assert len(e.mt5.sent) == 1
    e.mt5.positions = []
    e.mt5.trade_mode = FakeMT5.ACCOUNT_TRADE_MODE_REAL
    e.next_bar()
    e.bot.step()
    assert len(e.mt5.sent) == 1


def test_sortie_forcee_apres_max_minutes(tmp_path):
    e = Env(tmp_path)
    e.t = T0 + 5
    e.mt5.positions = [Position(555, "CrudeOIL", 0, 0.1, 70, 71, 0, 0, 0, 0, T0 - 20 * 60, MAGIC_SCALP),
                       Position(556, "CrudeOIL", 0, 0.1, 70, 71, 0, 0, 0, 0, T0 - 5 * 60, MAGIC_SCALP)]
    e.bot.tickets = {"555": {"max_minutes": 15}, "556": {"max_minutes": 15}}
    e.bot.step()
    fermetures = [r for r in e.mt5.sent if "position" in r]
    assert [r["position"] for r in fermetures] == [555]
    assert fermetures[0]["magic"] == MAGIC_SCALP and fermetures[0]["type"] == FakeMT5.ORDER_TYPE_SELL


def test_sortie_forcee_ticket_inconnu_defaut_prudent(tmp_path):
    e = Env(tmp_path, default_max_minutes=30)
    e.t = T0 + 5
    e.mt5.positions = [Position(700, "CrudeOIL", 0, 0.1, 70, 71, 0, 0, 0, 0, T0 - 31 * 60, MAGIC_SCALP)]
    e.bot.step()
    assert [r["position"] for r in e.mt5.sent if "position" in r] == [700]


def test_sortie_forcee_ignore_autre_magic(tmp_path):
    e = Env(tmp_path)
    e.t = T0 + 5
    e.mt5.positions = [Position(800, "CrudeOIL", 0, 0.1, 70, 71, 0, 0, 0, 0, T0 - 600 * 60, 20261120)]
    e.bot.step()
    assert not [r for r in e.mt5.sent if "position" in r]
    assert [p.ticket for p in e.mt5.positions if p.ticket == 800] == [800]


def test_etat_persiste_entre_redemarrages(tmp_path):
    e = Env(tmp_path)
    e.t = T0 + 5
    e.bot.step()
    tid = next(iter(e.bot.tickets))
    e.bot.close()
    bot2 = Bot(e.mt5, Settings(), e.bot.cfg, clock=lambda: e.t, sleep=lambda s: None,
               registry={"fake": e.strat})
    assert bot2.tickets[tid]["max_minutes"] == 15


def test_deals_alimentent_coupe_circuit_sans_compter_autre_magic(tmp_path):
    from fake_mt5 import Deal
    e = Env(tmp_path, report="absent")   # observation : seuls deals et coupe-circuits
    e.t = T0 + 5
    e.bot.step()                         # base de référence des deals
    for i in range(5):
        e.mt5.deals.append(Deal(100 + i, 1, 50 + i, T0, 1, 1, MAGIC_SCALP, 0.1, 70, -10.0, 0, 0, 0, "CrudeOIL", 0))
    e.mt5.deals.append(Deal(200, 1, 90, T0, 1, 1, 20261120, 0.1, 70, -10.0, 0, 0, 0, "CrudeOIL", 0))
    e.bot.step()
    assert e.bot.breaker.consecutive_losses == 5 and e.bot.breaker.halted_today
    assert e.status()["halted_today"] is True
    assert len(list(csv.DictReader(open(e.logs / "trades_scalp.csv")))) == 5


def test_reconnexion_apres_exception(tmp_path):
    e = Env(tmp_path)
    e.t = T0 + 5
    e.mt5.fail_next["copy_rates_from_pos"] = RuntimeError("terminal perdu")
    w1 = e.bot.step()
    assert w1 == 2 and not e.bot.connected and e.mt5.shut
    e.mt5.fail_next["account_info"] = RuntimeError("encore")
    w2 = e.bot.step()
    assert w2 == 4
    w3 = e.bot.step()
    assert w3 == e.bot.cfg.poll_s and e.bot.connected
    assert e.mt5.init_count == 3 and len(e.mt5.sent) == 1


def test_account_info_none_declenche_reconnexion(tmp_path):
    e = Env(tmp_path)
    e.t = T0 + 5
    e.bot.step()
    orig = e.mt5.account_info
    e.mt5.account_info = lambda: None
    assert e.bot.step() == 2 and not e.bot.connected
    e.mt5.account_info = orig
    e.bot.step()
    assert e.bot.connected
