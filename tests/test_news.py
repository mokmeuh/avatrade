import json
from datetime import datetime, timedelta, timezone

from ava_mt5 import news

UTC = timezone.utc


def utc(*a):
    return datetime(*a, tzinfo=UTC)


def evs(cfg, debut, fin):
    return [(e.name, e.time_utc) for e in news.events_between(cfg, debut, fin)]


def test_eia_ete_et_hiver():
    # mercredi 10:30 NY : 14:30 UTC en été, 15:30 UTC en hiver
    assert ("EIA", utc(2026, 7, 1, 14, 30)) in evs({}, utc(2026, 7, 1), utc(2026, 7, 2))
    assert ("EIA", utc(2026, 12, 2, 15, 30)) in evs({}, utc(2026, 12, 2), utc(2026, 12, 3))


def test_api_ete_et_hiver():
    assert ("API", utc(2026, 7, 7, 20, 30)) in evs({}, utc(2026, 7, 7), utc(2026, 7, 8))
    assert ("API", utc(2026, 12, 1, 21, 30)) in evs({}, utc(2026, 12, 1), utc(2026, 12, 2))


def test_nfp_premier_vendredi():
    assert ("NFP", utc(2026, 8, 7, 12, 30)) in evs({}, utc(2026, 8, 1), utc(2026, 8, 31))
    assert ("NFP", utc(2026, 12, 4, 13, 30)) in evs({}, utc(2026, 12, 1), utc(2026, 12, 31))
    noms = [n for n, _ in evs({}, utc(2026, 8, 1), utc(2026, 8, 31))]
    assert noms.count("NFP") == 1


def test_fomc_et_evenement_perso():
    cfg = {"events": [{"name": "OPEC+", "time_ny": "2026-12-03 09:00"}]}
    e = evs(cfg, utc(2026, 12, 1), utc(2026, 12, 10))
    assert ("FOMC", utc(2026, 12, 9, 19, 0)) in e
    assert ("OPEC+", utc(2026, 12, 3, 14, 0)) in e


def test_skip_et_shift():
    cfg = {"skip": ["2026-12-02 EIA"]}
    assert "EIA" not in [n for n, _ in evs(cfg, utc(2026, 12, 2), utc(2026, 12, 3))]
    cfg = {"shift": {"2026-12-02 EIA": "2026-12-03 12:00"}}
    assert ("EIA", utc(2026, 12, 3, 17, 0)) in evs(cfg, utc(2026, 12, 2), utc(2026, 12, 4))
    assert "EIA" not in [n for n, t in evs({"skip": ["2026-12-02"]}, utc(2026, 12, 2), utc(2026, 12, 3))]


def test_fenetre_no_trade():
    t = utc(2026, 7, 1, 14, 30)
    assert news.compute_bias(t - timedelta(minutes=5), {})["bias"] == "no_trade"
    assert news.compute_bias(t - timedelta(minutes=5, seconds=1), {})["bias"] == "neutral"
    assert news.compute_bias(t + timedelta(minutes=2), {})["bias"] == "no_trade"
    assert news.compute_bias(t + timedelta(minutes=2, seconds=1), {})["bias"] == "neutral"
    r = news.compute_bias(t, {})
    assert r["until"] == "2026-07-01T14:32:00+00:00"


def test_fenetre_parametrable():
    t = utc(2026, 7, 1, 14, 30)
    assert news.compute_bias(t - timedelta(minutes=10), {}, before_min=15)["bias"] == "no_trade"
    assert news.compute_bias(t + timedelta(minutes=10), {"after_min": 12})["bias"] == "no_trade"


def test_until_neutre_est_prochain_no_trade():
    r = news.compute_bias(utc(2026, 7, 1, 10, 0), {})
    assert r["bias"] == "neutral" and r["until"] == "2026-07-01T14:25:00+00:00"


def test_override():
    now = utc(2026, 7, 2, 10, 0)  # jeudi, rien
    ov = {"bias": "long", "until": "2026-07-02T12:00:00Z", "reason": "test"}
    r = news.compute_bias(now, {}, ov)
    assert r["bias"] == "long" and "test" in r["reasons"][0]
    assert news.compute_bias(now + timedelta(hours=3), {}, ov)["bias"] == "neutral"  # expiré
    assert news.compute_bias(now, {}, {"bias": "up", "until": "2026-07-02T12:00:00Z"})["bias"] == "neutral"
    # no_trade du calendrier prioritaire
    t = utc(2026, 7, 1, 14, 30)
    ov2 = {"bias": "short", "until": "2026-07-01T20:00:00Z"}
    assert news.compute_bias(t, {}, ov2)["bias"] == "no_trade"
    # l'override s'arrête avant le prochain no_trade
    r = news.compute_bias(utc(2026, 7, 1, 10, 0), {}, ov2)
    assert r["bias"] == "short" and r["until"] == "2026-07-01T14:25:00+00:00"


def _ecrire(p, bias, gen, until):
    p.write_text(json.dumps({"bias": bias, "generated": gen.isoformat(), "until": until.isoformat(),
                             "reasons": ["x"]}))


def test_read_bias(tmp_path):
    now = utc(2026, 7, 2, 10, 0)
    p = tmp_path / "b.json"
    assert news.read_bias(p, now)["bias"] == "neutral"               # absent
    _ecrire(p, "short", now - timedelta(minutes=5), now + timedelta(hours=1))
    assert news.read_bias(p, now)["bias"] == "short"
    _ecrire(p, "short", now - timedelta(minutes=31), now + timedelta(hours=1))
    assert news.read_bias(p, now)["bias"] == "neutral"               # trop vieux
    _ecrire(p, "no_trade", now - timedelta(minutes=1), now - timedelta(seconds=1))
    assert news.read_bias(p, now)["bias"] == "neutral"               # no_trade expiré
    _ecrire(p, "no_trade", now - timedelta(minutes=1), now + timedelta(minutes=1))
    assert news.read_bias(p, now)["bias"] == "no_trade"
    p.write_text("{pas du json")
    assert news.read_bias(p, now)["bias"] == "neutral"               # corrompu


def test_allows():
    assert not news.allows("no_trade", "buy") and not news.allows("no_trade", "sell")
    assert news.allows("long", "buy") and not news.allows("long", "sell")
    assert news.allows("short", "sell") and not news.allows("short", "buy")
    assert news.allows("neutral", "buy") and news.allows("neutral", "sell")
    assert not news.allows({"bias": "short"}, "buy")


def test_cli_ecrit_fichier(tmp_path):
    assert news.main(["--config", str(tmp_path), "--logs", str(tmp_path / "l")]) == 0
    d = json.loads((tmp_path / "l" / "news_bias.json").read_text())
    assert d["bias"] in ("neutral", "no_trade", "long", "short") and d["generated"]
