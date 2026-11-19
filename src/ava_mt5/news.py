"""Calendrier d'événements pétroliers et biais directionnel.

Les horaires du calendrier sont exprimés en heure de New York (America/New_York) puis
convertis en UTC, ce qui gère automatiquement l'heure d'été / d'hiver.

Événements récurrents intégrés :
  - EIA  : chaque mercredi 10:30 NY
  - API  : chaque mardi 16:30 NY
  - NFP  : premier vendredi du mois 8:30 NY
  - FOMC : décisions à 14:00 NY (dates 2026 intégrées ; dates supplémentaires dans
           config/events.json, clé "fomc_dates")
  - événements ponctuels (OPEP+, ...) : config/events.json, clé "events"

Format de config/events.json : voir ce fichier (clé "_doc").
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
UTC = timezone.utc

AVANT_MIN = 5.0   # minutes de no_trade avant l'événement
APRES_MIN = 2.0   # minutes de no_trade après l'événement

# Décisions FOMC 2026 (second jour de la réunion, 14:00 NY).
FOMC_2026 = ["2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
             "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09"]

BIAIS_VALIDES = ("long", "short", "neutral")


@dataclass(frozen=True)
class Event:
    name: str
    time_utc: datetime

    def key(self) -> str:
        return f"{self.time_utc.astimezone(NY):%Y-%m-%d} {self.name}"


# ------------------------------------------------------------------ utilitaires
def _parse_iso(txt: Any) -> datetime | None:
    """ISO 8601 -> datetime UTC (naïf = UTC) ; None si invalide."""
    if not isinstance(txt, str):
        return None
    try:
        d = datetime.fromisoformat(txt.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return d.replace(tzinfo=UTC) if d.tzinfo is None else d.astimezone(UTC)


def _iso(d: datetime) -> str:
    return d.astimezone(UTC).isoformat(timespec="seconds")


def _ny_to_utc(d: date, hh: int, mm: int) -> datetime:
    return datetime(d.year, d.month, d.day, hh, mm, tzinfo=NY).astimezone(UTC)


def _parse_time_ny(txt: str) -> datetime:
    naive = datetime.strptime(txt.strip(), "%Y-%m-%d %H:%M")
    return naive.replace(tzinfo=NY).astimezone(UTC)


def load_json(path: str | os.PathLike) -> Any:
    """Lit un JSON ; None si absent ou illisible."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _write_json_atomic(path: str | os.PathLike, data: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)


# ------------------------------------------------------------------ calendrier
def events_between(cfg: dict | None, start_utc: datetime, end_utc: datetime) -> list[Event]:
    """Événements dont l'heure UTC est dans [start_utc, end_utc], après skip/shift."""
    cfg = cfg if isinstance(cfg, dict) else {}
    fomc = set(FOMC_2026) | {str(x) for x in cfg.get("fomc_dates", []) or []}
    brut: list[tuple[str, datetime]] = []

    d = start_utc.astimezone(NY).date() - timedelta(days=3)
    fin = end_utc.astimezone(NY).date() + timedelta(days=3)
    while d <= fin:
        wd = d.weekday()
        if wd == 2:
            brut.append(("EIA", _ny_to_utc(d, 10, 30)))
        if wd == 1:
            brut.append(("API", _ny_to_utc(d, 16, 30)))
        if wd == 4 and d.day <= 7:
            brut.append(("NFP", _ny_to_utc(d, 8, 30)))
        if d.isoformat() in fomc:
            brut.append(("FOMC", _ny_to_utc(d, 14, 0)))
        d += timedelta(days=1)

    for ev in cfg.get("events", []) or []:
        try:
            brut.append((str(ev["name"]), _parse_time_ny(ev["time_ny"])))
        except (KeyError, TypeError, ValueError):
            continue

    skip = [str(s).strip().lower() for s in cfg.get("skip", []) or []]
    shift = {str(k).strip().lower(): v for k, v in (cfg.get("shift", {}) or {}).items()}

    out: list[Event] = []
    for name, t in brut:
        ev = Event(name, t)
        key = ev.key().lower()
        jour = key.split(" ", 1)[0]
        if key in skip or jour in skip:
            continue
        if key in shift:
            try:
                ev = Event(name, _parse_time_ny(shift[key]))
            except (TypeError, ValueError):
                pass
        if start_utc <= ev.time_utc <= end_utc:
            out.append(ev)
    return sorted(out, key=lambda e: e.time_utc)


# ------------------------------------------------------------------ biais
def _override_actif(override: dict | None, now: datetime) -> tuple[str, datetime, str] | None:
    if not isinstance(override, dict):
        return None
    biais = override.get("bias")
    fin = _parse_iso(override.get("until"))
    if biais not in BIAIS_VALIDES or fin is None or fin <= now:
        return None
    return biais, fin, str(override.get("reason") or "").strip()


def compute_bias(now_utc: datetime, events_cfg: dict | None, override: dict | None = None,
                 before_min: float | None = None, after_min: float | None = None) -> dict:
    """Biais du moment (fonction pure).

    no_trade du calendrier > override utilisateur > neutral.
    "until" : fin de la fenêtre no_trade, sinon prochain changement d'état
    (début du prochain no_trade ou expiration de l'override).
    """
    cfg = events_cfg if isinstance(events_cfg, dict) else {}
    now = now_utc.astimezone(UTC) if now_utc.tzinfo else now_utc.replace(tzinfo=UTC)
    avant = timedelta(minutes=before_min if before_min is not None
                      else float(cfg.get("before_min", AVANT_MIN)))
    apres = timedelta(minutes=after_min if after_min is not None
                      else float(cfg.get("after_min", APRES_MIN)))

    evs = events_between(cfg, now - timedelta(hours=2), now + timedelta(days=16))
    actifs = [e for e in evs if e.time_utc - avant <= now <= e.time_utc + apres]
    if actifs:
        fin = max(e.time_utc for e in actifs) + apres
        # chaînage : une fenêtre qui démarre avant la fin courante la prolonge
        etendu = True
        while etendu:
            etendu = False
            for e in evs:
                if e.time_utc - avant <= fin < e.time_utc + apres:
                    fin = e.time_utc + apres
                    etendu = True
        raisons = [f"fenêtre no_trade {e.name} ({e.time_utc.astimezone(NY):%Y-%m-%d %H:%M} NY)"
                   for e in actifs]
        if _override_actif(override, now):
            raisons.append("override ignoré : no_trade du calendrier prioritaire")
        return {"bias": "no_trade", "until": _iso(fin), "reasons": raisons,
                "generated": _iso(now)}

    prochain = next((e.time_utc - avant for e in evs if e.time_utc - avant > now), None)
    ov = _override_actif(override, now)
    if ov:
        biais, fin, raison = ov
        if prochain is not None:
            fin = min(fin, prochain)
        return {"bias": biais, "until": _iso(fin),
                "reasons": [f"override utilisateur : {raison or 'sans motif'}"],
                "generated": _iso(now)}
    return {"bias": "neutral", "until": _iso(prochain) if prochain else None,
            "reasons": ["aucun événement ni override actif"], "generated": _iso(now)}


def _neutre(raison: str) -> dict:
    return {"bias": "neutral", "until": None, "reasons": [raison], "generated": None}


def read_bias(path: str | os.PathLike, now_utc: datetime, max_age_min: float = 30) -> dict:
    """Lit logs/news_bias.json côté bot. Tout doute -> neutral."""
    now = now_utc.astimezone(UTC) if now_utc.tzinfo else now_utc.replace(tzinfo=UTC)
    data = load_json(path)
    if not isinstance(data, dict):
        return _neutre("fichier de biais absent ou illisible")
    gen = _parse_iso(data.get("generated"))
    if gen is None:
        return _neutre("date de génération invalide")
    if now - gen > timedelta(minutes=max_age_min):
        return _neutre(f"fichier de biais périmé (> {max_age_min:g} min)")
    biais = data.get("bias")
    if biais not in BIAIS_VALIDES + ("no_trade",):
        return _neutre("biais inconnu")
    fin = _parse_iso(data.get("until"))
    if biais != "neutral" and (fin is None or fin <= now):
        return _neutre(f"biais {biais} expiré")
    return {"bias": biais, "until": data.get("until"),
            "reasons": list(data.get("reasons") or []), "generated": data.get("generated")}


def allows(bias: dict | str, side: str) -> bool:
    """no_trade : rien ; long : buy seulement ; short : sell seulement ; neutral : tout."""
    b = bias.get("bias") if isinstance(bias, dict) else bias
    if b == "no_trade":
        return False
    if b == "long":
        return side == "buy"
    if b == "short":
        return side == "sell"
    return b == "neutral"


# ------------------------------------------------------------------ CLI
def run_once(now_utc: datetime, config_dir: str, logs_dir: str,
             before_min: float | None = None, after_min: float | None = None) -> dict:
    cfg = load_json(Path(config_dir) / "events.json") or {}
    ov = load_json(Path(config_dir) / "bias_override.json")
    res = compute_bias(now_utc, cfg, ov, before_min, after_min)
    _write_json_atomic(Path(logs_dir) / "news_bias.json", res)
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ava_mt5.news", description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="config", help="dossier de events.json / bias_override.json")
    ap.add_argument("--logs", default="logs", help="dossier d'écriture de news_bias.json")
    ap.add_argument("--loop", type=float, default=0, metavar="SECONDES",
                    help="réécrit le fichier toutes les N secondes (0 = une seule fois)")
    ap.add_argument("--before", type=float, default=None, help="minutes de no_trade avant")
    ap.add_argument("--after", type=float, default=None, help="minutes de no_trade après")
    a = ap.parse_args(argv)
    try:
        while True:
            res = run_once(datetime.now(UTC), a.config, a.logs, a.before, a.after)
            print(f"{res['generated']} biais={res['bias']} jusqu'à {res['until']} "
                  f"- {'; '.join(res['reasons'])}", flush=True)
            if a.loop <= 0:
                return 0
            time.sleep(a.loop)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
