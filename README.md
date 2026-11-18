# ava-mt5

Serveur MCP qui pilote un terminal **MetaTrader 5 AvaTrade** depuis Claude Code,
pour trader (notamment le pétrole) sur un **compte démo**.

Le serveur refuse de se connecter à un compte réel.

## Prérequis (Windows)

- Terminal **AvaTrade MetaTrader 5** installé, ouvert et connecté au compte démo.
- Python 3.10+ pour Windows (python.org).
- Claude Code installé sur la même machine.
- Dans MT5 : bouton **Algo Trading** activé (barre d'outils), et dans
  *Outils > Options > Expert Advisors*, cocher « Autoriser le trading algorithmique ».
  Sans cela, les ordres sont rejetés avec le code 10027.

## Installation

```powershell
git clone https://github.com/mokmeuh/avatrade.git
cd avatrade
py -m venv .venv
.venv\Scripts\pip install -e .[dev]
```

Vérification rapide :

```powershell
.venv\Scripts\python -m pytest -q
```

## Configuration

Par défaut, le serveur s'attache au terminal MT5 déjà ouvert et connecté. Aucun
identifiant n'est nécessaire.

Variables d'environnement facultatives (voir `.env.example`) :

| Variable | Rôle |
|---|---|
| `MT5_LOGIN`, `MT5_PASSWORD`, `MT5_SERVER` | Connexion explicite au compte démo |
| `MT5_PATH` | Chemin de `terminal64.exe` si plusieurs MT5 sont installés |
| `MT5_MAX_VOLUME` | Volume maximum par ordre, en lots (défaut : 1.0) |

## Utilisation avec Claude Code

Le fichier `.mcp.json` du dépôt déclare le serveur. Lancer Claude Code dans le
dossier du dépôt :

```powershell
claude
```

Accepter le serveur `ava-mt5` à la première ouverture, puis vérifier avec `/mcp`.

Exemples de demandes :

- « Montre-moi le solde du compte démo. »
- « Trouve le symbole du pétrole WTI et donne-moi sa spécification. »
- « Achète 100 barils de WTI avec un stop 1 $ sous le prix. »
- « Ferme toutes mes positions pétrole. »

## Outils exposés

| Outil | Description |
|---|---|
| `account_info` | Solde, équité, marge (vérifie que le compte est démo) |
| `search_symbols` | Recherche de symboles (ex. `oil`, `brent`) |
| `symbol_spec` | Taille du contrat (barils par lot), volumes min/max/pas |
| `get_quote` | Bid/ask courant |
| `get_rates` | Bougies OHLC (M1 à MN1) |
| `get_positions` | Positions ouvertes |
| `place_market_order` | Achat/vente au marché, en lots ou en barils, avec SL/TP |
| `close_position` | Fermeture totale ou partielle |
| `modify_position` | Modification du SL/TP |

## Garde-fous

- Connexion refusée si le compte n'est pas en mode démo.
- Volume plafonné par `MT5_MAX_VOLUME`.
- SL/TP vérifiés du bon côté du prix avant envoi.
- Claude Code demande une confirmation avant chaque appel d'outil, sauf si tu
  l'autorises explicitement.

## Bots de scalping pétrole (compte démo)

Deux bots indépendants, chacun avec son magic number, son journal et son heartbeat :

| Bot | Module | Magic | Journal | Heartbeat |
|---|---|---|---|---|
| Scalp (stratégies M1) | `ava_mt5.bot` | 20261119 | `logs/trades_scalp.csv`, `logs/bot.log` | `logs/status.json` |
| Rafale (ticks, à la seconde) | `ava_mt5.rafale` | 20261120 | `logs/trades_rafale.csv`, `logs/rafale.log` | `logs/status_rafale.json` |

Les deux bots refusent tout compte non démo et vérifient le mode à chaque ordre.
Ils restent en **mode observation** (aucun ordre) tant que leur stratégie n'a pas
passé les critères hors échantillon sur des données réelles (≥ 200 trades,
PF ≥ 1,15, DD ≤ 10 %, gain net > 0, perte moy./gain moy. ≤ 2). `--force` (rafale)
ou `--force-strategies orb,...` (scalp) passent outre, en démo uniquement.

### 1. Données et coûts réels

```powershell
.venv\Scripts\python -m ava_mt5.data collect --symbols CrudeOIL,BrentOIL --out data --weeks 4
.venv\Scripts\python -m ava_mt5.data profile --data data
```

Le marché doit être ouvert pendant la collecte (estimation du décalage horaire du
serveur ; sinon `--server-offset 2`). Résultat : `data/*.csv.gz`, `data/meta.json`,
`data/spread_profile.json` (spread médian par heure UTC, autour d'EIA et d'API).

### 2. Backtests

```powershell
.venv\Scripts\python -m ava_mt5.backtest --data data --out logs/backtest_report.json
.venv\Scripts\python -m ava_mt5.rafale check
.venv\Scripts\python -m ava_mt5.rafale backtest --data data --symbol CrudeOIL --out logs/rafale_report.json --jobs 4
```

Rapports lisibles : `logs/backtest_report.md`, `logs/rafale_report.md`.
Les montants des backtests sont en USD (devise de cotation du pétrole).

### 3. Lancement (trois fenêtres)

```powershell
.venv\Scripts\python -m ava_mt5.news --loop 60
.venv\Scripts\python -m ava_mt5.bot --symbols CrudeOIL,BrentOIL
.venv\Scripts\python -m ava_mt5.rafale live --symbol CrudeOIL --params-from logs/rafale_report.json
```

En arrière-plan (PowerShell) :

```powershell
Start-Process -WindowStyle Hidden .venv\Scripts\python.exe "-m ava_mt5.news --loop 60"
Start-Process -WindowStyle Hidden .venv\Scripts\python.exe "-m ava_mt5.bot --symbols CrudeOIL,BrentOIL"
Start-Process -WindowStyle Hidden .venv\Scripts\python.exe "-m ava_mt5.rafale live --symbol CrudeOIL --params-from logs/rafale_report.json"
```

Arrêt : `Ctrl+C` dans chaque fenêtre, ou en arrière-plan :

```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'ava_mt5\.(bot|rafale|news)' } | ForEach-Object { Stop-Process -Id $_.ProcessId }
```

Après un arrêt complet (drawdown 8 %), relancer avec `--reset-halt`.

### Filtre de nouvelles

`logs/news_bias.json` : `{"bias": "long"|"short"|"neutral"|"no_trade", "until", "reasons"}`.
No-trade de 5 min avant à 2 min après EIA (mer. 10:30 NY), API (mar. 16:30 NY),
NFP, FOMC et OPEP+. Les dates OPEP+, FOMC 2027 et les décalages de jours fériés se
règlent dans `config/events.json`. Biais manuel : copier
`config/bias_override.example.json` en `config/bias_override.json`. Un fichier
de plus de 30 minutes est lu comme `neutral`.

### Garde-fous

- Risque 0,5 % de l'équité par trade (scalp), stop et objectif posés à l'ouverture,
  une position par symbole, sortie forcée après N minutes.
- Pas d'entrée si le spread dépasse 1,5 × sa médiane horaire.
- Arrêt pour la journée à −2 % ou après 5 pertes consécutives ; arrêt complet à
  8 % de drawdown.
- Rafale : volume minimum fixe, 10 positions au plus, stop par position, durée de vie
  maximale, fermeture de tout à −2 % de perte latente. Mode « paires » refusé si le
  compte n'est pas en hedging.
- Pas de martingale, de grille ni de moyenne à la baisse.
