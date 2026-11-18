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
