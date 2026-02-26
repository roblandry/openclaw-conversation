# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Projet

**OpenClaw Conversation** - Intégration Home Assistant custom pour utiliser OpenClaw comme agent de conversation. Connecte le pipeline vocal de HA à un gateway OpenClaw (API compatible OpenAI).

## Stack

- **Plateforme** : Home Assistant (v2024.1+)
- **Langage** : Python 3.7+
- **Dépendances** : aiohttp, voluptuous
- **Type** : Custom component HACS
- **Protocole** : OpenAI-compatible `/v1/chat/completions`

## Structure

```
custom_components/openclaw_conversation/
├── __init__.py           # Setup du domaine
├── conversation.py       # Agent principal (OpenClawConversationAgent)
├── config_flow.py        # Configuration UI + validation connexion
├── const.py              # Constantes (URL par défaut 127.0.0.1:18789, timeout 30s)
├── manifest.json         # Métadonnées intégration (v0.1.0)
├── strings.json          # Traductions UI
└── translations/         # EN + FR
```

## Fonctionnement

1. `config_flow.py` : Demande URL gateway, token API, modèle, timeout ; teste la connexion
2. `__init__.py` : Enregistre l'agent de conversation au démarrage
3. `conversation.py` : Reçoit texte/voix → appelle `/v1/chat/completions` → retourne réponse (historique 20 derniers messages)

## Installation

```bash
# Copier dans HA
cp -r custom_components/openclaw_conversation /path/to/ha/config/custom_components/
```
