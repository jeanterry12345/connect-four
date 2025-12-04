# Projet Puissance 4

Projet Python M2 - Sorbonne Universite

## Description

Implementation d'agents intelligents pour le jeu Puissance 4 avec PettingZoo.

## Regles du jeu

- Plateau: 6 lignes x 7 colonnes
- Aligner 4 pions pour gagner
- Les pions tombent en bas de la colonne

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

```bash
# Jouer une partie
python main.py

# Executer les tests
pytest tests/ -v

# Lancer un tournoi
python scripts/tournament.py
```

## Structure

```
projet/
├── agent.py              # Agent ML-Arena
├── src/                  # Code source
│   ├── base_agent.py
│   ├── random_agent.py
│   ├── rule_based_agent.py
│   ├── minimax_agent.py
│   └── mcts_agent.py
├── tests/                # Tests
├── scripts/              # Scripts
└── docs/                 # Documentation
```

## Agents

1. **RandomAgent**: Joue au hasard
2. **RuleBasedAgent**: Regles simples (gagner, bloquer, centre)
3. **MinimaxAgent**: Algorithme Minimax + alpha-beta
4. **MCTSAgent**: Monte Carlo Tree Search

## Contraintes ML-Arena

- Temps: max 3 secondes/coup
- Memoire: max 384 Mo
- CPU: 1 coeur

