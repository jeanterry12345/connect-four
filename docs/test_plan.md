# Plan de test

## Objectifs
- Verifier le bon fonctionnement des agents
- Tester les regles du jeu
- Evaluer les performances

## Tests fonctionnels

### Mecaniques de jeu
- Detection des actions legales
- Placement des pions
- Detection de fin de partie

### Conditions de victoire
- Horizontal: 4 pions dans une ligne
- Vertical: 4 pions dans une colonne
- Diagonale: 4 pions en diagonale

### Tests par agent
| Agent | Test |
|-------|------|
| RandomAgent | Selection valide |
| RuleBasedAgent | Victoire, blocage, centre |
| MinimaxAgent | Recherche correcte |
| MCTSAgent | Simulations correctes |

## Tests de performance

- Temps par coup: < 3 secondes
- Memoire: < 384 Mo

## Tests de strategie

| Match | Resultat attendu |
|-------|------------------|
| RuleAgent vs Random | > 80% victoires |
| Minimax vs Random | > 95% victoires |

## Scenarios de test

### Victoire
```
5 |X X X . . . .|
Action attendue: 3
```

### Blocage
```
5 |O O O . X . .|
Action attendue: 3
```

## Execution

```bash
# Tous les tests
pytest tests/ -v

# Tests specifiques
pytest tests/test_random_agent.py -v

# Couverture
pytest tests/ --cov=src
```
