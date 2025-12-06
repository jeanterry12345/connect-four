# Analyse de l'agent aleatoire

## Experience

### Configuration
- 2 agents aleatoires
- 100 parties
- Environnement PettingZoo Connect Four

### Strategie
```python
def select_action(observation, action_mask):
    valid = [i for i, v in enumerate(action_mask) if v == 1]
    return random.choice(valid)
```

## Resultats typiques

### Victoires (sur 100 parties)
| Resultat | Nombre |
|----------|--------|
| Joueur 1 | ~55 |
| Joueur 2 | ~44 |
| Egalite | ~1 |

### Statistiques de coups
- Moyenne: ~28 coups
- Min: 7 (victoire rapide)
- Max: 42 (plateau plein)

### Temps
- Par partie: < 10 ms
- Par coup: < 0.1 ms

## Analyse

1. **Avantage premier joueur**: ~10% de plus
2. **Matchs nuls rares**: < 2%
3. **Parties courtes**: terminent avant plateau plein

## Comme reference

L'agent aleatoire sert de reference pour evaluer les autres agents:
- Agent base sur regles: > 80% contre aleatoire
- Agent Minimax: > 95% contre aleatoire
- Agent MCTS: > 95% contre aleatoire
