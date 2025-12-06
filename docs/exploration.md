# Notes d'exploration PettingZoo

## Environnement Connect Four

### Creation
```python
from pettingzoo.classic import connect_four_v3
env = connect_four_v3.env()
env.reset()
```

### Agents
- `env.possible_agents` : ['player_0', 'player_1']
- `env.agent_selection` : agent actuel

### Observation
- Shape: (6, 7, 2)
- Canal 0: mes pions
- Canal 1: pions adversaire

### Actions
- 0 a 6 : colonnes
- `action_mask` : 1 = colonne disponible

### Boucle de jeu
```python
for agent in env.agent_iter():
    obs, reward, done, trunc, info = env.last()
    if done or trunc:
        env.step(None)
        break
    action = choisir_action(obs)
    env.step(action)
```

### Recompenses
- 1 : victoire
- -1 : defaite
- 0 : nul ou en cours

## Coordonnees
```
Colonnes: 0 1 2 3 4 5 6
Lignes:   0 (haut) a 5 (bas)
```

## Erreurs courantes
- Oublier `env.step(None)` a la fin
- Choisir une colonne avec action_mask=0
