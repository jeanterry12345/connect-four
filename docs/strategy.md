# Strategie de l'agent base sur regles

## Priorites de decision

### 1. Gagner
Si un coup permet de gagner, le jouer immediatement.

### 2. Bloquer
Si l'adversaire peut gagner au prochain coup, bloquer.

### 3. Preferer le centre
La colonne 3 (centre) a le plus de possibilites d'alignement.

### 4. Colonnes proches du centre
Ordre de preference: 3, 2, 4, 1, 5, 0, 6

### 5. Aleatoire
Si aucune regle ne s'applique, choisir au hasard.

## Fonctions auxiliaires

### Actions valides
```python
def _get_valid_actions(action_mask):
    return [i for i, v in enumerate(action_mask) if v == 1]
```

### Ligne de chute
```python
def _get_next_row(board, col):
    for row in range(5, -1, -1):
        if board[row, col] == 0:
            return row
    return -1
```

### Detection victoire
Verifier les 4 directions depuis une position:
- Horizontal (0, 1)
- Vertical (1, 0)
- Diagonale (1, 1)
- Anti-diagonale (1, -1)

## Performance attendue

| Adversaire | Taux de victoire |
|------------|------------------|
| Aleatoire | > 80% |
| Regles | ~50% |
| Minimax | < 30% |

## Limitations

- Ne regarde qu'un coup a l'avance
- Vulnerable aux strategies planifiees
- Pas de detection des menaces a long terme
