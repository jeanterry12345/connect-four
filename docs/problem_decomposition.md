# Decomposition du probleme Puissance 4

## 1. Le jeu

### Regles
- Plateau 6x7
- Deux joueurs, tour par tour
- Placer un pion dans une colonne
- Premier a aligner 4 pions gagne

### Contraintes
- Pions soumis a la gravite
- Pas de placement si colonne pleine

## 2. Espace d'etats

### Observation PettingZoo
- observation: (6, 7, 2) - mes pions et ceux de l'adversaire
- action_mask: (7,) - colonnes disponibles

### Conditions de victoire
- Horizontal: 4 pions dans une ligne
- Vertical: 4 pions dans une colonne
- Diagonal: 4 pions en diagonale

## 3. Strategies

### Agent aleatoire
1. Obtenir le masque d'actions
2. Choisir aleatoirement une action valide

### Agent base sur regles
1. Gagner si possible
2. Bloquer l'adversaire
3. Preferer le centre
4. Sinon aleatoire

### Agent Minimax
- Recherche en profondeur
- Elagage alpha-beta
- Fonction d'evaluation

### Agent MCTS
1. Selection (UCB1)
2. Expansion
3. Simulation
4. Backpropagation

## 4. Fonction d'evaluation
- Bonus colonne centrale: +3
- Trois alignes: +5
- Deux alignes: +2

## 5. Contraintes de competition
- Temps: max 3 secondes/coup
- Memoire: max 384 Mo
- CPU: 1 coeur
