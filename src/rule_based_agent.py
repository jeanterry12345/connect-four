import random
from .base_agent import BaseAgent


class RuleBasedAgent(BaseAgent):
    """
    Agent qui suit des regles simples:
    1. Gagner si possible
    2. Bloquer l'adversaire
    3. Preferer le centre
    4. Sinon jouer au hasard
    """

    def __init__(self, name="RuleBasedAgent", player_id=None, seed=None):
        super().__init__(name=name, player_id=player_id)
        self.seed = seed
        self._rng = random.Random(seed)

    def select_action(self, observation, action_mask):
        """Choisit une action selon les regles."""
        valid = self._get_valid_actions(action_mask)

        if not valid:
            raise ValueError("Pas d'action valide")

        board = self._observation_to_board(observation)

        # 1. Chercher un coup gagnant
        win_move = self._find_winning_move(board, action_mask, player=1)
        if win_move != -1:
            return win_move

        # 2. Bloquer l'adversaire
        block_move = self._find_winning_move(board, action_mask, player=2)
        if block_move != -1:
            return block_move

        # 3. Preferer le centre
        center_order = [3, 2, 4, 1, 5, 0, 6]
        for col in center_order:
            if col in valid:
                return col

        # 4. Sinon hasard
        return self._rng.choice(valid)

    def reset(self):
        """Reset."""
        if self.seed is not None:
            self._rng = random.Random(self.seed)

    def set_seed(self, seed):
        """Change la graine."""
        self.seed = seed
        self._rng = random.Random(seed)
