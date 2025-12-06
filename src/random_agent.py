import random
from .base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """Agent qui joue au hasard."""

    def __init__(self, name="RandomAgent", player_id=None, seed=None):
        super().__init__(name=name, player_id=player_id)
        self.seed = seed
        self._rng = random.Random(seed)

    def select_action(self, observation, action_mask):
        """Choisit une action au hasard parmi les coups valides."""
        valid = self._get_valid_actions(action_mask)

        if not valid:
            raise ValueError("Pas d'action valide")

        return self._rng.choice(valid)

    def reset(self):
        """Reset le generateur aleatoire."""
        if self.seed is not None:
            self._rng = random.Random(self.seed)

    def set_seed(self, seed):
        """Change la graine aleatoire."""
        self.seed = seed
        self._rng = random.Random(seed)
