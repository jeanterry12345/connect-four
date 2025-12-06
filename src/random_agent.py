import random
from .base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """Agent that plays randomly."""

    def __init__(self, name="RandomAgent", player_id=None, seed=None):
        super().__init__(name=name, player_id=player_id)
        self.seed = seed
        self._rng = random.Random(seed)

    def select_action(self, observation, action_mask):
        """Select a random action among valid moves."""
        valid = self._get_valid_actions(action_mask)

        if not valid:
            raise ValueError("No valid action available")

        return self._rng.choice(valid)

    def reset(self):
        """Reset the random generator."""
        if self.seed is not None:
            self._rng = random.Random(self.seed)

    def set_seed(self, seed):
        """Change the random seed."""
        self.seed = seed
        self._rng = random.Random(seed)
