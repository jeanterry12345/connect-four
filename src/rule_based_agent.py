import random
from .base_agent import BaseAgent


class RuleBasedAgent(BaseAgent):
    """
    Agent that follows simple rules:
    1. Win if possible
    2. Block opponent
    3. Prefer center columns
    4. Otherwise play randomly
    """

    def __init__(self, name="RuleBasedAgent", player_id=None, seed=None):
        super().__init__(name=name, player_id=player_id)
        self.seed = seed
        self._rng = random.Random(seed)

    def select_action(self, observation, action_mask):
        """Select an action according to the rules."""
        valid = self._get_valid_actions(action_mask)

        if not valid:
            raise ValueError("No valid action available")

        board = self._observation_to_board(observation)

        # 1. Look for a winning move
        win_move = self._find_winning_move(board, action_mask, player=1)
        if win_move != -1:
            return win_move

        # 2. Block opponent
        block_move = self._find_winning_move(board, action_mask, player=2)
        if block_move != -1:
            return block_move

        # 3. Prefer center columns
        center_order = [3, 2, 4, 1, 5, 0, 6]
        for col in center_order:
            if col in valid:
                return col

        # 4. Otherwise random
        return self._rng.choice(valid)

    def reset(self):
        """Reset the agent."""
        if self.seed is not None:
            self._rng = random.Random(self.seed)

    def set_seed(self, seed):
        """Change the random seed."""
        self.seed = seed
        self._rng = random.Random(seed)
