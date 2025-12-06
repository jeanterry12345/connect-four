from abc import ABC, abstractmethod
import numpy as np


class BaseAgent(ABC):
    """Abstract base class for all agents."""

    def __init__(self, name="BaseAgent", player_id=None):
        self.name = name
        self.player_id = player_id

    @abstractmethod
    def select_action(self, observation, action_mask):
        """Select an action. Must be implemented in subclasses."""
        pass

    def _get_valid_actions(self, action_mask):
        """Return list of valid actions."""
        return [i for i, v in enumerate(action_mask) if v == 1]

    def _observation_to_board(self, observation):
        """Convert observation to simple board representation."""
        board = np.zeros((6, 7), dtype=np.int8)
        board[observation[:, :, 0] == 1] = 1  # my pieces
        board[observation[:, :, 1] == 1] = 2  # opponent pieces
        return board

    def _get_next_row(self, board, col):
        """Find the row where a piece will land."""
        for row in range(5, -1, -1):
            if board[row, col] == 0:
                return row
        return -1

    def _check_win_from_position(self, board, row, col, player):
        """Check if 4 pieces are aligned from this position."""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for dr, dc in directions:
            count = 1
            # forward
            r, c = row + dr, col + dc
            while 0 <= r < 6 and 0 <= c < 7 and board[r, c] == player:
                count += 1
                r += dr
                c += dc
            # backward
            r, c = row - dr, col - dc
            while 0 <= r < 6 and 0 <= c < 7 and board[r, c] == player:
                count += 1
                r -= dr
                c -= dc

            if count >= 4:
                return True
        return False

    def _find_winning_move(self, board, action_mask, player):
        """Find a winning move if available."""
        valid = self._get_valid_actions(action_mask)

        for col in valid:
            row = self._get_next_row(board, col)
            if row != -1:
                board[row, col] = player
                if self._check_win_from_position(board, row, col, player):
                    board[row, col] = 0
                    return col
                board[row, col] = 0
        return -1

    def reset(self):
        """Reset the agent."""
        pass

    def __str__(self):
        return f"{self.name}(player={self.player_id})"

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"
