from abc import ABC, abstractmethod
import numpy as np


class BaseAgent(ABC):
    """Classe abstraite pour tous les agents."""

    def __init__(self, name="BaseAgent", player_id=None):
        self.name = name
        self.player_id = player_id

    @abstractmethod
    def select_action(self, observation, action_mask):
        """Choisit une action. A implementer dans les sous-classes."""
        pass

    def _get_valid_actions(self, action_mask):
        """Retourne la liste des actions valides."""
        return [i for i, v in enumerate(action_mask) if v == 1]

    def _observation_to_board(self, observation):
        """Convertit l'observation en plateau simple."""
        board = np.zeros((6, 7), dtype=np.int8)
        board[observation[:, :, 0] == 1] = 1  # moi
        board[observation[:, :, 1] == 1] = 2  # adversaire
        return board

    def _get_next_row(self, board, col):
        """Trouve la ligne ou tombe le pion."""
        for row in range(5, -1, -1):
            if board[row, col] == 0:
                return row
        return -1

    def _check_win_from_position(self, board, row, col, player):
        """Verifie si 4 pions sont alignes depuis cette position."""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for dr, dc in directions:
            count = 1
            # avant
            r, c = row + dr, col + dc
            while 0 <= r < 6 and 0 <= c < 7 and board[r, c] == player:
                count += 1
                r += dr
                c += dc
            # arriere
            r, c = row - dr, col - dc
            while 0 <= r < 6 and 0 <= c < 7 and board[r, c] == player:
                count += 1
                r -= dr
                c -= dc

            if count >= 4:
                return True
        return False

    def _find_winning_move(self, board, action_mask, player):
        """Cherche un coup gagnant."""
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
        """Reset l'agent."""
        pass

    def __str__(self):
        return f"{self.name}(player={self.player_id})"

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"
