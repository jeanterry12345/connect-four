import time
import numpy as np
from .base_agent import BaseAgent


class MinimaxAgent(BaseAgent):
    """Agent utilisant l'algorithme Minimax avec elagage alpha-beta."""

    def __init__(self, name="MinimaxAgent", player_id=None, max_depth=4, time_limit=2.5):
        super().__init__(name=name, player_id=player_id)
        self.max_depth = max_depth
        self.time_limit = time_limit
        self._start_time = 0

    def select_action(self, observation, action_mask):
        """Choisit la meilleure action avec minimax."""
        valid = self._get_valid_actions(action_mask)

        if not valid:
            raise ValueError("Pas d'action valide")

        if len(valid) == 1:
            return valid[0]

        board = self._observation_to_board(observation)

        # verifier victoire immediate
        win = self._find_winning_move(board, action_mask, player=1)
        if win != -1:
            return win

        # bloquer adversaire
        block = self._find_winning_move(board, action_mask, player=2)
        if block != -1:
            return block

        # recherche minimax
        self._start_time = time.time()
        return self._search(board, valid)

    def _search(self, board, valid):
        """Recherche le meilleur coup."""
        best = valid[0]
        best_score = -99999

        # ordre: preferer le centre
        order = [3, 2, 4, 1, 5, 0, 6]
        sorted_valid = sorted(valid, key=lambda x: order.index(x) if x in order else 7)

        for col in sorted_valid:
            if time.time() - self._start_time > self.time_limit:
                break

            row = self._get_next_row(board, col)
            if row == -1:
                continue

            board[row, col] = 1
            score = -self._minimax(board, self.max_depth - 1, -99999, 99999, 2)
            board[row, col] = 0

            if score > best_score:
                best_score = score
                best = col

        return best

    def _minimax(self, board, depth, alpha, beta, player):
        """Minimax avec alpha-beta."""
        if time.time() - self._start_time > self.time_limit:
            return 0

        # verifier gagnant
        if self._has_won(board, 1):
            return 10000 + depth
        if self._has_won(board, 2):
            return -10000 - depth

        valid = [c for c in range(7) if board[0, c] == 0]

        if not valid or depth <= 0:
            return self._evaluate(board)

        best = -99999
        for col in valid:
            row = self._get_next_row(board, col)
            if row == -1:
                continue

            board[row, col] = player
            score = -self._minimax(board, depth - 1, -beta, -alpha, 3 - player)
            board[row, col] = 0

            best = max(best, score)
            alpha = max(alpha, score)

            if alpha >= beta:
                break

        return best

    def _evaluate(self, board):
        """Evalue la position."""
        score = 0

        # bonus centre
        for row in range(6):
            if board[row, 3] == 1:
                score += 3
            elif board[row, 3] == 2:
                score -= 3

        # compter alignements
        score += self._count_score(board, 1)
        score -= self._count_score(board, 2)

        return score

    def _count_score(self, board, player):
        """Compte les points des alignements."""
        score = 0
        opp = 3 - player

        # horizontal
        for row in range(6):
            for col in range(4):
                mine = sum(board[row, col + i] == player for i in range(4))
                theirs = sum(board[row, col + i] == opp for i in range(4))
                if theirs == 0:
                    if mine == 3:
                        score += 5
                    elif mine == 2:
                        score += 2

        # vertical
        for row in range(3):
            for col in range(7):
                mine = sum(board[row + i, col] == player for i in range(4))
                theirs = sum(board[row + i, col] == opp for i in range(4))
                if theirs == 0:
                    if mine == 3:
                        score += 5
                    elif mine == 2:
                        score += 2

        # diagonale
        for row in range(3):
            for col in range(4):
                mine = sum(board[row + i, col + i] == player for i in range(4))
                theirs = sum(board[row + i, col + i] == opp for i in range(4))
                if theirs == 0:
                    if mine == 3:
                        score += 5
                    elif mine == 2:
                        score += 2

        # anti-diagonale
        for row in range(3):
            for col in range(3, 7):
                mine = sum(board[row + i, col - i] == player for i in range(4))
                theirs = sum(board[row + i, col - i] == opp for i in range(4))
                if theirs == 0:
                    if mine == 3:
                        score += 5
                    elif mine == 2:
                        score += 2

        return score

    def _has_won(self, board, player):
        """Verifie si le joueur a gagne."""
        # horizontal
        for row in range(6):
            for col in range(4):
                if all(board[row, col + i] == player for i in range(4)):
                    return True
        # vertical
        for row in range(3):
            for col in range(7):
                if all(board[row + i, col] == player for i in range(4)):
                    return True
        # diagonales
        for row in range(3):
            for col in range(4):
                if all(board[row + i, col + i] == player for i in range(4)):
                    return True
        for row in range(3):
            for col in range(3, 7):
                if all(board[row + i, col - i] == player for i in range(4)):
                    return True
        return False

    def reset(self):
        """Reset."""
        pass
