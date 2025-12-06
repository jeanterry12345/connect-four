import math
import time
import random
import numpy as np
from .base_agent import BaseAgent


class MCTSNode:
    """Node in the MCTS tree."""

    def __init__(self, parent=None, action=None, player=1, valid_actions=None):
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.wins = 0.0
        self.player = player
        self.untried = valid_actions.copy() if valid_actions else []

    def is_fully_expanded(self):
        return len(self.untried) == 0

    def ucb1(self, c=1.414):
        """Calculate UCB1 value."""
        if self.visits == 0:
            return float('inf')
        return self.wins / self.visits + c * math.sqrt(math.log(self.parent.visits) / self.visits)

    def best_child(self, c=1.414):
        """Return the best child according to UCB1."""
        return max(self.children.values(), key=lambda n: n.ucb1(c))

    def best_action(self):
        """Return the most visited action."""
        return max(self.children.keys(), key=lambda a: self.children[a].visits)


class MCTSAgent(BaseAgent):
    """Agent using Monte Carlo Tree Search."""

    def __init__(self, name="MCTSAgent", player_id=None, time_limit=2.5, max_iter=100000):
        super().__init__(name=name, player_id=player_id)
        self.time_limit = time_limit
        self.max_iter = max_iter

    def select_action(self, observation, action_mask):
        """Select an action using MCTS."""
        valid = self._get_valid_actions(action_mask)

        if not valid:
            raise ValueError("No valid action available")

        if len(valid) == 1:
            return valid[0]

        board = self._observation_to_board(observation)

        # immediate win?
        win = self._find_winning_move(board, action_mask, player=1)
        if win != -1:
            return win

        # block opponent?
        block = self._find_winning_move(board, action_mask, player=2)
        if block != -1:
            return block

        return self._mcts(board, valid)

    def _mcts(self, board, valid):
        """Execute MCTS search."""
        root = MCTSNode(player=1, valid_actions=valid)
        start = time.time()
        iterations = 0

        while time.time() - start < self.time_limit and iterations < self.max_iter:
            sim_board = board.copy()

            # selection
            node = self._select(root, sim_board)

            # expansion
            if not self._is_terminal(sim_board) and not node.is_fully_expanded():
                node = self._expand(node, sim_board)

            # simulation
            result = self._simulate(sim_board, node.player)

            # backpropagation
            self._backprop(node, result)

            iterations += 1

        if root.children:
            return root.best_action()
        return valid[len(valid) // 2]

    def _select(self, node, board):
        """Selection: descend through the tree."""
        while node.is_fully_expanded() and node.children:
            node = node.best_child()
            row = self._get_next_row(board, node.action)
            if row != -1:
                board[row, node.action] = 3 - node.player
            if self._is_terminal(board):
                break
        return node

    def _expand(self, node, board):
        """Expansion: add a child node."""
        action = node.untried.pop()
        row = self._get_next_row(board, action)
        if row != -1:
            board[row, action] = node.player

        valid = [c for c in range(7) if board[0, c] == 0]
        child = MCTSNode(parent=node, action=action, player=3 - node.player, valid_actions=valid)
        node.children[action] = child
        return child

    def _simulate(self, board, player):
        """Simulation: random playout."""
        while True:
            winner = self._check_winner(board)
            if winner == 1:
                return 1
            if winner == 2:
                return -1

            valid = [c for c in range(7) if board[0, c] == 0]
            if not valid:
                return 0

            # play smart move
            action = self._smart_action(board, valid, player)
            row = self._get_next_row(board, action)
            if row == -1:
                break
            board[row, action] = player
            player = 3 - player

        return 0

    def _smart_action(self, board, valid, player):
        """Choose a smart action."""
        # can we win?
        for col in valid:
            row = self._get_next_row(board, col)
            if row != -1:
                board[row, col] = player
                if self._check_win_pos(board, row, col, player):
                    board[row, col] = 0
                    return col
                board[row, col] = 0

        # block opponent?
        opp = 3 - player
        for col in valid:
            row = self._get_next_row(board, col)
            if row != -1:
                board[row, col] = opp
                if self._check_win_pos(board, row, col, opp):
                    board[row, col] = 0
                    return col
                board[row, col] = 0

        return random.choice(valid)

    def _backprop(self, node, result):
        """Backpropagation: update node statistics."""
        while node is not None:
            node.visits += 1
            if node.player == 1:
                if result == 1:
                    node.wins += 1.0
                elif result == 0:
                    node.wins += 0.5
            else:
                if result == -1:
                    node.wins += 1.0
                elif result == 0:
                    node.wins += 0.5
            node = node.parent

    def _check_winner(self, board):
        """Check for a winner."""
        for row in range(6):
            for col in range(4):
                p = board[row, col]
                if p != 0 and all(board[row, col + i] == p for i in range(4)):
                    return p

        for row in range(3):
            for col in range(7):
                p = board[row, col]
                if p != 0 and all(board[row + i, col] == p for i in range(4)):
                    return p

        for row in range(3):
            for col in range(4):
                p = board[row, col]
                if p != 0 and all(board[row + i, col + i] == p for i in range(4)):
                    return p

        for row in range(3):
            for col in range(3, 7):
                p = board[row, col]
                if p != 0 and all(board[row + i, col - i] == p for i in range(4)):
                    return p

        return 0

    def _check_win_pos(self, board, row, col, player):
        """Check for win from a position."""
        dirs = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in dirs:
            count = 1
            r, c = row + dr, col + dc
            while 0 <= r < 6 and 0 <= c < 7 and board[r, c] == player:
                count += 1
                r += dr
                c += dc
            r, c = row - dr, col - dc
            while 0 <= r < 6 and 0 <= c < 7 and board[r, c] == player:
                count += 1
                r -= dr
                c -= dc
            if count >= 4:
                return True
        return False

    def _is_terminal(self, board):
        """Check if terminal state."""
        if self._check_winner(board) != 0:
            return True
        return np.all(board[0, :] != 0)

    def reset(self):
        """Reset the agent."""
        pass
