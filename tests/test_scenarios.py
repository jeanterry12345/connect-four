"""Scenarios de test specifiques"""

import pytest
import numpy as np
from src.rule_based_agent import RuleBasedAgent
from src.utils import check_winner


class TestWinningScenarios:
    """Tests scenarios de victoire"""

    def test_horizontal_win(self):
        """Victoire horizontale"""
        agent = RuleBasedAgent()

        obs = np.zeros((6, 7, 2), dtype=np.int8)
        obs[5, 0, 0] = 1
        obs[5, 1, 0] = 1
        obs[5, 2, 0] = 1

        mask = np.array([1, 1, 1, 1, 1, 1, 1])
        action = agent.select_action(obs, mask)

        assert action == 3

    def test_vertical_win(self):
        """Victoire verticale"""
        agent = RuleBasedAgent()

        obs = np.zeros((6, 7, 2), dtype=np.int8)
        obs[5, 0, 0] = 1
        obs[4, 0, 0] = 1
        obs[3, 0, 0] = 1

        mask = np.array([1, 1, 1, 1, 1, 1, 1])
        action = agent.select_action(obs, mask)

        assert action == 0


class TestBlockingScenarios:
    """Tests scenarios de blocage"""

    def test_block_horizontal(self):
        """Bloquer horizontal"""
        agent = RuleBasedAgent()

        obs = np.zeros((6, 7, 2), dtype=np.int8)
        obs[5, 0, 1] = 1
        obs[5, 1, 1] = 1
        obs[5, 2, 1] = 1

        mask = np.array([1, 1, 1, 1, 1, 1, 1])
        action = agent.select_action(obs, mask)

        assert action == 3

    def test_block_vertical(self):
        """Bloquer vertical"""
        agent = RuleBasedAgent()

        obs = np.zeros((6, 7, 2), dtype=np.int8)
        obs[5, 0, 1] = 1
        obs[4, 0, 1] = 1
        obs[3, 0, 1] = 1

        mask = np.array([1, 1, 1, 1, 1, 1, 1])
        action = agent.select_action(obs, mask)

        assert action == 0


class TestWinConditions:
    """Tests conditions de victoire"""

    def test_four_horizontal(self):
        """4 pions horizontaux"""
        board = np.zeros((6, 7), dtype=np.int8)
        board[5, 0:4] = 1
        assert check_winner(board) == 1

    def test_four_vertical(self):
        """4 pions verticaux"""
        board = np.zeros((6, 7), dtype=np.int8)
        board[2:6, 0] = 1
        assert check_winner(board) == 1

    def test_four_diagonal(self):
        """4 pions diagonaux"""
        board = np.zeros((6, 7), dtype=np.int8)
        for i in range(4):
            board[5 - i, i] = 1
        assert check_winner(board) == 1

    def test_three_not_win(self):
        """3 pions ne gagnent pas"""
        board = np.zeros((6, 7), dtype=np.int8)
        board[5, 0:3] = 1
        assert check_winner(board) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
