"""Tests pour l'agent base sur regles"""

import pytest
import numpy as np
from src.rule_based_agent import RuleBasedAgent


class TestRuleBasedAgent:
    """Tests pour RuleBasedAgent"""

    def test_init(self):
        """Test initialisation"""
        agent = RuleBasedAgent()
        assert agent.name == "RuleBasedAgent"

    def test_wins_when_possible(self):
        """Test victoire horizontale"""
        agent = RuleBasedAgent()

        obs = np.zeros((6, 7, 2), dtype=np.int8)
        obs[5, 0, 0] = 1
        obs[5, 1, 0] = 1
        obs[5, 2, 0] = 1

        mask = np.array([1, 1, 1, 1, 1, 1, 1])
        action = agent.select_action(obs, mask)

        assert action == 3

    def test_blocks_opponent(self):
        """Test blocage adversaire"""
        agent = RuleBasedAgent()

        obs = np.zeros((6, 7, 2), dtype=np.int8)
        obs[5, 0, 1] = 1
        obs[5, 1, 1] = 1
        obs[5, 2, 1] = 1

        mask = np.array([1, 1, 1, 1, 1, 1, 1])
        action = agent.select_action(obs, mask)

        assert action == 3

    def test_prefers_center(self):
        """Test preference centre"""
        agent = RuleBasedAgent()

        obs = np.zeros((6, 7, 2), dtype=np.int8)
        mask = np.array([1, 1, 1, 1, 1, 1, 1])

        action = agent.select_action(obs, mask)
        assert action == 3

    def test_win_priority_over_block(self):
        """Test priorite victoire sur blocage"""
        agent = RuleBasedAgent()

        obs = np.zeros((6, 7, 2), dtype=np.int8)
        # moi: 3 en ligne horizontal
        obs[5, 0, 0] = 1
        obs[5, 1, 0] = 1
        obs[5, 2, 0] = 1
        # adversaire: 3 en ligne vertical
        obs[5, 6, 1] = 1
        obs[4, 6, 1] = 1
        obs[3, 6, 1] = 1

        mask = np.array([1, 1, 1, 1, 1, 1, 1])
        action = agent.select_action(obs, mask)

        assert action == 3  # gagner plutot que bloquer


class TestHelperMethods:
    """Tests pour les methodes helper"""

    def test_get_valid_actions(self):
        """Test actions valides"""
        agent = RuleBasedAgent()
        mask = np.array([1, 0, 1, 0, 0, 1, 0])
        valid = agent._get_valid_actions(mask)
        assert valid == [0, 2, 5]

    def test_get_next_row(self):
        """Test ligne suivante"""
        agent = RuleBasedAgent()
        board = np.zeros((6, 7), dtype=np.int8)

        assert agent._get_next_row(board, 0) == 5

        board[5, 0] = 1
        assert agent._get_next_row(board, 0) == 4

    def test_check_win(self):
        """Test detection victoire"""
        agent = RuleBasedAgent()
        board = np.zeros((6, 7), dtype=np.int8)
        board[5, 0:4] = 1

        assert agent._check_win_from_position(board, 5, 0, 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
