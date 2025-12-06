"""Tests for rule-based agent"""

import pytest
import numpy as np
from src.rule_based_agent import RuleBasedAgent


class TestRuleBasedAgent:
    """Tests for RuleBasedAgent"""

    def test_init(self):
        """Test initialization"""
        agent = RuleBasedAgent()
        assert agent.name == "RuleBasedAgent"

    def test_wins_when_possible(self):
        """Test horizontal win"""
        agent = RuleBasedAgent()

        obs = np.zeros((6, 7, 2), dtype=np.int8)
        obs[5, 0, 0] = 1
        obs[5, 1, 0] = 1
        obs[5, 2, 0] = 1

        mask = np.array([1, 1, 1, 1, 1, 1, 1])
        action = agent.select_action(obs, mask)

        assert action == 3

    def test_blocks_opponent(self):
        """Test blocking opponent"""
        agent = RuleBasedAgent()

        obs = np.zeros((6, 7, 2), dtype=np.int8)
        obs[5, 0, 1] = 1
        obs[5, 1, 1] = 1
        obs[5, 2, 1] = 1

        mask = np.array([1, 1, 1, 1, 1, 1, 1])
        action = agent.select_action(obs, mask)

        assert action == 3

    def test_prefers_center(self):
        """Test center preference"""
        agent = RuleBasedAgent()

        obs = np.zeros((6, 7, 2), dtype=np.int8)
        mask = np.array([1, 1, 1, 1, 1, 1, 1])

        action = agent.select_action(obs, mask)
        assert action == 3

    def test_win_priority_over_block(self):
        """Test win priority over blocking"""
        agent = RuleBasedAgent()

        obs = np.zeros((6, 7, 2), dtype=np.int8)
        # me: 3 in a row horizontal
        obs[5, 0, 0] = 1
        obs[5, 1, 0] = 1
        obs[5, 2, 0] = 1
        # opponent: 3 in a row vertical
        obs[5, 6, 1] = 1
        obs[4, 6, 1] = 1
        obs[3, 6, 1] = 1

        mask = np.array([1, 1, 1, 1, 1, 1, 1])
        action = agent.select_action(obs, mask)

        assert action == 3  # win rather than block


class TestHelperMethods:
    """Tests for helper methods"""

    def test_get_valid_actions(self):
        """Test valid actions"""
        agent = RuleBasedAgent()
        mask = np.array([1, 0, 1, 0, 0, 1, 0])
        valid = agent._get_valid_actions(mask)
        assert valid == [0, 2, 5]

    def test_get_next_row(self):
        """Test next row"""
        agent = RuleBasedAgent()
        board = np.zeros((6, 7), dtype=np.int8)

        assert agent._get_next_row(board, 0) == 5

        board[5, 0] = 1
        assert agent._get_next_row(board, 0) == 4

    def test_check_win(self):
        """Test win detection"""
        agent = RuleBasedAgent()
        board = np.zeros((6, 7), dtype=np.int8)
        board[5, 0:4] = 1

        assert agent._check_win_from_position(board, 5, 0, 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
