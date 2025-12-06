"""Tests for random agent"""

import pytest
import numpy as np
from src.random_agent import RandomAgent


class TestRandomAgent:
    """Basic tests for RandomAgent"""

    def test_init(self):
        """Test initialization"""
        agent = RandomAgent()
        assert agent.name == "RandomAgent"

    def test_select_valid_action(self):
        """Test selection of a valid action"""
        agent = RandomAgent()
        obs = np.zeros((6, 7, 2), dtype=np.int8)
        mask = np.array([1, 1, 1, 1, 1, 1, 1])

        action = agent.select_action(obs, mask)
        assert 0 <= action <= 6

    def test_respects_mask(self):
        """Test that mask is respected"""
        agent = RandomAgent()
        obs = np.zeros((6, 7, 2), dtype=np.int8)
        mask = np.array([0, 1, 0, 1, 0, 1, 0])

        for _ in range(50):
            action = agent.select_action(obs, mask)
            assert action in [1, 3, 5]

    def test_no_valid_action_error(self):
        """Test error when no valid action"""
        agent = RandomAgent()
        obs = np.zeros((6, 7, 2), dtype=np.int8)
        mask = np.array([0, 0, 0, 0, 0, 0, 0])

        with pytest.raises(ValueError):
            agent.select_action(obs, mask)

    def test_seed_reproducibility(self):
        """Test reproducibility with seed"""
        obs = np.zeros((6, 7, 2), dtype=np.int8)
        mask = np.array([1, 1, 1, 1, 1, 1, 1])

        agent1 = RandomAgent(seed=42)
        agent2 = RandomAgent(seed=42)

        actions1 = [agent1.select_action(obs, mask) for _ in range(10)]
        actions2 = [agent2.select_action(obs, mask) for _ in range(10)]

        assert actions1 == actions2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
