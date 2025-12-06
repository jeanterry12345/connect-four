"""Tests pour l'agent aleatoire"""

import pytest
import numpy as np
from src.random_agent import RandomAgent


class TestRandomAgent:
    """Tests basiques pour RandomAgent"""

    def test_init(self):
        """Test initialisation"""
        agent = RandomAgent()
        assert agent.name == "RandomAgent"

    def test_select_valid_action(self):
        """Test selection d'une action valide"""
        agent = RandomAgent()
        obs = np.zeros((6, 7, 2), dtype=np.int8)
        mask = np.array([1, 1, 1, 1, 1, 1, 1])

        action = agent.select_action(obs, mask)
        assert 0 <= action <= 6

    def test_respects_mask(self):
        """Test que le masque est respecte"""
        agent = RandomAgent()
        obs = np.zeros((6, 7, 2), dtype=np.int8)
        mask = np.array([0, 1, 0, 1, 0, 1, 0])

        for _ in range(50):
            action = agent.select_action(obs, mask)
            assert action in [1, 3, 5]

    def test_no_valid_action_error(self):
        """Test erreur si pas d'action valide"""
        agent = RandomAgent()
        obs = np.zeros((6, 7, 2), dtype=np.int8)
        mask = np.array([0, 0, 0, 0, 0, 0, 0])

        with pytest.raises(ValueError):
            agent.select_action(obs, mask)

    def test_seed_reproducibility(self):
        """Test reproductibilite avec seed"""
        obs = np.zeros((6, 7, 2), dtype=np.int8)
        mask = np.array([1, 1, 1, 1, 1, 1, 1])

        agent1 = RandomAgent(seed=42)
        agent2 = RandomAgent(seed=42)

        actions1 = [agent1.select_action(obs, mask) for _ in range(10)]
        actions2 = [agent2.select_action(obs, mask) for _ in range(10)]

        assert actions1 == actions2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
