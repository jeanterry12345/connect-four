"""Complete test suite"""

import pytest
import time
import numpy as np
from pettingzoo.classic import connect_four_v3

from src.random_agent import RandomAgent
from src.rule_based_agent import RuleBasedAgent
from src.utils import check_winner, get_next_row, get_valid_actions


class TestGameMechanics:
    """Tests for game mechanics"""

    def test_valid_actions(self):
        """Test valid actions"""
        mask = np.array([1, 1, 1, 1, 1, 1, 1])
        assert get_valid_actions(mask) == [0, 1, 2, 3, 4, 5, 6]

        mask = np.array([1, 0, 1, 0, 1, 0, 1])
        assert get_valid_actions(mask) == [0, 2, 4, 6]

    def test_piece_placement(self):
        """Test piece placement"""
        board = np.zeros((6, 7), dtype=np.int8)
        assert get_next_row(board, 0) == 5

        board[5, 0] = 1
        assert get_next_row(board, 0) == 4

    def test_win_detection(self):
        """Test win detection"""
        board = np.zeros((6, 7), dtype=np.int8)
        board[5, 0:4] = 1
        assert check_winner(board) == 1


class TestPerformance:
    """Performance tests"""

    def test_random_agent_speed(self):
        """Test random agent speed"""
        agent = RandomAgent()
        obs = np.zeros((6, 7, 2), dtype=np.int8)
        mask = np.array([1, 1, 1, 1, 1, 1, 1])

        start = time.time()
        for _ in range(100):
            agent.select_action(obs, mask)
        elapsed = time.time() - start

        assert elapsed < 1.0  # 100 decisions in less than 1 second

    def test_rule_agent_speed(self):
        """Test rule agent speed"""
        agent = RuleBasedAgent()
        obs = np.zeros((6, 7, 2), dtype=np.int8)
        mask = np.array([1, 1, 1, 1, 1, 1, 1])

        start = time.time()
        for _ in range(100):
            agent.select_action(obs, mask)
        elapsed = time.time() - start

        assert elapsed < 1.0


class TestStrategy:
    """Strategy tests"""

    def test_rule_vs_random_winrate(self):
        """Test win rate"""
        wins = 0
        games = 20

        for _ in range(games):
            env = connect_four_v3.env()
            env.reset()

            agents = {
                "player_0": RuleBasedAgent(),
                "player_1": RandomAgent()
            }

            for name in env.agent_iter():
                obs, reward, done, trunc, info = env.last()

                if done or trunc:
                    env.step(None)
                    break

                agent = agents[name]
                action = agent.select_action(
                    obs["observation"],
                    obs["action_mask"]
                )
                env.step(action)

            env.close()

            # the last reward indicates who won
            if reward == 1 and name == "player_0":
                wins += 1
            elif reward == -1 and name == "player_1":
                wins += 1

        winrate = wins / games
        assert winrate > 0.5  # must win more than 50%


class TestIntegration:
    """Integration tests"""

    def test_full_game(self):
        """Test full game"""
        env = connect_four_v3.env()
        env.reset()

        agents = {
            "player_0": RandomAgent(seed=42),
            "player_1": RandomAgent(seed=123)
        }

        moves = 0
        for name in env.agent_iter():
            obs, reward, done, trunc, info = env.last()

            if done or trunc:
                env.step(None)
                break

            agent = agents[name]
            action = agent.select_action(
                obs["observation"],
                obs["action_mask"]
            )
            env.step(action)
            moves += 1

        env.close()
        assert moves > 0
        assert moves <= 42


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
