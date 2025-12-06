import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pettingzoo.classic import connect_four_v3
from src.random_agent import RandomAgent


def run_game(agent1, agent2):
    """Run a game."""
    env = connect_four_v3.env()
    env.reset()

    agents = {"player_0": agent1, "player_1": agent2}
    moves = 0

    for name in env.agent_iter():
        obs, reward, done, trunc, _ = env.last()

        if done or trunc:
            if reward == 1:
                winner = name
            elif reward == -1:
                winner = "player_1" if name == "player_0" else "player_0"
            else:
                winner = "draw"
            env.step(None)
            break

        agent = agents[name]
        action = agent.select_action(obs["observation"], obs["action_mask"])
        env.step(action)
        moves += 1

    env.close()
    return winner, moves


def main():
    """Main analysis."""
    print("=" * 50)
    print("Random Agents Analysis")
    print("=" * 50)

    num_games = 100
    agent1 = RandomAgent(name="Random1")
    agent2 = RandomAgent(name="Random2")

    stats = {"player_0": 0, "player_1": 0, "draw": 0}
    all_moves = []

    start = time.time()

    for i in range(num_games):
        winner, moves = run_game(agent1, agent2)
        stats[winner] += 1
        all_moves.append(moves)

        if (i + 1) % 20 == 0:
            print(f"{i + 1}/{num_games} games...")

    elapsed = time.time() - start

    # Results
    print(f"\n[Results over {num_games} games]")
    print(f"Player 1 wins: {stats['player_0']} ({stats['player_0']/num_games*100:.1f}%)")
    print(f"Player 2 wins: {stats['player_1']} ({stats['player_1']/num_games*100:.1f}%)")
    print(f"Draws: {stats['draw']} ({stats['draw']/num_games*100:.1f}%)")

    print(f"\n[Move Statistics]")
    print(f"Average: {np.mean(all_moves):.1f}")
    print(f"Min: {np.min(all_moves)}, Max: {np.max(all_moves)}")

    print(f"\n[Time]")
    print(f"Total: {elapsed:.2f}s")
    print(f"Per game: {elapsed/num_games*1000:.2f}ms")

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
