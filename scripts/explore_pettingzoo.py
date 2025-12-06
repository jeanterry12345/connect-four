import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pettingzoo.classic import connect_four_v3
from src.utils import print_board, get_valid_actions


def main():
    """Explore l'environnement Connect Four."""
    print("=" * 50)
    print("Exploration de l'environnement Connect Four")
    print("=" * 50)

    env = connect_four_v3.env()
    env.reset()

    # Infos de base
    print("\n[Infos de base]")
    print(f"Agents: {env.possible_agents}")

    agent = env.possible_agents[0]
    obs_space = env.observation_space(agent)
    act_space = env.action_space(agent)

    print(f"\n[Espace d'observation]")
    print(f"Shape: {obs_space['observation'].shape}")
    print("(6 lignes, 7 colonnes, 2 canaux)")

    print(f"\n[Espace d'action]")
    print(f"Actions: 0-6 (colonnes)")

    # Demo d'une partie
    print("\n[Demo partie]")
    observation, _, _, _, _ = env.last()
    print_board(observation['observation'])

    moves = [3, 3, 4, 4, 5, 5, 2]
    for move in moves:
        obs, reward, done, trunc, _ = env.last()
        if done or trunc:
            break

        if obs['action_mask'][move] == 1:
            print(f"\n{env.agent_selection} joue colonne {move}")
            env.step(move)
            obs, reward, done, trunc, _ = env.last()
            print_board(obs['observation'])

            if done:
                print(f"Partie terminee! Reward: {reward}")

    env.close()
    print("\nExploration terminee!")


if __name__ == "__main__":
    main()
