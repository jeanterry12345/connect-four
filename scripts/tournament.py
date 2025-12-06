import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pettingzoo.classic import connect_four_v3
from src.random_agent import RandomAgent
from src.rule_based_agent import RuleBasedAgent


def run_game(agent1, agent2):
    """Run a game."""
    env = connect_four_v3.env()
    env.reset()

    agents = {"player_0": agent1, "player_1": agent2}

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

    env.close()
    return winner


def run_match(agent1, agent2, num_games=50):
    """Run multiple games between 2 agents."""
    stats = {"agent1": 0, "agent2": 0, "draw": 0}

    for i in range(num_games):
        # alternate first player
        if i % 2 == 0:
            first, second = agent1, agent2
            is_first = True
        else:
            first, second = agent2, agent1
            is_first = False

        winner = run_game(first, second)

        if winner == "player_0":
            if is_first:
                stats["agent1"] += 1
            else:
                stats["agent2"] += 1
        elif winner == "player_1":
            if is_first:
                stats["agent2"] += 1
            else:
                stats["agent1"] += 1
        else:
            stats["draw"] += 1

    return stats


def main():
    """Main tournament."""
    print("=" * 50)
    print("Connect Four Tournament")
    print("=" * 50)

    # create agents
    agents = {
        "RandomAgent": RandomAgent(),
        "RuleBasedAgent": RuleBasedAgent(),
    }

    # import advanced agents if available
    try:
        from src.minimax_agent import MinimaxAgent
        agents["MinimaxAgent"] = MinimaxAgent()
    except ImportError:
        pass

    try:
        from src.mcts_agent import MCTSAgent
        agents["MCTSAgent"] = MCTSAgent()
    except ImportError:
        pass

    names = list(agents.keys())
    print(f"\nAgents: {', '.join(names)}")

    # run matches
    results = {}
    for i, name1 in enumerate(names):
        for name2 in names[i+1:]:
            print(f"\n{name1} vs {name2}...")

            agent1 = agents[name1]
            agent2 = agents[name2]
            agent1.reset()
            agent2.reset()

            stats = run_match(agent1, agent2, num_games=50)
            results[(name1, name2)] = stats

            print(f"  {name1}: {stats['agent1']} wins")
            print(f"  {name2}: {stats['agent2']} wins")
            print(f"  Draws: {stats['draw']}")

    # ranking
    print("\n" + "=" * 50)
    print("Final Ranking")
    print("=" * 50)

    total = {name: 0 for name in names}
    for (n1, n2), stats in results.items():
        total[n1] += stats["agent1"]
        total[n2] += stats["agent2"]

    ranking = sorted(total.items(), key=lambda x: x[1], reverse=True)
    for i, (name, wins) in enumerate(ranking, 1):
        print(f"{i}. {name}: {wins} wins")

    print("\nTournament complete!")


if __name__ == "__main__":
    main()
