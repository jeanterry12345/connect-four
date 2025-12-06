import sys
import argparse
from pettingzoo.classic import connect_four_v3

from src.random_agent import RandomAgent
from src.rule_based_agent import RuleBasedAgent
from src.utils import print_board


def create_agent(agent_type, player_id):
    """Create an agent based on type."""
    if agent_type == "random":
        return RandomAgent(name="RandomAgent", player_id=player_id)
    elif agent_type == "rule":
        return RuleBasedAgent(name="RuleBasedAgent", player_id=player_id)
    elif agent_type == "minimax":
        from src.minimax_agent import MinimaxAgent
        return MinimaxAgent(name="MinimaxAgent", player_id=player_id)
    elif agent_type == "mcts":
        from src.mcts_agent import MCTSAgent
        return MCTSAgent(name="MCTSAgent", player_id=player_id)
    elif agent_type == "human":
        return None
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def get_human_action(action_mask):
    """Get action from human player."""
    valid = [i for i, v in enumerate(action_mask) if v == 1]

    while True:
        try:
            action = int(input(f"Choose a column {valid}: "))
            if action in valid:
                return action
            print(f"Invalid choice, choose from {valid}")
        except ValueError:
            print("Enter a valid number")
        except KeyboardInterrupt:
            print("\nGame interrupted")
            sys.exit(0)


def run_game(agent1, agent2, verbose=True):
    """Run a game."""
    env = connect_four_v3.env()
    env.reset()

    agents = {"player_0": agent1, "player_1": agent2}

    if verbose:
        print("\nGame started!")
        print(f"Player 0 (X): {agent1.name if agent1 else 'Human'}")
        print(f"Player 1 (O): {agent2.name if agent2 else 'Human'}")

    move = 0
    for name in env.agent_iter():
        obs, reward, done, trunc, _ = env.last()

        if done or trunc:
            if verbose:
                print("\nFinal board:")
                print_board(obs["observation"])

                if reward == 1:
                    print(f"\n{name} wins!")
                elif reward == -1:
                    winner = "player_1" if name == "player_0" else "player_0"
                    print(f"\n{winner} wins!")
                else:
                    print("\nDraw!")

            env.step(None)
            break

        if verbose:
            print(f"\nMove {move + 1}")
            print_board(obs["observation"])
            print(f"Turn: {name}")

        agent = agents[name]
        if agent is None:
            action = get_human_action(obs["action_mask"])
        else:
            action = agent.select_action(obs["observation"], obs["action_mask"])
            if verbose:
                print(f"{agent.name} plays column {action}")

        env.step(action)
        move += 1

    env.close()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Connect Four")
    parser.add_argument("--player1", type=str, default="rule",
                        choices=["random", "rule", "minimax", "mcts", "human"])
    parser.add_argument("--player2", type=str, default="random",
                        choices=["random", "rule", "minimax", "mcts", "human"])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--games", type=int, default=1)

    args = parser.parse_args()

    print("\n" + "=" * 40)
    print("Connect Four")
    print("=" * 40)

    agent1 = create_agent(args.player1, "player_0")
    agent2 = create_agent(args.player2, "player_1")

    if args.games == 1:
        run_game(agent1, agent2, verbose=not args.quiet)
    else:
        for i in range(args.games):
            print(f"\nGame {i + 1}...")
            run_game(agent1, agent2, verbose=False)
        print(f"\n{args.games} games completed!")


if __name__ == "__main__":
    main()
