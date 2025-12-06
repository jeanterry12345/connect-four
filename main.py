import sys
import argparse
from pettingzoo.classic import connect_four_v3

from src.random_agent import RandomAgent
from src.rule_based_agent import RuleBasedAgent
from src.utils import print_board


def create_agent(agent_type, player_id):
    """Cree un agent selon le type."""
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
        raise ValueError(f"Type d'agent inconnu: {agent_type}")


def get_human_action(action_mask):
    """Obtient l'action du joueur humain."""
    valid = [i for i, v in enumerate(action_mask) if v == 1]

    while True:
        try:
            action = int(input(f"Choisir une colonne {valid}: "))
            if action in valid:
                return action
            print(f"Choix invalide, choisir parmi {valid}")
        except ValueError:
            print("Entrer un nombre valide")
        except KeyboardInterrupt:
            print("\nPartie interrompue")
            sys.exit(0)


def run_game(agent1, agent2, verbose=True):
    """Execute une partie."""
    env = connect_four_v3.env()
    env.reset()

    agents = {"player_0": agent1, "player_1": agent2}

    if verbose:
        print("\nPartie lancee!")
        print(f"Joueur 0 (X): {agent1.name if agent1 else 'Humain'}")
        print(f"Joueur 1 (O): {agent2.name if agent2 else 'Humain'}")

    move = 0
    for name in env.agent_iter():
        obs, reward, done, trunc, _ = env.last()

        if done or trunc:
            if verbose:
                print("\nPlateau final:")
                print_board(obs["observation"])

                if reward == 1:
                    print(f"\n{name} gagne!")
                elif reward == -1:
                    winner = "player_1" if name == "player_0" else "player_0"
                    print(f"\n{winner} gagne!")
                else:
                    print("\nMatch nul!")

            env.step(None)
            break

        if verbose:
            print(f"\nCoup {move + 1}")
            print_board(obs["observation"])
            print(f"Tour: {name}")

        agent = agents[name]
        if agent is None:
            action = get_human_action(obs["action_mask"])
        else:
            action = agent.select_action(obs["observation"], obs["action_mask"])
            if verbose:
                print(f"{agent.name} joue colonne {action}")

        env.step(action)
        move += 1

    env.close()


def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(description="Puissance 4")
    parser.add_argument("--player1", type=str, default="rule",
                        choices=["random", "rule", "minimax", "mcts", "human"])
    parser.add_argument("--player2", type=str, default="random",
                        choices=["random", "rule", "minimax", "mcts", "human"])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--games", type=int, default=1)

    args = parser.parse_args()

    print("\n" + "=" * 40)
    print("Puissance 4")
    print("=" * 40)

    agent1 = create_agent(args.player1, "player_0")
    agent2 = create_agent(args.player2, "player_1")

    if args.games == 1:
        run_game(agent1, agent2, verbose=not args.quiet)
    else:
        for i in range(args.games):
            print(f"\nPartie {i + 1}...")
            run_game(agent1, agent2, verbose=False)
        print(f"\n{args.games} parties terminees!")


if __name__ == "__main__":
    main()
