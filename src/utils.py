import numpy as np


def print_board(observation, player_names=("Player1", "Player2")):
    """Display the game board."""
    print("\n   0  1  2  3  4  5  6")
    print("  " + "-" * 21)

    for row in range(6):
        print(f"{row} |", end="")
        for col in range(7):
            if observation[row, col, 0] == 1:
                print(" X ", end="")
            elif observation[row, col, 1] == 1:
                print(" O ", end="")
            else:
                print(" . ", end="")
        print("|")

    print("  " + "-" * 21)
    print(f"  X = {player_names[0]}, O = {player_names[1]}")


def print_board_simple(board):
    """Display a simple board (6x7)."""
    symbols = {0: '.', 1: 'X', 2: 'O'}

    print("\n   0  1  2  3  4  5  6")
    print("  " + "-" * 21)

    for row in range(6):
        print(f"{row} |", end="")
        for col in range(7):
            s = symbols.get(board[row, col], '?')
            print(f" {s} ", end="")
        print("|")

    print("  " + "-" * 21)


def get_valid_actions(action_mask):
    """Return playable columns."""
    return [i for i, v in enumerate(action_mask) if v == 1]


def get_next_row(board, col):
    """Find where the piece will land."""
    for row in range(5, -1, -1):
        if board[row, col] == 0:
            return row
    return -1


def check_winner(board):
    """Check if there is a winner. Returns 1, 2, or 0."""
    # horizontal
    for row in range(6):
        for col in range(4):
            player = board[row, col]
            if player != 0:
                if all(board[row, col + i] == player for i in range(4)):
                    return player

    # vertical
    for row in range(3):
        for col in range(7):
            player = board[row, col]
            if player != 0:
                if all(board[row + i, col] == player for i in range(4)):
                    return player

    # diagonal
    for row in range(3):
        for col in range(4):
            player = board[row, col]
            if player != 0:
                if all(board[row + i, col + i] == player for i in range(4)):
                    return player

    # anti-diagonal
    for row in range(3):
        for col in range(3, 7):
            player = board[row, col]
            if player != 0:
                if all(board[row + i, col - i] == player for i in range(4)):
                    return player

    return 0


def is_board_full(board):
    """Check if the board is full."""
    return np.all(board[0, :] != 0)


def observation_to_board(observation, current_player):
    """Convert PettingZoo observation to board."""
    board = np.zeros((6, 7), dtype=np.int8)
    board[observation[:, :, 0] == 1] = current_player
    opponent = 3 - current_player
    board[observation[:, :, 1] == 1] = opponent
    return board


def board_to_observation(board, current_player):
    """Convert board to PettingZoo observation."""
    obs = np.zeros((6, 7, 2), dtype=np.int8)
    opponent = 3 - current_player
    obs[:, :, 0] = (board == current_player).astype(np.int8)
    obs[:, :, 1] = (board == opponent).astype(np.int8)
    return obs


def evaluate_position(board, player):
    """Evaluate board position for a player."""
    opponent = 3 - player

    winner = check_winner(board)
    if winner == player:
        return 100000
    if winner == opponent:
        return -100000

    score = 0

    # center bonus
    for row in range(6):
        if board[row, 3] == player:
            score += 3
        elif board[row, 3] == opponent:
            score -= 3

    return score
