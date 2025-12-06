import time


class Agent:

    def __init__(self, env, player_name=None):
        self.env = env
        self.player_name = player_name
        self.time_limit = 2.5  # limite de temps avec marge
        self.max_depth = 4
        self.start_time = 0

    def choose_action(
        self,
        observation,
        reward=0.0,
        terminated=False,
        truncated=False,
        info=None,
        action_mask=None,
    ):
        """Choisit la meilleure action."""
        _ = reward, terminated, truncated, info

        self.start_time = time.time()
        board = observation
        valid = [i for i, v in enumerate(action_mask) if v == 1]

        if not valid:
            return 0
        if len(valid) == 1:
            return valid[0]

        # verifier si on peut gagner
        for col in valid:
            if self._can_win(board, col, 0):
                return col

        # bloquer l'adversaire
        for col in valid:
            if self._can_win(board, col, 1):
                return col

        # sinon minimax
        return self._search(board, valid)

    def _can_win(self, board, col, player):
        """Verifie si jouer dans col fait gagner."""
        row = self._get_row(board, col)
        if row is None:
            return False

        # directions: horizontal, vertical, 2 diagonales
        dirs = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for dr, dc in dirs:
            count = 1
            # avant
            r, c = row + dr, col + dc
            while 0 <= r < 6 and 0 <= c < 7 and board[r, c, player] == 1:
                count += 1
                r += dr
                c += dc
            # arriere
            r, c = row - dr, col - dc
            while 0 <= r < 6 and 0 <= c < 7 and board[r, c, player] == 1:
                count += 1
                r -= dr
                c -= dc

            if count >= 4:
                return True
        return False

    def _search(self, board, valid):
        """Recherche minimax simple."""
        best = valid[0]
        best_score = -99999

        # preferer le centre
        order = [3, 2, 4, 1, 5, 0, 6]
        sorted_valid = sorted(valid, key=lambda x: order.index(x) if x in order else 7)

        for col in sorted_valid:
            if time.time() - self.start_time > self.time_limit:
                break

            new_board = board.copy()
            row = self._get_row(board, col)
            if row is not None:
                new_board[row, col, 0] = 1

            score = -self._minimax(new_board, self.max_depth - 1, -99999, 99999, 1)

            if score > best_score:
                best_score = score
                best = col

        return best

    def _minimax(self, board, depth, alpha, beta, player):
        """Minimax avec alpha-beta."""
        if time.time() - self.start_time > self.time_limit:
            return 0

        # check winner
        if self._has_won(board, 0):
            return 10000 + depth
        if self._has_won(board, 1):
            return -10000 - depth

        valid = [c for c in range(7) if board[0, c, 0] == 0 and board[0, c, 1] == 0]

        if not valid or depth <= 0:
            return self._eval(board)

        best = -99999
        for col in valid:
            new_board = board.copy()
            row = self._get_row(board, col)
            if row is not None:
                new_board[row, col, player] = 1

            score = -self._minimax(new_board, depth - 1, -beta, -alpha, 1 - player)
            best = max(best, score)
            alpha = max(alpha, score)

            if alpha >= beta:
                break

        return best

    def _eval(self, board):
        """Evalue la position."""
        score = 0

        # bonus colonne centrale
        for row in range(6):
            if board[row, 3, 0] == 1:
                score += 3
            elif board[row, 3, 1] == 1:
                score -= 3

        # compter les alignements
        score += self._count_score(board, 0)
        score -= self._count_score(board, 1)

        return score

    def _count_score(self, board, player):
        """Compte le score des alignements."""
        score = 0
        opp = 1 - player

        # horizontal
        for row in range(6):
            for col in range(4):
                mine = sum(board[row, col+i, player] for i in range(4))
                theirs = sum(board[row, col+i, opp] for i in range(4))
                if theirs == 0:
                    if mine == 3:
                        score += 5
                    elif mine == 2:
                        score += 2

        # vertical
        for row in range(3):
            for col in range(7):
                mine = sum(board[row+i, col, player] for i in range(4))
                theirs = sum(board[row+i, col, opp] for i in range(4))
                if theirs == 0:
                    if mine == 3:
                        score += 5
                    elif mine == 2:
                        score += 2

        # diagonale principale
        for row in range(3):
            for col in range(4):
                mine = sum(board[row+i, col+i, player] for i in range(4))
                theirs = sum(board[row+i, col+i, opp] for i in range(4))
                if theirs == 0:
                    if mine == 3:
                        score += 5
                    elif mine == 2:
                        score += 2

        # anti-diagonale
        for row in range(3):
            for col in range(3, 7):
                mine = sum(board[row+i, col-i, player] for i in range(4))
                theirs = sum(board[row+i, col-i, opp] for i in range(4))
                if theirs == 0:
                    if mine == 3:
                        score += 5
                    elif mine == 2:
                        score += 2

        return score

    def _has_won(self, board, player):
        """Verifie si le joueur a gagne."""
        # horizontal
        for row in range(6):
            for col in range(4):
                if all(board[row, col+i, player] == 1 for i in range(4)):
                    return True
        # vertical
        for row in range(3):
            for col in range(7):
                if all(board[row+i, col, player] == 1 for i in range(4)):
                    return True
        # diagonales
        for row in range(3):
            for col in range(4):
                if all(board[row+i, col+i, player] == 1 for i in range(4)):
                    return True
        for row in range(3):
            for col in range(3, 7):
                if all(board[row+i, col-i, player] == 1 for i in range(4)):
                    return True
        return False

    def _get_row(self, board, col):
        """Trouve la ligne ou le pion tombe."""
        for row in range(5, -1, -1):
            if board[row, col, 0] == 0 and board[row, col, 1] == 0:
                return row
        return None
