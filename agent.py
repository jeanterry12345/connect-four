import time


class Agent:
    """Connect Four agent using Minimax with Alpha-Beta pruning."""

    def __init__(self, env, player_name=None):
        self.env = env
        self.player_name = player_name
        self.time_limit = 2.5  # time limit with margin
        self.max_depth = 5
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
        """Choose the best action using minimax strategy."""
        self.start_time = time.time()
        board = observation
        valid = [i for i, v in enumerate(action_mask) if v == 1]

        if not valid:
            return 0
        if len(valid) == 1:
            return valid[0]

        # 1. Win immediately if possible
        for col in valid:
            if self._is_winning_move(board, col, 0):
                return col

        # 2. Block opponent's winning move
        for col in valid:
            if self._is_winning_move(board, col, 1):
                return col

        # 3. Check for double threat (create two winning opportunities)
        for col in valid:
            if self._creates_double_threat(board, col, 0):
                return col

        # 4. Block opponent's double threat
        for col in valid:
            if self._creates_double_threat(board, col, 1):
                return col

        # 5. Avoid moves that give opponent a winning move on top
        safe_moves = []
        for col in valid:
            row = self._get_row(board, col)
            if row is not None and row > 0:
                # Check if opponent can win by playing on top
                if not self._is_winning_move(board, col, 1, row - 1):
                    safe_moves.append(col)
            elif row == 0:
                safe_moves.append(col)

        if not safe_moves:
            safe_moves = valid

        # 6. Use minimax for remaining decisions
        return self._search(board, safe_moves)

    def _is_winning_move(self, board, col, player, forced_row=None):
        """Check if playing in col wins the game."""
        if forced_row is not None:
            row = forced_row
        else:
            row = self._get_row(board, col)

        if row is None:
            return False

        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for dr, dc in directions:
            count = 1
            # forward
            r, c = row + dr, col + dc
            while 0 <= r < 6 and 0 <= c < 7 and board[r, c, player] == 1:
                count += 1
                r += dr
                c += dc
            # backward
            r, c = row - dr, col - dc
            while 0 <= r < 6 and 0 <= c < 7 and board[r, c, player] == 1:
                count += 1
                r -= dr
                c -= dc

            if count >= 4:
                return True
        return False

    def _creates_double_threat(self, board, col, player):
        """Check if move creates two or more winning opportunities."""
        row = self._get_row(board, col)
        if row is None:
            return False

        # Simulate placing the piece
        board[row, col, player] = 1
        threats = 0

        # Check all columns for potential winning moves
        for c in range(7):
            r = self._get_row(board, c)
            if r is not None:
                if self._is_winning_move(board, c, player):
                    threats += 1

        # Undo the move
        board[row, col, player] = 0

        return threats >= 2

    def _search(self, board, valid):
        """Minimax search with move ordering."""
        best = valid[0]
        best_score = -99999

        # Prefer center columns (better strategic position)
        order = [3, 2, 4, 1, 5, 0, 6]
        sorted_valid = sorted(valid, key=lambda x: order.index(x) if x in order else 7)

        for col in sorted_valid:
            if time.time() - self.start_time > self.time_limit:
                break

            row = self._get_row(board, col)
            if row is None:
                continue

            board[row, col, 0] = 1
            score = -self._minimax(board, self.max_depth - 1, -99999, 99999, 1)
            board[row, col, 0] = 0

            if score > best_score:
                best_score = score
                best = col

        return best

    def _minimax(self, board, depth, alpha, beta, player):
        """Minimax with alpha-beta pruning."""
        if time.time() - self.start_time > self.time_limit:
            return 0

        # Check for winner
        if self._has_won(board, 0):
            return 10000 + depth
        if self._has_won(board, 1):
            return -10000 - depth

        valid = [c for c in range(7) if board[0, c, 0] == 0 and board[0, c, 1] == 0]

        if not valid or depth <= 0:
            return self._evaluate(board)

        # Move ordering for better pruning
        order = [3, 2, 4, 1, 5, 0, 6]
        sorted_valid = sorted(valid, key=lambda x: order.index(x) if x in order else 7)

        best = -99999
        for col in sorted_valid:
            row = self._get_row(board, col)
            if row is None:
                continue

            board[row, col, player] = 1
            score = -self._minimax(board, depth - 1, -beta, -alpha, 1 - player)
            board[row, col, player] = 0

            best = max(best, score)
            alpha = max(alpha, score)

            if alpha >= beta:
                break

        return best

    def _evaluate(self, board):
        """Evaluate board position."""
        score = 0

        # Center column bonus (strong strategic position)
        center_weight = [0, 1, 2, 3, 2, 1, 0]
        for row in range(6):
            for col in range(7):
                if board[row, col, 0] == 1:
                    score += center_weight[col]
                elif board[row, col, 1] == 1:
                    score -= center_weight[col]

        # Count potential alignments
        score += self._count_alignments(board, 0)
        score -= self._count_alignments(board, 1)

        return score

    def _count_alignments(self, board, player):
        """Count alignment scores for potential wins."""
        score = 0
        opp = 1 - player

        # All directions: horizontal, vertical, diagonal, anti-diagonal
        directions = [
            (0, 1, 6, 4),   # horizontal
            (1, 0, 3, 7),   # vertical
            (1, 1, 3, 4),   # diagonal
            (1, -1, 3, 4),  # anti-diagonal (start from col 3)
        ]

        # Horizontal
        for row in range(6):
            for col in range(4):
                window = [board[row, col + i, player] for i in range(4)]
                opp_window = [board[row, col + i, opp] for i in range(4)]
                score += self._score_window(window, opp_window)

        # Vertical
        for row in range(3):
            for col in range(7):
                window = [board[row + i, col, player] for i in range(4)]
                opp_window = [board[row + i, col, opp] for i in range(4)]
                score += self._score_window(window, opp_window)

        # Diagonal
        for row in range(3):
            for col in range(4):
                window = [board[row + i, col + i, player] for i in range(4)]
                opp_window = [board[row + i, col + i, opp] for i in range(4)]
                score += self._score_window(window, opp_window)

        # Anti-diagonal
        for row in range(3):
            for col in range(3, 7):
                window = [board[row + i, col - i, player] for i in range(4)]
                opp_window = [board[row + i, col - i, opp] for i in range(4)]
                score += self._score_window(window, opp_window)

        return score

    def _score_window(self, window, opp_window):
        """Score a window of 4 positions."""
        mine = sum(window)
        theirs = sum(opp_window)

        if theirs > 0:
            return 0  # blocked, no potential

        if mine == 3:
            return 50
        elif mine == 2:
            return 10
        elif mine == 1:
            return 1
        return 0

    def _has_won(self, board, player):
        """Check if player has won."""
        # Horizontal
        for row in range(6):
            for col in range(4):
                if all(board[row, col + i, player] == 1 for i in range(4)):
                    return True
        # Vertical
        for row in range(3):
            for col in range(7):
                if all(board[row + i, col, player] == 1 for i in range(4)):
                    return True
        # Diagonal
        for row in range(3):
            for col in range(4):
                if all(board[row + i, col + i, player] == 1 for i in range(4)):
                    return True
        # Anti-diagonal
        for row in range(3):
            for col in range(3, 7):
                if all(board[row + i, col - i, player] == 1 for i in range(4)):
                    return True
        return False

    def _get_row(self, board, col):
        """Find the row where piece will land."""
        for row in range(5, -1, -1):
            if board[row, col, 0] == 0 and board[row, col, 1] == 0:
                return row
        return None
