class Othello:
    def __init__(self):
        self.board = self.create_board()
        self.current_player = 'B'  # B for Black, W for White

    def create_board(self):
        board = [[' ' for _ in range(8)] for _ in range(8)]
        board[3][3] = 'W'
        board[3][4] = 'B'
        board[4][3] = 'B'
        board[4][4] = 'W'
        return board

    def print_board(self):
        for row in self.board:
            print(' '.join(row))
        print()

    def is_valid_move(self, row, col, player):
        if self.board[row][col] != ' ':
            return False
        opponent = 'W' if player == 'B' else 'B'
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
        for dr, dc in directions:
            r, c = row + dr, col + dc
            if 0 <= r < 8 and 0 <= c < 8 and self.board[r][c] == opponent:
                while 0 <= r < 8 and 0 <= c < 8:
                    r += dr
                    c += dc
                    if 0 <= r < 8 and 0 <= c < 8 and self.board[r][c] == player:
                        return True
                    if not (0 <= r < 8 and 0 <= c < 8) or self.board[r][c] == ' ':
                        break
        return False

    def make_move(self, row, col):
        if not self.is_valid_move(row, col, self.current_player):
            return False
        self.board[row][col] = self.current_player
        self.flip_discs(row, col)
        self.current_player = 'W' if self.current_player == 'B' else 'B'
        return True

    def flip_discs(self, row, col):
        opponent = 'W' if self.current_player == 'B' else 'B'
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
        for dr, dc in directions:
            r, c = row + dr, col + dc
            discs_to_flip = []
            while 0 <= r < 8 and 0 <= c < 8 and self.board[r][c] == opponent:
                discs_to_flip.append((r, c))
                r += dr
                c += dc
            if 0 <= r < 8 and 0 <= c < 8 and self.board[r][c] == self.current_player:
                for rr, cc in discs_to_flip:
                    self.board[rr][cc] = self.current_player

    def has_valid_moves(self):
        for row in range(8):
            for col in range(8):
                if self.is_valid_move(row, col, self.current_player):
                    return True
        return False

    def play_game(self):
        self.print_board()
        while self.has_valid_moves():
            row, col = map(int, input(f"Player {self.current_player}, enter your move (row col): ").split())
            if self.make_move(row, col):
                self.print_board()
            else:
                print("Invalid move. Try again.")
        print("Game over!")


if __name__ == "__main__":
    game = Othello()
    game.play_game()
