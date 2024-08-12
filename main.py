import json

class Othello:
    def __init__(self, size=8):
        self.size = size
        self.board = self.create_board()
        self.current_player = 'B'  # B for Black, W for White
        self.score = {'B': 2, 'W': 2}  # Initial score with 2 pieces each
        self.history = []

    def create_board(self):
        board = [[' ' for _ in range(self.size)] for _ in range(self.size)]
        mid = self.size // 2
        board[mid-1][mid-1] = 'W'
        board[mid-1][mid] = 'B'
        board[mid][mid-1] = 'B'
        board[mid][mid] = 'W'
        return board

    def print_board(self):
        print("  " + " ".join(map(str, range(self.size))))
        for i, row in enumerate(self.board):
            print(str(i) + " " + ' '.join(row))
        print(f"Score -> Black: {self.score['B']}, White: {self.score['W']}")
        print()

    def is_valid_move(self, row, col, player):
        if self.board[row][col] != ' ':
            return False
        opponent = 'W' if player == 'B' else 'B'
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
        for dr, dc in directions:
            r, c = row + dr, col + dc
            if 0 <= r < self.size and 0 <= c < self.size and self.board[r][c] == opponent:
                while 0 <= r < self.size and 0 <= c < self.size:
                    r += dr
                    c += dc
                    if 0 <= r < self.size and 0 <= c < self.size and self.board[r][c] == player:
                        return True
                    if not (0 <= r < self.size and 0 <= c < self.size) or self.board[r][c] == ' ':
                        break
        return False

    def make_move(self, row, col):
        if not self.is_valid_move(row, col, self.current_player):
            return False
        self.history.append((self.board_copy(), self.score.copy(), self.current_player))
        self.board[row][col] = self.current_player
        self.flip_discs(row, col)
        self.update_score()
        self.current_player = 'W' if self.current_player == 'B' else 'B'
        return True

    def flip_discs(self, row, col):
        opponent = 'W' if self.current_player == 'B' else 'B'
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
        for dr, dc in directions:
            r, c = row + dr, col + dc
            discs_to_flip = []
            while 0 <= r < self.size and 0 <= c < self.size and self.board[r][c] == opponent:
                discs_to_flip.append((r, c))
                r += dr
                c += dc
            if 0 <= r < self.size and 0 <= c < self.size and self.board[r][c] == self.current_player:
                for rr, cc in discs_to_flip:
                    self.board[rr][cc] = self.current_player

    def update_score(self):
        self.score = {'B': 0, 'W': 0}
        for row in self.board:
            for cell in row:
                if cell in self.score:
                    self.score[cell] += 1

    def has_valid_moves(self):
        for row in range(self.size):
            for col in range(self.size):
                if self.is_valid_move(row, col, self.current_player):
                    return True
        return False

    def get_valid_moves(self):
        valid_moves = []
        for row in range(self.size):
            for col in range(self.size):
                if self.is_valid_move(row, col, self.current_player):
                    valid_moves.append((row, col))
        return valid_moves

    def suggest_moves(self):
        valid_moves = self.get_valid_moves()
        for row, col in valid_moves:
            print(f"Suggested move: ({row}, {col})")
        print()

    def undo_move(self):
        if not self.history:
            print("No moves to undo.")
            return
        self.board, self.score, self.current_player = self.history.pop()
        print("Move undone.")
        self.print_board()

    def save_game(self, filename):
        game_state = {
            'size': self.size,
            'board': self.board,
            'score': self.score,
            'current_player': self.current_player,
            'history': self.history
        }
        with open(filename, 'w') as f:
            json.dump(game_state, f)
        print(f"Game saved to {filename}.")

    def load_game(self, filename):
        try:
            with open(filename, 'r') as f:
                game_state = json.load(f)
            self.size = game_state['size']
            self.board = game_state['board']
            self.score = game_state['score']
            self.current_player = game_state['current_player']
            self.history = game_state['history']
            print(f"Game loaded from {filename}.")
            self.print_board()
        except FileNotFoundError:
            print(f"File {filename} not found.")

    def board_copy(self):
        return [row.copy() for row in self.board]

    def play_game(self):
        self.print_board()
        while self.has_valid_moves():
            self.suggest_moves()
            action = input(f"Player {self.current_player}, enter your move (row col) or 'undo', 'save', or 'load': ")
            if action == 'undo':
                self.undo_move()
            elif action == 'save':
                filename = input("Enter filename to save: ")
                self.save_game(filename)
            elif action == 'load':
                filename = input("Enter filename to load: ")
                self.load_game(filename)
            else:
                row, col = map(int, action.split())
                if self.make_move(row, col):
                    self.print_board()
                else:
                    print("Invalid move. Try again.")
        self.declare_winner()

    def declare_winner(self):
        if self.score['B'] > self.score['W']:
            print("Black wins!")
        elif self.score['W'] > self.score['B']:
            print("White wins!")
        else:
            print("It's a tie!")
        print("Final Score -> Black: {}, White: {}".format(self.score['B'], self.score['W']))


if __name__ == "__main__":
    size = int(input("Enter board size (4-16): "))
    if 4 <= size <= 16:
        game = Othello(size)
        game.play_game()
    else:
        print("Invalid board size.")
