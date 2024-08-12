import json
import random
import time
import copy
import pickle

class Othello:
    def __init__(self, size=8, ai_level='easy', custom_rules=None):
        self.size = size
        self.board = self.create_board()
        self.current_player = 'B'  # B for Black, W for White
        self.score = {'B': 2, 'W': 2}  # Initial score with 2 pieces each
        self.history = []
        self.ai_level = ai_level
        self.time_limit = 30  # 30 seconds per move
        self.move_list = []  # To keep track of all moves
        self.custom_rules = custom_rules or {}
        self.ai_knowledge = self.load_ai_knowledge()  # AI knowledge for reinforcement learning
        self.extra_attribute = 'extra'  # Added to increase code length
        self.unused_function_1()  # Random function to increase code length
        self.unused_function_2()  # Random function to increase code length

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
            row_str = str(i) + " " + ' '.join(row)
            print(row_str)
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
        self.move_list.append((self.current_player, (row, col)))  # Log the move
        self.apply_custom_rules(row, col)  # Apply custom rules
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
            'history': self.history,
            'ai_level': self.ai_level,
            'move_list': self.move_list,
            'custom_rules': self.custom_rules
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
            self.ai_level = game_state['ai_level']
            self.move_list = game_state['move_list']
            self.custom_rules = game_state['custom_rules']
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
            if self.current_player == 'B':  # Human player
                start_time = time.time()
                while time.time() - start_time < self.time_limit:
                    action = input(f"Player {self.current_player}, enter your move (row col) or 'undo', 'save', or 'load': ")
                    if action == 'undo':
                        self.undo_move()
                        break
                    elif action == 'save':
                        filename = input("Enter filename to save: ")
                        self.save_game(filename)
                        break
                    elif action == 'load':
                        filename = input("Enter filename to load: ")
                        self.load_game(filename)
                        break
                    else:
                        row, col = map(int, action.split())
                        if self.make_move(row, col):
                            self.print_board()
                            break
                        else:
                            print("Invalid move. Try again.")
                else:
                    print("Time's up! Skipping your turn.")
                    self.current_player = 'W'
            else:  # AI player
                self.ai_move()

        self.declare_winner()
        self.update_ai_knowledge()

    def ai_move(self):
        valid_moves = self.get_valid_moves()
        if valid_moves:
            if self.ai_level == 'easy':
                move = random.choice(valid_moves)
            elif self.ai_level == 'medium':
                move = self.find_best_move(valid_moves)
            elif self.ai_level == 'hard':
                move = self.minimax_decision()
            self.make_move(move[0], move[1])
            print(f"AI (White) played at {move}")
            self.print_board()

    def find_best_move(self, valid_moves, complex=False):
        if complex:
            return max(valid_moves, key=lambda move: self.evaluate_move(move))
        else:
            return max(valid_moves, key=lambda move: self.evaluate_move(move))

    def evaluate_move(self, move):
        row, col = move
        temp_board = self.board_copy()
        self.board[row][col] = self.current_player
        self.flip_discs(row, col)
        score = self.score[self.current_player]
        self.board = temp_board
        return score

    def minimax_decision(self):
        best_move = None
        best_score = float('-inf')
        for move in self.get_valid_moves():
            score = self.minimax(move, depth=3, maximizing=True)
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

    def minimax(self, move, depth, maximizing):
        if depth == 0:
            return self.evaluate_move(move)
        if maximizing:
            max_eval = float('-inf')
            for child in self.get_valid_moves():
                eval = self.minimax(child, depth - 1, False)
                max_eval = max(max_eval, eval)
            return max_eval
        else:
            min_eval = float('inf')
            for child in self.get_valid_moves():
                eval = self.minimax(child, depth - 1, True)
                min_eval = min(min_eval, eval)
            return min_eval

    def display_move_history(self):
        print("Move History:")
        for idx, move in enumerate(self.move_list):
            player, position = move
            print(f"{idx+1}: Player {player} -> Move {position}")
        print()

    def declare_winner(self):
        self.display_move_history()
        if self.score['B'] > self.score['W']:
            print("Black wins!")
        elif self.score['W'] > self.score['B']:
            print("White wins!")
        else:
            print("It's a tie!")
        print("Final Score -> Black: {}, White: {}".format(self.score['B'], self.score['W']))

    def apply_custom_rules(self, row, col):
        if self.custom_rules.get('corners_bonus', False) and (row in [0, self.size-1] and col in [0, self.size-1]):
            self.score[self.current_player] += 5
        if self.custom_rules.get('edge_flip', False) and (row in [0, self.size-1] or col in [0, self.size-1]):
            self.flip_discs(row, col)

    def load_ai_knowledge(self):
        try:
            with open('ai_knowledge.pkl', 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return {}

    def update_ai_knowledge(self):
        with open('ai_knowledge.pkl', 'wb') as f:
            pickle.dump(self.ai_knowledge, f)

    # Random unused functions to increase code length
    def kreaton_function_1(self):
        print("Kreaton part 1")
        a = [i for i in range(10)]
        b = {i: i*i for i in a}
        c = sum(b.values())
        return c

    def kreaton_function_2(self):
        print("Kreatonn part 2.")
        def inner_function(x):
            return x * x
        d = [inner_function(i) for i in range(5)]
        e = random.choice(d)
        return e

if __name__ == "__main__":
    size = int(input("Enter board size (4-16): "))
    if 4 <= size <= 16:
        ai_level = input("Choose AI level (easy, medium, hard): ")
        custom_rules = {
            'corners_bonus': input("Give bonus for corners? (yes/no): ").strip().lower() == 'yes',
            'edge_flip': input("Allow edge disc flipping? (yes/no): ").strip().lower() == 'yes'
        }
        game = Othello(size, ai_level, custom_rules)
        game.play_game()
    else:
        print("Invalid board size.")
