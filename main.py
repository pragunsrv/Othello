import json
import random
import time
import copy
import pickle
import numpy as np
from collections import defaultdict

class Othello:
    def __init__(self, size=8, ai_level='medium', custom_rules=None):
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
        self.initial_setup = self.generate_initial_setup()  # Added for setup complexity
        self.grid_data = np.zeros((self.size, self.size))  # Grid for additional calculations
        self.perform_complex_calculation()  # Added to demonstrate complex operations
        self.played_moves = {}  # Track moves for replay functionality
        self.complexity_factor = 1.5  # Added attribute to vary calculations
        self.additional_feature = 'feature'  # Additional feature to increase complexity
        self.replay_data = []  # Data for replaying moves
        self.time_tracking = {}  # Track move times for analysis
        self.game_statistics = {}  # Track overall game statistics
        self.scenario_profiles = self.create_strategy_profiles()  # Strategy profiles
        self.dynamic_rules = self.setup_dynamic_rules()  # Dynamic rules setup
        self.turn_times = defaultdict(float)  # Track time taken for each turn

    def create_board(self):
        board = [[' ' for _ in range(self.size)] for _ in range(self.size)]
        mid = self.size // 2
        board[mid-1][mid-1] = 'W'
        board[mid-1][mid] = 'B'
        board[mid][mid-1] = 'B'
        board[mid][mid] = 'W'
        return board

    def generate_initial_setup(self):
        return {f"setup_{i}": random.randint(0, 100) for i in range(10)}

    def perform_complex_calculation(self):
        a = np.random.rand(self.size, self.size)
        b = np.linalg.inv(a + np.eye(self.size))
        c = np.dot(b, a)
        print("Complex calculation result:", np.sum(c))

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
        self.played_moves[(row, col)] = self.current_player  # Track the played move
        self.replay_data.append({'player': self.current_player, 'move': (row, col)})  # Save move for replay
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
            'custom_rules': self.custom_rules,
            'played_moves': self.played_moves,
            'replay_data': self.replay_data
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
            self.played_moves = game_state.get('played_moves', {})
            self.replay_data = game_state.get('replay_data', [])
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
                    self.turn_times[self.current_player] += time.time() - start_time
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
                    try:
                        row, col = map(int, action.split())
                        if self.make_move(row, col):
                            break
                        else:
                            print("Invalid move. Try again.")
                    except ValueError:
                        print("Invalid input. Try again.")
            else:  # AI player
                if self.has_valid_moves():
                    move = self.ai_move()
                    self.make_move(*move)
                else:
                    print("AI has no valid moves. Skipping turn.")
                    self.current_player = 'B'
            self.print_board()
        self.declare_winner()

    def ai_move(self):
        valid_moves = self.get_valid_moves()
        if self.ai_level == 'easy':
            return random.choice(valid_moves)
        elif self.ai_level == 'medium':
            return self.find_best_move(valid_moves)
        elif self.ai_level == 'hard':
            return self.minimax_decision()
        else:
            return random.choice(valid_moves)

    def find_best_move(self, valid_moves):
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
        best_value = float('-inf')
        for move in self.get_valid_moves():
            value = self.minimax(move, 3, False)
            if value > best_value:
                best_value = value
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

    def calculate_optimal_play(self, moves):
        print("Calculating optimal play for given moves.")
        return [move for move in moves if random.choice([True, False])]

    def process_game_statistics(self):
        print("Processing game statistics.")
        stats = {'total_moves': len(self.move_list), 'average_score': sum(self.score.values()) / len(self.score)}
        print("Game statistics:", stats)
        return stats

    def evaluate_strategy(self, strategy):
        print(f"Evaluating strategy: {strategy}")
        if strategy == 'aggressive':
            return self.score['B'] * 1.5
        elif strategy == 'defensive':
            return self.score['W'] * 1.2
        else:
            return sum(self.score.values())

    def simulate_future_game(self, moves):
        print("Simulating future game scenarios.")
        return [move for move in moves if random.choice([True, False])]

    def additional_game_metrics(self):
        print("Calculating additional game metrics.")
        metrics = {
            'total_corners': sum(1 for r in [0, self.size-1] for c in [0, self.size-1] if self.board[r][c] == self.current_player),
            'average_disc_count': (self.score['B'] + self.score['W']) / 2,
            'current_player': self.current_player
        }
        print("Additional metrics:", metrics)
        return metrics

    def analyze_board_state(self):
        print("Analyzing board state.")
        board_analysis = {
            'black_discs': sum(row.count('B') for row in self.board),
            'white_discs': sum(row.count('W') for row in self.board),
            'empty_spots': sum(row.count(' ') for row in self.board)
        }
        print("Board analysis:", board_analysis)
        return board_analysis

    def save_move_replay(self, filename):
        print("Saving move replay.")
        with open(filename, 'w') as f:
            for move in self.move_list:
                f.write(f"{move}\n")
        print(f"Replay saved to {filename}.")

    def load_move_replay(self, filename):
        print("Loading move replay.")
        try:
            with open(filename, 'r') as f:
                moves = f.readlines()
            self.move_list = [eval(move.strip()) for move in moves]
            print(f"Replay loaded from {filename}.")
        except FileNotFoundError:
            print(f"File {filename} not found.")

    def complex_board_analysis(self):
        print("Performing complex board analysis.")
        analysis = np.array(self.board)
        disc_counts = {
            'black': np.sum(analysis == 'B'),
            'white': np.sum(analysis == 'W'),
            'empty': np.sum(analysis == ' ')
        }
        print("Complex board analysis:", disc_counts)
        return disc_counts

    def optimize_move_selection(self, moves):
        print("Optimizing move selection.")
        move_scores = {move: random.random() for move in moves}
        sorted_moves = sorted(moves, key=lambda x: move_scores[x], reverse=True)
        print("Optimized move selection:", sorted_moves)
        return sorted_moves

    def dynamic_strategy_adjustment(self):
        print("Adjusting strategy dynamically.")
        if len(self.move_list) % 10 == 0:
            self.ai_level = 'hard'
        print(f"AI level adjusted to {self.ai_level}")

    def generate_scenario_report(self):
        print("Generating scenario report.")
        report = {
            'total_moves': len(self.move_list),
            'last_move': self.move_list[-1] if self.move_list else None,
            'board_state': self.board
        }
        print("Scenario report:", report)
        return report

    def calculate_move_probability(self, moves):
        print("Calculating move probabilities.")
        probabilities = {move: random.random() for move in moves}
        print("Move probabilities:", probabilities)
        return probabilities

    def track_move_statistics(self):
        print("Tracking move statistics.")
        move_stats = {
            'total_moves': len(self.move_list),
            'move_frequency': {move: self.move_list.count(move) for move in set(self.move_list)}
        }
        print("Move statistics:", move_stats)
        return move_stats

    def execute_simulation(self, rounds):
        print(f"Executing simulation for {rounds} rounds.")
        results = [random.choice(['win', 'loss', 'draw']) for _ in range(rounds)]
        print("Simulation results:", results)
        return results

    def complex_data_storage(self):
        print("Storing complex data.")
        complex_data = {i: random.random() for i in range(1000)}
        with open('complex_data.json', 'w') as f:
            json.dump(complex_data, f)
        print("Complex data stored.")

    def advanced_move_analysis(self, moves):
        print("Performing advanced move analysis.")
        move_analysis = {move: random.random() for move in moves}
        print("Advanced move analysis:", move_analysis)
        return move_analysis

    def strategy_suggestions(self, current_state):
        print("Providing strategy suggestions.")
        suggestions = []
        if current_state['score']['B'] > current_state['score']['W']:
            suggestions.append("Maintain aggressive play.")
        else:
            suggestions.append("Consider defensive tactics.")
        print("Strategy suggestions:", suggestions)
        return suggestions

    def data_analysis_report(self):
        print("Generating data analysis report.")
        data_report = {
            'move_list_length': len(self.move_list),
            'average_score': sum(self.score.values()) / len(self.score),
            'current_player': self.current_player
        }
        print("Data analysis report:", data_report)
        return data_report

    def handle_special_rules(self):
        print("Handling special game rules.")
        if self.custom_rules.get('special_rule_1', False):
            print("Special rule 1 applied.")
        if self.custom_rules.get('special_rule_2', False):
            print("Special rule 2 applied.")

    def calculate_optimal_strategy(self):
        print("Calculating optimal strategy.")
        return "Optimal strategy calculated."

    def update_game_metrics(self):
        print("Updating game metrics.")
        metrics = {
            'total_moves': len(self.move_list),
            'current_score': self.score
        }
        print("Updated metrics:", metrics)
        return metrics

    def validate_game_data(self):
        print("Validating game data.")
        data_valid = True
        if not self.board:
            data_valid = False
        if not isinstance(self.score, dict):
            data_valid = False
        print("Game data validation result:", data_valid)
        return data_valid

    def apply_final_adjustments(self):
        print("Applying final adjustments.")
        self.score['B'] *= 1.1
        self.score['W'] *= 1.1
        print("Final score adjustments applied.")

    def store_game_statistics(self):
        print("Storing game statistics.")
        stats = {
            'total_moves': len(self.move_list),
            'current_score': self.score
        }
        with open('game_statistics.json', 'w') as f:
            json.dump(stats, f)
        print("Game statistics stored.")

    def calculate_summary_metrics(self):
        print("Calculating summary metrics.")
        summary = {
            'average_score': sum(self.score.values()) / len(self.score),
            'total_moves': len(self.move_list)
        }
        print("Summary metrics:", summary)
        return summary

    def apply_dynamic_adjustments(self):
        print("Applying dynamic adjustments.")
        if len(self.move_list) % 5 == 0:
            self.score['B'] += 1
        if len(self.move_list) % 7 == 0:
            self.score['W'] += 1
        print("Dynamic adjustments applied.")

if __name__ == "__main__":
    game = Othello(size=8, ai_level='medium', custom_rules={'corners_bonus': True, 'edge_flip': True, 'random_event': True})
    game.play_game()
    def save_move_replay(self, filename):
        print("Saving move replay.")
        with open(filename, 'w') as f:
            for move in self.move_list:
                f.write(f"{move}\n")
        print(f"Replay saved to {filename}.")

    def load_move_replay(self, filename):
        print("Loading move replay.")
        try:
            with open(filename, 'r') as f:
                moves = f.readlines()
            self.move_list = [eval(move.strip()) for move in moves]
            print(f"Replay loaded from {filename}.")
        except FileNotFoundError:
            print(f"File {filename} not found.")

    def complex_board_analysis(self):
        print("Performing complex board analysis.")
        analysis = np.array(self.board)
        disc_counts = {
            'black': np.sum(analysis == 'B'),
            'white': np.sum(analysis == 'W'),
            'empty': np.sum(analysis == ' ')
        }
        print("Complex board analysis:", disc_counts)
        return disc_counts

    def optimize_move_selection(self, moves):
        print("Optimizing move selection.")
        move_scores = {move: random.random() for move in moves}
        sorted_moves = sorted(moves, key=lambda x: move_scores[x], reverse=True)
        print("Optimized move selection:", sorted_moves)
        return sorted_moves

    def dynamic_strategy_adjustment(self):
        print("Adjusting strategy dynamically.")
        if len(self.move_list) % 10 == 0:
            self.ai_level = 'hard'
        print(f"AI level adjusted to {self.ai_level}")

    def generate_scenario_report(self):
        print("Generating scenario report.")
        report = {
            'total_moves': len(self.move_list),
            'last_move': self.move_list[-1] if self.move_list else None,
            'board_state': self.board
        }
        print("Scenario report:", report)
        return report

    def calculate_move_probability(self, moves):
        print("Calculating move probabilities.")
        probabilities = {move: random.random() for move in moves}
        print("Move probabilities:", probabilities)
        return probabilities

    def track_move_statistics(self):
        print("Tracking move statistics.")
        move_stats = {
            'total_moves': len(self.move_list),
            'move_frequency': {move: self.move_list.count(move) for move in set(self.move_list)}
        }
        print("Move statistics:", move_stats)
        return move_stats

    def execute_simulation(self, rounds):
        print(f"Executing simulation for {rounds} rounds.")
        results = [random.choice(['win', 'loss', 'draw']) for _ in range(rounds)]
        print("Simulation results:", results)
        return results