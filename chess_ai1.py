import os
import chess
import chess.pgn
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from multiprocessing import Pool
import multiprocessing
import sys
import logging
from collections import deque

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("entrenamiento.log"),
        logging.StreamHandler()
    ]
)

sys.setrecursionlimit(1500)
torch.backends.cudnn.benchmark = True  # Optimiza convoluciones
multiprocessing.set_start_method("spawn", force=True)

# Parámetros globales
BOARD_SIZE = 8
INPUT_CHANNELS = 12
CAPTURE_FACTOR = 10.0
REPLAY_BUFFER_SIZE = 50000
BATCH_SIZE = 4096
mcts_simulations = 100

# Buffer de replay
replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}

ALL_MOVES = [
    chess.Move(from_sq, to_sq)
    for from_sq in range(64)
    for to_sq in range(64)
    if from_sq != to_sq
] + [
    chess.Move(from_sq, to_sq, promotion=promo)
    for from_sq in range(64)
    for to_sq in range(64)
    if from_sq != to_sq
    for promo in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
]

ACTION_SIZE = len(ALL_MOVES)

def move_to_index(move):
    try:
        return ALL_MOVES.index(move)
    except ValueError:
        return -1

def index_to_move(index):
    if 0 <= index < len(ALL_MOVES):
        return ALL_MOVES[index]
    return None

def board_to_tensor(board, device):
    piece_map = board.piece_map()
    tensor = torch.zeros((INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=torch.float32, device=device)
    for square, piece in piece_map.items():
        piece_type = piece.piece_type - 1
        color_offset = 0 if piece.color == chess.WHITE else 6
        row, col = divmod(square, 8)
        tensor[piece_type + color_offset][row][col] = 1.0
    return tensor

def calculate_material_gain(board, move):
    if board.is_capture(move):
        captured_piece = board.piece_at(move.to_square)
        if captured_piece:
            return PIECE_VALUES.get(captured_piece.piece_type, 0)
    return 0

class ChessNet(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.conv = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU()
        )
        self.policy_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * BOARD_SIZE * BOARD_SIZE, 2048),
            nn.ReLU(),
            nn.Linear(2048, ACTION_SIZE),
            nn.Softmax(dim=1),
        )
        self.value_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * BOARD_SIZE * BOARD_SIZE, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.conv(x)
        return self.policy_head(x), self.value_head(x)




class MCTSNode:
    def __init__(self, board, parent=None):
        self.board = board
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value_sum = 0
        self.material_gain = 0
        self.visited_positions = set()

    def backup(self, value):
        # Aumentamos el peso de la ganancia material
        material_factor = 0.5  # Incrementar este valor para darle más prioridad a la ganancia material
        adjusted_value = value + (self.material_gain * material_factor)
        self.value_sum += adjusted_value
        self.visits += 1
        if self.parent:
            self.parent.backup(-adjusted_value)

    def select(self, alpha=1.0):
        # Aumentamos alpha para elevar el peso de la ganancia material
        best_score = -float('inf')
        best_child = None
        for move, child in self.children.items():
            exploitation = (child.value_sum + alpha * child.material_gain) / (child.visits + 1e-6)
            exploration = np.sqrt(2 * np.log(self.visits + 1) / (child.visits + 1e-6))
            # Penalización por repetir posiciones
            repetition_penalty = -0.5 if child.board.fen() in self.visited_positions else 0
            noise = np.random.normal(0, 0.1)
            score = exploitation + exploration + repetition_penalty + noise
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def expand(self, model, device, alpha=0.5):
        # Resto de la lógica permanece igual    
        for move in self.board.legal_moves:
            new_board = self.board.copy()   
            mg = calculate_material_gain(self.board, move) * CAPTURE_FACTOR
            new_board.push(move)
            child_node = MCTSNode(new_board, parent=self)
            child_node.material_gain = mg
            self.children[move] = child_node

        with torch.no_grad():
            inp = board_to_tensor(self.board, device).unsqueeze(0).to(device, non_blocking=True)
            policy, val = model(inp)

        rollout_val = self.rollout_simulation(model, device, depth=3)
        return val.item() + rollout_val

    def rollout_simulation(self, model, device, depth=3):
        # Resto de la lógica permanece igual
        temp_board = self.board.copy()
        total_value = 0.0
        for _ in range(depth):
            if temp_board.is_game_over():
                break
            move_list = list(temp_board.legal_moves)
            if not move_list:
                break
            move = random.choice(move_list)
            temp_board.push(move)
        with torch.no_grad():
            inp = board_to_tensor(temp_board, device).unsqueeze(0)
            policy, val = model(inp)
        total_value += val.item()
        return total_value
    

def train_from_replay_buffer(model, optimizer, device):
    if len(replay_buffer) < BATCH_SIZE:
        return

    batch = random.sample(replay_buffer, BATCH_SIZE)
    states, actions, rewards = zip(*batch)

    states_tensor = torch.stack(states).to(device, non_blocking=True)
    actions_tensor = torch.tensor(actions, dtype=torch.long, device=device)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)

    model.train()
    optimizer.zero_grad()

    policy, values = model(states_tensor)
    loss = nn.CrossEntropyLoss()(policy, actions_tensor) + 0.5 * nn.MSELoss()(values.view(-1), rewards_tensor)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()  # <--- Cambiado al lugar correcto
    # scheduler.step()  # <--- Eliminado de aquí, se llamará después del optimizer.step() fuera de esta función

    logging.info(f"Entrenado desde Replay Buffer con pérdida: {loss.item():.4f}")


def load_model_weights(model, checkpoint_path, device):
    if os.path.exists(checkpoint_path):
        try:
            # Cargar los pesos del modelo
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if isinstance(model, nn.DataParallel):
                model = model.module
            model.load_state_dict(checkpoint)
            logging.info(f"Pesos cargados desde {checkpoint_path}")
        except Exception as e:
            logging.error(f"Error al cargar los pesos del modelo: {e}")
    else:
        logging.warning(f"No se encontró el archivo {checkpoint_path}. Se iniciará el entrenamiento desde cero.")


def initialize_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def train_model(model, optimizer, data, device, batch_size=4096):
    model.train()
    optimizer.zero_grad()

    states, actions, rewards = zip(*data)
    states_tensor = torch.stack(states).to(device, non_blocking=True)
    actions_tensor = torch.tensor(actions, dtype=torch.long, device=device)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)

    policy, values = model(states_tensor)
    loss = nn.CrossEntropyLoss()(policy, actions_tensor) + 0.5 * nn.MSELoss()(values.view(-1), rewards_tensor)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    logging.info(f"Batch entrenado con pérdida: {loss.item():.4f}")

def mate_in_one_move(board):
    for move in board.legal_moves:
        board.push(move)
        if board.is_checkmate():
            board.pop()
            return move
        board.pop()
    return None



def play_single_game(model, episode, device):
    global mcts_simulations
    board = chess.Board()
    node = MCTSNode(board)

    game_moves = []

    while not board.is_game_over():
        for _ in range(mcts_simulations):
            leaf = node
            while leaf.children:
                leaf = leaf.select()
            val = leaf.expand(model, device)
            leaf.backup(val)

        best_move = mate_in_one_move(board)
        if best_move is None:
            best_move = max(node.children, key=lambda m: node.children[m].visits)
        
        mgain = calculate_material_gain(board, best_move)
        action_idx = move_to_index(best_move)

        if action_idx != -1:
            replay_buffer.append((board_to_tensor(board, device), action_idx, float(mgain)))
        else:
            logging.warning(f"Movimiento no válido encontrado: {best_move}")
            break

        game_moves.append(best_move.uci())
        board.push(best_move)

        if len(game_moves) >= 200:
            logging.info("Partida excede 200 jugadas. Se declara tablas.")
            break
        node = node.children[best_move]

    return game_moves
    
def save_game_to_pgn(game_data, filename):
    game = chess.pgn.Game()
    node = game

    board = chess.Board()
    for move_uci in game_data:
        move = board.push_uci(move_uci)
        node = node.add_variation(move)

    with open(filename, 'w') as pgn_file:
        pgn_file.write(str(game))

def train_chess_ai_parallel(num_episodes, num_processes=4, save_interval=50):
    """
    Entrena el modelo de ajedrez con múltiples procesos y almacenamiento incremental.
    
    Args:
        num_episodes (int): Número total de episodios a jugar.
        num_processes (int): Número de procesos paralelos a usar.
        save_interval (int): Intervalo en episodios para guardar el modelo.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessNet(device).to(device)
    model.apply(initialize_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Usar un programador de tasa de aprendizaje para ajustes dinámicos
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

    # Cargar pesos del modelo si existen
    load_model_weights(model, "chess_model.pth", device)

    os.makedirs("partidas", exist_ok=True)
    global replay_buffer
    replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)
    pool = Pool(processes=num_processes)

    try:
        for batch_start in range(0, num_episodes, num_processes):
            batch_end = min(batch_start + num_processes, num_episodes)
            episodes = range(batch_start, batch_end)

            # Jugar múltiples juegos en paralelo
            results = pool.starmap(play_single_game, [(model, ep, device) for ep in episodes])

            # Recopilar datos de entrenamiento
            for i, game_moves in enumerate(results):
                    if game_moves:
                        temp_board = chess.Board()
                        temp_replay_data = []
                        for move_uci in game_moves:
                            temp_board.push_uci(move_uci)
                            state_tensor = board_to_tensor(temp_board, device)
                            action_idx = move_to_index(chess.Move.from_uci(move_uci))
                            reward = float(calculate_material_gain(temp_board, chess.Move.from_uci(move_uci)))
                            temp_replay_data.append((state_tensor, action_idx, reward))
                        # Asignar recompensa final basada en el resultado de la partida
                        if temp_board.is_checkmate():
                            final_reward = 1.0  # Recompensa por ganar
                        elif temp_board.is_stalemate() or temp_board.is_insufficient_material():
                            final_reward = -0.5  # Recompensa por empate
                        else:
                            final_reward = -1.0  # Recompensa por perder
                        temp_replay_data.append((state_tensor, action_idx, final_reward))
                        replay_buffer.extend(temp_replay_data)  
                        # Removed duplicate line below
                        # replay_buffer.extend(temp_replay_data)
                        pgn_filename = f"partidas/partida_{batch_start + i + 1}.pgn"
                        save_game_to_pgn(game_moves, pgn_filename)

            logging.info(f"Episodios {batch_start + 1} a {batch_end} procesados. "
                         f"Replay Buffer actual: {len(replay_buffer)}")

            # Entrenar desde el Replay Buffer cuando hay suficientes datos
            if len(replay_buffer) >= BATCH_SIZE:
                train_from_replay_buffer(model, optimizer, device)
                # Ahora se llama al scheduler después del optimizer.step()
                scheduler.step()

            # Guardar el modelo periódicamente
            if (batch_start + num_processes) % save_interval == 0:
                torch.save(model.state_dict(), f"chess_model_{batch_start + num_processes}.pth")
                logging.info(f"Modelo guardado como chess_model_{batch_start + num_processes}.pth")

    except KeyboardInterrupt:
        logging.warning("Entrenamiento interrumpido por el usuario. Guardando el modelo...")
        torch.save(model.state_dict(), "chess_model_interrupt.pth")
        logging.info("Modelo guardado como chess_model_interrupt.pth")
    except Exception as e:
        logging.error(f"Ocurrió un error inesperado: {e}")
    finally:
        pool.close()
        pool.join()
        torch.save(model.state_dict(), "chess_model.pth")
        logging.info("Entrenamiento finalizado. Modelo guardado como chess_model.pth")


def load_games(file_path):
    games = []
    with open(file_path, 'r') as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            games.append(game)
    logging.info(f"{len(games)} partidas cargadas desde {file_path}")
    return games

def train_from_pgn_file(file_path, model, optimizer, device, batch_size=15000):
    games = load_games(file_path)
    accumulated_data = []
    
    for idx, game in enumerate(games, 1):
        board = game.board()
        reward = 0.0
        mgain = 0.0
        temp_replay_data = []
        for move in game.mainline_moves():
            board.push(move)
            captured_piece = board.piece_at(move.to_square)
            if captured_piece:
                mgain += PIECE_VALUES.get(captured_piece.piece_type, 0)
            total_reward = reward + mgain
            state_tensor = board_to_tensor(board, device)
            action_idx = move_to_index(move)
            if action_idx != -1:
                temp_replay_data.append((state_tensor, action_idx, total_reward))
        
        # Asignar recompensa final basada en el resultado del juego
        if board.is_checkmate():
            final_reward = 1.0  # Ganar
        elif board.is_stalemate() or board.is_insufficient_material():
            final_reward = 0.0  # Empate
        else:
            final_reward = -1.0  # Perder
        
        temp_replay_data.append((state_tensor, action_idx, final_reward))
        processed_data = process_game_rewards(temp_replay_data)
        
        # Añadir al buffer priorizado con prioridad
        for state, action, reward in processed_data:
            priority = 2.0 if reward > 0 else 1.0
            replay_buffer.add(state, action, reward, priority=priority)
            accumulated_data.append((state, action, reward))
        
        if len(accumulated_data) >= batch_size:
            train_from_replay_buffer(model, optimizer, device)
            accumulated_data = []
    
    # Entrenar con datos restantes
    if accumulated_data:
        train_from_replay_buffer(model, optimizer, device)
    
    logging.info(f"Entrenamiento completado desde el archivo {file_path}")


if __name__ == "__main__":
    try:
     
        # Device configuration
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")
        if device.type == 'cuda':
            logging.info(f"Number of GPUs available: {torch.cuda.device_count()}")
        
        # Model initialization with error handling
        try:
            model = ChessNet(device)
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model.to(device)
            model.apply(initialize_weights)
        except Exception as e:
            logging.error(f"Error initializing model: {e}")
            raise
        
        # Optimizer and scheduler
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
        
        # Load existing weights with validation
        try:
            load_model_weights(model, "chess_model.pth", device)
        except Exception as e:
            logging.warning(f"Could not load weights: {e}")
        
        # Training parameters
        num_episodes = 20  # Ajusta según sea necesario
        num_processes = 3   # Ajusta basado en los núcleos de CPU
        save_interval = 10  # Guardar modelo cada 10 episodios
        
        # Start training with proper buffer handling
        try:
            train_chess_ai_parallel(num_episodes, num_processes, save_interval)
        except Exception as e:
            logging.error(f"Training error: {e}")
            raise
                
    except Exception as e:
        logging.error(f"Critical error: {e}")
        # Save emergency backup
        try:
            torch.save(model.state_dict(), "chess_model_emergency.pth")
            logging.info("Emergency model backup saved")
        except:
            pass
        raise
