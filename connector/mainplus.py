from fastapi import FastAPI
from pydantic import BaseModel
import random
import numpy as np

app = FastAPI()

class Data(BaseModel):
    message: str

class Position(BaseModel):
    x: int
    y: int

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/send")
def send_data(data: Data):
    return {"received_message": data.message}

def get_random_cell(m, n):
    return (np.random.randint(0, m), np.random.randint(0, n))

def generate_cases(m, n):
    mat = np.arange(m * n)
    return mat.reshape(m, n)

def generate_vertical_walls(m, n):
    return np.ones((m - 1, n))

def generate_horizontal_walls(m, n):
    return np.ones((m, n - 1))

def generate_maze_internal(m, n):
    cases = generate_cases(m, n)
    vwalls = generate_vertical_walls(m, n)
    hwalls = generate_horizontal_walls(m, n)

    steps = 0
    while steps < (m * n) - 1:
        x, y = get_random_cell(m, n)
        walls = None
        c = (0, 0)
    
        if x >= vwalls.shape[0] and y >= hwalls.shape[1]:
            continue
        elif x >= vwalls.shape[0]:
            walls = hwalls
            c = (cases[x][y], cases[x][y + 1])
        elif y >= hwalls.shape[1]:
            walls = vwalls
            c = (cases[x][y], cases[x + 1][y])
        else:
            choice = random.randint(0, 1)
            if choice == 0:
                walls = hwalls
                c = (cases[x][y], cases[x][y + 1])
            else:
                walls = vwalls
                c = (cases[x][y], cases[x + 1][y])

        if walls is None:
            continue
    
        if c[0] == c[1]:
            continue

        for i in range(cases.shape[0]):
            for j in range(cases.shape[1]):
                if cases[i][j] == c[1]:
                    cases[i][j] = c[0]

        walls[x][y] = 0
        steps += 1

    return (vwalls, hwalls)

def maze_to_string(vwalls, hwalls, player_position, exit_position):
    m = len(vwalls) + 1
    n = len(hwalls[0]) + 1

    maze = [["#" for _ in range(n * 2 + 1)] for _ in range(m * 2 + 1)]

    for i in range(m):
        for j in range(n):
            maze[i * 2 + 1][j * 2 + 1] = " "
    
    for i in range(len(vwalls)):
        for j in range(len(vwalls[i])):
            if vwalls[i][j] == 0:
                maze[i * 2 + 2][j * 2 + 1] = " "

    for i in range(len(hwalls)):
        for j in range(len(hwalls[i])):
            if hwalls[i][j] == 0:
                maze[i * 2 + 1][j * 2 + 2] = " "
    
    # Ensure player and exit positions are within bounds
    player_i, player_j = player_position
    exit_i, exit_j = exit_position
    maze[player_i][player_j] = "P"

    # Remove the wall perimeter of the maze next to the exit
    if exit_i == 0:
        maze[0][exit_j] = " "
        maze[1][exit_j] = "E"
    elif exit_i == m * 2:
        maze[m * 2][exit_j] = " "
        maze[m * 2 - 1][exit_j] = "E"
    elif exit_j == 0:
        maze[exit_i][0] = " "
        maze[exit_i][1] = "E"
    elif exit_j == n * 2:
        maze[exit_i][n * 2] = " "
        maze[exit_i][n * 2 - 1] = "E"
    else:
        maze[exit_i][exit_j] = "E"

    return "\n".join("".join(row) for row in maze)

def create_exit(vwalls, hwalls):
    perimeter_walls = []

    # Collect the perimeter walls
    for j in range(len(hwalls[0])):
        perimeter_walls.append((0, j, 'h'))
        perimeter_walls.append((len(hwalls)-1, j, 'h'))
    for i in range(len(vwalls)):
        perimeter_walls.append((i, 0, 'v'))
        perimeter_walls.append((i, len(vwalls[0])-1, 'v'))

    # Select a random wall to remove for the exit
    exit_wall = random.choice(perimeter_walls)
    if exit_wall[2] == 'h':
        hwalls[exit_wall[0]][exit_wall[1]] = 0
    else:
        vwalls[exit_wall[0]][exit_wall[1]] = 0

    if exit_wall[2] == 'h':
        return (exit_wall[0] * 2 + 1, exit_wall[1] * 2 + 2 if exit_wall[0] == 0 else exit_wall[1] * 2)
    else:
        return (exit_wall[0] * 2 + 2 if exit_wall[1] == 0 else exit_wall[0] * 2, exit_wall[1] * 2 + 1)

def get_accessible_positions(vwalls, hwalls, start):
    m = len(vwalls) + 1
    n = len(hwalls[0]) + 1
    queue = [start]
    visited = set()
    visited.add(start)

    while queue:
        x, y = queue.pop(0)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < m and 0 <= ny < n and (nx, ny) not in visited:
                if dx == -1 and x > 0 and vwalls[x-1][y] == 0 or dx == 1 and x < len(vwalls) and vwalls[x][y] == 0 or dy == -1 and y > 0 and hwalls[x][y-1] == 0 or dy == 1 and y < len(hwalls[0]) and hwalls[x][y] == 0:
                    queue.append((nx, ny))
                    visited.add((nx, ny))
    return visited

class MazeEnvironment:
    def __init__(self, maze, start, exit):
        self.maze = maze
        self.start = np.array(start, dtype=float)
        self.exit = np.array(exit, dtype=float)
        self.position = self.start.copy()
        self.angle = 0  # Angle de vision de l'agent
        self.reset()

    def reset(self):
        self.position = self.start.copy()
        self.angle = 0
        return self.get_state()

    def get_state(self):
        lidar_data = self.get_lidar_data()
        return lidar_data

    def get_lidar_data(self, num_rays=8):
        lidar_data = np.zeros(num_rays)
        angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
        for i, angle in enumerate(angles):
            distance = self.cast_ray(angle)
            lidar_data[i] = distance
        return lidar_data

    def cast_ray(self, angle, max_distance=10):
        x, y = self.position
        for d in np.linspace(0, max_distance, int(max_distance * 10)):
            x_new = x + d * np.cos(angle)
            y_new = y + d * np.sin(angle)
            if x_new < 0 or x_new >= self.maze.shape[0] or y_new < 0 or y_new >= self.maze.shape[1] or self.maze[int(x_new), int(y_new)] == 1:
                return d
        return max_distance

    def step(self, action):
        move_forward, rotate = action
        self.angle += rotate
        new_x = self.position[0] + move_forward * np.cos(self.angle)
        new_y = self.position[1] + move_forward * np.sin(self.angle)

        reward = self.penalty_move
        if np.linalg.norm([new_x - self.exit[0], new_y - self.exit[1]]) < 1.0:
            reward = self.reward_exit
            done = True
        elif new_x < 0 or new_x >= self.maze.shape[0] or new_y < 0 or new_y >= self.maze.shape[1] or self.maze[int(new_x), int(new_y)] == 1:
            reward = self.penalty_impossible_move
            done = False
        else:
            self.position[0] = new_x
            self.position[1] = new_y
            done = False

        return self.get_state(), reward, done

    @property
    def reward_exit(self):
        return 10.0

    @property
    def penalty_move(self):
        return -0.05

    @property
    def penalty_impossible_move(self):
        return -0.75

@app.get("/generate_maze")
def generate_maze():
    WIDTH = 3
    HEIGHT = 3
    v, h = generate_maze_internal(WIDTH, HEIGHT)
    exit_position = create_exit(v, h)

    # Convert exit position to the correct coordinates for the maze array
    exit_x, exit_y = exit_position
    if exit_x == 0:
        exit_x = 1
    elif exit_x == (HEIGHT - 1) * 2:
        exit_x = HEIGHT * 2 - 1
    if exit_y == 0:
        exit_y = 1
    elif exit_y == (WIDTH - 1) * 2:
        exit_y = WIDTH * 2 - 1

    # Place the player at a random position ensuring it's accessible to the exit
    accessible_positions = get_accessible_positions(v, h, (exit_x // 2, exit_y // 2))
    player_position = random.choice(list(accessible_positions))

    # Convert player position to the correct coordinates for the maze array
    player_x, player_y = player_position
    player_x = player_x * 2 + 1
    player_y = player_y * 2 + 1

    maze_string = maze_to_string(v, h, (player_x, player_y), (exit_x, exit_y))
    return {"maze": maze_string, "player_position": (player_x, player_y), "exit_position": (exit_x, exit_y)}

@app.post("/get_raycast")
def get_raycast(position: Position):
    player_position = (position.x, position.y)
    v, h = generate_maze_internal(5, 5)
    env = MazeEnvironment(generate_maze_internal(5, 5), start=player_position, exit=(4,4))
    lidar_data = env.get_lidar_data()
    return {"raycast": lidar_data.tolist()}

@app.post("/get_state")
def get_state(position: Position):
    player_position = (position.x, position.y)
    v, h = generate_maze_internal(5, 5)
    env = MazeEnvironment(generate_maze_internal(5, 5), start=player_position, exit=(4,4))
    state = env.get_state()
    return {"state": state.tolist()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
