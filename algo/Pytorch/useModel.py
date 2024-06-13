import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# Définir l'environnement du labyrinthe
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

# Créer le modèle ANN avec PyTorch
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Définir l'algorithme de reinforcement learning
class Agent:
    def __init__(self, input_dim, output_dim):
        self.model = DQN(input_dim, output_dim)
        self.target_model = DQN(input_dim, output_dim)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=2000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return [np.random.uniform(-1, 1), np.random.uniform(-0.1, 0.1)]
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return q_values[0].numpy()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).unsqueeze(0)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            target = reward
            if not done:
                target += self.gamma * torch.max(self.target_model(next_state)).item()
            target_f = self.model(state).detach().numpy()
            target_f[0] = target
            target_f = torch.FloatTensor(target_f)
            self.optimizer.zero_grad()
            output = self.model(state)
            loss = self.criterion(output, target_f)
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)

# Entraînement de l'agent
if __name__ == "__main__":
    maze = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0]
    ])
    env = MazeEnvironment(maze, start=(0,0), exit=(4,4))
    agent = Agent(input_dim=env.get_state().shape[0], output_dim=2)
    
    # Charger un modèle sauvegardé (si nécessaire)
    # agent.load('maze-dqn-50.h5')  # Décommentez et remplacez par le fichier de votre choix

    episodes = 1000
    batch_size = 32

    for e in range(episodes):
        state = env.reset()
        for time in range(500):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                agent.update_target_model()
                print(f"Episode: {e}/{episodes}, score: {time}, e: {agent.epsilon:.2}")
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        if e % 50 == 0:
            agent.save(f"maze-dqn-{e}.h5")
