from fastapi import FastAPI
from pydantic import BaseModel
import random
import numpy as np

app = FastAPI()

class Data(BaseModel):
    message: str

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

    return (vwalls.tolist(), hwalls.tolist())

@app.get("/generate_maze")
def generate_maze():
    WIDTH = 10
    HEIGHT = 10
    v, h = generate_maze_internal(WIDTH, HEIGHT)
    return {"Vertical walls": v, "Horizontal walls": h}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
