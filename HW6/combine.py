import numpy as np
import matplotlib.pyplot as plt
import pickle

from rl import *

with open('pickle_files/litter_qtable.pickle', 'rb') as handle:
    litter_qtable = pickle.load(handle)
with open('pickle_files/obstacles_qtable.pickle', 'rb') as handle:
    obstacles_qtable = pickle.load(handle)
with open('pickle_files/sidewalk_qtable.pickle', 'rb') as handle:
    sidewalk_qtable = pickle.load(handle)
with open('pickle_files/forward_qtable.pickle', 'rb') as handle:
    forward_qtable = pickle.load(handle)

def create_full_state():
    init_position = train_initial_position()
    litter_grid = np.zeros((FULL_GRID_X, FULL_GRID_Y))
    obstacles_grid = np.zeros((FULL_GRID_X, FULL_GRID_Y))
    for i in range(FULL_GRID_X * FULL_GRID_Y):
        position = (i // FULL_GRID_Y, i % FULL_GRID_Y)
        r1 = np.random.rand()
        r2 = np.random.rand()
        if r1 < LITTER_PROB and (position[0] != init_position[0] or position[1] != init_position[1]):
            litter_grid[position[0], position[1]] = 1
        if r2 < OBSTACLES_PROB:
            obstacles_grid[position[0], position[1]] = 1

    return FullState(init_position, litter_grid, obstacles_grid)

def show_full_state(x, y, full_state):
    plt.figure(69)
    litter_x = []
    litter_y = []
    obstacles_x = []
    obstacles_y = []
    for i in range(FULL_GRID_X * FULL_GRID_Y):
        position = (i // FULL_GRID_Y, i % FULL_GRID_Y)
        if full_state.litter[position[0], position[1]] == 1:
            litter_x.append(position[0])
            litter_y.append(position[1])
        if full_state.obstacles[position[0], position[1]] == 1:
            obstacles_x.append(position[0])
            obstacles_y.append(position[1])

    plt.scatter(obstacles_y, obstacles_x, c='b')
    plt.scatter(litter_y, litter_x, c='g', marker='D')
    plt.plot(y, x)
    plt.show()

full_state = create_full_state()
print(full_state)

show_full_state([], [], full_state)
