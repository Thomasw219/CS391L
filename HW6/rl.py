import numpy as np

from collections import namedtuple

FullState = namedtuple('FullState', 'position obstacles litter')

SidewalkState = namedtuple('SidewalkState', 'position')
ForwardState = namedtuple('ForwardState', 'position')
ObstaclesState = namedtuple('ObstaclesState', 'obstacles_grid')
LitterState = namedtuple('LitterState', 'litter_grid')

LITTER_GRID_DIM = 5

action_map = {0 : 'idle', 1 : 'up', 2 : 'up right', 3 : 'right', 4 : 'down right', 5 : 'down', 6 : 'down left', 7 : 'left', 8 : 'up left'}

def ndarray_to_tuple(array):
    return tuple(np.ravel(array).tolist())

def act_position(position, action):
    if action == 0:
        return position
    elif action == 1:
        return (position[0] - 1, position[1])
    elif action == 2:
        return (position[0] - 1, position[1] + 1)
    elif action == 3:
        return (position[0], position[1] + 1)
    elif action == 4:
        return (position[0] + 1, position[1] + 1)
    elif action == 5:
        return (position[0] + 1, position[1])
    elif action == 6:
        return (position[0] + 1, position[1] - 1)
    elif action == 7:
        return (position[0], position[1] - 1)
    elif action == 8:
        return (position[0] - 1, position[1] - 1)

def in_bounds(position, x_min, x_max, y_min, y_max):
    if position[0] >= x_min and position[0] <= x_max and position[1] >= y_min and position[1] <= y_max:
        return True
    return False

def train_litter_act(litter_grid, action):
    new_litter_grid = np.zeros((LITTER_GRID_DIM, LITTER_GRID_DIM))
    for i in range(LITTER_GRID_DIM**2):
        position = (i // LITTER_GRID_DIM, i % LITTER_GRID_DIM)
        new_position = act_position(position, action)
        if position[0] == LITTER_GRID_DIM / 2 and position[1] == LITTER_GRID_DIM / 2:
            # Pick up litter case
            new_litter_grid[position[0], position[1]] = 0
        if in_bounds(new_position, 0, LITTER_GRID_DIM - 1, 0, LITTER_GRID_DIM - 1):
            new_litter_grid[position[0], position[1]] = litter_grid[new_position[0], new_position[1]]
        else:
            new_litter_grid[position[0], position[1]] = 0
    return new_litter_grid

def train_litter(num_iter):
    q_table = {}
    for i in range(num_iter):

mat = np.zeros((3, 3))
for i in range(3**2):
    mat[i // 3, i % 3] = i

print(ndarray_to_tuple(mat))
