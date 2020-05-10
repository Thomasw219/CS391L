import numpy as np

from collections import namedtuple

FullState = namedtuple('FullState', 'position obstacles litter')

SidewalkState = namedtuple('SidewalkState', 'position')
ForwardState = namedtuple('ForwardState', 'position')
ObstaclesState = namedtuple('ObstaclesState', 'obstacles_grid')
LitterState = namedtuple('LitterState', 'litter_grid')

EPSILON = 0.25

LITTER_GRID_DIM = 5
LITTER_PROB = 1 / 10

num_actions = 9
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

def train_litter_initial_grid(p):
    grid = np.zeros((LITTER_GRID_DIM, LITTER_GRID_DIM))
    for i in range(LITTER_GRID_DIM**2):
        position = (i // LITTER_GRID_DIM, i % LITTER_GRID_DIM)
        r = np.random.rand()
        if position[0] == LITTER_GRID_DIM // 2 and position[1] == LITTER_GRID_DIM // 2:
            # Center of the grid, no litter initially here
            continue
        elif r < p:
            grid[position[0], position[1]] = 1
    return grid

def train_litter_act(litter_grid, action):
    new_litter_grid = np.zeros((LITTER_GRID_DIM, LITTER_GRID_DIM))
    for i in range(LITTER_GRID_DIM**2):
        position = (i // LITTER_GRID_DIM, i % LITTER_GRID_DIM)
        new_position = act_position(position, action)
        if position[0] == LITTER_GRID_DIM // 2 and position[1] == LITTER_GRID_DIM // 2:
            # Pick up litter case
            new_litter_grid[position[0], position[1]] = 0
        elif in_bounds(new_position, 0, LITTER_GRID_DIM - 1, 0, LITTER_GRID_DIM - 1):
            new_litter_grid[position[0], position[1]] = litter_grid[new_position[0], new_position[1]]
        else:
            new_litter_grid[position[0], position[1]] = 0
    return new_litter_grid

def train_litter_reward(state, action, new_state):
    position = (LITTER_GRID_DIM // 2, LITTER_GRID_DIM // 2)
    new_position = act_position(position, action)
    if state[new_position[0], new_position[1]] == 1:
        return 1
    else:
        return 0

def get_q_val(q_table, key):
    if key in q_val.keys():
        q_val = q_table[key]
    else:
        q_val = 0
    return q_val

def get_argmax(q_table, state):
    maxes = []
    max_q = -1 * np.inf
    for i in range(num_actions):
        key = (ndarray_to_tuple(state), i)
        if q_val > max_q:
            maxes = [i]
            max_q = q_val
        elif q_val == max_q:
            maxes.append(i)
    opt = np.random.choice(maxes)
    return opt

def litter_select_action(state, q_table, eps):
    opt = get_argmax(q_table, state)
    dist = eps / n_actions * np.ones((num_actions,))
    dist[opt] += 1 - eps
    return np.random.choice(n_actions, p=dist)

def train_litter(num_iter):
    q_table = {}
    state = train_litter_initial_grid(LITTER_PROB)
    terminal_state = np.zeros((LITTER_GRID_DIM, LITTER_GRID_DIM))
    for i in range(num_iter):
        if np.array_equal(state, terminal_state):
            state = train_litter_initial_grid(LITTER_PROB)
        action = litter_select_action(state, q_table, EPSILON)
        key = (ndarray_to_tuple(state), action)
        new_state = train_litter_act(state, action)
        reward = train_litter_reward(state, action, new_state)
        argmax_q = get_q_val(q_table, (ndarray_to_tuple(new_state), get_argmax(q_table, new_state)))
        update = reward + GAMMA * argmax_q
        old_q = get_q_val(q_table, key)
        new_q = (1 - ALPHA) * old_q + ALPHA * update
        q_table[key] = new_q

mat = np.zeros((3, 3))
for i in range(3**2):
    mat[i // 3, i % 3] = i


# Just verifying the movement works as desired
init = train_litter_initial_grid(LITTER_PROB)
for i in range(num_actions):
    after_act = train_litter_act(init, i)
    print("Action: {}".format(action_map[i]))
    print("Before: ")
    print(init)
    print("After: ")
    print(after_act)
    print("Reward: {}".format(train_litter_reward(init, i, after_act)))
