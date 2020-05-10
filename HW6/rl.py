import numpy as np
import matplotlib.pyplot as plt
import pickle

from collections import namedtuple

FullState = namedtuple('FullState', 'position obstacles litter')

SidewalkState = namedtuple('SidewalkState', 'position')
ForwardState = namedtuple('ForwardState', 'position')
ObstaclesState = namedtuple('ObstaclesState', 'obstacles_grid')
LitterState = namedtuple('LitterState', 'litter_grid')

EPSILON = 0.25
GAMMA = 0.90
ALPHA = 0.05

LITTER_GRID_DIM = 5
OBSTACLES_GRID_DIM = 3
LITTER_PROB = 1 / 10
OBSTACLES_PROB = 1 / 5

num_actions = 9
action_map = {0 : 'idle', 1 : 'up', 2 : 'up right', 3 : 'right', 4 : 'down right', 5 : 'down', 6 : 'down left', 7 : 'left', 8 : 'up left'}

def ndarray_to_tuple(array):
    return tuple(np.ravel(array).tolist())

def tuple_to_ndarray(tup, d):
    l = [*tup]
    return np.reshape(np.array(l), (d, d))

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

def train_litter_initial_grid():
    p = LITTER_PROB
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

def train_obstacles_initial_grid():
    p = OBSTACLES_PROB
    grid = np.zeros((OBSTACLES_GRID_DIM, OBSTACLES_GRID_DIM))
    for i in range(OBSCTACLES_GRID_DIM**2):
        position = (i // OBSTACLES_GRID_DIM, i % OBSTACLES_GRID_DIM)
        r = np.random.rand()
        if position[0] == OBSTACLES_GRID_DIM // 2 and position[1] == OBSTACLES_GRID_DIM // 2:
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

def train_obstacles_act(grid, action):
    new_grid = np.zeros((OBSTACLES_GRID_DIM, OBSTACLES_GRID_DIM))
    for i in range(OBSCTACLES_GRID_DIM**2):
        position = (i // OBSTACLES_GRID_DIM, i % OBSTACLES_GRID_DIM)
        new_position = act_position(position, action)
        if in_bounds(new_position, 0, OBSTACLES_GRID_DIM - 1, 0, OBSTACLES_GRID_DIM - 1):
            new_grid[position[0], position[1]] = grid[new_position[0], new_position[1]]
        else:
            new_grid[position[0], position[1]] = 0
    return new_grid

def train_litter_reward(state, action, new_state):
    position = (LITTER_GRID_DIM // 2, LITTER_GRID_DIM // 2)
    new_position = act_position(position, action)
    if state[new_position[0], new_position[1]] == 1:
        return 1
    else:
        return 0

def train_obstacles_reward(state, action, new_state):
    position = (OBSTACLES_GRID_DIM // 2, OBSTACLES_GRID_DIM // 2)
    new_position = act_position(position, action)
    if state[new_position[0], new_position[1]] == 1:
        return -1
    else:
        return 0

def get_q_val(q_table, key):
    if key in q_table.keys():
        q_val = q_table[key]
    else:
        q_val = 0
    return q_val

def get_argmax(q_table, state):
    maxes = []
    max_q = -1 * np.inf
    for i in range(num_actions):
        key = (ndarray_to_tuple(state), i)
        q_val = get_q_val(q_table, key)
        if q_val > max_q:
            maxes = [i]
            max_q = q_val
        elif q_val == max_q:
            maxes.append(i)
    opt = np.random.choice(maxes)
    return opt

def select_action(state, q_table, eps):
    opt = get_argmax(q_table, state)
    dist = eps / num_actions * np.ones((num_actions,))
    dist[opt] += 1 - eps
    return np.random.choice(num_actions, p=dist)

litter_terminal_state = np.zeros((LITTER_GRID_DIM, LITTER_GRID_DIM))
def litter_is_terminal(state):
    if np.array_equal(state, litter_terminal_state):
        return True
    return False

obstacles_terminal_state = np.zeros((OBSTACLES_GRID_DIM, OBSTACLES_GRID_DIM))
def litter_is_terminal(state):
    if np.array_equal(state, obstacles_terminal_state):
        return True
    return False

def train(num_iter, q_table, act, reward_func, action_selection, initialize_func, is_terminal_state):
    state = initialize_func()
    total_delta = 0
    for i in range(num_iter):
        if is_terminal_state(state):
            state = initialize_func()
        action = action_selection(state, q_table, EPSILON)
        key = (ndarray_to_tuple(state), action)
        new_state = act(state, action)
        reward = reward_func(state, action, new_state)
        argmax_q = get_q_val(q_table, (ndarray_to_tuple(new_state), get_argmax(q_table, new_state)))
        update = reward + GAMMA * argmax_q
        old_q = get_q_val(q_table, key)
        new_q = (1 - ALPHA) * old_q + ALPHA * update
        total_delta += np.abs(old_q - new_q)
        q_table[key] = new_q
        state = new_state
    return q_table, total_delta / num_iter


"""
# Verifying matrix to tuple conversion and back
mat = np.zeros((LITTER_GRID_DIM, LITTER_GRID_DIM))
for i in range(LITTER_GRID_DIM**2):
    mat[i // LITTER_GRID_DIM, i % LITTER_GRID_DIM] = i

tup = ndarray_to_tuple(mat)
new_mat = tuple_to_ndarray(tup, LITTER_GRID_DIM)

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
"""

ITERS_PER = 10000
q_table = {}
avg_deltas = []
episodes = []

for i in range(200):
    """
    # The training for litter collection
    q_table, avg_delta = train(ITERS_PER, q_table, train_litter_act, train_litter_reward, select_action, train_litter_initial_grid, litter_is_terminal)
    """
    q_table, avg_delta = train(ITERS_PER, q_table, train_obstacles_act, train_obstacles_reward, select_action, train_obstacles_initial_grid, obstacles_is_terminal)
    avg_deltas.append(avg_delta)
    episodes.append(i + 1)
    print("Iterations Trained: {}".format((i + 1) * ITERS_PER))
    print("Q table entries: {}".format(len(q_table.keys())))
    pair = q_table.popitem()
    q_table[pair[0]] = pair[1]
    state = pair[0][0]
    print("State:\n{}".format(tuple_to_ndarray(state, LITTER_GRID_DIM)))
    opt = get_argmax(q_table, state)
    print("Optimal Action: {}".format(action_map[opt]))
    print("Q value: {}".format(get_q_val(q_table, (ndarray_to_tuple(state), opt))))

    """
    with open('pickle_files/litter_qtable.pickle', 'wb') as handle:
        pickle.dump(q_table, handle, protocol=pickle.HIGHEST_PROTOCOL)

    plt.figure(0)
    plt.plot(episodes, avg_deltas)
    plt.savefig('figures/litter_training.png')
    plt.close()
    """
    with open('pickle_files/obstacles_qtable.pickle', 'wb') as handle:
        pickle.dump(q_table, handle, protocol=pickle.HIGHEST_PROTOCOL)

    plt.figure(0)
    plt.plot(episodes, avg_deltas)
    plt.savefig('figures/obstacles_training.png')
    plt.close()
