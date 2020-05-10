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

    plt.scatter(obstacles_y, obstacles_x, c='b', label='Obstacles')
    plt.scatter(litter_y, litter_x, c='g', marker='D', label='Litter')
    plt.plot(y, x, label="Path")
    plt.legend()
    plt.show()

def rescale_qval(q_val, max_q, min_q):
    if max_q == min_q:
        return 0.5
    return (q_val - min_q) / (max_q - min_q)

def full_state_act(full_state, action):
    new_position = act_position(full_state.position, action)
    if not in_bounds(new_position, 0, FULL_GRID_X - 1, 0, FULL_GRID_Y - 1):
        new_position = full_state.position
    full_state.litter[new_position[0], new_position[1]] = 0
    return FullState(new_position, full_state.litter, full_state.obstacles)

np.random.seed(1)
full_state = create_full_state()
init_litter = np.copy(full_state.litter)

W_LITTER = 0.75
W_OBSTACLES = 0
W_SIDEWALK = 0
W_FORWARD = 0.25

litter_padding = LITTER_GRID_DIM // 2
obstacles_padding = OBSTACLES_GRID_DIM // 2
padded_litter = np.zeros((FULL_GRID_X + 2 * litter_padding, FULL_GRID_Y + 2 * litter_padding))
padded_obstacles = np.zeros((FULL_GRID_X + 2 * obstacles_padding, FULL_GRID_Y + 2 * obstacles_padding))
padded_obstacles[obstacles_padding:obstacles_padding + FULL_GRID_X, obstacles_padding:obstacles_padding + FULL_GRID_Y] = full_state.obstacles
position_xs = [full_state.position[0]]
position_ys = [full_state.position[1]]
while not position_is_terminal(full_state.position):
    x, y = full_state.position
    padded_litter[litter_padding:litter_padding + FULL_GRID_X, litter_padding:litter_padding + FULL_GRID_Y] = full_state.litter
    litter_grid = padded_litter[x: x + 2 * litter_padding + 1, y: y + 2 * litter_padding + 1]
    obstacles_grid = padded_obstacles[x: x + 2 * obstacles_padding + 1, y: y + 2 * obstacles_padding + 1]
    q_val_mins = np.ones((num_modules,)) * 100000
    q_val_maxes = np.ones((num_modules,)) * -100000
    for action in range(num_actions):
        litter_q_val = get_q_val(litter_qtable, (ndarray_to_tuple(litter_grid), action))
        obstacles_q_val = get_q_val(obstacles_qtable, (ndarray_to_tuple(obstacles_grid), action))
        sidewalk_q_val = get_q_val(sidewalk_qtable, (full_state.position, action))
        forward_q_val = get_q_val(forward_qtable, (full_state.position, action))
        if litter_q_val < q_val_mins[0]:
            q_val_mins[0] = litter_q_val
        if obstacles_q_val < q_val_mins[1]:
            q_val_mins[1] = obstacles_q_val
        if sidewalk_q_val < q_val_mins[2]:
            q_val_mins[2] = sidewalk_q_val
        if forward_q_val < q_val_mins[3]:
            q_val_mins[3] = forward_q_val

        if litter_q_val > q_val_maxes[0]:
            q_val_maxes[0] = litter_q_val
        if obstacles_q_val > q_val_maxes[1]:
            q_val_maxes[1] = obstacles_q_val
        if sidewalk_q_val > q_val_maxes[2]:
            q_val_maxes[2] = sidewalk_q_val
        if forward_q_val > q_val_maxes[3]:
            q_val_maxes[3] = forward_q_val

    maxes = []
    max_s = -1 * np.inf
    for action in range(num_actions):
        litter_val = rescale_qval(get_q_val(litter_qtable, (ndarray_to_tuple(litter_grid), action)), q_val_maxes[0], q_val_mins[0])
        obstacles_val = rescale_qval(get_q_val(obstacles_qtable, (ndarray_to_tuple(obstacles_grid), action)), q_val_maxes[1], q_val_mins[1])
        sidewalk_val = rescale_qval(get_q_val(sidewalk_qtable, (full_state.position, action)), q_val_maxes[2], q_val_mins[2])
        forward_val = rescale_qval(get_q_val(forward_qtable, (full_state.position, action)), q_val_maxes[3], q_val_mins[3])

        s = litter_val * W_LITTER + obstacles_val * W_OBSTACLES + sidewalk_val * W_SIDEWALK + forward_val * W_FORWARD
        print("{} weighted value: {}".format(action_map[action], s))
        if s > max_s:
            maxes = [action]
            max_s = s
        elif s == max_s:
            maxes.append(action)

    action = np.random.choice(maxes)
    print("Selected action: {}".format(action_map[action]))
    new_full_state = full_state_act(full_state, action)
    full_state = new_full_state
    position_xs.append(full_state.position[0])
    position_ys.append(full_state.position[1])
    print(position_xs, position_ys)
    show_full_state(position_xs, position_ys, full_state)

