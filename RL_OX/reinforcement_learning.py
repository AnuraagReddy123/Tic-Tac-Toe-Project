import sys
sys.path.append('..')
sys.path.append('.')

import numpy as np
import Constants as C
import utils
import os

join = os.path.join

states = []
state_to_index = {}
index_to_state = {}
absorbing_states = []
action_to_index = {}
index_to_action = {}
all_actions = []


# Get all actions

for i in range(C.BOARD_DIM):
    for j in range(C.BOARD_DIM):
        all_actions.append((i, j))
        action_to_index[(i, j)] = len(action_to_index)
        index_to_action[len(index_to_action)] = (i, j)



# State representation = C.BOARD_DIM x C.BOARD_DIM matrix
# 0 = empty space, 1 = player 1, -1 = player 2
# Total possible states = 3^(C.BOARD_DIM x C.BOARD_DIM)
# Get all possible states
# States are what the agent sees
cnt = 0
for i in range(3**(C.BOARD_DIM * C.BOARD_DIM)):
    temp = np.base_repr(i, base=3)
    temp = temp.zfill(C.BOARD_DIM * C.BOARD_DIM)
    temp = np.array(list(temp), dtype=int)
    temp[temp == 2] = -1

    # If state is not valid, (no. of 1s != no. of -1s or no. of 1s != no. of -1s + 1), skip it
    if np.count_nonzero(temp == 1) != np.count_nonzero(temp == -1) and np.count_nonzero(temp == 1) != np.count_nonzero(temp == -1) + 1:
        continue

    # If X wins but # of 1s == # of -1s, skip it
    if utils.check_if_wins(temp.reshape(C.BOARD_DIM, C.BOARD_DIM), 1) and np.count_nonzero(temp == 1) == np.count_nonzero(temp == -1):
        continue

    # If O wins but # of 1s == # of -1s + 1, skip it
    if utils.check_if_wins(temp.reshape(C.BOARD_DIM, C.BOARD_DIM), -1) and np.count_nonzero(temp == 1) == np.count_nonzero(temp == -1) + 1:
        continue



    temp = temp.reshape(C.BOARD_DIM, C.BOARD_DIM)
    states.append(temp)
    state_to_index[tuple(temp.flatten())] = cnt
    index_to_state[cnt] = temp
    cnt += 1



# Get all absorbing states (states where game is over)
for state in states:
    if utils.check_if_wins(state, 1) or utils.check_if_wins(state, -1) or utils.check_draw(state):
        absorbing_states.append(state_to_index[tuple(state.flatten())])


# Build probability transition matrix and reward matrix
# Shape = (total possible states, 9 actions, total possible states)
# P(s, a, s') = probability of transitioning from state s to state s' when taking action a
# R(s, a, s') = reward of transitioning from state s to state s' when taking action a

# Get O's policy
x_policy = np.load(join('..', 'RL_X', 'Params', 'policy.npy')) # Maps state to action (action is a tuple)
x_state_to_index = np.load(join('..', 'RL_X', 'Params', 'state_to_index.npy'), allow_pickle=True).item()
x_index_to_state = np.load(join('..', 'RL_X', 'Params', 'index_to_state.npy'), allow_pickle=True).item()


probability_transition_matrix = np.zeros((len(states), C.BOARD_DIM**2, len(states)))
reward_matrix = np.zeros((len(states), C.BOARD_DIM**2, len(states)))

for state in states:
    # If # of Xs == # of Os
    if np.count_nonzero(state.flatten() == 1) == np.count_nonzero(state.flatten() == -1):
        # If O does not win and game is not a draw, skip it
        if not utils.check_if_wins(state, -1) and not utils.check_draw(state):
            continue
    state_index = state_to_index[tuple(state.flatten())]
    if state_index not in absorbing_states:
        actions_int = utils.available_actions(state)
        for action_int in actions_int:
            intermediate_state = state.copy()
            intermediate_state[action_int[0]][action_int[1]] = -1
            if utils.check_if_wins(intermediate_state, -1):
                probability_transition_matrix[state_index][action_to_index[tuple(action_int)]][state_index] = 1
                reward_matrix[state_index][action_to_index[tuple(action_int)]][state_index] = 1
            elif utils.check_draw(intermediate_state):
                probability_transition_matrix[state_index][action_to_index[tuple(action_int)]][state_index] = 1
                reward_matrix[state_index][action_to_index[tuple(action_int)]][state_index] = 0
            else:
                x_policy_action = x_policy[x_state_to_index[tuple(intermediate_state.flatten())]]
                x_policy_action = index_to_action[x_policy_action]
                intermediate_state[x_policy_action[0]][x_policy_action[1]] = 1
                probability_transition_matrix[state_index][action_to_index[tuple(action_int)]][state_index] = 1
                if utils.check_if_wins(intermediate_state, 1):
                    reward_matrix[state_index][action_to_index[tuple(action_int)]][state_index] = -1
                elif utils.check_draw(intermediate_state):                    
                    reward_matrix[state_index][action_to_index[tuple(action_int)]][state_index] = 0
    else:
        for action in all_actions:
            probability_transition_matrix[state_index][action_to_index[tuple(action)]][state_index] = 1


# Value function
# V(s) = expected return when starting from state s

# Initialize value function
value_function = np.zeros(len(states))

# Initialize policy
policy = np.zeros(len(states))

# Discount factor
gamma = 0.9

# Number of iterations
num_iterations = 10

# Epsilon
epsilon = 0.0001

# Run value iteration
for i in range(num_iterations):
    print("Iteration: ", i)
    new_value_function = np.zeros(len(states))
    for state in states:
        # If # of Xs == # of Os
        if np.count_nonzero(state.flatten() == 1) == np.count_nonzero(state.flatten() == -1):
            # If O does not win and game is not a draw, skip it
            if not utils.check_if_wins(state, -1) and not utils.check_draw(state):
                continue
        state_index = state_to_index[tuple(state.flatten())]
        if state_index not in absorbing_states:
            value = -np.inf
            for action in utils.available_actions(state):
                next_state_value = 0
                intermediate_state = state.copy()
                intermediate_state[action[0]][action[1]] = -1
                if utils.check_if_wins(intermediate_state, -1):
                    next_state_value = 1
                elif utils.check_draw(intermediate_state):
                    next_state_value = 0
                else:
                    x_policy_action = x_policy[x_state_to_index[tuple(intermediate_state.flatten())]]
                    x_policy_action = index_to_action[x_policy_action]
                    intermediate_state[x_policy_action[0]][x_policy_action[1]] = 1
                    intermediate_state_index = state_to_index[tuple(intermediate_state.flatten())]
                    if utils.check_if_wins(intermediate_state, 1):
                        next_state_value = -1
                    elif utils.check_draw(intermediate_state):
                        next_state_value = 0
                    else:
                        next_state_value = gamma * value_function[intermediate_state_index]

                if next_state_value > value:
                    value = next_state_value
                    new_policy = action_to_index[tuple(action)]
            new_value_function[state_index] = value
            policy[state_index] = new_policy
        else:
            new_value_function[state_index] = 0

    if np.sum(np.abs(new_value_function - value_function)) <= epsilon:
        break
    value_function = new_value_function
    print("Value function: ", value_function[0:10])

# Save policy
np.save(join("Params", "policy.npy"), policy)
# Save states
np.save(join("Params", "states.npy"), states)
# Save state_to_index
np.save(join("Params", "state_to_index.npy"), state_to_index)
# Save index_to_state
np.save(join("Params", "index_to_state.npy"), index_to_state)
# Save action_to_index
np.save(join("Params", "action_to_index.npy"), action_to_index)
# Save index_to_action
np.save(join("Params", "index_to_action.npy"), index_to_action)



