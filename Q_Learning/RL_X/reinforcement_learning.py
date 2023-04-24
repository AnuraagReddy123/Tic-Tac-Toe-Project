import sys
sys.path.append('../..')

import numpy as np
import Constants as C
import utils
import os
import random

# Seed for reproducibility
np.random.seed(0)
random.seed(0)

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

    # If # of Xs == # of Os + 1
    if np.count_nonzero(temp == 1) == np.count_nonzero(temp == -1) + 1:
        # If not X win and not draw, skip it
        if not utils.check_if_wins(temp.reshape(C.BOARD_DIM, C.BOARD_DIM), 1) and not utils.check_draw(temp.reshape(C.BOARD_DIM, C.BOARD_DIM)):
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


# ----------------------------------- Q Learning ----------------------------------- #

# Reward function
def get_reward(state, action, next_state):
    if utils.check_if_wins(next_state, 1):
        return 1
    elif utils.check_if_wins(next_state, -1):
        return -1
    elif utils.check_draw(next_state):
        return 0
    else:
        return 0
    

# Q Learning
# Initialize Q(s, a) arbitrarily
# For each episode:
#   Initialize S
#   While S is not terminal:
#       Choose A from S using policy derived from Q (e.g. epsilon-greedy)
#       Take action A, observe R, S'
#       Q(S, A) = Q(S, A) + alpha * (R + gamma * max_a' Q(S', a') - Q(S, A))
#       S = S'
#   Until S is terminal

o_policy = np.load(join('..', '..', 'ValueIteration', 'RL_OX', 'Params', 'policy.npy'))
o_state_to_index = np.load(join('..', '..', 'ValueIteration', 'RL_OX', 'Params', 'state_to_index.npy'), allow_pickle=True).item()

def get_next_state(state, action):
    intermediate_state = state.copy()
    intermediate_state[action[0]][action[1]] = 1
    available_actions = utils.available_actions(intermediate_state)
    if len(available_actions) == 0:
        return intermediate_state
    else:
        if utils.check_if_wins(intermediate_state, 1) or utils.check_draw(intermediate_state):
            return intermediate_state
        else:
            # action = o_policy[o_state_to_index[tuple(intermediate_state.flatten())]]
            # action = index_to_action[action]
            action = random.choice(available_actions)
            intermediate_state[action[0]][action[1]] = -1
            return intermediate_state

gamma = C.GAMMA
alpha = C.ALPHA
iterations = C.NUM_ITERATIONS
policy = np.zeros(len(states)) # State->action_index
Q = np.zeros((len(states), len(all_actions))) # State, action->Q value

# For each episode:
for episode in range(100000):
    print('Episode: ', episode)
    # Initialize S
    S = random.choice(states)
    S_index = state_to_index[tuple(S.flatten())]

    # While S is not terminal:
    while S_index not in absorbing_states:
        # Choose A from S using policy derived from Q (e.g. epsilon-greedy)
        # epsilon-greedy
        valid_actions = utils.available_actions(S)
        if random.random() < 0.1:
            A = random.choice(valid_actions)
        else:
            A = all_actions[np.argmax(Q[S_index, :])]
            # Check if A is valid
            while A not in valid_actions:
                A = random.choice(valid_actions)

        A_index = action_to_index[tuple(A)]

        # Take action A, observe R, S'
        S_prime = get_next_state(S, A)
        S_prime_index = state_to_index[tuple(S_prime.flatten())]
        R = get_reward(S, A, S_prime)

        # Q(S, A) = Q(S, A) + alpha * (R + gamma * max_a' Q(S', a') - Q(S, A))
        Q[S_index, A_index] = Q[S_index, A_index] + alpha * (R + gamma * np.max(Q[S_prime_index, :]) - Q[S_index, A_index])

        # S = S'
        S = S_prime
        S_index = S_prime_index

# Get policy from Q
for state in states:
    state_index = state_to_index[tuple(state.flatten())]
    valid_actions = utils.available_actions(state)
    valid_actions = [tuple(action) for action in valid_actions]
    if len(valid_actions) == 0:
        continue
    action_index = np.argmax(Q[state_index, :])
    action = all_actions[action_index]
    while action not in valid_actions:
        Q[state_index, action_index] = -999999
        action_index = np.argmax(Q[state_index, :])
        action = all_actions[action_index]
        action = tuple(action)
    policy[state_index] = action_index

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

# # Breaking condition
# if np.array_equal(policy, prev_policy):
#     break
# else:
#     # Mention where they are different
#     if i % 100 == 0:
#         for i in range(len(policy)):
#             if policy[i] != prev_policy[i]:
#                 print('Different at index: ', i)
#                 print('Previous policy: ', prev_policy[i])
#                 print('Current policy: ', policy[i])
#                 print('State: ', index_to_state[i])
#                 print('Action: ', index_to_action[policy[i]])
#                 print('--------------------------------------')

# # Save policy
# np.save(join("Params", "policy.npy"), policy)
# # Save states
# np.save(join("Params", "states.npy"), states)
# # Save state_to_index
# np.save(join("Params", "state_to_index.npy"), state_to_index)
# # Save index_to_state
# np.save(join("Params", "index_to_state.npy"), index_to_state)
# # Save action_to_index
# np.save(join("Params", "action_to_index.npy"), action_to_index)
# # Save index_to_action
# np.save(join("Params", "index_to_action.npy"), index_to_action)



