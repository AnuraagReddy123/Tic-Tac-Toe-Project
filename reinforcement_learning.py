import numpy as np
import Constants as C



# State representation = C.BOARD_DIM x C.BOARD_DIM matrix
# 0 = empty space, 1 = player 1, -1 = player 2
# Total possible states = 3^(C.BOARD_DIM x C.BOARD_DIM)
# Get all possible states
states = []
state_to_index = {}
index_to_state = {}
absorbing_states = []

cnt = 0
for i in range(3**(C.BOARD_DIM * C.BOARD_DIM)):
    temp = np.base_repr(i, base=3)
    temp = temp.zfill(C.BOARD_DIM * C.BOARD_DIM)
    temp = np.array(list(temp), dtype=int)
    temp[temp == 2] = -1

    # If state is not valid, (no. of 1s != no. of -1s or no. of 1s != no. of -1s + 1), skip it
    if np.count_nonzero(temp == 1) != np.count_nonzero(temp == -1) and np.count_nonzero(temp == 1) != np.count_nonzero(temp == -1) + 1:
        continue

    temp = temp.reshape(C.BOARD_DIM, C.BOARD_DIM)
    states.append(temp)
    state_to_index[tuple(temp.flatten())] = cnt
    index_to_state[cnt] = temp
    cnt += 1


# Get all absorbing states (states where game is over)
for state in states:
    x_pos = 0
    for x in state:
        # check columns
        if sum(x) == C.BOARD_DIM:
            absorbing_states.append(state_to_index[tuple(state.flatten())])
        if sum(x) == -C.BOARD_DIM:
            absorbing_states.append(state_to_index[tuple(state.flatten())])
        # check rows
        sum_check = 0
        for y in state:
            sum_check += y[x_pos]
        if sum_check == C.BOARD_DIM:
            absorbing_states.append(state_to_index[tuple(state.flatten())])
        if sum_check == -C.BOARD_DIM:
            absorbing_states.append(state_to_index[tuple(state.flatten())])
        x_pos += 1

    # check cross
    sum_check = 0
    for x in range(C.BOARD_DIM):
        sum_check += state[x][x]
    if sum_check == C.BOARD_DIM:
        absorbing_states.append(state_to_index[tuple(state.flatten())])
    if sum_check == -C.BOARD_DIM:
        absorbing_states.append(state_to_index[tuple(state.flatten())])
        
    sum_check = 0
    for x, y in zip(range(C.BOARD_DIM), reversed(range(C.BOARD_DIM))):
        sum_check += state[x][y]
    if sum_check == C.BOARD_DIM:
        absorbing_states.append(state_to_index[tuple(state.flatten())])
    if sum_check == -C.BOARD_DIM:
        absorbing_states.append(state_to_index[tuple(state.flatten())])

    # check for tie
    if state_to_index[tuple(state.flatten())] not in absorbing_states:
        tie = True
        for row in state:
            for i in row:
                if i == 0:
                    tie = False
        if tie == True:
            absorbing_states.append(state_to_index[tuple(state.flatten())])


# Build probability transition matrix
# Shape = (total possible states, total possible states)
# Each row is a state, each column is a possible next state
# Each cell is the probability of transitioning from the row state to the column state

def available_actions(markers:list):
    positions = []
    for x in range(C.BOARD_DIM):
        for y in range(C.BOARD_DIM):
            if markers[x][y] == 0:
                positions.append([x,y])

    return positions

probability_transition_matrix = np.zeros((len(states), len(states)))

for state in states:
    state_index = state_to_index[tuple(state.flatten())]
    if state_index not in absorbing_states:
        actions = available_actions(state)
        for action in actions:
            next_state = state.copy()
            # Count number of 1's in state and -1's in state
            # If number of 1's > number of -1's, next state is -1
            # If number of 1's < number of -1's, next state is 1
            if sum(state.flatten()) > 0:
                next_state[action[0]][action[1]] = -1
            else:
                next_state[action[0]][action[1]] = 1
            probability_transition_matrix[state_index][state_to_index[tuple(next_state.flatten())]] = 1/len(actions)
    
    else:
        probability_transition_matrix[state_index][state_index] = 1


def choose_action(markers:list, policy=True):
    markers = np.array(markers)
    state_index = state_to_index[tuple(markers.flatten())]
    if policy == False:
        # use Probability transition matrix
        next_state_index = np.random.choice(len(states), p=probability_transition_matrix[state_index])
        return index_to_state[next_state_index]
    else:
        # Argmax of probability transition matrix
        next_state_index = np.argmax(probability_transition_matrix[state_index])
        return index_to_state[next_state_index]