import numpy as np

def available_actions(markers:list):
    positions = []
    for x in range(3):
        for y in range(3):
            if markers[x][y] == 0:
                positions.append([x,y])

    return positions

def choose_action(markers:list, policy=True):
    actions = available_actions(markers)
    # Choose a random action
    if policy == False:
        return actions[np.random.randint(0, len(actions))]

    # Follow the policy select first action
    else:
        return actions[0]