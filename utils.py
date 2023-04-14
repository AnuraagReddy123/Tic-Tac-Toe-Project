import Constants as C

def check_if_wins(markers, player):
    '''
    Check if the player wins the game
    markers: the markers on the board 
    player: the player to check (-1 or 1)
    '''
    #check columns  
    for x in markers:
        if sum(x) == C.BOARD_DIM * player:
            return True
    #check rows
    for x in range(C.BOARD_DIM):
        sum_check = 0
        for y in markers:
            sum_check += y[x]
        if sum_check == C.BOARD_DIM * player:
            return True
    #check cross
    sum_check = 0
    for x in range(C.BOARD_DIM):
        sum_check += markers[x][x]
    if sum_check == C.BOARD_DIM * player:
        return True
    sum_check = 0
    for x, y in zip(range(C.BOARD_DIM), reversed(range(C.BOARD_DIM))):
        sum_check += markers[x][y]
    if sum_check == C.BOARD_DIM * player:
        return True
    return False

def check_draw(markers):
    '''
    Check if the game is a draw
    markers: the markers on the board 
    '''
    for row in markers:
        for i in row:
            if i == 0:
                return False
    return True