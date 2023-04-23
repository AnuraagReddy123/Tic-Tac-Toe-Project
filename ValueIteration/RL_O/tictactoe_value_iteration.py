#import modules
import sys
sys.path.append('..')
sys.path.append('../..')

import pygame
import pygame.locals as pl
import time
import utils
import numpy as np
import os

import Constants as C

join = os.path.join

pygame.init()

screen_height = 100 * C.BOARD_DIM
screen_width = 100 * C.BOARD_DIM
line_width = 6
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('Tic Tac Toe')

#define colours
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)

#define font
font = pygame.font.SysFont(None, 40)

#define variables
clicked = False
player = 1
pos = (0,0)
markers = []
game_over = False
winner = 0

#setup a rectangle for "Play Again" Option
again_rect = pl.Rect(screen_width // 2 - 80, screen_height // 2, 160, 50)

#create empty 3 x 3 list to represent the grid
for x in range (C.BOARD_DIM):
    row = [0] * C.BOARD_DIM
    markers.append(row)

#------------------------------------------ Board functions --------------------------------------------

def draw_board():
    bg = (255, 255, 210)
    grid = (50, 50, 50)
    screen.fill(bg)
    for x in range(1,C.BOARD_DIM):
        pygame.draw.line(screen, grid, (0, 100 * x), (screen_width,100 * x), line_width)
        pygame.draw.line(screen, grid, (100 * x, 0), (100 * x, screen_height), line_width)

def draw_markers():
    x_pos = 0
    for x in markers:
        y_pos = 0
        for y in x:
            if y == 1:
                pygame.draw.line(screen, red, (x_pos * 100 + 15, y_pos * 100 + 15), (x_pos * 100 + 85, y_pos * 100 + 85), line_width)
                pygame.draw.line(screen, red, (x_pos * 100 + 85, y_pos * 100 + 15), (x_pos * 100 + 15, y_pos * 100 + 85), line_width)
            if y == -1:
                pygame.draw.circle(screen, green, (x_pos * 100 + 50, y_pos * 100 + 50), 38, line_width)
            y_pos += 1
        x_pos += 1    

def check_game_over():
    global game_over
    global winner

    x_pos = 0
    for x in markers:
        #check columns
        if sum(x) == C.BOARD_DIM:
            winner = 1
            game_over = True
        if sum(x) == -C.BOARD_DIM:
            winner = 2
            game_over = True
        #check rows
        sum_check = 0
        for y in markers:
            sum_check += y[x_pos]
        if sum_check == C.BOARD_DIM:
            winner = 1
            game_over = True
        if sum_check == -C.BOARD_DIM:
            winner = 2
            game_over = True
        x_pos += 1

    #check cross
    sum_check = 0
    for x in range(C.BOARD_DIM):
        sum_check += markers[x][x]
    if sum_check == C.BOARD_DIM:
        winner = 1
        game_over = True
    if sum_check == -C.BOARD_DIM:
        winner = 2
        game_over = True
    
    sum_check = 0
    for x, y in zip(range(C.BOARD_DIM), reversed(range(C.BOARD_DIM))):
        sum_check += markers[x][y]
    if sum_check == C.BOARD_DIM:
        winner = 1
        game_over = True
    if sum_check == -C.BOARD_DIM:
        winner = 2
        game_over = True
        
    #check for tie
    if game_over == False:
        tie = True
        for row in markers:
            for i in row:
                if i == 0:
                    tie = False
        #if it is a tie, then call game over and set winner to 0 (no one)
        if tie == True:
            game_over = True
            winner = 0

def draw_game_over(winner):

    if winner != 0:
        end_text = "Player " + str(winner) + " wins!"
    elif winner == 0:
        end_text = "You have tied!"

    end_img = font.render(end_text, True, blue)
    pygame.draw.rect(screen, green, (screen_width // 2 - 100, screen_height // 2 - 60, 200, 50))
    screen.blit(end_img, (screen_width // 2 - 100, screen_height // 2 - 50))

    again_text = 'Play Again?'
    again_img = font.render(again_text, True, blue)
    pygame.draw.rect(screen, green, again_rect)
    screen.blit(again_img, (screen_width // 2 - 80, screen_height // 2 + 10))

# ------------------------------------------- RL functions -----------------------------------------------

state_to_index = np.load(join('Params', 'state_to_index.npy'), allow_pickle=True).item()
index_to_action = np.load(join('Params', 'index_to_action.npy'), allow_pickle=True).item()
policy = np.load(join('Params', 'policy.npy'), allow_pickle=True)

def choose_action(markers:list, policy_bool=True):
    markers = np.array(markers)
    if policy_bool == False:
        # choose random action
        actions = utils.available_actions(markers)
        action = actions[np.random.randint(len(actions))]
        intermediate_state = markers.copy()
        intermediate_state[action[0]][action[1]] = 1
        return intermediate_state
    else:
        # use policy to determine best action
        state_index = state_to_index[tuple(markers.flatten())]
        action = index_to_action[int(policy[state_index])]
        intermediate_state = markers.copy()
        intermediate_state[action[0]][action[1]] = -1
        return intermediate_state
    


#main loop
run = True
while run:

    #draw board and markers first
    draw_board()
    draw_markers()

    #handle events
    for event in pygame.event.get():
        #handle game exit
        if event.type == pygame.QUIT:
            run = False
        
        #run new game
        if game_over == False:
            #check for mouseclick
            if player == 1:
                if event.type == pygame.MOUSEBUTTONDOWN and clicked == False:
                    clicked = True
                if event.type == pygame.MOUSEBUTTONUP and clicked == True:
                    clicked = False
                    pos = pygame.mouse.get_pos()
                    x = pos[0] // 100
                    y = pos[1] // 100
                    if markers[x][y] == 0:
                        markers[x][y] = 1
                        player *= -1
                        check_game_over()
                        pygame.event.post(pygame.event.Event(pygame.MOUSEMOTION))
            
            elif player == -1:
                markers = choose_action(markers, policy_bool=True)
                player *= -1
                check_game_over()
                pygame.event.post(pygame.event.Event(pygame.MOUSEMOTION))

    #check if game has been won
    if game_over == True:
        draw_game_over(winner)
        #check for mouseclick to see if we clicked on Play Again
        if event.type == pygame.MOUSEBUTTONDOWN and clicked == False:
            clicked = True
        if event.type == pygame.MOUSEBUTTONUP and clicked == True:
            clicked = False
            pos = pygame.mouse.get_pos()
            if again_rect.collidepoint(pos):
                #reset variables
                game_over = False
                player = 1
                pos = (0,0)
                markers = []
                winner = 0
                #create empty 3 x 3 list to represent the grid
                for x in range (C.BOARD_DIM):
                    row = [0] * C.BOARD_DIM
                    markers.append(row)

    #update display
    pygame.display.update()

pygame.quit()
