# Adapted from https://mblogscode.com/2016/06/03/python-naughts-crossestic-tac-toe-coding-unbeatable-ai/

import gym
import numpy as np
import json   

import os
import time

import config

from stable_baselines import logger
import pandas as pd
from tabulate import tabulate

os.environ["DISPLAY"] = ":0"

import pygame, sys
from pygame.locals import *
import time
import pygame

import random

def grid(window, size, rows, cols):

    offset = size #offset grilla

    total_y = size * rows #tamano total eje y
    total_x = size * cols #tamano total eje x
    # distanceBtwRows = size // rows  esto es size ahora

    x = offset
    y = offset

    print('dibujando lineas horizontales')
    cont_cols = 0
    cont_rows = 0
    for l in range(rows+1):
        print('x1', 'y1',x, y)
        print('x2', 'y2',total_x, y)


        pygame.draw.line(window, (0,0,0), (x,y), (offset + total_x, y))
        y += size
        cont_cols+=1

    x = offset
    y = offset

    print('dibujando lineas verticales')

    for z in range(cols+1):

        print('x1', 'y1',x, y)
        print('x2', 'y2',x, total_y)

        pygame.draw.line(window, (0,0,0), (x,y), (x, offset + total_y))
        x += size
        cont_rows+=1



def circle(window, size, var_1, var_2):


    circle_x = size + var_2*size + (size/2)
    circle_y = size + var_1*size + (size/2)
    pygame.draw.circle(window, (0, 255, 0),
                   [circle_x, circle_y], 10, 2)

def label (window, size, rows_name, cols_name):

    myfont = pygame.font.SysFont("monospace", 15)

    # render text
    label = myfont.render(cols_name, 1, (0,0,0))
    label2 = myfont.render(rows_name, 2, (0,0,0))
    window.blit(label, (size, size/4))
    window.blit(label2, (0, size))

def options(window, size, options_1, options_2):


    myfont =  pygame.font.SysFont("monospace", 10)
    x = size
    y = size/2
    for i, value in enumerate(options_2):
        label = myfont.render(value, 1, (0,0,0))
        window.blit(label, (x*(i+1) + size/2, y))

    x = size/2
    y = size

    for i, value in enumerate(options_1):
        label = myfont.render(value, 1, (0,0,0))
        window.blit(label, (x, y*(i+1) + size/2))

def draw_grid(window, size, row, cols, labels, opt1, opt2, x, y):

    window.fill((255,255,255))
    grid(window, size, rows, cols)
    circle(window, size, x, y)
    label(window, size, labels[0], labels[1])
    options(window, size, opt1, opt2)
    pygame.display.update()


def game(running,size, rows, cols, x, y):
    window_width = cols*size + 2*size
    window_height = rows*size + 2*size 
    pygame.init()

    self.window = pygame.display.set_mode((window_width, window_height))
    labels = [self.board1.get_var_1(), self.board1.get_var_2()]
    opt1 = self.board1.get_options_var_1()
    opt2 = self.board1.get_options_var_2()
    draw_grid(self.window, size, rows, cols, labels, opt1, opt2, 0,0)

    while running.value:
    
        # Check for event if user has pushed
        # any event in queue
        for event in pygame.event.get():
        
            # if event is of type quit then set
            # running bool to false
            if event.type == pygame.QUIT:
                print('finalizado')
                running.value = False
        draw_grid(self.window, size, rows, cols, labels, opt1, opt2, x.value,y.value)

'''
def redraw(window, x, y):

    circle(window, x, y)
    pygame.display.update()

'''

class Player():
    def __init__(self, id, token):
        self.id = id
        self.token = token
        

class Token():
    def __init__(self, symbol, number):
        self.number = number  #vacio number 0, solo para player
        self.symbol = symbol

class Board():
        def __init__(self, empty_token, number, var_x, var_y, vals_x, vals_y, rewards_csv_filepath):

            #vals_x, vals_y -> valores que pueden tomar x e y
            #var_x, var_y -> variables x e y del tablero
            self.empty_token = empty_token  #token vacio que se le pasa por parametro cuando se inicializa
            self.number = number
            self.var_x = var_x
            self.var_y = var_y
            self.vals_x = vals_x
            self.vals_y = vals_y
            self.player_x = 0
            self.player_y = 0
            self.rewards_grid = read_reward_board_csv(rewards_csv_filepath,
                                                            varx_name=self.var_x, 
                                                            vary_name=self.var_y,
                                                            width=len(vals_x),
                                                            height=len(vals_y))

        def get_grid(self):
            grid_shape = (len(self.vals_y),len(self.vals_x))
            #grid = np.ones(grid_shape) * self.empty_token

            grid = [self.empty_token] * (len(self.vals_y)*len(self.vals_x)) #se inicializa board con tokens vacios
            grid = np.array(grid).reshape(grid_shape) #board de posicion actual 
            #if(self.player_x is not None):
            grid[self.player_y,self.player_x] = self.player_token

            return grid

        def get_symbol_grid(self):
            grid_shape = (len(self.vals_y),len(self.vals_x))
            #grid = np.ones(grid_shape) * self.empty_token

            grid = [self.empty_token.symbol] * (len(self.vals_y)*len(self.vals_x)) #se inicializa board con tokens vacios
            grid = np.array(grid).reshape(grid_shape) #board de posicion actual 
            #if(self.player_x is not None):
            
            grid[self.player_y,self.player_x] = self.player_token.symbol

            return grid

        def get_position_grid(self):
            grid_shape = (len(self.vals_y),len(self.vals_x))
            position_grid = np.zeros(grid_shape) #se inicializa board con tokens vacios
            #if(self.player_x is not None):

            position_grid[self.player_y,self.player_x] = self.player_token.number
          
            return position_grid

        def get_la_grid(self, legal_positions):
            grid_shape = (len(self.vals_y),len(self.vals_x))
            la_grid = np.zeros(grid_shape)  #board de legal actions, a que posiciones te puedes mover 
            # update with legal positions
            for x in legal_positions[self.var_x]:
                y = self.player_y

                la_grid[y,x] = 1
            for y in legal_positions[self.var_y]:
                x = self.player_x

                la_grid[y,x] = 1
            return la_grid

        def get_player_x(self):
            return self.player_x 

        def get_player_y(self):
            return self.player_y

        def get_rewards_grid(self):
            return self.rewards_grid

        def set_player_x(self, value):
            self.player_x = value

        def set_player_y(self, value):
            self.player_y = value

        def set_player_position(self, x,y, token):
            self.player_x = x
            self.player_y = y
            self.player_token = token

        def get_options_var_1(self):
            return self.vals_x

        def get_options_var_2(self):
            return self.vals_y

        def get_var_1(self):
            return self.var_x

        def get_var_2(self):
            return self.var_y
        

class EvolutionEnv(gym.Env):


    def __init__(self, verbose = False, manual = False):


        super(EvolutionEnv, self).__init__()
        self.name = 'evolution'
        self.manual = manual
        
        self.n_players = 1
            
        # find way to automatize shapes
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(3,12,6), dtype=np.int32)
        
        self.verbose = verbose
        
        # import premes json
        f=open('./environments/evolution/evolution/envs/premes.json', "r")
        self.all_premes = json.loads(f.read())
        self.premes_quantity = len(list(self.all_premes.keys()))
        self.action_space = gym.spaces.Discrete(self.premes_quantity) # number of premes


    @property
    def observation(self):  #metodos de la clase, toma estado del juego y actualiza tablero posicion y legal positions. 
        position_grid_board1 = self.board1.get_position_grid()
        position_grid_board2 = self.board2.get_position_grid()
        position_grid_board3 = self.board3.get_position_grid()
        #position_grid = np.array([token.number for token in self.board]).reshape(self.grid_shape) #board de posicion actual 
        position_grid_board1=np.lib.pad(position_grid_board1, ((0,6-position_grid_board1.shape[0]),(0,6-position_grid_board1.shape[1])), 'constant', constant_values=(0))
        position_grid_board2=np.lib.pad(position_grid_board2, ((0,6-position_grid_board2.shape[0]),(0,6-position_grid_board2.shape[1])), 'constant', constant_values=(0))
        position_grid_board3=np.lib.pad(position_grid_board3, ((0,6-position_grid_board3.shape[0]),(0,6-position_grid_board3.shape[1])), 'constant', constant_values=(0))
        # print('## RESHAPED position_grids ##')
        final_position_grid = np.block([[[position_grid_board1]],[[position_grid_board2]],[[position_grid_board3]]])
        # print(final_position_grid.shape)
        # print(final_position_grid)

        # in board se t 1 to legal positions
        # print('## legal_positions ##')
        # print(self.legal_positions,'\n')
        la_grid_board1 = self.board1.get_la_grid(self.legal_positions)
        la_grid_board2 = self.board2.get_la_grid(self.legal_positions)
        la_grid_board3 = self.board3.get_la_grid(self.legal_positions)
        la_grid_board1=np.lib.pad(la_grid_board1, ((0,6-la_grid_board1.shape[0]),(0,6-la_grid_board1.shape[1])), 'constant', constant_values=(0))
        la_grid_board2=np.lib.pad(la_grid_board2, ((0,6-la_grid_board2.shape[0]),(0,6-la_grid_board2.shape[1])), 'constant', constant_values=(0))
        la_grid_board3=np.lib.pad(la_grid_board3, ((0,6-la_grid_board3.shape[0]),(0,6-la_grid_board3.shape[1])), 'constant', constant_values=(0))

        final_la_grid = np.block([[[la_grid_board1]],[[la_grid_board2]],[[la_grid_board3]]])
        # print(final_la_grid.shape)
        # print(final_la_grid)
        total_observation = np.hstack((final_position_grid,final_la_grid))
        if self.verbose:
            print('## Total Observation ##')
            print(total_observation)
        # print(total_observation.shape)
        # print(total_observation)

        return total_observation

    @property
    def legal_actions(self):
        legal_actions = []
        for preme_name in list(self.premes.keys()):
            # TODO check if restriction is complete
            preme_legality = True
            restrictions = self.premes[preme_name]["restrictions"]
            for restriction in restrictions:
                # if restriction["var_name"] == "":
                #     continue 
                if restriction["condition"] == "lower":
                    preme_legality = preme_legality and self.position[restriction["var_name"]]<restriction["value"]
                elif restriction["condition"] == "equal":
                    preme_legality = preme_legality and self.position[restriction["var_name"]]==restriction["value"]
                else: #restriction["condition"] == "higher"
                    preme_legality = preme_legality and self.position[restriction["var_name"]]>restriction["value"]
            if preme_legality:
                legal_actions.append({preme_name:self.premes[preme_name]})
        return np.array(legal_actions)
    
    @property
    def legal_positions(self):
        # position = [board1_x,board1_y,board2_x,board2_y,board3_x,board3_y]
        legal_positions = {"avance_solucion":[],
                            "modelo_negocio":[],
                            "total_fundadores":[],
                            "horas_dedicacion":[],
                            "problema_organico":[],
                            "punto_equilibrio":[]
                            }
        for legal_preme in self.legal_actions:
            legal_preme=list(legal_preme.values())[0]
            for preme_effect in legal_preme["effects"]:
                preme_effect["value"]=int(preme_effect["value"])
                if preme_effect["operator"] == "increase":
                    legal_positions[preme_effect["var_name"]].append(self.position[preme_effect["var_name"]]+preme_effect["value"])
                if preme_effect["operator"] == "decrease":
                    legal_positions[preme_effect["var_name"]].append(self.position[preme_effect["var_name"]]-preme_effect["value"])
                if preme_effect["operator"] == "set":
                   legal_positions[preme_effect["var_name"]].append(preme_effect["value"])
                if preme_effect["operator"] == "random":
                    legal_positions[preme_effect["var_name"]].append(random.randint(preme_effect["start"], preme_effect["stop"]))

        # keep only set of positions
        legal_positions = {k:list(set(v)) for k, v in legal_positions.items()}
        return legal_positions

    '''
    function to look into all the boards and returns a reward vector with the 
    corresponding reward for each board position
    '''
    def get_new_position_reward(self):
        game_over = False
        all_boards_rewards = 0
        for board in self.boards:
            # get board position
            x = self.position[board.var_x]
            y = self.position[board.var_y]
            # get reward in actual position
            all_boards_rewards+=int(board.rewards_grid[y,x])
            # consume reward: set actual position reward to 0
            board.rewards_grid[y,x] = 0

        return all_boards_rewards, game_over

    @property
    def current_player(self):
        return self.players[self.current_player_num]


    '''
    function to execute one movement
    handles rewards received for the action taken
    '''
    def step(self, action):
        # write executed action to actions_log.txt
        self.actions_logs.append(action)

        reward = [0]
        # check effect of selected action
        ids = []
        names = []
        for i,legal_preme in enumerate(self.legal_actions):
            preme_name = list(legal_preme.keys())[0]
            ids.append(list(legal_preme.values())[0]['id'])
            names.append(preme_name)
            if(action == list(legal_preme.values())[0]['id']):
                pos = i
                break

            #pos = i if action == list(legal_preme.values())[0]['id'] else 0  #modifique esto por lo de arriba



        if action not in ids:  # ilegal action, ends game, punishment
            if self.verbose:
                print("Action not in list")
            done = True
            reward = [-100] # TODO dejar en -1 o cambiar
        else: # legal action proceed 
            # apply all effects related to chosen action preme
            action_preme_name = names[pos]
            effects = self.premes[action_preme_name]["effects"] 
            for effect in effects:
                if effect["operator"] == "increase":
                    self.position[effect["var_name"]]=self.position[effect["var_name"]]+effect["value"]
                elif effect["operator"] == "decrease":
                    self.position[effect["var_name"]]=self.position[effect["var_name"]]-effect["value"]
                else:
                    self.position[effect["var_name"]]=effect["value"]
            self.turns_taken += 1

            # get rewards from new position in all grids
            r, done = self.get_new_position_reward()
            reward = [r]
        
            # update board
            if self.verbose:
                print('new position dict: ',self.position)

            #self.board[old_x*(old_y+1)] = Token('🔳', 0)
            #self.board[new_x*(new_y+1)] = self.players[0].token
            self.board1.set_player_position(self.position["avance_solucion"],
                                            self.position["modelo_negocio"], 
                                            self.players[0].token)
            self.board2.set_player_position(self.position["total_fundadores"],
                                            self.position["horas_dedicacion"], 
                                            self.players[0].token)
            self.board3.set_player_position(self.position["problema_organico"],
                                            self.position["punto_equilibrio"], 
                                            self.players[0].token)
            self.done = done
            # remove preme once used if not repetitive
            if not list(legal_preme.values())[0]['repetitive']:
                self.premes.pop(preme_name, None)

            # game over condition: no more premes for now
            if not self.premes:
                if self.verbose:
                    print('## GAME OVER: NO MORE PREMES ##')
                done = True

        # if done write self.actions_log to file
        if done:
            # timestr = time.strftime("%Y%m%d-%H%M%S")
            # f = open(timestr+"-actions_log.txt", "a")
            f = open("actions_log.txt", "a")
            f.write("\n"+str(self.actions_logs))
            f.close()

        return self.observation, reward, done, {}

    def reset(self):


        # paths de matrices de rewards etiquetadas por expertos
        rewards_csv_filepath='./environments/evolution/evolution/envs/evo2_reinforcement_learning_matrices - rewards de exploracion 20.csv'
        
        # inicializar boards
        
        self.board1 = Board(Token('⬜', 0), 1, 
                            'avance_solucion', 'modelo_negocio', 
                            ['idea', 'concepto', 'prototipo', 'mvp', 'ventas', 'crecimiento'],
                            ['producto', 'servicio', 'plataforma', 'ecosistema'],
                            rewards_csv_filepath=rewards_csv_filepath)
        self.board2 = Board(Token('⬜', 0), 2, 
                            'total_fundadores', 'horas_dedicacion', 
                            ['1', '2', '3', '4', '5', '6'],
                            ['0-5', '6-10', '10-30', '30-45'],
                            rewards_csv_filepath=rewards_csv_filepath)
        self.board3 = Board(Token('⬜', 0), 3, 
                            'problema_organico', 'punto_equilibrio', 
                            ['no', 'si'],
                            ['no', 'pronto', 'si'],
                            rewards_csv_filepath=rewards_csv_filepath)
        self.boards = [self.board1,self.board2,self.board3]
        self.players = [Player('Startup1', Token('🔴', 1))] #se inicializan los players con su toquen y numero de jugador

        # start player at certain default position in vevery board
        # define attribute that contains position coords
        self.position = {"avance_solucion":0,
                        "modelo_negocio":0,
                        "total_fundadores":0,
                        "horas_dedicacion":0,
                        "problema_organico":0,
                        "punto_equilibrio":0
        
                        }
        if self.verbose:                        
            print(self.position)
        self.board1.set_player_position(0,0,self.players[0].token) #se posiciona el token circulo en la posicion 0,0 para el player 1 (solo hay un player)
        self.board2.set_player_position(0,0,self.players[0].token) #se posiciona el token circulo en la posicion 0,0 para el player 1 (solo hay un player)
        self.board3.set_player_position(0,0,self.players[0].token) #se posiciona el token circulo en la posicion 0,0 para el player 1 (solo hay un player)
        
        
        # Start grid 

        global rows, cols, size
        size = 100
        rows = len(self.board1.get_options_var_1())
        cols = len(self.board1.get_options_var_2())

        x = multiprocessing.Value('i')
        y = multiprocessing.Value('i')
        running = multiprocessing.Value('i')

        x.value = 0
        y.value = 0
        running.value = 1

        print(rows)
        print(cols)

        game = multiprocessing.Process(target=game, args(running,size, rows, cols, x, y))





        # set starting position rewards to 0
        r, done = self.get_new_position_reward()

        
        self.current_player_num = 0  #player que esta en el turno
        self.turns_taken = 0  #la cantidad de turnos 
        self.done = False #cuando se acabo el juego total
        self.actions_logs=[] # reset actions log
        self.premes = self.all_premes.copy()
        logger.debug(f'\n\n---- NEW GAME ----')
        return self.observation


    def render(self, mode='human', close=False, verbose = False):
        global cols, rows, size
        logger.debug('')
        if close:
            return
        if self.done:
            logger.debug(f'GAME OVER')
            running.value = 0 #se finaliza render
        else:
            logger.debug(f"It is Player {self.current_player.id}'s turn to move")
        board1 = self.board1.get_symbol_grid()
        board2 = self.board2.get_symbol_grid()
        board3 = self.board3.get_symbol_grid()
        
        board1_df = pd.DataFrame(data =board1,
                                    index=['producto', 'servicio', 'plataforma', 'ecosistema'],
                                    columns=['idea', 'concepto', 'prototipo', 'mvp', 'ventas', 'crecimiento'])
        board2_df = pd.DataFrame(data =board2,
                                    index=['0-5', '6-10', '10-30', '30-45'],
                                    columns=['1', '2', '3', '4', '5', '6'])
        board3_df = pd.DataFrame(data =board3,
                                    index=['no', 'pronto', 'si'],
                                    columns=['no', 'si'])
        #Render on window

        #labels = [self.board1.get_var_1(), self.board1.get_var_2()]
        #opt1 = self.board1.get_options_var_1()
        #opt2 = self.board1.get_options_var_2()

        x.value = self.board1.get_player_x()
        y.value = self.board1.get_player_y()

        #draw_grid(self.window, size, rows, cols, labels, opt1, opt2, self.board1.get_player_x(),self.board1.get_player_y())
        #pygame.event.get()

        if self.verbose:
            print("## Reward Boards ##")
            print(self.board1.get_rewards_grid(),'\n')                            
            print(self.board2.get_rewards_grid(),'\n')                            
            print(self.board3.get_rewards_grid(),'\n')                            
        # print("## Game Boards ##")
        # print("X: avance_solucion, Y: modelo_negocio")
        # print(tabulate(board1_df, headers='keys', tablefmt='grid'))
        # print("X: total_fundadores, Y: horas_dedicacion")
        # print(tabulate(board2_df, headers='keys', tablefmt='grid'))
        # print("X: problema_organico, Y: punto_equilibrio")
        # print(tabulate(board3_df, headers='keys', tablefmt='grid'))

        if self.verbose:
            logger.debug(f'\nObservation: \n{self.observation}')
        
        if not self.done:
            legal_actions_ui_list = []
            for legal_preme in self.legal_actions:
                id = list(legal_preme.values())[0]['id']
                name = list(legal_preme.keys())[0]
                legal_actions_ui_list.append(str(id)+': '+name)
            logger.debug(f"\nLegal actions: {legal_actions_ui_list}")

def read_reward_board_csv(filepath,varx_name, vary_name,width,height):
    board_array = [line.split(',') for line in open(filepath)]
    new_board = []
    for line in board_array:
        new_line = [word.replace('\n','') for word in line]
        new_board.append(new_line)
    board_array=np.array(new_board)

    # find the variable row
    varx_itemindex = np.where(board_array==varx_name)
    vary_itemindex = np.where(board_array==vary_name)
    # extract the values
    horizontal_values = board_array[varx_itemindex[0][0],varx_itemindex[1][0]+1:varx_itemindex[1][0]+1+width]
    vertical_values = board_array[vary_itemindex[0][0],vary_itemindex[1][0]+1:vary_itemindex[1][0]+1+height]

    # convert to matrix
    a=np.tile(horizontal_values, (height, 1)).astype(int)
    b=np.tile(vertical_values, (width,1)).T.astype(int)
    
    rewards_board = a+b
    return rewards_board