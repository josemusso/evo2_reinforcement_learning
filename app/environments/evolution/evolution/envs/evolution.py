# Adapted from https://mblogscode.com/2016/06/03/python-naughts-crossestic-tac-toe-coding-unbeatable-ai/

from matplotlib.pyplot import xlabel
import gym
import numpy as np
import json
import cv2

import os
import time

from stable_baselines import logger
import pandas as pd
from tabulate import tabulate
import pygame
import sys
from pygame.locals import *
import time
import pygame
import random
import multiprocessing

# Variables Globales
var_tableros_path = './environments/evolution/evolution/envs/tableros.csv' # csv con los tableros, no deberia ser necesario modificar
premes_path = './environments/evolution/evolution/envs/premes.json' # json con las premes, cambiar segun sea necesario
cantidad_tableros_por_fila = 9 #numero de tableros por fila, no deberia ser necesario modificar
var_max_path = './environments/evolution/evolution/envs/var_max.json' # diccionario con los valores maximos de cada preme, modificar al cambiar el json de premes
assestment_json = './environments/evolution/evolution/envs/init_preme.json' #json con el assesment inicial
use_custom_assestment = True # si es True, se usa el json de assestment, si es False los tableros se inicializan en 0


class Player():
    def __init__(self, id, token):
        self.id = id
        self.token = token


class Token():
    def __init__(self, symbol, number):
        self.number = number  # vacio number 0, solo para player
        self.symbol = symbol


class Board():
    def __init__(self, empty_token, number, var_x, var_y, vals_x, vals_y, rewards_csv_filepath):

        # token vacio que se le pasa por parametro cuando se inicializa
        self.empty_token = empty_token
        self.number = number
        self.var_x = var_x  # var_x -> variables x del tablero
        self.var_y = var_y  # var_y -> variables y del tablero
        self.vals_x = vals_x  # vals_x-> valores que pueden tomar x
        self.vals_y = vals_y  # vals_y-> valores que pueden tomar y
        self.player_x = 0
        self.player_y = 0
        self.rewards_grid = read_reward_board_csv(rewards_csv_filepath,
                                                  varx_name=self.var_x,
                                                  vary_name=self.var_y,
                                                  width=len(vals_x),
                                                  height=len(vals_y))

    def get_grid(self):
        grid_shape = (len(self.vals_y), len(self.vals_x))

        # se inicializa board con tokens vacios
        grid = [self.empty_token] * (len(self.vals_y)*len(self.vals_x))
        grid = np.array(grid).reshape(grid_shape)  # board de posicion actual
        # if(self.player_x is not None):

        grid[self.player_y, self.player_x] = self.player_token

        return grid

    def get_symbol_grid(self):
        grid_shape = (len(self.vals_y), len(self.vals_x))

        # se inicializa board con tokens vacios
        grid = [self.empty_token.symbol] * (len(self.vals_y)*len(self.vals_x))
        grid = np.array(grid).reshape(grid_shape)  # board de posicion actual
        # if(self.player_x is not None):

        grid[self.player_y, self.player_x] = self.player_token.symbol

        return grid

    def get_position_grid(self):
        grid_shape = (len(self.vals_y), len(self.vals_x))

        # se inicializa board con tokens vacios
        position_grid = np.zeros(grid_shape)

        position_grid[self.player_y, self.player_x] = self.player_token.number

        return position_grid

    def get_la_grid(self, legal_positions):
        grid_shape = (len(self.vals_y), len(self.vals_x))

        # board de legal actions, a que posiciones te puedes mover
        la_grid = np.zeros(grid_shape)

        # update with legal positions for var_x
        for x in legal_positions[self.var_x]:
            y = self.player_y
            la_grid[y, x] = 1

        # update with legal positions for var_y
        for y in legal_positions[self.var_y]:
            x = self.player_x
            la_grid[y, x] = 1
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

    def set_player_position(self, x, y, token):
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

    def __init__(self, verbose=False, manual=False, env_test=False, initial_state=None):

        super(EvolutionEnv, self).__init__()

        self.name = 'evolution'
        self.manual = manual
        self.render_lib = 'opencv'
        
        self.env_test = env_test
        self.initial_state = initial_state
                
        self.n_players = 1

        f = open('logs/current_actions_log.txt', 'w')
        f.close()

        with open(var_tableros_path) as csv:
            lines = csv.readlines()

        self.data = lines

        self.n_tableros = int((len(lines)+1)/6)

        self.max_features = int(len(lines[0].split(',')))

        self.dim_complete = {1.0: 0, 2.0: 0, 3.0: 0, 4.0: 0, 5.0: 0}
        self.dim_name = {1.0: [], 2.0: [], 3.0: [], 4.0: [], 5.0: []}

        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(
            self.n_tableros, self.max_features*2, self.max_features), dtype=np.int32)

        self.verbose = verbose

        f = open(premes_path, "r")  # import premes json
        self.all_premes = json.loads(f.read())
        self.premes_quantity = len(list(self.all_premes.keys()))
        self.action_space = gym.spaces.Discrete(
            self.premes_quantity)  # number of premes

    @property
    # metodos de la clase, toma estado del juego y actualiza tablero posicion y legal positions.
    def observation(self):
        board_size = self.max_features

        position_grid = {}  # board de posicion actual
        for number in range(1, self.n_tableros+1):
            position_grid[f'position_grid_board{number}'] = self.boards_dict[f"board{number}"].get_position_grid(
            )
            position_grid[f'position_grid_board{number}'] = np.lib.pad(position_grid[f'position_grid_board{number}'], ((
                0, board_size-position_grid[f'position_grid_board{number}'].shape[0]), (0, board_size-position_grid[f'position_grid_board{number}'].shape[1])), 'constant', constant_values=(0))

        grid_list = []
        for position_grid_value in position_grid.values():
            grid_list.append([[position_grid_value]])
        final_position_grid = np.block(grid_list)

        la_grid = {}  # in board set 1 to legal positions
        for number in range(1, self.n_tableros+1):
            la_grid[f"la_grid_board{number}"] = self.boards_dict[f"board{number}"].get_la_grid(
                self.legal_positions)  # dim_error: cuando se pasa de los limites de la dimension genera un error
            la_grid[f"la_grid_board{number}"] = np.lib.pad(la_grid[f"la_grid_board{number}"], ((
                0, board_size-la_grid[f"la_grid_board{number}"].shape[0]), (0, board_size-la_grid[f"la_grid_board{number}"].shape[1])), 'constant', constant_values=(0))

        la_list = []
        for la_value in la_grid.values():
            la_list.append([[la_value]])

        final_la_grid = np.block(la_list)

        total_observation = np.hstack((final_position_grid, final_la_grid))
        if self.verbose:
            print('## Total Observation ##')
            print(total_observation)

        return total_observation

    @property
    def legal_actions(self):
        legal_actions = []
        for preme_name in list(self.premes.keys()):
            # TODO check if restriction is complete
            preme_legality = True
            restrictions = self.premes[preme_name]["restrictions"]

            for restriction in restrictions:
                if restriction["condition"] == "dim_complete":
                    preme_legality = preme_legality and self.dim_complete[restriction["value"]] >= 0.8
                elif restriction["condition"] == "not_null":
                    preme_legality = preme_legality and self.position[
                        restriction["var_name"]] > 0
                elif restriction["condition"] == "lower":
                    preme_legality = preme_legality and self.position[
                        restriction["var_name"]] < restriction["value"]
                elif restriction["condition"] == "equal":
                    preme_legality = preme_legality and self.position[
                        restriction["var_name"]] == restriction["value"]
                elif restriction["condition"] == "higher":
                    preme_legality = preme_legality and self.position[
                        restriction["var_name"]] > restriction["value"]
            if preme_legality:
                legal_actions.append({preme_name: self.premes[preme_name]})
        return np.array(legal_actions)

    @property
    def legal_positions(self):

        with open(var_max_path, 'r') as f:
            max_var = json.loads(f.read())

        legal_positions = {}
        for label_name in self.label_list:
            legal_positions[label_name] = []

        for legal_preme in self.legal_actions:
            legal_preme = list(legal_preme.values())[0]
            for preme_effect in legal_preme["effects"]:
                preme_effect["value"] = int(preme_effect["value"])

                if preme_effect["operator"] == "increase":
                    if self.position[preme_effect["var_name"]] >= max_var[preme_effect["var_name"]]-1:
                        continue
                    else:
                        legal_positions[preme_effect["var_name"]].append(
                            self.position[preme_effect["var_name"]]+preme_effect["value"])
                if preme_effect["operator"] == "decrease":
                    legal_positions[preme_effect["var_name"]].append(
                        self.position[preme_effect["var_name"]]-preme_effect["value"])
                if preme_effect["operator"] == "set":
                    legal_positions[preme_effect["var_name"]].append(
                        preme_effect["value"])
                if preme_effect["operator"] == "random":
                    value = random.randint(1, preme_effect["value"])
                    legal_positions[preme_effect["var_name"]].append(value)
                if preme_effect["operator"] == "exclusive_random":
                    value = random.randint(2, preme_effect["value"])
                    legal_positions[preme_effect["var_name"]].append(value)
        



        # keep only set of positions
        legal_positions = {k: list(set(v)) for k, v in legal_positions.items()}

        return legal_positions
    '''
    function to look into all the boards and returns a reward vector with the 
    corresponding reward for each board position
    '''

    def get_new_position_reward(self):
        game_over = False
        all_boards_rewards = 0
        for board in self.boards:  # get board position

            x = self.position[board.var_x]
            y = self.position[board.var_y]

            # get reward in actual position
            all_boards_rewards += int(board.rewards_grid[y, x])

            # consume reward: set actual position reward to 0
            board.rewards_grid[y, x] = 0

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
        ids = []
        names = []

        # check effect of selected action
        for i, legal_preme in enumerate(self.legal_actions):
            preme_name = list(legal_preme.keys())[0]
            ids.append(list(legal_preme.values())[0]['id'])
            names.append(preme_name)

            if (action == list(legal_preme.values())[0]['id']):
                pos = i
                break

        with open(var_max_path, 'r') as f:
            max_var = json.loads(f.read())

        if action not in ids:  # ilegal action, ends game, punishment
            if self.verbose:
                print(action, "Action not in list")
            done = True
            reward = [-100]  # TODO Evaluate negative reward
        else:  # legal action proceed, apply all effects related to chosen action preme
            action_preme_name = names[pos]
            effects = self.premes[action_preme_name]["effects"]
            for effect in effects:
                if effect["var_name"] not in self.dim_name[effect["dimension"]]:
                    self.dim_name[effect["dimension"]].append(
                        effect["var_name"])
                    self.dim_complete[effect["dimension"]] += effect["inc_dim"]
                if effect["operator"] == "increase":
                    if self.position[effect["var_name"]] >= max_var[effect["var_name"]]-1:
                        continue
                    else:
                        self.position[effect["var_name"]
                                      ] = self.position[effect["var_name"]]+effect["value"]
                if effect["operator"] == "decrease":
                    self.position[effect["var_name"]
                                  ] = self.position[effect["var_name"]]-effect["value"]
                if effect["operator"] == "set":
                    self.position[effect["var_name"]] = effect["value"]
                if effect["operator"] == "random":
                    value = random.randint(1, effect["value"])
                    self.position[effect["var_name"]] = value
                if effect["operator"] == "exclusive_random":
                    value = random.randint(2, effect["value"])
                    self.position[effect["var_name"]] = value

            self.turns_taken += 1

            # get rewards from new position in all grids
            r, done = self.get_new_position_reward()
            reward = [r]

            if self.verbose:  # update board
                print('new position dict: ', self.position)

            for number in range(1, self.n_tableros+1):
                self.boards_dict[f"board{number}"].set_player_position(self.position[self.label_list[2*number-2]],
                                                                       self.position[self.label_list[2*number-1]],
                                                                       self.players[0].token)

            self.done = done

            # substract repeated actions
            # TODO Evaluate substract reward for repeated actions
            if legal_preme[preme_name]['repetitive'] > 0:
                legal_preme[preme_name]['repetitive'] -= 1
            else:
                self.premes.pop(preme_name, None)

            if not self.premes:  # game over condition: no more premes for now
                if self.verbose:
                    print('## GAME OVER: NO MORE PREMES ##')
                done = True

        if done:  # if done write self.actions_log to file
            with open('logs/current_actions_log.txt', 'a') as new_log:
                new_log.write("\n"+str(self.actions_logs))

        return self.observation, reward, done, {}

    def reset(self):
        # paths de matrices de rewards etiquetadas por expertos
        rewards_csv_filepath = var_tableros_path

        label_list = []
        boards_dict = {}

        if self.env_test == False:
            if use_custom_assestment == True:
                with open(assestment_json, 'r') as f:
                    assestment = json.loads(f.read())

                with open('./environments/evolution/evolution/envs/init_state.json', 'r') as f:
                    init_state_preme = json.loads(f.read())

                effects = assestment['pr_initial_assestment']["effects"]
                for effect in effects:
                    if effect["operator"] == "set":
                        value = int(effect["value"])
                    elif effect["operator"] == "random":
                        value = int(random.randint(1, effect["value"]))
                    elif effect["operator"] == "exclusive_random":
                        value = int(random.randint(2, effect["value"]))
                    init_state_preme[effect["var_name"]] = value

        for number in range(1, self.n_tableros+1):  # inicializar boards
            list_data = self.data[(3*2*(number-1)):(3*2*(number-1))+5]

            x_value_list = list_data[0].split(',')
            y_value_list = list_data[3].split(',')

            c_x = 0
            for value in x_value_list:
                if value != '' and value != '\n':
                    c_x += 1
            c_y = 0
            for value in y_value_list:
                if value != '' and value != '\n':
                    c_y += 1

            x_label = list_data[1].split(',')[1]
            y_label = list_data[4].split(',')[1]
            label_list.append(x_label)
            label_list.append(y_label)

            x_values = x_value_list[2:2+(c_x-1)]
            y_values = y_value_list[2:2+(c_y-1)]

            boards_dict[f"board{number}"] = Board(Token('⬜', 0), number,
                                                  x_label, y_label,
                                                  x_values, y_values,
                                                  rewards_csv_filepath=rewards_csv_filepath)

        self.label_list = label_list
        self.boards_dict = boards_dict

        board_list = []
        for board in boards_dict.values():
            board_list.append(board)

        self.boards = board_list

        # se inicializan los players con su toquen y numero de jugador
        self.players = [Player('Startup1', Token('🔴', 1))]

        position_dict = {}  # start player at certain default position in vevery board, define attribute that contains position coords

        # Inicializacion de los tableros, por defecto se inicializa en la posicion 0, si es test se inicializa segun el initial_state, y si es custom se inicializa segun el assestment
        for label_name in self.label_list:
            if self.env_test: # test environment
                position_dict[label_name] = self.initial_state[label_name]
            elif use_custom_assestment: # custom initial state
                position_dict[label_name] = init_state_preme[label_name]
            else: #default state
                position_dict[label_name] = 0
        self.position = position_dict

        if self.verbose:
            print(self.position)

        # se posiciona el token circulo en la posicion 0,0 para el player 1 (solo hay un player)
        for number in range(1, self.n_tableros+1):
            self.boards_dict[f"board{number}"].set_player_position(
                0, 0, self.players[0].token)

        if self.render_lib == 'pygame':  # Start grid
            global rows, cols, size, x, y, running
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

            game = multiprocessing.Process(
                target=game, args=(running, size, rows, cols, x, y))

        r, done = self.get_new_position_reward()  # set starting position rewards to 0

        self.current_player_num = 0  # player que esta en el turno
        self.turns_taken = 0  # la cantidad de turnos
        self.done = False  # cuando se acabo el juego total
        self.actions_logs = []  # reset actions log
        self.premes = self.all_premes.copy()
        logger.debug(f'\n\n---- NEW GAME ----')
        return self.observation

    def render(self, mode='human', close=False, verbose=False):
        global cols, rows, size
        logger.debug('')
        if close:
            return
        if self.done:
            logger.debug(f'GAME OVER')
            if self.render_lib == 'pygame':
                running.value = 0  # se finaliza render
        else:
            logger.debug(
                f"It is Player {self.current_player.id}'s turn to move")

        if self.render_lib == 'pygame':
            x.value = self.board1.get_player_x()
            y.value = self.board1.get_player_y()

        if self.verbose:  # actualizar tablero
            for number in range(1, self.n_tableros+1):
                print(
                    self.boards_dict[f"board{number}"].get_rewards_grid(), '\n')

        pos_size = self.max_features

        pos_grid_dict = {}  # OPENCV render approach
        for number in range(1, self.n_tableros+1):
            pos_grid_dict[f"pos_grid_{number}"] = self.boards_dict[f"board{number}"].get_position_grid(
            )*255
            pos_grid_dict[f"pos_grid_{number}"] = np.lib.pad(pos_grid_dict[f"pos_grid_{number}"], ((
                0, pos_size-pos_grid_dict[f"pos_grid_{number}"].shape[0]), (0, pos_size-pos_grid_dict[f"pos_grid_{number}"].shape[1])), 'constant', constant_values=(0))

        board_pixel_size = self.max_features*2*10

        resized_dict = {}  # resize and inverse to show position in black
        for number in range(1, self.n_tableros+1):
            resized_dict[f"resized_{number}"] = cv2.resize(pos_grid_dict[f"pos_grid_{number}"], (
                board_pixel_size, board_pixel_size), interpolation=cv2.INTER_AREA)

        boards = list(resized_dict.values())

        cont = 0
        for board in boards:  # draw lines on every board
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.7
            fontColor = (255, 0, 0)
            thickness = 2

            x_label = self.label_list[cont]
            y_label = self.label_list[cont+1]

            x_label_cont = 0
            for label in x_label.split('_'):  # draw x labels
                cv2.putText(board, label,
                            (0, 60+x_label_cont*20),
                            font,
                            fontScale,
                            fontColor,
                            thickness)
                x_label_cont += 1

            y_label_cont = 0
            for label in y_label.split('_'):  # draw y labels
                cv2.putText(board, label,
                            (0, 200+y_label_cont*20),
                            font,
                            fontScale,
                            fontColor,
                            thickness)
                y_label_cont += 1

            cv2.putText(board, 'VS',
                        (100, 140),
                        font,
                        fontScale,
                        fontColor,
                        thickness)
            cont += 2
            # ojo con el 20 y con el range ya que va desde [0,board_pixel_size[
            for i in range(0, board_pixel_size, 20):
                if i == board_pixel_size-20:
                    cv2.line(board, (i, 0), (i, board_pixel_size),
                             (255, 255, 255), thickness=2)  # vertical lines
                    cv2.line(board, (0, i), (board_pixel_size, i),
                             (255, 255, 255), thickness=2)  # horizontal lines
                else:
                    cv2.line(board, (i, 0), (i, board_pixel_size),
                             (255, 255, 255), thickness=1)  # vertical lines
                    cv2.line(board, (0, i), (board_pixel_size, i),
                             (255, 255, 255), thickness=1)  # horizontal lines

        n_tableros = cantidad_tableros_por_fila
        tuple_list = []
        for i in range(int(len(boards)/n_tableros)):  # concatenate horizontal boards
            tuple_list.append(
                np.hstack(tuple(boards[(n_tableros*i):(n_tableros*i)+n_tableros])))

        concat_boards = np.vstack(tuple_list)  # concatenate vertical boards

        # inverse to show black square over white grid
        inverse = (255-concat_boards)

        cv2.imshow('grid', inverse)
        cv2.setWindowProperty('grid', cv2.WND_PROP_TOPMOST, 1)

        if self.env_test:  # if env is test, wait for key press
            wait_key_value = 100
        else:  # if env is train, wait for 1 second
            wait_key_value = 1
        k = cv2.waitKey(wait_key_value) & 0XFF

        if self.verbose:
            logger.debug(f'\nObservation: \n{self.observation}')

        if not self.done:
            legal_actions_ui_list = []
            for legal_preme in self.legal_actions:
                id = list(legal_preme.values())[0]['id']
                name = list(legal_preme.keys())[0]
                legal_actions_ui_list.append(str(id)+': '+name)
            logger.debug(f"\nLegal actions: {legal_actions_ui_list}")


def read_reward_board_csv(filepath, varx_name, vary_name, width, height):
    board_array = [line.split(',') for line in open(filepath)]

    new_board = []
    for line in board_array:
        new_line = [word.replace('\n', '') for word in line]
        new_board.append(new_line)
    board_array = np.array(new_board)

    # find the variable row
    varx_itemindex = np.where(board_array == varx_name)
    # find the variable column
    vary_itemindex = np.where(board_array == vary_name)

    horizontal_values = board_array[varx_itemindex[0][0],
                                    varx_itemindex[1][0]+1:varx_itemindex[1][0]+1+width]  # extract the values of the variable
    vertical_values = board_array[vary_itemindex[0][0],
                                  vary_itemindex[1][0]+1:vary_itemindex[1][0]+1+height]  # extract the values of the variable

    a = np.tile(horizontal_values, (height, 1)).astype(
        float)  # convert to matrix
    b = np.tile(vertical_values, (width, 1)).T.astype(
        float)  # convert to matrix

    rewards_board = a+b
    return rewards_board
