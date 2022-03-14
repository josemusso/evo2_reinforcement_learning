# Adapted from https://mblogscode.com/2016/06/03/python-naughts-crossestic-tac-toe-coding-unbeatable-ai/

import gym
import numpy as np
import json   

import config

from stable_baselines import logger
import pandas as pd
from tabulate import tabulate

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


class EvolutionEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, verbose = False, manual = False):
        super(EvolutionEnv, self).__init__()
        self.name = 'tictactoe'
        self.manual = manual
        
        self.grid_length = 6
        self.n_players = 1
        self.num_squares = self.grid_length * self.grid_length
        self.grid_shape = (self.grid_length, self.grid_length)
        self.action_space = gym.spaces.Discrete(self.num_squares)
        self.observation_space = gym.spaces.Box(-1, 1, self.grid_shape+(2,))
        self.verbose = verbose

        # board variable names
        self.m1_colnames = ['idea', 'concepto', 'prototipo', 'mvp', 'ventas', 'crecimiento']
        self.m1_rownames = ['producto', 'servicio', 'plataforma', 'ecosistema']
        
        # import premes json
        f=open('/app/environments/evolution/evolution/envs/premes.json', "r")
        self.premes = json.loads(f.read())

        # define attribute that contains position coords
        self.position = {"avance_solucion":0,
                        "modelo_negocio":0,
                        "total_fundadores":0,
                        "horas_dedicacion":0,
                        "problema_organico":0,
                        "punto_equilibrio":0
                        }

    @property
    def observation(self):  #metodos de la clase, toma estado del juego y actualiza tablero posicion y legal positions. 
        position_grid_board1 = self.board1.get_position_grid()
        position_grid_board2 = self.board2.get_position_grid()
        position_grid_board3 = self.board3.get_position_grid()
        #position_grid = np.array([token.number for token in self.board]).reshape(self.grid_shape) #board de posicion actual 
        print('## position_grids ##')
        print(position_grid_board1,'\n')
        print(position_grid_board2,'\n')
        print(position_grid_board3,'\n')

        # in board se t 1 to legal positions
        print('## legal_positions ##')
        print(self.legal_positions,'\n')
        la_grid_board1 = self.board1.get_la_grid(self.legal_positions)
        la_grid_board2 = self.board2.get_la_grid(self.legal_positions)
        la_grid_board3 = self.board3.get_la_grid(self.legal_positions)
        print(la_grid_board1,'\n')
        #la_grid = np.array([0 for x in self.board]).reshape(self.grid_shape)  #board de legal actions, a que posiciones te puedes mover 
        # update with legal positions
        out = np.stack([position_grid_board1,la_grid_board1], axis = -1)
        #print(out)
        return out  #

    @property
    def legal_actions(self):
        legal_actions = []
        for preme_name in list(self.premes.keys()):
            # TODO check if restriction is complete
            preme_legality = True
            restrictions = self.premes[preme_name]["restrictions"]
            for restriction in restrictions:
                if restriction["var_name"] == "":
                    continue 
                if restriction["condition"] == "lower":
                    preme_legality = preme_legality and self.position[restriction["var_name"]]<restriction["value"]
                if restriction["condition"] == "equal":
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
        # keep only set of positions
        legal_positions = {k:list(set(v)) for k, v in legal_positions.items()}
        return legal_positions



    def square_is_player(self, square, player):
        return self.board1[square].number == self.players[player].token.number


    '''
    function to look into all the boards and returns a reward vector with the 
    corresponding reward for each board position
    '''
    def get_new_position_reward(self):

        board = self.board1
        current_player_num = self.current_player_num
        players = self.players


        # check game over
        for i in range(self.grid_length):
            # horizontals and verticals

            #grid = self.get_grid()
            #grid[0,:] == 
            if ((self.square_is_player(i*self.grid_length,current_player_num) and self.square_is_player(i*self.grid_length+1,current_player_num) and self.square_is_player(i*self.grid_length+2,current_player_num))
                or (self.square_is_player(i+0,current_player_num) and self.square_is_player(i+self.grid_length,current_player_num) and self.square_is_player(i+self.grid_length*2,current_player_num))):
                return  1, True

        # diagonals
        if((self.square_is_player(0,current_player_num) and self.square_is_player(4,current_player_num) and self.square_is_player(8,current_player_num))
            or (self.square_is_player(2,current_player_num) and self.square_is_player(4,current_player_num) and self.square_is_player(6,current_player_num))):
                return  1, True

        if self.turns_taken == self.num_squares:
            logger.debug("Board full")
            return  0, True

        return 0, False

    @property
    def current_player(self):
        return self.players[self.current_player_num]


    '''
    function to execute one movement
    handles rewards received for the action taken
    '''
    def step(self, action):
        
        reward = [0]
        
        # check move legality
        print('old position dict: ',self.position)

        # check effect of selected action
        ids = []
        names = []
        for i,legal_preme in enumerate(self.legal_actions):
            ids.append(list(legal_preme.values())[0]['id'])
            names.append(list(legal_preme.keys())[0])
            pos = i if action == list(legal_preme.values())[0]['id'] else 0
        
        action_preme_name = names[pos]

        if action not in ids:  # ilegal action, ends game, punishment
            print("Action not in list")
            done = True
            reward = [-1]
        else: # legal action proceed
            print("Action in list")
            # apply all effects related to chosen action preme
            effects = self.premes[action_preme_name]["effects"] 
            for effect in effects:
                print(effect["operator"])
                if effect["operator"] == "increase":
                    self.position[effect["var_name"]]=self.position[effect["var_name"]]+effect["value"]
                elif effect["operator"] == "decrease":
                    self.position[effect["var_name"]]=self.position[effect["var_name"]]-effect["value"]
                else:
                    self.position[effect["var_name"]]=effect["value"]
            self.turns_taken += 1
            r, done = self.get_new_position_reward()
            reward = [r]
        
        # update board
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

        if not done:
            self.current_player_num = self.current_player_num

        return self.observation, reward, done, {}

    def reset(self):
        # paths de matrices de rewards etiquetadas por expertos
        board_1_rewards_csv_filepath='/app/environments/evolution/evolution/envs/evo2_reinforcement_learning_matrices - matrix_1.csv'
        board_2_rewards_csv_filepath='/app/environments/evolution/evolution/envs/evo2_reinforcement_learning_matrices - matrix_2.csv'
        board_3_rewards_csv_filepath='/app/environments/evolution/evolution/envs/evo2_reinforcement_learning_matrices - matrix_3.csv'
        
        # inicializar boards
        self.board1 = Board(Token('⬜', 0), 1, 
                            'avance_solucion', 'modelo_negocio', 
                            ['idea', 'concepto', 'prototipo', 'mvp', 'ventas', 'crecimiento'],
                            ['producto', 'servicio', 'plataforma', 'ecosistema'],
                            rewards_csv_filepath=board_1_rewards_csv_filepath)
        self.board2 = Board(Token('⬜', 0), 2, 
                            'total_fundadores', 'horas_dedicacion', 
                            ['1', '2', '3', '4', '5', '6'],
                            ['0-5', '6-10', '10-30', '30-45'],
                            rewards_csv_filepath=board_2_rewards_csv_filepath)
        self.board3 = Board(Token('⬜', 0), 3, 
                            'problema_organico', 'punto_equilibrio', 
                            ['no', 'si'],
                            ['no', 'si'],
                            rewards_csv_filepath=board_3_rewards_csv_filepath)
        

        self.players = [Player('Startup1', Token('🔴', 1))] #se inicializan los players con su toquen y numero de jugador

        # start player at certain default position in vevery board
        self.board1.set_player_position(0,0,self.players[0].token) #se posiciona el token circulo en la posicion 0,0 para el player 1 (solo hay un player)
        self.board2.set_player_position(0,0,self.players[0].token) #se posiciona el token circulo en la posicion 0,0 para el player 1 (solo hay un player)
        self.board3.set_player_position(0,0,self.players[0].token) #se posiciona el token circulo en la posicion 0,0 para el player 1 (solo hay un player)

        self.current_player_num = 0  #player que esta en el turno
        self.turns_taken = 0  #la cantidad de turnos 
        self.done = False #cuando se acabo el juego total 
        logger.debug(f'\n\n---- NEW GAME ----')
        return self.observation


    def render(self, mode='human', close=False, verbose = True):
        logger.debug('')
        if close:
            return
        if self.done:
            logger.debug(f'GAME OVER')
        else:
            logger.debug(f"It is Player {self.current_player.id}'s turn to move")
            
        # colnames
        # logger.debug(' '.join(['\t\t'] +[x for x in self.m1_colnames]))
        # rownames
        # logger.debug(' '.join([x.symbol for x in self.board[:self.grid_length]]))
        
        # board
        # logger.debug('\t '.join([self.m1_rownames[0]] +[x.symbol for x in self.board[:self.grid_length]]))
        # logger.debug('\t '.join([self.m1_rownames[1]] +[x.symbol for x in self.board[self.grid_length:self.grid_length*2]]))
        # logger.debug('\t '.join([self.m1_rownames[2]] +[x.symbol for x in self.board[(self.grid_length*2):(self.grid_length*3)]]))
        # logger.debug('\t '.join([self.m1_rownames[3]] +[x.symbol for x in self.board[(self.grid_length*3):(self.grid_length*4)]]))
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
                                    index=['no', 'si'],
                                    columns=['no', 'si'])

        print("## REWARD BOARDS ##")
        print(self.board1.get_rewards_grid(),'\n')                            
        print(self.board2.get_rewards_grid(),'\n')                            
        print(self.board3.get_rewards_grid(),'\n')                            
        print("## GAME BOARDS ##")
        print(tabulate(board1_df, headers='keys', tablefmt='grid'))
        print(tabulate(board2_df, headers='keys', tablefmt='grid'))
        print(tabulate(board3_df, headers='keys', tablefmt='grid'))
        # logger.debug('\t '.join([self.m1_rownames[4]] +[x.symbol for x in self.board[(self.grid_length*4):(self.grid_length*5)]]))
        # logger.debug('\t '.join([self.m1_rownames[5]] +[x.symbol for x in self.board[(self.grid_length*5):(self.grid_length*6)]]))

        if self.verbose:
            logger.debug(f'\nObservation: \n{self.observation}')
        
        if not self.done:
            legal_actions_ui_list = []
            for legal_preme in self.legal_actions:
                id = list(legal_preme.values())[0]['id']
                name = list(legal_preme.keys())[0]
                legal_actions_ui_list.append(str(id)+': '+name)
            logger.debug(f"\nLegal actions: {legal_actions_ui_list}")


    def rules_move(self):
        grid_board1 = self.board1.get_grid() 
        grid_board2 = self.board1.get_grid() 
        grid_board3 = self.board1.get_grid() 
        #grid = grid.flatten()
        if self.current_player.token.number == 1:
            b = [token.number for token in grid_board1]
        else:
            b = [-token.number for token in grid_board1]

    
        # Check computer win moves
        for i in range(0, self.num_squares):
            if b[i] == 0 and testWinMove(b, 1, i):
                logger.debug('Winning move')
                return self.create_action_probs(i)
        # Check player win moves
        for i in range(0, self.num_squares):
            if b[i] == 0 and testWinMove(b, -1, i):
                logger.debug('Block move')
                return self.create_action_probs(i)
        # Check computer fork opportunities
        for i in range(0, self.num_squares):
            if b[i] == 0 and testForkMove(b, 1, i):
                logger.debug('Create Fork')
                return self.create_action_probs(i)
        # Check player fork opportunities, incl. two forks
        playerForks = 0
        for i in range(0, self.num_squares):
            if b[i] == 0 and testForkMove(b, -1, i):
                playerForks += 1
                tempMove = i
        if playerForks == 1:
            logger.debug('Block One Fork')
            return self.create_action_probs(tempMove)
        elif playerForks == 2:
            for j in [1, 3, 5, 7]:
                if b[j] == 0:
                    logger.debug('Block 2 Forks')
                    return self.create_action_probs(j)
        # Play center
        if b[4] == 0:
            logger.debug('Play Centre')
            return self.create_action_probs(4)
        # Play a corner
        for i in [0, 2, 6, 8]:
            if b[i] == 0:
                logger.debug('Play Corner')
                return self.create_action_probs(i)
        #Play a side
        for i in [1, 3, 5, 7]:
            if b[i] == 0:
                logger.debug('Play Side')
                return self.create_action_probs(i)


    def create_action_probs(self, action):
        action_probs = [0.01] * self.action_space.n
        action_probs[action] = 0.92
        return action_probs   


def checkWin(b, m):
    return ((b[0] == m and b[1] == m and b[2] == m) or  # H top
            (b[3] == m and b[4] == m and b[5] == m) or  # H mid
            (b[6] == m and b[7] == m and b[8] == m) or  # H bot
            (b[0] == m and b[3] == m and b[6] == m) or  # V left
            (b[1] == m and b[4] == m and b[7] == m) or  # V centre
            (b[2] == m and b[5] == m and b[8] == m) or  # V right
            (b[0] == m and b[4] == m and b[8] == m) or  # LR diag
            (b[2] == m and b[4] == m and b[6] == m))  # RL diag


def checkDraw(b):
    return 0 not in b

def getBoardCopy(b):
    # Make a duplicate of the board. When testing moves we don't want to 
    # change the actual board
    dupeBoard = b.copy()
    return dupeBoard

def testWinMove(b, mark, x, y):
    # b = the board
    # mark = 0 or X
    # i = the square to check if makes a win 
    bCopy = getBoardCopy(b)
    bCopy[x][y] = mark
    return checkWin(bCopy, mark)

'''    #Tenemos que arreglarlo para que considere que b es una matriz 
def testForkMove(b, mark, x, y):
    # Determines if a move opens up a fork
    bCopy = getBoardCopy(b)
    bCopy[x][y] = mark
    winningMoves = 0
    for j in range(0, 9):
        if testWinMove(bCopy, mark, j) and bCopy[j] == 0:
            winningMoves += 1
    return winningMoves >= 2
    '''

def read_reward_board_csv(filepath,width,height):
    board_array = [line.split(',') for line in open(filepath)]
    new_board = []
    for line in board_array:
        new_line = [word.replace('\n','') for word in line]
        new_board.append(new_line)
    board_array=new_board
    board_array=np.array(board_array[3:3+height])[:,1:] # start reading in line 2
    return board_array