# Adapted from https://mblogscode.com/2016/06/03/python-naughts-crossestic-tac-toe-coding-unbeatable-ai/

import gym
import numpy as np
import json   

import config

from stable_baselines import logger


class Player():
    def __init__(self, id, token):
        self.id = id
        self.token = token
        

class Token():
    def __init__(self, symbol, number):
        self.number = number  #vacio number 0, solo para player
        self.symbol = symbol

class Board():
        def __init__(self, empty_token, number, var_x, var_y, vals_x, vals_y):

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


        def get_grid(self):
            grid_shape = (len(self.vals_x),len(self.vals_y))
            #grid = np.ones(grid_shape) * self.empty_token

            grid = [self.empty_token] * (len(self.vals_x) * len(self.vals_y))#se inicializa board con tokens vacios
            grid = np.array(grid).reshape(grid_shape) #board de posicion actual 
            print('posicion x: ', self.player_x)
            print('posicion y: ', self.player_y)
            #if(self.player_x is not None):
            grid[self.player_y,self.player_x] = self.player_token

            return grid

        def get_symbol_grid(self):
            grid_shape = (len(self.vals_x),len(self.vals_y))
            #grid = np.ones(grid_shape) * self.empty_token

            grid = [self.empty_token.symbol] * (len(self.vals_x) * len(self.vals_y))#se inicializa board con tokens vacios
            grid = np.array(grid).reshape(grid_shape) #board de posicion actual 
            #if(self.player_x is not None):
            print('posicion x: ', self.player_x)
            print('posicion y: ', self.player_y)
            
            grid[self.player_y][self.player_x] = self.player_token.symbol

            return grid

        def get_position_grid(self):
            grid_shape = (len(self.vals_x),len(self.vals_y))
            position_grid = np.zeros(grid_shape) #se inicializa board con tokens vacios
            #if(self.player_x is not None):

            print('posicion x: ', self.player_x)
            print('posicion y: ', self.player_y)
            position_grid[self.player_y][self.player_x] = self.player_token.number
          
            return position_grid

        def get_la_grid(self, legal_positions):
            grid_shape = (len(self.vals_x),len(self.vals_y))
            la_grid = np.zeros(grid_shape)  #board de legal actions, a que posiciones te puedes mover 
            # update with legal positions
            for y in legal_positions["avance_solucion"]:
                x = self.player_x
                la_grid[y,x] = 1
            for x in legal_positions["modelo_negocio"]:
                y = self.player_y
                la_grid[y,x] = 1
            return la_grid

        def get_player_x(self):
            return self.player_x 

        def get_player_y(self):
            return self.player_y

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
        position_grid = self.board.get_position_grid()
        #position_grid = np.array([token.number for token in self.board]).reshape(self.grid_shape) #board de posicion actual 
        print(position_grid)

        # in board se t 1 to legal positions
        print(self.legal_positions)
        la_grid = self.board.get_la_grid(self.legal_positions)
        #la_grid = np.array([0 for x in self.board]).reshape(self.grid_shape)  #board de legal actions, a que posiciones te puedes mover 
        # update with legal positions
        out = np.stack([position_grid,la_grid], axis = -1)
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
                elif preme_effect["operator"] == "decrease":
                    legal_positions[preme_effect["var_name"]].append(self.position[preme_effect["var_name"]]-preme_effect["value"])
                else: # "set"
                    legal_positions[preme_effect["var_name"]].append(preme_effect["value"])
        # keep only set of positions
        legal_positions = {k:list(set(v)) for k, v in legal_positions.items()}
        return legal_positions



    def square_is_player(self, square, player):
        return self.board[square].number == self.players[player].token.number


    def check_game_over(self):

        '''

        board = self.board
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
        '''

        return 0, False

    @property
    def current_player(self):
        return self.players[self.current_player_num]


    def step(self, action):
        
        reward = [0]
        
        # check move legality
        old_x = self.position["avance_solucion"]
        old_y = self.position["modelo_negocio"]

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
            r, done = self.check_game_over()
            reward = [r]
        
        # update board
        new_x = self.position["avance_solucion"]
        new_y = self.position["modelo_negocio"]
        print(old_x,old_y)
        print(new_x,new_y)

        #self.board[old_x*(old_y+1)] = Token('🔳', 0)
        #self.board[new_x*(new_y+1)] = self.players[0].token

        self.board.set_player_position(new_x, new_y, self.players[0].token)

        self.done = done

        if not done:
            self.current_player_num = self.current_player_num

        return self.observation, reward, done, {}

    def reset(self):
        # cambiard board de lista a matriz xy

        self.board = Board(Token('🔳', 0), 1, 'avance_solucion', 'modelo_negocio', 
                    ['idea', 'concepto', 'prototipo', 'mvp', 'ventas', 'crecimiento'],
                    ['producto', 'servicio', 'plataforma', 'ecosistema'])
        #self.board = [Token('🔳', 0)] * self.num_squares #se inicializa board con tokens vacios

        self.players = [Player('Startup1', Token('🟠', 1))] #se inicializan los players con su toquen y numero de jugador

        # start player at certain default position
        self.board.set_player_position(0,0,self.players[0].token) #se posiciona el token circulo en la posicion 0,0 para el player 1 (solo hay un player)
        #self.board[0] = self.players[0].token #board se ocupa con un puro indice 
        #board de 0 a 35 

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
        logger.debug(' '.join(['\t\t'] +[x for x in self.m1_colnames]))
        # rownames
        # logger.debug(' '.join([x.symbol for x in self.board[:self.grid_length]]))
        
        # board
        '''
        logger.debug('\t '.join([self.m1_rownames[0]] +[x.symbol for x in self.board[:self.grid_length]]))
        logger.debug('\t '.join([self.m1_rownames[1]] +[x.symbol for x in self.board[self.grid_length:self.grid_length*2]]))
        logger.debug('\t '.join([self.m1_rownames[2]] +[x.symbol for x in self.board[(self.grid_length*2):(self.grid_length*3)]]))
        logger.debug('\t '.join([self.m1_rownames[3]] +[x.symbol for x in self.board[(self.grid_length*3):(self.grid_length*4)]]))
        '''

        logger.debug(self.board.get_symbol_grid())
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
        grid = self.board.get_grid() 
        #grid = grid.flatten()
        if self.current_player.token.number == 1:
            b = [token.number for token in grid]
        else:
            b = [-token.number for token in grid]

    
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