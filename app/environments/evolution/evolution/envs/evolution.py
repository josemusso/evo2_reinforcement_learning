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
        self.number = number
        self.symbol = symbol
        

    

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


    @property
    def observation(self):
        position = {"avance_solucion":0,
                    "modelos_negocio":0,
                    "total_fundadores":0,
                    "horas_dedicacion":0,
                    "problema_organico":0,
                    "punto_equilibrio":0
                    }
        position_grid = np.array([x.number for x in self.board]).reshape(self.grid_shape)
        print(position_grid)

        # in board se t 1 to legal positions
        print(self.legal_positions)
        la_grid = np.array([0 for x in self.board]).reshape(self.grid_shape)
        # update with legal positions
        for y in self.legal_positions["avance_solucion"]:
            x = position["modelos_negocio"]
            la_grid[x,y] = 1
        for x in self.legal_positions["modelos_negocio"]:
            y = position["avance_solucion"]
            la_grid[x,y] = 1
        print(la_grid)
        out = np.stack([position_grid,la_grid], axis = -1)
        #print(out)
        return out

    @property
    def legal_actions(self):
        legal_actions = []
        for preme_name in list(self.premes.keys()):
            # TODO check if restriction is complete
            legal_actions.append(self.premes[preme_name])
        return np.array(legal_actions)
    
    @property
    def legal_positions(self):
        # position = [board1_x,board1_y,board2_x,board2_y,board3_x,board3_y]
        position = {"avance_solucion":0,
                    "modelos_negocio":0,
                    "total_fundadores":0,
                    "horas_dedicacion":0,
                    "problema_organico":0,
                    "punto_equilibrio":0
                    }
        legal_positions = {"avance_solucion":[],
                    "modelos_negocio":[],
                    "total_fundadores":[],
                    "horas_dedicacion":[],
                    "problema_organico":[],
                    "punto_equilibrio":[]
                    }
        for legal_preme in self.legal_actions:
            for preme_effect in legal_preme["effects"]:
                preme_effect["value"]=int(preme_effect["value"])
                if preme_effect["operator"] == "increase":
                    legal_positions[preme_effect["var_name"]].append(position[preme_effect["var_name"]]+preme_effect["value"])
                elif preme_effect["operator"] == "decrease":
                    legal_positions[preme_effect["var_name"]].append(position[preme_effect["var_name"]]-preme_effect["value"])
                else: # "set"
                    legal_positions[preme_effect["var_name"]].append(preme_effect["value"])
        # keep only set of positions
        legal_positions = {k:list(set(v)) for k, v in legal_positions.items()}
        return legal_positions



    def square_is_player(self, square, player):
        return self.board[square].number == self.players[player].token.number

    def check_game_over(self):

        board = self.board
        current_player_num = self.current_player_num
        players = self.players


        # check game over
        for i in range(self.grid_length):
            # horizontals and verticals
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


    def step(self, action):
        
        reward = [0,0]
        
        # check move legality
        board = self.board
        
        if (board[action].number != 0):  # not empty
            done = True
            reward = [1, 1]
            reward[self.current_player_num] = -1
        else:
            board[action] = self.current_player.token
            self.turns_taken += 1
            r, done = self.check_game_over()
            reward = [-r,-r]
            reward[self.current_player_num] = r

        self.done = done

        if not done:
            self.current_player_num = self.current_player_num

        return self.observation, reward, done, {}

    def reset(self):
        self.board = [Token('.', 0)] * self.num_squares
        self.players = [Player('Startup1', Token('X', 4))]

        # start player at certain default position
        self.board[0] = self.players[0].token

        self.current_player_num = 0
        self.turns_taken = 0
        self.done = False
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
        logger.debug(' '.join([x for x in self.m1_colnames]))
        # rownames
        # logger.debug(' '.join([x.symbol for x in self.board[:self.grid_length]]))
        
        # board
        logger.debug(' '.join([x.symbol for x in self.board[:self.grid_length]]))
        logger.debug(' '.join([x.symbol for x in self.board[self.grid_length:self.grid_length*2]]))
        logger.debug(' '.join([x.symbol for x in self.board[(self.grid_length*2):(self.grid_length*3)]]))
        logger.debug(' '.join([x.symbol for x in self.board[(self.grid_length*3):(self.grid_length*4)]]))
        logger.debug(' '.join([x.symbol for x in self.board[(self.grid_length*4):(self.grid_length*5)]]))
        logger.debug(' '.join([x.symbol for x in self.board[(self.grid_length*5):(self.grid_length*6)]]))

        if self.verbose:
            logger.debug(f'\nObservation: \n{self.observation}')
        
        if not self.done:
            logger.debug(f'\nLegal actions: {[i for i,o in enumerate(self.legal_actions) if o != 0]}')


    def rules_move(self):
        if self.current_player.token.number == 1:
            b = [x.number for x in self.board]
        else:
            b = [-x.number for x in self.board]

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
    dupeBoard = []
    for j in b:
        dupeBoard.append(j)
    return dupeBoard

def testWinMove(b, mark, i):
    # b = the board
    # mark = 0 or X
    # i = the square to check if makes a win 
    bCopy = getBoardCopy(b)
    bCopy[i] = mark
    return checkWin(bCopy, mark)


def testForkMove(b, mark, i):
    # Determines if a move opens up a fork
    bCopy = getBoardCopy(b)
    bCopy[i] = mark
    winningMoves = 0
    for j in range(0, 9):
        if testWinMove(bCopy, mark, j) and bCopy[j] == 0:
            winningMoves += 1
    return winningMoves >= 2