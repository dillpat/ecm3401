from cmath import polar
import random
from collections import deque
from queue import PriorityQueue
from re import L, T
from tkinter import Label
from unittest.case import _AssertWarnsContext

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
from skopt import forest_minimize, gp_minimize
from skopt.plots import plot_convergence
from skopt.space import Real

from Coin import Coin
from Enemy import Enemy
from pyagame import COLOR, agent, maze, textLabel
from Quadrant import Quadrant

mpl.rcParams['axes.grid'] = True
mpl.rcParams['axes.grid.which'] = 'both'
mpl.rcParams['grid.linestyle'] = '--'

MAZE_ROWS = 20
MAZE_COLS = MAZE_ROWS
RAND_LOW   = 0
RAND_HIGH  = MAZE_ROWS/2
MAX_ASSET  = 8
MAZE_DIFFICULTY = 7
MAZE_EXIT = (1, 1)
MAZE_START = (MAZE_ROWS, MAZE_COLS)
MAZE_LEVEL_FIXED = False
MAX_COIN = MAX_ASSET
MAX_ENEMY = MAX_ASSET
MIN_PROB = (1 / MAX_ASSET)
ENEMY_TARGET = 8
COIN_TARGET = 80
WEIGHT_GAIN = 10

PL_COIN_CELLS  = 'coin_cells'
PL_COIN_DIST   = 'coin_dist'
PL_ENEMY_CELLS = 'enemy_cells'
PL_ENEMY_DIST  = 'enemy_dist'
PL_GP_PARAMS   = 'gp_params'
PL_SCORE       = 'score'
PL_QUADS       = ['qNW', 'qNE', 'qSW', 'qSE']


class PlayLog:
    """
    
    """
    def __init__(self, player=''):
        """
        
        """
        self._player = player
        self._log = {PL_COIN_CELLS:[], PL_COIN_DIST:[], 
                     PL_ENEMY_CELLS:[], PL_ENEMY_DIST:[],
                     PL_GP_PARAMS:[], PL_SCORE:[]}

    def add_asset(self, asset, cells):
        self._log[asset].append(cells)
    
    def title(self):
        return self._player

    @property
    def coin_cells(self):
        return (self._log[PL_COIN_CELLS])

    @coin_cells.setter
    def coin_cells(self, cells):
        self._log[PL_COIN_CELLS].append(cells)

    @property
    def coin_dist(self):
        return (self._log[PL_COIN_DIST])

    @coin_dist.setter
    def coin_dist(self, dist):
        self._log[PL_COIN_DIST].append(dist)

    @property
    def enemy_cells(self):
        return (self._log[PL_ENEMY_CELLS])

    @enemy_cells.setter
    def enemy_cells(self, cells):
        self._log[PL_ENEMY_CELLS].append(cells)

    @property
    def enemy_dist(self):
        return (self._log[PL_ENEMY_DIST])

    @enemy_dist.setter
    def enemy_dist(self, dist):
        self._log[PL_ENEMY_DIST].append(dist)

    @property
    def gp_params(self):
        return (self._log[PL_GP_PARAMS])

    @gp_params.setter
    def gp_params(self, params):
        self._log[PL_GP_PARAMS].append(params)

    @property
    def score(self):
        return (self._log[PL_SCORE])

    @score.setter
    def score(self, score):
        self._log[PL_SCORE].append(score)

greedy_log = PlayLog('Greedy Player')
neutral_log = PlayLog('Neutral Player')
aggressive_log = PlayLog('Aggressive Player')
run_log = PlayLog('All Players')


def BFS(m, start=None, goal=None):
    if start is None:
        start=(m.rows,m.cols)
    if goal is None:
        goal = m._goal
    frontier = deque()
    frontier.append(start)
    bfsPath = {}
    explored = [start]
    bSearch=[]

    while len(frontier)>0:
        currCell=frontier.popleft()
        if currCell==goal:
            break
        for d in 'ESNW':
            if m.maze_map[currCell][d]==True:
                if d=='E':
                    nextCell=(currCell[0],currCell[1]+1)
                elif d=='W':
                    nextCell=(currCell[0],currCell[1]-1)
                elif d=='S':
                    nextCell=(currCell[0]+1,currCell[1])
                elif d=='N':
                    nextCell=(currCell[0]-1,currCell[1])
                
                if nextCell in explored:
                    continue

                frontier.append(nextCell)
                explored.append(nextCell)
                bfsPath[nextCell] = currCell
                bSearch.append(nextCell)
        
    fwdPath={}
    cell=goal
    while cell!=(start):
        fwdPath[bfsPath[cell]]=cell
        cell=bfsPath[cell]
        
    weight = 0
    for cell in fwdPath:
        if 'A' in m.maze_map[cell].keys() and not m.maze_map[cell]['A'].defeated:
            weight += ((len(fwdPath)+1) + WEIGHT_GAIN)
        else:
            weight = (len(fwdPath)+1)

    return bSearch, bfsPath, fwdPath, weight


def aStar(m, start=None, goal=None):

    def h(cell1, cell2):
        """ Manhattan method """
        x1, y1 = cell1
        x2, y2 = cell2
        return (abs(x1 - x2) + abs(y1 - y2))

    if start is None:
        start=(m.rows,m.cols)
    if goal is None:
        goal = m._goal
    open = PriorityQueue()
    open.put((h(start, goal), h(start, goal), start))
    aPath = {}
    g_score = {row: float("inf") for row in m.grid}
    g_score[start] = 0
    f_score = {row: float("inf") for row in m.grid}
    f_score[start] = h(start, goal)
    searchPath=[start]

    while not open.empty():
        currCell = open.get()[2]
        searchPath.append(currCell)
        if currCell == goal:
            break        
        for d in 'ESNW':
            if m.maze_map[currCell][d]==True:
                if d=='E':
                    nextCell=(currCell[0],currCell[1]+1)
                elif d=='W':
                    nextCell=(currCell[0],currCell[1]-1)
                elif d=='N':
                    nextCell=(currCell[0]-1,currCell[1])
                elif d=='S':
                    nextCell=(currCell[0]+1,currCell[1])

                temp_g_score = g_score[currCell] + 1
                temp_f_score = temp_g_score + h(nextCell, goal)

                if temp_f_score < f_score[nextCell]:   
                    aPath[nextCell] = currCell
                    g_score[nextCell] = temp_g_score
                    f_score[nextCell] = temp_g_score + h(nextCell, goal)
                    open.put((f_score[nextCell], h(nextCell, goal), nextCell))

    fwdPath={}
    cell=goal
    while cell!=start:
        fwdPath[aPath[cell]]=cell
        cell=aPath[cell]

    weight = 0
    for cell in fwdPath:
        if 'A' in m.maze_map[cell].keys() and not m.maze_map[cell]['A'].defeated:
            weight += ((len(fwdPath)+1) + WEIGHT_GAIN)
        else:
            weight = (len(fwdPath)+1)

    return searchPath, aPath, fwdPath, weight


def addCoins(m, quadrant, coin_position_list = {}, number=MAX_COIN):
    coin_count = 0
    while coin_count < number:
        # add coin to maze at given random cell
        cell = np.random.randint(low=RAND_LOW, high=RAND_HIGH, size=2)
        cell[0] += quadrant.base[0]
        cell[1] += quadrant.base[1]
        x=cell[0]
        y=cell[1]
        if 'C' not in m.maze_map[x,y].keys() and  \
           'A' not in m.maze_map[x,y].keys():       # cell not occupied by coin/enemy
            coin = Coin(m, x, y)
            coin_count += 1
            m.maze_map[x,y]['C'] = coin
            coin_position_list[x,y]=cell

            # add coin to quadrant
            quadrant.coin_positions[x,y]=cell


def findNearestCoin(m, start_position, coin_position_list):
    """
    """
    nearest_coin_cell=(1,1) # Default position maze exit
    distance = 100000
    for cell in coin_position_list:
        if not m.maze_map[cell]['C'].collected: 
            bSearch,aPath,fwdPath,weight = m.findPath(m, start_position, cell)
            if (len(fwdPath)+1) < distance:
                distance = (len(fwdPath)+1)
                nearest_coin_cell = cell

    return nearest_coin_cell


def collectNearestCoins(m, coin_position_list, start_cell = None, coin_target = COIN_TARGET):
    if start_cell is None:
        start_position = (m.rows,m.cols)
    else:
        start_position = start_cell
    currentScore = 0
    steps = 0
    for i in range(len(coin_position_list)):
        nearest_coin = findNearestCoin(m, start_position, coin_position_list)
        bSearch,aPath,fwdPath,weight = m.findPath(m, start_position, nearest_coin)
        if ('C' in m.maze_map[nearest_coin]):
            coin = m.maze_map[nearest_coin]['C']
            if not coin.collected:
                coin.collected = True
                currentScore += coin.value
                steps += (len(fwdPath)+1)
        start_position = nearest_coin
        if currentScore >= coin_target:
            break

    return start_position, steps


def addEnemy(m, quadrant, enemy_list = {}, number = MAX_ENEMY):
    enemy_count = 0
    while enemy_count < number:
        # add enemy to maze at given random cell
        cell = np.random.randint(low=RAND_LOW, high=RAND_HIGH, size=2)
        cell[0] += quadrant.base[0]
        cell[1] += quadrant.base[1]
        x=cell[0]
        y=cell[1]
        if 'A' not in m.maze_map[x,y].keys() and \
           'C' not in m.maze_map[x,y].keys() :      # cell not occupied by coin/enemy
            enemy = Enemy(m, x, y)
            enemy_count += 1
            m.maze_map[x,y]['A'] = enemy
            enemy_list[x,y]=cell

            # add coin to quadrant
            quadrant.enemy_positions[x,y]=cell


def findNearestEnemy(m, start_position, enemy_list):
    """
    """
    nearest_enemy_cell=(1,1) # Default position maze exit
    distance = 100000
    for cell in enemy_list:
        if not m.maze_map[cell]['A'].defeated: 
            bSearch,aPath,fwdPath,weight = m.findPath(m, start_position, cell)
            if (len(fwdPath)+1) < distance:
                distance = (len(fwdPath)+1)
                nearest_enemy_cell = cell

    return nearest_enemy_cell


def combatNearestEnemy(m, enemy_list, start_cell = None, player_health = 80, enemy_target = ENEMY_TARGET):
    current_health = player_health
    if start_cell is None:
        start_position = (m.rows,m.cols)
    else:
        start_position = start_cell
    enemy_killed = 0
    steps = 0
    while enemy_killed < len(enemy_list) and enemy_killed < enemy_target:
        nearest_enemy = findNearestEnemy(m, start_position, enemy_list)
        bSearch,aPath,fwdPath,weight = m.findPath(m, start_position, nearest_enemy)
        if ('A' in m.maze_map[nearest_enemy]): # Checks for an enemy in cell
            enemy = m.maze_map[nearest_enemy]['A']
            if not enemy.defeated:
                current_health = combat(enemy, current_health)
                if enemy.defeated:
                    enemy_killed += 1
                    steps += (len(fwdPath)+1)
                # current health will tell you if enemy is defeated
                if current_health == 0:
                    current_health, stepsRH, home = restoreHealth(m, current_health, enemy)
                    steps += stepsRH
                    nearest_enemy = home
        start_position = nearest_enemy

    return start_position, steps


def combat(enemy, player_health):
    while player_health > 0:
        if random.randint(0,10) > MAZE_DIFFICULTY:
            enemy.defeated = True
            break
        else:
            player_health -= enemy.health

    return player_health


"""
return back to start method from current position then restore player_health to 80
"""
def restoreHealth(m, current_health, enemy):
    home = (m.rows,m.cols)
    current_position = enemy.position
    steps = 0
    if current_health == 0:
        bSearch,aPath,fwdPath,weight = m.findPath(m, current_position, home)
        steps += (len(fwdPath)+1)
        current_health = 80
    else:
        home = current_position
    return current_health, steps, home


def divideQuadrants(m):
    cNW = []
    cNE = []
    cSW = []
    cSE = []
    width_max = m.rows
    height_max = m.cols
    half_width = width_max // 2
    half_height = height_max // 2

    # Creating the NW quadrant
    cNW.append((1,1)) # (1,1) for 10x10
    cNW.append((1, half_width))
    cNW.append((half_height, 1))
    cNW.append((half_height, half_width))

    # Creating the NE quadrant
    cNE.append((1, half_width + 1)) # (1,6) for 10x10
    cNE.append((1, width_max))
    cNE.append((half_height, half_width + 1))
    cNE.append((half_height, width_max))

    # Creating the SW quadrant
    cSW.append((half_height + 1, 1)) # (6, 1) for 10x10
    cSW.append((half_height + 1, half_width))
    cSW.append((height_max, 1))
    cSW.append((height_max, half_width))

    # Creating the SE quadrant
    cSE.append((half_height + 1, half_width + 1)) # (6,6) for 10x10
    cSE.append((half_height + 1, width_max))
    cSE.append((height_max, half_width + 1))
    cSE.append((height_max, width_max))

    return cNW, cNE, cSW, cSE


def createQuadrantDictionary(m,cNW, cNE, cSW, cSE):
    distribution_dict = {"qNW":Quadrant(m, cNW),\
                         "qNE":Quadrant(m, cNE),\
                         "qSW":Quadrant(m, cSW),\
                         "qSE":Quadrant(m, cSE)}

    return distribution_dict


def setProbability(m, distribution_dict, pNW = 0.25, pNE = 0.25, pSW = 0.25, pSE = 0.25):
    if sum([pNW, pNE, pSW]) < 1:
        pSE = 1 - sum([pNW, pNE, pSW])

    # reset quadrant probabilities proportionally
    pSum = sum([pNW, pNE, pSW, pSE])
    distribution_dict["qNW"].probability = pNW/pSum
    distribution_dict["qNE"].probability = pNE/pSum
    distribution_dict["qSW"].probability = pSW/pSum
    distribution_dict["qSE"].probability = pSE/pSum


def distributeCoinAssets(m,  cpNW, cpNE, cpSW, cpSE):
    # Setting the probability of coins in each quadrant
    cNW, cNE, cSW, cSE = divideQuadrants(m)
    coin_quadrant_dict = createQuadrantDictionary(m, cNW, cNE, cSW, cSE)
    setProbability(m, coin_quadrant_dict, cpNW, cpNE, cpSW, cpSE)

    # order quadrants highest probability to lowest
    coin_dict=dict(sorted(coin_quadrant_dict.items(), key=lambda x:x[1].probability, reverse=True))

    # Populate quadrants with coins
    coin_cells = {}
    while len(coin_cells) < MAX_COIN:
        for quadrant in coin_dict.values():
            if len(coin_cells) >= MAX_COIN:
                break
            if int(MAX_COIN * quadrant.probability):
                number = int(MAX_COIN * quadrant.probability) if not len(quadrant.coin_positions) else 1
                total_coins = len(coin_cells)
                if total_coins + number > MAX_COIN:
                    number = MAX_COIN - total_coins
                addCoins(m, quadrant, coin_cells, number)

    run_log.coin_cells = list(coin_cells)
    return list(coin_cells), coin_quadrant_dict

def distributeEnemyAssets(m, epNW, epNE, epSW, epSE):
    # Setting the probability of enemies in each quadrant
    eNW, eNE, eSW, eSE = divideQuadrants(m)
    enemy_quadrant_dict = createQuadrantDictionary(m, eNW, eNE, eSW, eSE)
    setProbability(m, enemy_quadrant_dict, epNW, epNE, epSW, epSE)

    # order quadrants highest probability to lowest
    enemy_dict=dict(sorted(enemy_quadrant_dict.items(), key=lambda x:x[1].probability, reverse=True))

    # Populate quadrants with enemies
    enemy_cells = {}
    while len(enemy_cells) < MAX_ENEMY:
        for quadrant in enemy_dict.values():
            if len(enemy_cells) >= MAX_ENEMY:
                break
            if int(MAX_COIN * quadrant.probability):
                number = int(MAX_COIN * quadrant.probability) if not len(quadrant.coin_positions) else 1
                total_enemy = len(enemy_cells)
                if total_enemy + number > MAX_ENEMY:
                    number = MAX_ENEMY - total_enemy
                addEnemy(m, quadrant, enemy_cells, number)
    
    run_log.enemy_cells = list(enemy_cells)
    return list(enemy_cells), enemy_quadrant_dict



def findOnlyCoin(m, start_position, coin_position_list):
    """
    """
    nearest_coin_cell=(1,1) # Default position maze exit
    distance = 100000
    for cell in coin_position_list:
        if not m.maze_map[cell]['C'].collected: 
            bSearch,aPath,fwdPath,weight = m.findPath(m, start_position, cell)
            if (weight) < distance:
                    distance = (weight)
                    nearest_coin_cell = cell

    return nearest_coin_cell


def collectOnlyCoins(m, coin_position_list, start_cell = None, coin_target = COIN_TARGET):
    if start_cell is None:
        start_position = (m.rows,m.cols)
    else:
        start_position = start_cell
    currentScore = 0
    steps = 0
    for i in range(len(coin_position_list)):
        nearest_coin = findOnlyCoin(m, start_position, coin_position_list)
        bSearch,aPath,fwdPath,weight = m.findPath(m, start_position, nearest_coin)
        if ('C' in m.maze_map[nearest_coin]):
            coin = m.maze_map[nearest_coin]['C']
            if not coin.collected:
                coin.collected = True
                currentScore += coin.value
                steps += (len(fwdPath)+1)
        start_position = nearest_coin
        if currentScore >= coin_target:
            break

    return start_position, steps


## def greedy 
'''
The objective of this player is to collect all the coins without fighting any enemies. This means collecting coins 1-8, without 
fighting any enemies. So all enemies must be avoided and pathed around. If no path then choose the quickest route.
'''
def greedy_player(m, coin_list, enemy_list, target = 0):
    total_steps = 0
    new_position, steps_coins = collectOnlyCoins(m, coin_list, start_cell=MAZE_START, coin_target=80)
    new_position, steps_enemy = combatNearestEnemy(m, enemy_list, start_cell=new_position, enemy_target=0)
    if new_position != MAZE_EXIT:
        bSearch,aPathPath,fwdPath,weight = m.findPath(m, new_position, MAZE_EXIT)
        total_steps = (len(fwdPath)+1)
    total_steps += steps_coins + steps_enemy
    score = total_steps - target
    return score


## def aggressive
''''
This player wants to fight all the enemies, this player only cares about fighting 1-8 enemies, it does not matter if the player
collects any coins, they just care about fighting the enemy
'''
def aggressive_player(m, coin_list, enemy_list, target = 0):
    total_steps = 0
    new_position, steps_enemy = combatNearestEnemy(m, enemy_list, start_cell=MAZE_START, enemy_target=8)
    new_position, steps_coins = collectNearestCoins(m, coin_list, start_cell=new_position, coin_target=0)
    if new_position != MAZE_EXIT:
        bSearch,aPath,fwdPath,weight = m.findPath(m, new_position, MAZE_EXIT)
        total_steps = (len(fwdPath)+1)
    total_steps = steps_coins + steps_enemy
    score = total_steps - target
    return score


## def neutral
'''
This player wants to partake in both objectives, fighting 1-8 enemies or collecting 1-8 coins, the player fights the enemies
first, the enemies that are set as defeated are allowed to be passed over by the player when he is collecting the coins. 
Otherwise he will have to path around
'''
def neutral_player(m, coin_list, enemy_list, target = 0):
    total_steps = 0
    new_position, steps_enemy = combatNearestEnemy(m, enemy_list, start_cell=MAZE_START, enemy_target=5)
    new_position, steps_coins = collectOnlyCoins(m, coin_list, start_cell=new_position, coin_target=40)
    if new_position != MAZE_EXIT:
        bSearch,aPath,fwdPath,weight = m.findPath(m, new_position, MAZE_EXIT)
        total_steps = (len(fwdPath)+1)
    total_steps = steps_coins + steps_enemy
    score = total_steps - target
    return score


g_maze = None
def init_maze(m = None):
    if m is None:
        m=maze(MAZE_ROWS, MAZE_COLS)
        m.CreateMaze(loopPercent=20, findPath=aStar, displayMaze=False)
        return m

    # clear any coin records in maze cells
    for cell in m.maze_map.values():
        if 'C' in cell:
            cell.pop('C')

    # clear any enemy records in maze cells
    for cell in m.maze_map.values():
        if 'A' in cell:
            cell.pop('A')

    return m


def objective_greedy(dimensions2x2):
    cpNW = dimensions2x2[0]
    cpNE = dimensions2x2[1]
    cpSW = dimensions2x2[2]
    cpSE = 0.0

    epNW = dimensions2x2[3]
    epNE = dimensions2x2[4]
    epSW = dimensions2x2[5]
    epSE = 0.0

    # create and initialise maze object
    global g_maze
    m = init_maze(g_maze)
    g_maze = m if MAZE_LEVEL_FIXED else None

    coin_list, coin_quad = distributeCoinAssets(m, cpNW, cpNE, cpSW, cpSE)
    greedy_log.coin_cells = coin_list
    greedy_log.coin_dist = [len(coin_quad[k].coin_positions) for k in PL_QUADS]

    enemy_list, enemy_quad = distributeEnemyAssets(m, epNW, epNE, epSW, epSE)
    greedy_log.enemy_cells = enemy_list
    greedy_log.enemy_dist = [len(enemy_quad[k].enemy_positions) for k in PL_QUADS]

    greedy_log.gp_params = [cpNW, cpNE, cpSW, coin_quad["qSE"].probability,
                            epNW, epNE, epSW, enemy_quad["qSE"].probability]

    total_steps = greedy_player(m, coin_list, enemy_list)
    greedy_log.score = total_steps

    print('Greedy steps:', total_steps)
    return total_steps


def objective_neutral(dimensions2x2):
    cpNW = dimensions2x2[0]
    cpNE = dimensions2x2[1]
    cpSW = dimensions2x2[2]
    cpSE = 0.0

    epNW = dimensions2x2[3]
    epNE = dimensions2x2[4]
    epSW = dimensions2x2[5]
    epSE = 0.0

    # create and initialise maze
    global g_maze
    m = init_maze(g_maze)
    g_maze = m if MAZE_LEVEL_FIXED else None

    coin_list, coin_quad = distributeCoinAssets(m, cpNW, cpNE, cpSW, cpSE)
    neutral_log.coin_cells = coin_list
    neutral_log.coin_dist = [len(coin_quad[k].coin_positions) for k in PL_QUADS]

    enemy_list, enemy_quad = distributeEnemyAssets(m, epNW, epNE, epSW, epSE)
    neutral_log.enemy_cells = enemy_list
    neutral_log.enemy_dist = [len(enemy_quad[k].enemy_positions) for k in PL_QUADS]

    neutral_log.gp_params = [cpNW, cpNE, cpSW, coin_quad["qSE"].probability,
                             epNW, epNE, epSW, enemy_quad["qSE"].probability]

    total_steps = neutral_player(m, coin_list, enemy_list)
    neutral_log.score = total_steps

    print('Neutral steps:', total_steps)
    return total_steps
    

def objective_aggressive(dimensions2x2):
    cpNW = dimensions2x2[0]
    cpNE = dimensions2x2[1]
    cpSW = dimensions2x2[2]
    cpSE = 0.0

    epNW = dimensions2x2[3]
    epNE = dimensions2x2[4]
    epSW = dimensions2x2[5]
    epSE = 0.0
    
    # create and initialise maze
    global g_maze
    m = init_maze(g_maze)
    g_maze = m if MAZE_LEVEL_FIXED else None

    coin_list, coin_quad = distributeCoinAssets(m, cpNW, cpNE, cpSW, cpSE)
    aggressive_log.coin_cells = coin_list
    aggressive_log.coin_dist = [len(coin_quad[k].coin_positions) for k in PL_QUADS]

    enemy_list, enemy_quad = distributeEnemyAssets(m, epNW, epNE, epSW, epSE)
    aggressive_log.enemy_cells = enemy_list
    aggressive_log.enemy_dist = [len(enemy_quad[k].enemy_positions) for k in PL_QUADS]

    aggressive_log.gp_params = [cpNW, cpNE, cpSW, coin_quad["qSE"].probability,
                                epNW, epNE, epSW, enemy_quad["qSE"].probability]

    total_steps = aggressive_player(m, coin_list, enemy_list)
    aggressive_log.score = total_steps
    
    print('Aggressive steps:', total_steps)
    return total_steps

        
def play_game(player, style='', n_iter = 1):
    print("Player Style: ", style)

    #  Biased
    # cpNW = Real(name = 'cpNW', low= 0.500, high = 0.999)
    # cpNE = Real(name = 'cpNE', low= 0.250, high = 0.999)
    # cpSW = Real(name = 'cpSW', low= 0.125, high = 0.999)

    # epNW = Real(name = 'epNW', low= 0.500, high = 0.999)
    # epNE = Real(name = 'epNE', low= 0.250, high = 0.999)
    # epSW = Real(name = 'epSW', low= 0.125, high = 0.999)

    # Non-biased
    cpNW = Real(name = 'cpNW', low= 0.001, high = 0.999)
    cpNE = Real(name = 'cpNE', low= 0.001, high = 0.999)
    cpSW = Real(name = 'cpSW', low= 0.001, high = 0.999)

    epNW = Real(name = 'epNW', low= 0.001, high = 0.999)
    epNE = Real(name = 'epNE', low= 0.001, high = 0.999)
    epSW = Real(name = 'epSW', low= 0.001, high = 0.999)

    dimensions2x2 = [cpNW, cpNE, cpSW, epNW, epNE, epSW]
    results = []
    for n in range(n_iter):
        # Fixing random state for reproducibility of each play level
        np.random.seed(78912345)
        # res = forest_minimize(player, dimensions = dimensions2x2, 
        #                       base_estimator='ET', acq_func='EI', initial_point_generator='grid', 
        #                       random_state=1, n_points=5000, n_initial_points = 10, n_calls = 510, xi=0.005, verbose=True)

        # fairly good results(scale: 1-10)  6.5-7.0 (date:02-05-2022)
        # res = forest_minimize(player, dimensions = dimensions2x2,
        #                   base_estimator='ET', acq_func='EI', initial_point_generator='lhs',
        #                   random_state=1, n_points=5000, n_initial_points = 10, n_calls = 510, xi=0.005, verbose=True)

        # ## fairly good results(scale: 1-10)  7 but SE has zero entries most times
        res = gp_minimize(player, 
                          dimensions=dimensions2x2, 
                          n_calls=210, 
                          n_initial_points=10, 
                          acq_func="EI",  
                          initial_point_generator="grid", 
                          noise=1e-10,
                          verbose=True
                          )

        # ## fairly good results(scale: 1-10)  5
        # res = gp_minimize(
        #          player,                
        #          dimensions = dimensions2x2, 
        #          n_calls=210,
        #          n_initial_points=10,
        #          acq_func='EI',
        #          n_points=5000,
        #          initial_point_generator='lhs',
        #          noise=1e-10,
        #          verbose=True
        #          )

        # ## fairly good results(scale: 1-10)  6
        # res = gp_minimize(
        #          player,                
        #          dimensions = dimensions2x2, 
        #          n_calls=210,
        #          n_initial_points=10,
        #          acq_func='EI',
        #          n_points=5000,
        #          initial_point_generator='random',
        #          noise=1e-10,
        #          verbose=True
        #          )

        print(n, ":res.fun", res.fun)
        results.append(res)

    print("==============")
    return results


def plot_coin_probability(play_log=None, show=True, legend=True):
    if not play_log:
        return

    scatter = 0
    x_val = [x+1 for x in range(len(play_log.gp_params))]
    for y in range(0,4):
        y_val = [play_log.gp_params[x][y] for x in range(len(play_log.gp_params))]
        scatter = plt.scatter(x_val, y_val)

    # produce a legend with the unique colors from the scatter
    if legend:
        plt.legend(scatter.legend_elements()[0], labels=['cpnw', 'cpne', 'cpsw', 'cpse'],
                   title="Quadrant Prob", bbox_to_anchor=(1.01, 1), borderaxespad=0)

    plt.title(play_log.title())
    plt.xlabel('GP Run')
    plt.ylabel('Coin Quadrant Probability')
    if show:
        plt.show()


def plot_start_probability(title = "GP", asset = '', bound = range(0), play_res=None, show=True, legend=True):
    if not play_res:
        return

    if asset == 'Coin':
        tab = ['cpnw', 'cpne', 'cpsw', 'cpse']
    elif asset == 'Enemy':
        tab = ['epnw', 'epne', 'epsw', 'epse']
    else:
        tab = []

    scatter = 0
    x_val = [x+1 for x in range(len(play_res))]
    for y in bound:
        y_val = [play_res[i].x[y] for i in range(len(play_res))]
        scatter = plt.scatter(x_val, y_val)

    # produce a legend with the unique colors from the scatter
    if legend:
        plt.legend(scatter.legend_elements()[0], labels=tab,
                   title="Quadrant Prob", bbox_to_anchor=(1.01, 1), borderaxespad=0)

    plt.title(title + " " + asset)
    plt.xlabel('GP Run')
    plt.ylabel('Best Start Probability')
    if (show):   
        plt.show()


def plot_enemy_probability(play_log=None, show=True, legend=True):
    if not play_log:
        return

    scatter = 0
    x_val = [x+1 for x in range(len(play_log.gp_params))]
    for y in range(4,8):
        y_val = [play_log.gp_params[x][y] for x in range(len(play_log.gp_params))]
        scatter = plt.scatter(x_val, y_val)

    # produce a legend with the unique colors from the scatter
    if legend:
        plt.legend(scatter.legend_elements()[0], labels=['epnw', 'epne', 'epsw', 'epse'],
                   title="Quadrant Prob", bbox_to_anchor=(1.01, 1), borderaxespad=0)

    plt.title(play_log.title())
    plt.xlabel('GP Run')
    plt.ylabel('Enemy Quadrant Probability') 
    if show:  
        plt.show()

def plot_3d_cells(play_log=None, show=True):
    if not play_log:
        return

    # Projection
    fig = plt.figure()
    ax = plt.axes(projection="3d")

    x_val = [r[1] for cells in play_log.coin_cells for r in cells]
    y_val = [c[0] for cells in play_log.coin_cells for c in cells]
    z_val = [z for z in range(len(play_log.coin_cells)) for _ in range(MAX_COIN)]
    ax.scatter(x_val, y_val, z_val, marker='o')

    x_val = [r[1] for cells in play_log.enemy_cells for r in cells]
    y_val = [c[0] for cells in play_log.enemy_cells for c in cells]
    z_val = [z for z in range(len(play_log.enemy_cells)) for _ in range(MAX_ENEMY)]
    ax.scatter(x_val, y_val, z_val, marker='^')

    ax.legend(['Coin', 'Enemy'])
    ax.set_title(play_log.title())
    ax.set_xlabel('Cell Column')
    ax.set_ylabel('Cell Row')
    ax.set_zlabel('Game Level')
    if show:
        plt.show()
 

def plot_coin_cells(play_log=None, show=True):
    if not play_log:
        return

    # Projection
    fig = plt.figure()
    ax = plt.axes(projection="3d")

    x_val = [r[1] for cells in play_log.coin_cells for r in cells]
    y_val = [c[0] for cells in play_log.coin_cells for c in cells]
    z_val = [z for z in range(len(play_log.coin_cells)) for _ in range(MAX_COIN)]
    ax.scatter(x_val, y_val, z_val, marker='o')

    ax.set_title(play_log.title())
    ax.set_xlabel('Coin Column')
    ax.set_ylabel('Coin Row')
    ax.set_zlabel('Game Level')
    if show:
        plt.show()


def plot_enemy_cells(play_log=None, show=True):
    if not play_log:
        return

    # Projection
    fig = plt.figure()
    ax = plt.axes(projection="3d")

    x_val = [r[1] for cells in play_log.enemy_cells for r in cells]
    y_val = [c[0] for cells in play_log.enemy_cells for c in cells]
    z_val = [z for z in range(len(play_log.enemy_cells)) for _ in range(MAX_ENEMY)]
    ax.scatter(x_val, y_val, z_val, marker='^', c='orange')

    ax.set_title(play_log.title())
    ax.set_xlabel('Enemy Column')
    ax.set_ylabel('Enemy Row')
    ax.set_zlabel('Game Level')
    if show:
        plt.show()


def plot_asset_dist(dist, title='', show=True, legend=True):
    def count(id, dist):
        return sum([quad[id] for quad in dist])
    # sum the coin/enemy distribution in each quadrant over all calls
    dNW = [count(0, d) for d in dist]
    dNE = [count(1, d) for d in dist]
    dSW = [count(2, d) for d in dist]
    dSE = [count(3, d) for d in dist]

    # setup bar labels
    labels = ['Greedy', 'Neutral', 'Aggressive']
    x = np.arange(len(labels))  # the label locations
    width = 0.4                 # the width of the bars

    # plot each quadrant bar--aligned using left edge of bar for positioning
    rects1 = plt.bar(x - width/2, dNW, width/4, label='North West', align='edge')
    rects2 = plt.bar(x - width/4, dNE, width/4, label='North East', align='edge')
    rects3 = plt.bar(x, dSW, width/4, label='South West', align='edge')
    rects4 = plt.bar(x + width/4, dSE, width/4, label='South East', align='edge')

    # add some text for labels, title and custom x-axis tick labels, etc.
    plt.ylabel('Distribution')
    plt.title(title + ' ' + 'Distribution by Player and Quadrant')
    plt.xticks(x, labels)

    # label each bar with total counts
    plt.bar_label(rects1, padding=3)
    plt.bar_label(rects2, padding=3)
    plt.bar_label(rects3, padding=3)
    plt.bar_label(rects4, padding=3)
    if legend:
        plt.legend(bbox_to_anchor=(1.01, 1))  # add keys legend top left of figure 
    if show:
        plt.show()


def plot_dist(show=True):
    plt.subplot(1, 2, 1)
    plot_asset_dist([greedy_log.coin_dist, neutral_log.coin_dist, aggressive_log.coin_dist], title='Coin', show=False, legend=False)
    plt.subplot(1, 2, 2)
    plot_asset_dist([greedy_log.enemy_dist, neutral_log.enemy_dist, aggressive_log.enemy_dist], title='Enemy', show=True, legend=True)


def plot_rescmp(greedy_res, neutral_res, aggressive_res):
    plt.subplot(1, 3, 1)
    plot_start_probability("Greedy res", "Coin", range(0,3), greedy_res, show=False, legend=False)
    plt.subplot(1, 3, 2)
    plot_start_probability("Neutral res", "Coin", range(0,3), neutral_res, show=False, legend=False)
    plt.subplot(1, 3, 3)
    plot_start_probability("Aggressive res", "Coin", range(0,3), aggressive_res, show=True, legend=True)

    plt.subplot(1, 3, 1)
    plot_start_probability("Greedy res", "Enemy", range(3,6), greedy_res, show=False, legend=False)
    plt.subplot(1, 3, 2)
    plot_start_probability("Neutral res", "Enemy", range(3,6), neutral_res, show=False, legend=False)
    plt.subplot(1, 3, 3)
    plot_start_probability("Aggressive res", "Enemy", range(3,6), aggressive_res, show=True, legend=True)

    plt.subplot(1, 3, 1)
    plot_coin_probability(greedy_log, show=False, legend=False)
    plt.subplot(1, 3, 2)
    plot_coin_probability(neutral_log, show=False, legend=False)
    plt.subplot(1, 3, 3)
    plot_coin_probability(aggressive_log, show=True, legend=True)

    plt.subplot(1, 3, 1)
    plot_enemy_probability(greedy_log, show=False, legend=False)
    plt.subplot(1, 3, 2)
    plot_enemy_probability(neutral_log, show=False, legend=False)
    plt.subplot(1, 3, 3)
    plot_enemy_probability(aggressive_log, show=True, legend=True)

    plot_3d_cells(greedy_log, show=False)
    plot_3d_cells(neutral_log, show=False)
    plot_3d_cells(aggressive_log, show=False)

    plot_coin_cells(greedy_log, show=False)
    plot_coin_cells(neutral_log, show=False)
    plot_coin_cells(aggressive_log, show=False)

    plot_enemy_cells(greedy_log, show=False)
    plot_enemy_cells(neutral_log, show=False)
    plot_enemy_cells(aggressive_log, show=True)


if __name__=='__main__':
    # play the game with three player profiles
    greedy_res = play_game(objective_greedy, "Greedy")
    neutral_res = play_game(objective_neutral, "Neutral")
    aggressive_res = play_game(objective_aggressive, "Aggressive")

    # import pdb; pdb.set_trace()

    # plot optimization convergence results per player profile
    plot_convergence(("Greedy res", greedy_res), ("Neutral res", neutral_res), ("Aggressive res", aggressive_res))
    plt.grid(); plt.show()

    # plot per quadrant coin/enemy distribution per player profile
    plot_dist()

    # plot objective function parameter space optimization results per player profile
    # and 3d visualise coin/enemy distribution 
    plot_rescmp(greedy_res, neutral_res, aggressive_res)
