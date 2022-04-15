from pyagame import maze,agent,textLabel,COLOR
from collections import deque
import numpy as np
from Coin import Coin
from Enemy import Enemy
from Quadrant import Quadrant
import random
from skopt import gp_minimize, forest_minimize
from skopt.space import Real
from skopt.plots import plot_convergence, plot_evaluations, plot_objective, plot_objective_2D
import matplotlib.pyplot as plt


MAZE_ROWS = 20
MAZE_COLS = MAZE_ROWS
RAND_LOW   = 0
RAND_HIGH  = MAZE_ROWS/2
MAX_ASSET  = 8
MAZE_DIFFICULTY = 7
MAZE_EXIT = (1, 1)
MAZE_START = (MAZE_ROWS, MAZE_COLS)
MAX_COIN = MAX_ASSET
MAX_ENEMY = MAX_ASSET
MIN_PROB = (1 / MAX_COIN)
ENEMY_TARGET = 8
COIN_TARGET = 80
WEIGHT_GAIN = 10


PL_COIN_CELLS  = 'coin_cells'
PL_ENEMY_CELLS = 'enemy_cells'
PL_GP_PARAMS   = 'gp_params'
PL_SCORE       = 'score'
class PlayLog:
    """
    
    """
    def __init__(self, player=''):
        """
        
        """
        self._player = player
        self._log = {PL_COIN_CELLS:[], PL_ENEMY_CELLS:[], PL_GP_PARAMS:[], PL_SCORE:[]}

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
    def enemy_cells(self):
        return (self._log[PL_ENEMY_CELLS])

    @enemy_cells.setter
    def enemy_cells(self, cells):
        self._log[PL_ENEMY_CELLS].append(cells)

    @property
    def gp_params(self):
        return (self._log[PL_GP_PARAMS])

    @gp_params.setter
    def gp_params(self, params):
        self._log[PL_GP_PARAMS].append(params)

    @property
    def score(self):
        return (self._log[PL_GP_PARAMS])

    @score.setter
    def score(self, score):
        self._log[PL_SCORE].append(score)

greedy_log = PlayLog('Greedy Player')
neutral_log = PlayLog('Neutral Player')
aggressive_log = PlayLog('aggressive Player')
run_log = PlayLog('All Players')

#FIXME: when do you use avoid_enemy?
def BFS(m, start=None, goal=None, avoid_enemy = False):
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
        
    #print("BFS PATH----")
    #print(f'{bfsPath}')
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


from queue import PriorityQueue
def h(cell1, cell2):
    x1, y1 = cell1
    x2, y2 = cell2
    return (abs(x1 - x2) + abs(y1 - y2))

def aStar(m, start=None, goal=None, avoid_enemy = False):
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
        cell = np.random.randint(low=RAND_LOW, high=RAND_HIGH, size=2)
        cell[0] += quadrant.base[0]
        cell[1] += quadrant.base[1]
        x=cell[0]
        y=cell[1]
        if 'C' not in m.maze_map[x,y].keys():
            coin = Coin(m, x, y)
            coin_count += 1
            m.maze_map[x,y]['C'] = coin
            coin_position_list[x,y]=cell
            #print(coin.cell)
    #print("Coin quadrant list", len(coin_position_list)," : ", coin_position_list)


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
    #print("Distance", distance)
    #print("Nearest Coin Cell", nearest_coin_cell)
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
                #print("found coin", coin)
                coin.collected = True
                currentScore += coin.value
                steps += (len(fwdPath)+1)
                #print("Coin Value: ", currentScore, coin.value)
        start_position = nearest_coin
        if currentScore >= coin_target:
            break
        #print("steps: ", steps)
    return start_position, steps

def addEnemy(m, quadrant, enemy_list = {}, number = MAX_ENEMY):
    #import pdb; pdb.set_trace()
    enemy_count = 0
    while enemy_count < number:
        cell = np.random.randint(low=RAND_LOW, high=RAND_HIGH, size=2)
        cell[0] += quadrant.base[0]
        cell[1] += quadrant.base[1]
        x=cell[0]
        y=cell[1]
        #FIXME: can an enemy and coin be in the same cell?
        if 'A' not in m.maze_map[x,y].keys():
            enemy = Enemy(m, x, y)
            enemy_count += 1
            m.maze_map[x,y]['A'] = enemy
            enemy_list[x,y]=cell
            #print(enemy)
    ##print("quadrant_enemy_list: ", len(enemy_list)," : ", enemy_list)
    #print("maze map: ", m.maze_map)


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
    ##print("Distance", distance)
    ##print("Nearest Enemy Cell", nearest_enemy_cell)
    #if nearest_enemy_cell == (1,1) and distance == 100000:
        #import pdb; pdb.set_trace()
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
                    #print("Enemy Path: ", fwdPath)
                # current health will tell you if enemy is defeated
                if current_health == 0:
                    current_health, stepsRH, home = restoreHealth(m, current_health, enemy)
                    steps += stepsRH
                    nearest_enemy = home
                    ##print("End health :", current_health)
                    #print("Number of enemies killed: ", enemy_killed_track)
                    #combatNearestEnemy(m, enemy_list)
                    #if nearest_enemy == (1,1):
                        #break
                ##print( "Player_health", current_health)
                ##print("Number of enemies killed: ", enemy_killed)
        start_position = nearest_enemy
    ##print("TOTAL STEP: ", steps)
    return start_position, steps

def combat(enemy, player_health):
    #player_health = 80
    while player_health > 0:
        if random.randint(0,10) > MAZE_DIFFICULTY:
            ##print("enemy defeated")
            enemy.defeated = True
            break
        else:
            player_health -= enemy.health
    ##print("fighting health: ", player_health)
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
        #print("Path: ", fwdPath)
    else:
        home = current_position
    return current_health, steps, home

# def fightEnemy(m, start_position, enemy_list):
#     nearest_enemy_to_start = findNearestEnemy(m, start_position, enemy_list)
#     player_health = 80
#     #while True:
#     #if agent.position == nearest_enemy_to_start:
#     for i in range(len(enemy_list)):
#         if random.randint(0,10) > 3:
#             nearest_enemy = findNearestEnemy(m, start_position, enemy_list)
#         else:
#             player_health -= 10


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

    #print("NW", cNW)
    #print("NE", cNE)
    #print("sW", cSW)
    #print("SE", cSE)

    return cNW, cNE, cSW, cSE

def createQuadrantDictionary(m,cNW, cNE, cSW, cSE):
    distribution_dict = {"qNW":Quadrant(m, cNW),\
                         "qNE":Quadrant(m, cNE),\
                         "qSW":Quadrant(m, cSW),\
                         "qSE":Quadrant(m, cSE)}

    return distribution_dict

def setProbability(m, distribution_dict, pNW = 0.25, pNE = 0.25, pSW = 0.25, pSE = 0.25):
    def prob(p1, p2, p3):
        return abs(1-(p1 + p2 + p3)) #+ 0.125
    distribution_dict["qNW"].probability = prob(pNE, pSW, pSE) #pNW
    distribution_dict["qNE"].probability = prob(pNW, pSW, pSE) #pNE
    distribution_dict["qSW"].probability = prob(pNW, pNE, pSE) #pSW
    distribution_dict["qSE"].probability = prob(pNW, pNE, pSW) #pSE
    

def distributeCoinAssets(m,  cpNW, cpNE, cpSW, cpSE):

    # Setting the probability of coins in each quadrant
    cNW, cNE, cSW, cSE = divideQuadrants(m)
    coin_quadrant_dict = createQuadrantDictionary(m, cNW, cNE, cSW, cSE)
    setProbability(m, coin_quadrant_dict, cpNW, cpNE, cpSW, cpSE)

    # Populate quadrants with coins
    coin_cells = {}
    while len(coin_cells) < MAX_COIN:
        for quadrant in coin_quadrant_dict.values():
            #import pdb; pdb.set_trace()
            #print("Coin Quad:", quadrant)
            number = int(MAX_COIN * quadrant.probability)
            #print("Add Coins:", number, " Prob:", quadrant.probability)
            if len(coin_cells) >= MAX_COIN:
                break
            if number:
                total_coins = len(coin_cells)
                if total_coins + number > MAX_COIN:
                    number = MAX_COIN - total_coins
                addCoins(m, quadrant, coin_cells, number)
                #print("Quad: ", quadrant.base, "Number: ", number)
            #print("Coin Quadrant: ", quadrant.base)
            #print("Coin dict: ", coin_cells)

    ##print("Coin list: ", list(coin_cells))       
    run_log.coin_cells = list(coin_cells)
    return list(coin_cells)

def distributeEnemyAssets(m, epNW, epNE, epSW, epSE):
    # Setting the probability of enemies in each quadrant
    eNW, eNE, eSW, eSE = divideQuadrants(m)
    enemy_quadrant_dict = createQuadrantDictionary(m, eNW, eNE, eSW, eSE)
    setProbability(m, enemy_quadrant_dict, epNW, epNE, epSW, epSE)

    # Populate quadrants with enemies
    enemy_cells = {}
    while len(enemy_cells) < MAX_ENEMY:
        for quadrant in enemy_quadrant_dict.values():
            #print("Enemy Quad:", quadrant)
            number = int(MAX_ENEMY * quadrant.probability)
            #print("Add Enemy:", number, " Prob:", quadrant.probability)
            if len(enemy_cells) >= MAX_ENEMY:
                break
            if number:
                total_enemy = len(enemy_cells)
                if total_enemy + number > MAX_ENEMY:
                    number = MAX_ENEMY - total_enemy
                addEnemy(m, quadrant, enemy_cells, number)
            #print("Enemy Quadrant: ", quadrant.base)
            #print("Enemy dict: ", enemy_cells)
    
    ##print("enemy List: ", list(enemy_cells))
    run_log.enemy_cells = list(enemy_cells)
    return list(enemy_cells)

def player_style1(m, coin_list, enemy_list):
    '''
    Player style where you collect all the coins and defeat all the enemies
    '''
    _, steps_coins = collectNearestCoins(m, coin_list)
    
    _, steps_enemy = combatNearestEnemy(m, enemy_list)

    total_steps = steps_coins + steps_enemy
    return total_steps

## def greedy 
'''
The objective of this player is to collect all the coins without fighting any enemies. This means collecting coins 1-8, without 
fighting any enemies. So all enemies must be avoided and pathed around. If no path then choose the quickest route.
'''
def findOnlyCoin(m, start_position, coin_position_list):
    """
    """
    nearest_coin_cell=(1,1) # Default position maze exit
    distance = 100000
    for cell in coin_position_list:
        if not m.maze_map[cell]['C'].collected: 
            bSearch,aPath,fwdPath,weight = m.findPath(m, start_position, cell, avoid_enemy=False)
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
                #print("found coin", coin)
                coin.collected = True
                #print("collected")
                currentScore += coin.value
               # print("score: ", currentScore)
                steps += (len(fwdPath)+1)
        #print(" coin path: ", fwdPath)
        #print("Coin Value: ", currentScore, coin.value)
        start_position = nearest_coin
        if currentScore >= coin_target:
            break
        #print("steps: ", steps)
    return start_position, steps


def greedy_player(m, coin_list, enemy_list, target = 0): #35):
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
def aggressive_player(m, coin_list, enemy_list, target = 0): #50):
    total_steps = 0
    new_position, steps_enemy = combatNearestEnemy(m, enemy_list, start_cell=MAZE_START, enemy_target=8)
    new_position, steps_coins = collectNearestCoins(m, coin_list, start_cell=new_position, coin_target=0)
    if new_position != MAZE_EXIT:
        bSearch,aPath,fwdPath,weight = m.findPath(m, new_position, MAZE_EXIT)
        total_steps = (len(fwdPath)+1)
    total_steps = steps_coins + steps_enemy
    score = total_steps - target
    return score

## def middle
'''
This player wants to partake in both objectives, fighting 1-8 enemies or collecting 1-8 coins, the player fights the enemies
first, the enemies that are set as defeated are allowed to be passed over by the player when he is collecting the coins. 
Otherwise he will have to path around
'''

def neutral_player(m, coin_list, enemy_list, target = 0): #60):
    total_steps = 0
    new_position, steps_enemy = combatNearestEnemy(m, enemy_list, start_cell=MAZE_START, enemy_target=5)
    new_position, steps_coins = collectOnlyCoins(m, coin_list, start_cell=new_position, coin_target=40)
    if new_position != MAZE_EXIT:
        bSearch,aPath,fwdPath,weight = m.findPath(m, new_position, MAZE_EXIT)
        total_steps = (len(fwdPath)+1)
    total_steps = steps_coins + steps_enemy
    score = total_steps - target
    return score

#def objective(cpNW, cpNE, cpSW, cpSE, epNW, epNE, epSW, epSE, *args, **kwargs):
def objective_greedy(dimensions2x2):
    #import pdb; pdb.set_trace()
    cpNW = dimensions2x2[0]
    cpNE = dimensions2x2[1]
    cpSW = dimensions2x2[2]
    cpSE = dimensions2x2[3]

    epNW = dimensions2x2[4]
    epNE = dimensions2x2[5]
    epSW = dimensions2x2[6]
    epSE = dimensions2x2[7]
    greedy_log.gp_params = dimensions2x2

    m=maze(MAZE_ROWS, MAZE_COLS)
    m.CreateMaze(loopPercent=20, findPath=aStar, displayMaze=False)

    coin_list = distributeCoinAssets(m, cpNW, cpNE, cpSW, cpSE)
    greedy_log.coin_cells = coin_list

    enemy_list = distributeEnemyAssets(m, epNW, epNE, epSW, epSE)
    greedy_log.enemy_cells = enemy_list

    total_steps = greedy_player(m, coin_list, enemy_list)
    greedy_log.score = total_steps

    if 1.0 - (cpNW + cpNE + cpSW + cpSE) < 1.0:
        total_steps += 300

    if 1.0 - (epNW + epNE + epSW + epSE) < 1.0:
        total_steps += 300

    return total_steps

def objective_neutral(dimensions2x2):
    #import pdb; pdb.set_trace()
    cpNW = dimensions2x2[0]    
    cpNE = dimensions2x2[1]
    cpSW = dimensions2x2[2]
    cpSE = dimensions2x2[3]

    epNW = dimensions2x2[4]
    epNE = dimensions2x2[5]
    epSW = dimensions2x2[6]
    epSE = dimensions2x2[7]
    neutral_log.gp_params = dimensions2x2

    m=maze(MAZE_ROWS, MAZE_COLS)
    m.CreateMaze(loopPercent=20, findPath=aStar, displayMaze=False)

    coin_list = distributeCoinAssets(m, cpNW, cpNE, cpSW, cpSE)
    neutral_log.coin_cells = coin_list

    enemy_list = distributeEnemyAssets(m, epNW, epNE, epSW, epSE)
    neutral_log.enemy_cells = enemy_list

    total_steps = neutral_player(m, coin_list, enemy_list)
    neutral_log.score = total_steps

    if 1.0 - (cpNW + cpNE + cpSW + cpSE) < 1:
        total_steps += 300

    if 1.0 - (epNW + epNE + epSW + epSE) < 1:
        total_steps += 300

    return total_steps
    
def objective_aggressive(dimensions2x2):
    #import pdb; pdb.set_trace()
    cpNW = dimensions2x2[0]
    cpNE = dimensions2x2[1]
    cpSW = dimensions2x2[2]
    cpSE = dimensions2x2[3]

    epNW = dimensions2x2[4]
    epNE = dimensions2x2[5]
    epSW = dimensions2x2[6]
    epSE = dimensions2x2[7]
    aggressive_log.gp_params = dimensions2x2
    
    m=maze(MAZE_ROWS, MAZE_COLS)
    m.CreateMaze(loopPercent=20, findPath=aStar, displayMaze=False)

    coin_list = distributeCoinAssets(m, cpNW, cpNE, cpSW, cpSE)
    aggressive_log.coin_cells = coin_list

    enemy_list = distributeEnemyAssets(m, epNW, epNE, epSW, epSE)
    aggressive_log.enemy_cells = enemy_list

    total_steps = aggressive_player(m, coin_list, enemy_list)
    aggressive_log.score = total_steps
    
    if 1.0 - (cpNW + cpNE + cpSW + cpSE) < 1:
        total_steps += 300

    if 1.0 - (epNW + epNE + epSW + epSE) < 1:
        total_steps += 300

    return total_steps

# def play_game(player, n_iter = 5):
#     # cpNW = Real(name = 'cpNW', low= 0.125, high = 0.625)
#     # cpNE = Real(name = 'cpNE', low= 0.125, high = 0.375)
#     # cpSW = Real(name = 'cpSW', low= 0.125, high = 0.375)
#     # cpSE = Real(name = 'cpSE', low= 0.001, high = 0.250)

#     # epNW = Real(name = 'epNW', low= 0.001, high = 0.250)
#     # epNE = Real(name = 'epNE', low= 0.125, high = 0.375)
#     # epSW = Real(name = 'epSW', low= 0.125, high = 0.375)
#     # epSE = Real(name = 'epSE', low= 0.125, high = 0.625)

#     cpNW = Real(name = 'cpNW', low= 0.125, high = 0.875)
#     cpNE = Real(name = 'cpNE', low= 0.001, high = 0.875)
#     cpSW = Real(name = 'cpSW', low= 0.001, high = 0.875)
#     cpSE = Real(name = 'cpSE', low= 0.001, high = 0.875)

#     epNW = Real(name = 'epNW', low= 0.125, high = 0.875)
#     epNE = Real(name = 'epNE', low= 0.001, high = 0.875)
#     epSW = Real(name = 'epSW', low= 0.001, high = 0.875)
#     epSE = Real(name = 'epSE', low= 0.001, high = 0.875)

#     dimensions2x2 = [cpNW, cpNE , cpSW, cpSE , epNW, epNE, epSW, epSE]

#     np.random.seed(12345)
#     return [forest_minimize(player, dimensions = dimensions2x2, n_initial_points = 10, n_calls = 50)
#             for n in range(n_iter)]
        
def play_game(player, n_iter = 1):
    # cpNW = Real(name = 'cpNW', low= 0.125, high = 0.625)
    # cpNE = Real(name = 'cpNE', low= 0.125, high = 0.375)
    # cpSW = Real(name = 'cpSW', low= 0.125, high = 0.375)
    # cpSE = Real(name = 'cpSE', low= 0.001, high = 0.250)

    # epNW = Real(name = 'epNW', low= 0.001, high = 0.250)
    # epNE = Real(name = 'epNE', low= 0.125, high = 0.375)
    # epSW = Real(name = 'epSW', low= 0.125, high = 0.375)
    # epSE = Real(name = 'epSE', low= 0.125, high = 0.625)

    # cpNW = Real(name = 'cpNW', low= 0.125, high = 0.875)
    # cpNE = Real(name = 'cpNE', low= 0.001, high = 0.875)
    # cpSW = Real(name = 'cpSW', low= 0.001, high = 0.875)
    # cpSE = Real(name = 'cpSE', low= 0.001, high = 0.875)

    # epNW = Real(name = 'epNW', low= 0.125, high = 0.875)
    # epNE = Real(name = 'epNE', low= 0.001, high = 0.875)
    # epSW = Real(name = 'epSW', low= 0.001, high = 0.875)
    # epSE = Real(name = 'epSE', low= 0.001, high = 0.875)

 
    cpNW = Real(name = 'cpNW', low= 0.001, high = 0.999)
    cpNE = Real(name = 'cpNE', low= 0.001, high = 0.999)
    cpSW = Real(name = 'cpSW', low= 0.001, high = 0.999)
    cpSE = Real(name = 'cpSE', low= 0.001, high = 0.999)

    epNW = Real(name = 'epNW', low= 0.001, high = 0.999)
    epNE = Real(name = 'epNE', low= 0.001, high = 0.999)
    epSW = Real(name = 'epSW', low= 0.001, high = 0.999)
    epSE = Real(name = 'epSE', low= 0.001, high = 0.999)

    dimensions2x2 = [cpNW, cpNE , cpSW, cpSE , epNW, epNE, epSW, epSE]

    # Fixing random state for reproducibility
    #np.random.seed(78912345)
    results = []
    for n in range(n_iter):
        # Fixing random state for reproducibility
        #np.random.seed(78912345)
        res = gp_minimize(player, dimensions = dimensions2x2, n_initial_points = 10, n_calls = 300)
        print("res.fun", res.fun)
        for i in range(len(res.x)):
            if i > len(dimensions2x2):
                break
            dimensions2x2[i].high = res.x[i]
        results.append(res)

    return results



def plot_coin_probability(play_log=None):
    if not play_log:
        return

    scatter = 0
    x_val = [x+1 for x in range(len(play_log.gp_params))]
    for y in range(0,4):
        y_val = [play_log.gp_params[x][y] for x in range(len(play_log.gp_params))]
        scatter = plt.scatter(x_val, y_val)

    # produce a legend with the unique colors from the scatter
    plt.legend(scatter.legend_elements(num=4)[0], labels=['cpnw', 'cpne', 'cpsw', 'cpse'],
                    loc="lower left", title="Quadrant Prob")

    plt.title(play_log.title())
    plt.xlabel('GP Run')
    plt.ylabel('Coin Quadrant Probability')   
    plt.show()

def plot_start_probability(title = "GP", asset = '', bound = range(0), play_res=None):
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
        scatter = plt.plot(x_val, y_val)
        scatter = plt.scatter(x_val, y_val)

    # produce a legend with the unique colors from the scatter
    plt.legend(scatter.legend_elements(num=4)[0], labels=tab,
                    loc="lower left", title="Quadrant Prob")

    plt.title(title + " " + asset)
    plt.xlabel('GP Run')
    plt.ylabel('Probability')   
    plt.show()

# plt.figure(figsize=(8,6))
# sp_names = ['Adelie', 'Gentoo', 'Chinstrap']
# scatter = plt.scatter(df.culmen_length_mm, 
#             df.culmen_depth_mm,
#             s=150,
#             c=df.species.astype('category').cat.codes)
# plt.xlabel("Culmen Length", size=24)
# plt.ylabel("Culmen Depth", size=24)
# # add legend to the plot with names
# plt.legend(handles=scatter.legend_elements()[0], 
#            labels=sp_names,
#            title="species")
# plt.savefig("scatterplot_colored_by_variable_with_legend_matplotlib_Python.png",
#                     format='png',dpi=

def plot_enemy_probability(play_log=None):
    if not play_log:
        return

    scatter = 0
    x_val = [x+1 for x in range(len(play_log.gp_params))]
    for y in range(4,8):
        y_val = [play_log.gp_params[x][y] for x in range(len(play_log.gp_params))]
        scatter = plt.scatter(x_val, y_val)

    # produce a legend with the unique colors from the scatter
    plt.legend(scatter.legend_elements(num=4)[0], labels=['epnw', 'epne', 'epsw', 'epse'],
                    loc="lower left", title="Quadrant Prob")

    plt.title(play_log.title())
    plt.xlabel('GP Run')
    plt.ylabel('Enemy Quadrant Probability')   
    plt.show()

def plot_3d_cells(play_log=None):
    if not play_log:
        return

    # Import Library
    from mpl_toolkits import mplot3d

    # Projection
    ax = plt.axes(projection="3d")

    x_val = [x[1] for cells in play_log.coin_cells for x in cells]
    y_val = [x[0] for cells in play_log.coin_cells for x in cells]
    z_val = [z for z in range(len(play_log.coin_cells)*MAX_COIN)]
    ax.scatter(x_val, y_val, z_val, marker='o')

    x_val = [x[1] for cells in play_log.enemy_cells for x in cells]
    y_val = [x[0] for cells in play_log.enemy_cells for x in cells]
    z_val = [z for z in range(len(play_log.coin_cells)*MAX_ENEMY)]
    ax.scatter(x_val, y_val, z_val, marker='^')
    #plt.scatter(x_val, y_val)

    plt.title(play_log.title())
    #plt.xlabel('Enemy Cell X')
    #plt.ylabel('Enemy Cell Y')   

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

def plot_3dshade_cells(play_log=None):
    if not play_log:
        return

    from matplotlib import cm
    from matplotlib.colors import LightSource

    # Import Library
    from mpl_toolkits import mplot3d

    # Projection
    ax = plt.axes(projection="3d")
    ls = LightSource(270, 45)

    x_val = [x[1] for cells in play_log.coin_cells for x in cells]
    y_val = [x[0] for cells in play_log.coin_cells for x in cells]
    z_val = [z for z in range(len(play_log.coin_cells)*MAX_COIN)]
    #ax.scatter(x_val, y_val, z_val, marker='o')
    surf = ax.plot_surface(x_val, y_val, z_val, rstride=1, cstride=1, facecolors=rgb,
                       linewidth=0, antialiased=False, shade=False)

    x_val = [x[1] for cells in play_log.enemy_cells for x in cells]
    y_val = [x[0] for cells in play_log.enemy_cells for x in cells]
    z_val = [z for z in range(len(play_log.coin_cells)*MAX_ENEMY)]
    surf = ax.plot_surface(x_val, y_val, z_val, rstride=1, cstride=1, facecolors=rgb,
                       linewidth=0, antialiased=False, shade=False)
    #ax.scatter(x_val, y_val, z_val, marker='^')
    #plt.scatter(x_val, y_val)

    plt.title(play_log.title())
    #plt.xlabel('Enemy Cell X')
    #plt.ylabel('Enemy Cell Y')   

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()



def plot_coin_cells(play_log=None):
    if not play_log:
        return
    import pdb; pdb.set_trace()

    # Import Library
    from mpl_toolkits import mplot3d

    # Projection
    ax = plt.axes(projection="3d")

    #fig = plt.figure()
    #ax = plt.axes(projection ='3d')

    x_val = [x[1] for cells in play_log.coin_cells for x in cells]
    y_val = [x[0] for cells in play_log.coin_cells for x in cells]
    z_val = [z for z in range(len(play_log.coin_cells)*MAX_COIN)]
    ax.scatter(x_val, y_val, z_val)
    #plt.scatter(x_val, y_val)

    plt.title(play_log.title())
    #plt.xlabel('Coin Cell X')
    #plt.ylabel('Coin Cell Y')   

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

def plot_enemy_cells(play_log=None):
    if not play_log:
        return

    # Import Library
    from mpl_toolkits import mplot3d

    # Projection
    ax = plt.axes(projection="3d")

    x_val = [x[1] for cells in play_log.enemy_cells for x in cells]
    y_val = [x[0] for cells in play_log.enemy_cells for x in cells]
    z_val = [z for z in range(len(play_log.coin_cells)*MAX_ENEMY)]
    ax.scatter(x_val, y_val, z_val)
    #plt.scatter(x_val, y_val)

    plt.title(play_log.title())
    #plt.xlabel('Enemy Cell X')
    #plt.ylabel('Enemy Cell Y')   

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

if __name__=='__main__':

    #m=maze(MAZE_ROWS, MAZE_COLS)
    #m.CreateMaze(loopPercent=70, displayMaze=False)
    #coin_position_list = addCoins(m)
    #print("Coin position list", coin_position_list)
    #print(m.maze_map)
    #print("Coin objects: ", m._coins)
    #bSearch,bfsPath,fwdPath,currentScore=BFS(m)
    #randomlyCollectAllCoins(m, coin_position_list)
    #score = collectNearestCoins(m, coin_position_list)
    #print("Score: ", score)
    #start_position = (m.rows,m.cols)
    #enemy_list = addEnemy(m)
    #print("enemy list", enemy_list)

    #divideQuadrants(m)

    #Quadrant probabilities for coin
    # cpNW = 0.123
    # cpNE = 0.453
    # cpSW = 0.345
    # cpSE = 0.079
    # coin_list = distributeCoinAssets(m, cpNW, cpNE, cpSW, cpSE)

    # #Quadrant probabiltiies for enemy
    # epNW = 0.244
    # epNE = 0.700
    # epSW = 0.020
    # epSE = 0.036
    # enemy_list = distributeEnemyAssets(m, epNW, epNE, epSW, epSE)

    # combatNearestEnemy(m, enemy_list)

    
    # cpNW = Real(name = 'cpNW', low= 0.001, high = 0.999)
    # cpNE = Real(name = 'cpNE', low= 0.001, high = 0.999)
    # cpSW = Real(name = 'cpSW', low= 0.001, high = 0.999)
    # cpSE = Real(name = 'cpSE', low= 0.001, high = 0.999)

    # epNW = Real(name = 'epNW', low= 0.001, high = 0.999)
    # epNE = Real(name = 'epNE', low= 0.001, high = 0.999)
    # epSW = Real(name = 'epSW', low= 0.001, high = 0.999)
    # epSE = Real(name = 'epSE', low= 0.001, high = 0.999)

    # dimensions2x2 = [cpNW, cpNE , cpSW, cpSE , epNW, epNE, epSW, epSE]

    # res = gp_minimize(objective, dimensions = dimensions2x2, n_calls = 20)

    # print("Best Result: ", res.fun)

    # print("Best Paramters: ", res.x)

    # plot_convergence(res)

    #_ = plot_evaluations(res) # this one works plots samples vs probability 
    #_ = plot_objective(res, n_samples=40)
    #_ = plot_objective_2D(result = res,
    #                      dimension_identifier1='epNW',
    #                      dimension_identifier2='cpNW'
    #                      )
    greedy_res = play_game(objective_greedy)
    neutral_res = play_game(objective_neutral)
    aggressive_res = play_game(objective_aggressive)
    plot_convergence(("Greedy res", greedy_res),
                    ("Neutral res", neutral_res),
                    ("aggressive res", aggressive_res))
    plt.show()

    import pdb; pdb.set_trace()

    plot_start_probability("Greedy res", "Coin", range(0,4), greedy_res)
    plot_start_probability("Greedy res", "Enemy", range(4,8), greedy_res)

    plot_start_probability("Neutral res", "Coin", range(0,4), greedy_res)
    plot_start_probability("Neutral res", "Enemy", range(4,8), greedy_res)

    plot_start_probability("aggressive res", "Coin", range(0,4), greedy_res)
    plot_start_probability("aggressive res", "Enemy", range(4,8), greedy_res)

    plot_coin_probability(greedy_log)
    plot_enemy_probability(greedy_log)

    plot_coin_probability(neutral_log)
    plot_enemy_probability(neutral_log)

    plot_coin_probability(aggressive_log)
    plot_enemy_probability(aggressive_log)

    plot_3d_cells(greedy_log)
    plot_3dshade_cells(greedy_log)
    plot_coin_cells(greedy_log)
    plot_enemy_cells(greedy_log)

    plot_3d_cells(neutral_log)
    plot_coin_cells(neutral_log)
    plot_enemy_cells(neutral_log)

    plot_3d_cells(aggressive_log)
    plot_coin_cells(aggressive_log)
    plot_enemy_cells(aggressive_log)

    import pdb; pdb.set_trace()
    