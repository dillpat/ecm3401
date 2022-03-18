from pyagame import maze,agent,textLabel,COLOR
from collections import deque
import numpy as np
from Coin import Coin
from Enemy import Enemy
from Quadrant import Quadrant
import random
from skopt import gp_minimize
from skopt.space import Real

MAZE_ROWS = 10
MAZE_COLS = 10
MAX_ASSET = 8
MAZE_DIFFICULTY = 7
MAX_COIN = MAX_ASSET
MAX_ENEMY = MAX_ASSET
MIN_PROB = (1 / MAX_COIN)

def BFS(m,start=None, goal=None):
    if start is None:
        start=(m.rows,m.cols)
    if goal is None:
        goal = m._goal
    frontier = deque()
    frontier.append(start)
    bfsPath = {}
    explored = [start]
    bSearch=[]
    currentScore = 0

    while len(frontier)>0:
        currCell=frontier.popleft()
        # if ('C' in m.maze_map[currCell]):
        #     coin = m.maze_map[currCell]['C']
        #     if not coin.collected:
        #         #print("found coin", coin)
        #         coin.collected = True
        #         currentScore += coin.value
        #         #print("Coin Value: ", currentScore, coin.value)
        

        if currCell==goal:
            break
        for d in 'ESNW':
            if m.maze_map[currCell][d]==True:
                if d=='E':
                    childCell=(currCell[0],currCell[1]+1)
                elif d=='W':
                    childCell=(currCell[0],currCell[1]-1)
                elif d=='S':
                    childCell=(currCell[0]+1,currCell[1])
                elif d=='N':
                    childCell=(currCell[0]-1,currCell[1])

                if childCell in explored:
                    continue


                frontier.append(childCell)
                explored.append(childCell)
                bfsPath[childCell] = currCell
                bSearch.append(childCell)
    # print(f'{bfsPath}')
    fwdPath={}
    cell=goal
    while cell!=(start):
        fwdPath[bfsPath[cell]]=cell
        cell=bfsPath[cell]
    return bSearch,bfsPath,fwdPath


def addCoins(m, quadrant, coin_position_list = {}, number=MAX_COIN):
    coin_count = 0
    while coin_count < number:
        cell = np.random.randint(low=0, high=5, size=2)
        cell[0] += quadrant.base[0]
        cell[1] += quadrant.base[1]
        x=cell[0]
        y=cell[1]
        coin = Coin(m, x, y)
        if 'C' not in m.maze_map[x,y].keys():
            coin_count += 1
        m.maze_map[x,y]['C'] = coin
        #print(coin.cell)
        coin_position_list[x,y]=cell
    print("Coin quadrant list", len(coin_position_list)," : ", coin_position_list)


def randomlyCollectAllCoins(m, coin_position_list):
    start=(m.rows,m.cols)
    currentScore = 0
    #print(coin_position_list)
    for cell in coin_position_list:
        if not m.maze_map[cell]['C'].collected:
            m.maze_map[cell]['C'].collected = True
            currentScore += m.maze_map[cell]['C'].collected.value
            #print("Start and end ", start, cell)
            bSearch,bfsPath,fwdPath,score = BFS(m, start, cell)
            currentScore += score
            #print("running score", score)
            #print("forward path", fwdPath)
            start = cell
    bSearch,bfsPath,fwdPath,score = BFS(m, start)
    currentScore += score
    #print(m._goal)
    #print(currentScore)

def findNearestCoin(m, start_position, coin_position_list):
    """
    """
    nearest_coin_cell=(1,1) # Default position maze exit
    distance = 100000
    for cell in coin_position_list:
        if not m.maze_map[cell]['C'].collected: 
            bSearch,bfsPath,fwdPath = BFS(m, start_position, cell)
            if (len(fwdPath)+1) < distance:
                distance = (len(fwdPath)+1)
                nearest_coin_cell = cell
    print("Distance", distance)
    print("Nearest Coin Cell", nearest_coin_cell)
    return nearest_coin_cell

def collectNearestCoins(m, coin_position_list, coin_target = 80):
    start_position = (m.rows,m.cols)
    currentScore = 0
    steps = 0
    for i in range(len(coin_position_list)):
        nearest_coin = findNearestCoin(m, start_position, coin_position_list)
        bSearch,bfsPath,fwdPath = BFS(m, start_position, nearest_coin)
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
        print("steps: ", steps)
    return currentScore, steps

def addEnemy(m, quadrant, enemy_list = {}, number = MAX_ENEMY):
    #import pdb; pdb.set_trace()
    enemy_count = 0
    while enemy_count < number:
        cell = np.random.randint(low=1, high=5, size=2)
        cell[0] += quadrant.base[0]
        cell[1] += quadrant.base[1]
        x=cell[0]
        y=cell[1]
        enemy = Enemy(m, x, y)
        if 'A' not in m.maze_map[x,y].keys():
            enemy_count += 1
        m.maze_map[x,y]['A'] = enemy
        #print(enemy)
        enemy_list[x,y]=cell
    print("quadrant_enemy_list: ", len(enemy_list)," : ", enemy_list)
    #print("maze map: ", m.maze_map)


def findNearestEnemy(m, start_position, enemy_list):
    """
    """
    nearest_enemy_cell=(1,1) # Default position maze exit
    distance = 100000
    for cell in enemy_list:
        if not m.maze_map[cell]['A'].defeated: 
            bSearch,bfsPath,fwdPath = BFS(m, start_position, cell)
            if (len(fwdPath)+1) < distance:
                distance = (len(fwdPath)+1)
                nearest_enemy_cell = cell
    print("Distance", distance)
    print("Nearest Enemy Cell", nearest_enemy_cell)
    #if nearest_enemy_cell == (1,1) and distance == 100000:
        #import pdb; pdb.set_trace()
    return nearest_enemy_cell

def combatNearestEnemy(m, enemy_list, player_health = 80):
    current_health = player_health
    start_position = (m.rows,m.cols)
    enemy_killed = 0
    steps = 0
    while enemy_killed < len(enemy_list):
        nearest_enemy = findNearestEnemy(m, start_position, enemy_list)
        bSearch,bfsPath,fwdPath = BFS(m, start_position, nearest_enemy)
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
                    print("End health :", current_health)
                    #print("Number of enemies killed: ", enemy_killed_track)
                    #combatNearestEnemy(m, enemy_list)
                    #if nearest_enemy == (1,1):
                        #break
                print( "Player_health", current_health)
                print("Number of enemies killed: ", enemy_killed)
        start_position = nearest_enemy
    print("TOTAL STEP: ", steps)
    return current_health, steps

def combat(enemy, player_health):
    #player_health = 80
    while player_health > 0:
        if random.randint(0,10) > MAZE_DIFFICULTY:
            print("enemy defeated")
            enemy.defeated = True
            break
        else:
            player_health -= enemy.health
    print("fighting health: ", player_health)
    return player_health

"""
return back to start method from current position then restore player_health to 80
"""
def restoreHealth(m, current_health, enemy):
    home = (m.rows,m.cols)
    current_position = enemy.position
    steps = 0
    if current_health == 0:
        bSearch,bfsPath,fwdPath = BFS(m, current_position, home)
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
    cNW.append((1,1)) # (1,1)
    cNW.append((half_width, half_height))
    cNW.append((1, half_height))
    cNW.append((half_width, 1))

    # Creating the NE quadrant
    cNE.append((half_width + 1, 1)) # (6,1)
    cNE.append((width_max, 1))
    cNE.append((width_max, half_height))
    cNE.append((half_width + 1, half_height))

    # Creating the SW quadrant
    cSW.append((1, half_height + 1)) # (1,6)
    cSW.append((half_width, half_height + 1))
    cSW.append((1, height_max))
    cSW.append((half_width, height_max))

    # Creating the SE quadrant
    cSE.append((half_width + 1, half_height + 1)) # (6,6)
    cSE.append((width_max, half_height + 1))
    cSE.append((width_max, height_max))
    cSE.append((half_width + 1, height_max))

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
    distribution_dict["qNW"].probability = (pNW // MIN_PROB) * MIN_PROB
    distribution_dict["qNE"].probability = (pNE // MIN_PROB) * MIN_PROB
    distribution_dict["qSW"].probability = (pSW // MIN_PROB) * MIN_PROB
    distribution_dict["qSE"].probability = (pSE // MIN_PROB) * MIN_PROB

def distributeCoinAssets(m,  cpNW, cpNE, cpSW, cpSE):

    # Setting the probability of coins in each quadrant
    cNW, cNE, cSW, cSE = divideQuadrants(m)
    coin_quadrant_dict = createQuadrantDictionary(m, cNW, cNE, cSW, cSE)
    setProbability(m, coin_quadrant_dict, cpNW, cpNE, cpSW, cpSE)

    # Populate quadrants with coins
    coin_cells = {}
    while len(coin_cells) < MAX_COIN:
        for quadrant in coin_quadrant_dict.values():
            print("Coin Quad:", quadrant)
            number = int(MAX_COIN * quadrant.probability)
            if len(coin_cells) >= MAX_COIN:
                break
            if number:
                total_coins = len(coin_cells)
                if total_coins + number > MAX_COIN:
                    number = MAX_COIN - total_coins
                addCoins(m, quadrant, coin_cells, number)
            print("Coin Quadrant: ", quadrant.base)
            print("Coin dict: ", coin_cells)

    print("Coin list: ", list(coin_cells))        
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
            print("Enemy Quad:", quadrant)
            number = int(MAX_ENEMY * quadrant.probability)
            if len(enemy_cells) >= MAX_ENEMY:
                break
            if number:
                total_enemy = len(enemy_cells)
                if total_enemy + number > MAX_ENEMY:
                    number = MAX_ENEMY - total_enemy
                addEnemy(m, quadrant, enemy_cells, number)
            print("Enemy Quadrant: ", quadrant.base)
            print("Enemy dict: ", enemy_cells)
    
    print("enemy List: ", list(enemy_cells))
    return list(enemy_cells)


#def objective(cpNW, cpNE, cpSW, cpSE, epNW, epNE, epSW, epSE, *args, **kwargs):
def objective(dimensions):
    #import pdb; pdb.set_trace()
    cpNW = dimensions[0]
    cpNE = dimensions[1]
    cpSW = dimensions[2]
    cpSE = dimensions[3]

    epNW = dimensions[4]
    epNE = dimensions[5]
    epSW = dimensions[6]
    epSE = dimensions[7]

    m=maze(MAZE_ROWS, MAZE_COLS)
    m.CreateMaze(loopPercent=60)

    # Quadrant probabilities for coin
    # cpNW = 0.123
    # cpNE = 0.453
    # cpSW = 0.345
    # cpSE = 0.079
    coin_list = distributeCoinAssets(m, cpNW, cpNE, cpSW, cpSE)

    # Quadrant probabiltiies for enemy
    # epNW = 0.244
    # epNE = 0.700
    # epSW = 0.020
    # epSE = 0.036
    enemy_list = distributeEnemyAssets(m, epNW, epNE, epSW, epSE)

    ##steps_coins = collectNearestCoins(m, coin_list)[1]
    # or
    _, steps_coins = collectNearestCoins(m, coin_list)

    #steps_enemy = combatNearestEnemy(m, enemy_list)[1]
    _, steps_enemy = combatNearestEnemy(m, enemy_list)

    total_steps = steps_coins + steps_enemy
    return total_steps

if __name__=='__main__':

    m=maze(MAZE_ROWS, MAZE_COLS)
    m.CreateMaze(loopPercent=60)
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

    
    cpNW = Real(name = 'cpNW', low= 0.001, high = 0.999)
    cpNE = Real(name = 'cpNE', low= 0.001, high = 0.999)
    cpSW = Real(name = 'cpSW', low= 0.001, high = 0.999)
    cpSE = Real(name = 'cpSE', low= 0.001, high = 0.999)

    epNW = Real(name = 'epNW', low= 0.001, high = 0.999)
    epNE = Real(name = 'epNE', low= 0.001, high = 0.999)
    epSW = Real(name = 'epSW', low= 0.001, high = 0.999)
    epSE = Real(name = 'epSE', low= 0.001, high = 0.999)

    dimensions = [cpNW, cpNE , cpSW, cpSE , epNW, epNE, epSW, epSE]

    res = gp_minimize(objective, dimensions = dimensions, n_calls = 10)

    print("Best Result: ", res.fun)

    print("Best Paramters: ", res.x)