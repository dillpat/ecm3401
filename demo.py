from xml.etree.ElementTree import C14NWriterTarget
from pyagame import maze,agent,textLabel,COLOR
from collections import deque
import numpy as np
from Coin import Coin
from Enemy import Enemy
import random

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
    return bSearch,bfsPath,fwdPath,currentScore


def addCoins(m, number=8):
    coin_position_list = {}
    while len(coin_position_list) < number:
        cell = np.random.randint(low=1, high=11, size=2)
        x=cell[0]
        y=cell[1]
        coin = Coin(m, x, y)
        m.maze_map[x,y]['C'] = coin
        #print(coin.cell)
        coin_position_list[x,y]=cell
    #print(len(coin_position_list)," : ", coin_position_list)
    return coin_position_list

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
            bSearch,bfsPath,fwdPath,score = BFS(m, start_position, cell)
            if (len(fwdPath)+1) < distance:
                distance = (len(fwdPath)+1)
                nearest_coin_cell = cell
    print("Distance", distance)
    print("Nearest Coin Cell", nearest_coin_cell)
    return nearest_coin_cell

def collectNearestCoins(m, coin_position_list, coin_target = 80):
    start_position = (m.rows,m.cols)
    currentScore = 0
    for i in range(len(coin_position_list)):
        nearest_coin = findNearestCoin(m, start_position, coin_position_list)
        bSearch,bfsPath,fwdPath,score = BFS(m, start_position, nearest_coin)
        if ('C' in m.maze_map[nearest_coin]):
            coin = m.maze_map[nearest_coin]['C']
            if not coin.collected:
                #print("found coin", coin)
                coin.collected = True
                currentScore += coin.value
                #print("Coin Value: ", currentScore, coin.value)
        start_position = nearest_coin
        if currentScore >= coin_target:
            break
    return currentScore

def addEnemy(m, number=8):
    enemy_list = {}
    while len(enemy_list) < number:
        cell = np.random.randint(low=1, high=11, size=2)
        x=cell[0]
        y=cell[1]
        enemy = Enemy(m, x, y)
        m.maze_map[x,y]['A'] = enemy
        #print(enemy)
        enemy_list[x,y]=cell
    #print("enemy_list: ", len(enemy_list)," : ", enemy_list)
    #print("maze map: ", m.maze_map)
    return enemy_list

def findNearestEnemy(m, start_position, enemy_list):
    """
    """
    nearest_enemy_cell=(1,1) # Default position maze exit
    distance = 100000
    for cell in enemy_list:
        if not m.maze_map[cell]['A'].defeated: 
            bSearch,bfsPath,fwdPath,score = BFS(m, start_position, cell)
            if (len(fwdPath)+1) < distance:
                distance = (len(fwdPath)+1)
                nearest_enemy_cell = cell
    #print("Distance", distance)
    print("Nearest Enemy Cell", nearest_enemy_cell)
    return nearest_enemy_cell

def combatNearestEnemy(m, enemy_list, player_health = 80):
    current_health = player_health
    start_position = (m.rows,m.cols)
    enemy_killed = 0
    while enemy_killed < len(enemy_list):
        nearest_enemy = findNearestEnemy(m, start_position, enemy_list)
        bSearch,bfsPath,fwdPath,score = BFS(m, start_position, nearest_enemy)
        if ('A' in m.maze_map[nearest_enemy]): # Checks for an enemy in cell
            enemy = m.maze_map[nearest_enemy]['A']
            if not enemy.defeated:
                current_health = combat(enemy, current_health)
                if enemy.defeated:
                    enemy_killed += 1
                current_health = restoreHealth(m, current_health, enemy)
                # current health will tell you if enemy is defeated
                if current_health == 0:
                    print("End health :", current_health)
                    #print("Number of enemies killed: ", enemy_killed_track)
                    break
                    #combatNearestEnemy(m, enemy_list)
                    #if nearest_enemy == (1,1):
                        #break
                print( "Player_health", current_health)
                print("Number of enemies killed: ", enemy_killed)
        start_position = nearest_enemy
    return current_health
# ERROR It is fighting 8 times, so when it loses to an enemy restores health and travels back that counts as 2.
def combat(enemy, player_health):
    #player_health = 80
    while player_health > 0:
        if random.randint(0,10) > 8:
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
    if current_health == 0:
        bSearch,bfsPath,fwdPath,score = BFS(m, current_position, home)
        current_health = 80
        #print("Path: ", fwdPath)
    return current_health

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
    half_width = width_max / 2
    half_height = height_max /2

    # Creating the NW quadrant
    cNW.append((1,1))
    cNW.append((half_width, half_height))
    cNW.append((1, half_height))
    cNW.append((half_width, 1))

    # Creating the NE quadrant
    cNE.append((width_max, 1))
    cNE.append((width_max, half_height))
    cNE.append((half_width, half_height))
    cNE.append((half_width,1))

    # Creating the SW quadrant
    cSW.append((1, half_height))
    cSW.append((half_width, half_height))
    cSW.append((1, height_max))
    cSW.append((half_width, height_max))

    # Creating the SE quadrant
    cSE.append((width_max, half_height))
    cSE.append((width_max, height_max))
    cSE.append((half_width, height_max))
    cSE.append((half_width, half_height))

    print("NW", cNW)
    print("NE", cNE)
    print("sW", cSW)
    print("SE", cSE)

    return cNW, cNE, cSW, cSE




if __name__=='__main__':

    m=maze(10,10)
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
    #findNearestEnemy(m, start_position, enemy_list)
    #combatNearestEnemy(m, enemy_list)

    divideQuadrants(m)
    #m.run()


    """
    How to implement the coins and enemeies as entropy values
    How to define the maze into quadrants
    Read up to page 36 in python book
    Do some leading change
    """


    """
    Divide the maze into 4 quadrants
    Give one quadrant a random probability bias from 0.5-1 to have a bias
    Divide the remaining probability into the other 3 quadrants
    4 parameters for the random probabiltiy of coins spawning
    4 parameters for the random probabiltiy of enemies spawning
    During bayeisian you run this configuration 10 times so the algorithm can learn from it before you run the optimization
    """

    """
    Genrate 4 values where the sum is 1
    Each quadrant has the weight(which is = to the probability)

    """