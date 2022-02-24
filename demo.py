from pyagame import maze,agent,textLabel,COLOR
from collections import deque
import numpy as np
from Coin import Coin

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
        if ('C' in m.maze_map[currCell]):
            coin = m.maze_map[currCell]['C']
            if not coin.collected:
                #print("found coin", coin)
                coin.collected = True
                currentScore += coin.value
                #print("Coin Value: ", currentScore, coin.value)
        

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
    coin_position = {}
    while len(coin_position) != 8:
        cell = np.random.randint(low=1, high=11, size=2)
        x=cell[0]
        y=cell[1]
        coin = Coin(m, x, y)
        m.maze_map[x,y]['C'] = coin
        #print(coin.cell)
        coin_position[x,y]=cell
    #print(len(coin_position)," : ", coin_position)
    return coin_position

def setGoal(m, coin_position):
    start=(m.rows,m.cols)
    currentScore = 0
    #print(coin_position)
    for cell in coin_position:
        if not m.maze_map[cell]['C'].collected: 
            print("Start and end ", start, cell)
            bSearch,bfsPath,fwdPath,score = BFS(m, start, cell)
            currentScore += score
            print("running score", score)
            print("forward path", fwdPath)
            start = cell
    bSearch,bfsPath,fwdPath,score = BFS(m, start)
    currentScore += score
    print(m._goal)
    print(currentScore)


if __name__=='__main__':

    m=maze(10,10)
    m.CreateMaze(loopPercent=60)
    coin_position = addCoins(m)
    #print(m.maze_map)
    #print(m._coins)
    #bSearch,bfsPath,fwdPath,currentScore=BFS(m)
    setGoal(m, coin_position)

    #m.run()