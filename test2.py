from pyamaze import maze,agent,COLOR,textLabel
from warnings import catch_warnings
from warnings import simplefilter
from math import sin
from math import pi
from numpy import arange
from numpy import argmax
from numpy.random import normal
from matplotlib import pyplot
from sklearn.gaussian_process import GaussianProcessRegressor

def BFS(m):
    start=(m.rows,m.cols)
    frontier=[start]
    explored=[start]
    bfsPath={}
    while len(frontier)>0:
        currCell=frontier.pop(0)
        if currCell==(1,1):
            break
        for d in 'ESNW':
            if m.maze_map[currCell][d]==True:
                if d=='E':
                    childCell=(currCell[0],currCell[1]+1)
                elif d=='W':
                    childCell=(currCell[0],currCell[1]-1)
                elif d=='N':
                    childCell=(currCell[0]-1,currCell[1])
                elif d=='S':
                    childCell=(currCell[0]+1,currCell[1])
                if childCell in explored:
                    continue
                frontier.append(childCell)
                explored.append(childCell)
                bfsPath[childCell]=currCell
    fwdPath={}
    cell=(1,1)
    while cell!=start:
        fwdPath[bfsPath[cell]]=cell
        cell=bfsPath[cell]
    return fwdPath

def objective(differentPercent=0, noise = 0.1):
    noise = normal(loc=0, scale=noise)

    m=maze(10,10)
    m.CreateMaze(loopPercent=differentPercent)
    path=BFS(m)

    #a=agent(m)
    #m.tracePath({a:path})
    #l=textLabel(m,'Length of Shortest Path',len(path)+1)
    print('Length of Shortest Path', len(path)+1)
    return(len(path)+1) + noise




if __name__=='__main__':
    X = arange(0, 100, 5)
    print("X = ", X)
    # sample the domain without noise
    y = [objective(x, 0) for x in X]
    # sample the domain with noise
    ynoise = [objective(x) for x in X]
    # find best result
    ix = argmax(y)
    print('Optima: x=%.3f, y=%.3f' % (X[ix], y[ix]))
    # plot the points with noise
    pyplot.scatter(X, ynoise)
    # plot the points without noise
    pyplot.plot(X, y)
    # show the plot
    pyplot.show()



    #model = GaussianProcessRegressor()

    objective()

    #m.run()