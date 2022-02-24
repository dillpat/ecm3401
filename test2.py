from pyamaze import maze,agent,COLOR,textLabel
from math import sin
from math import pi
from numpy import arange
from numpy import vstack
from numpy import argmax
from numpy import asarray
from numpy.random import normal
from numpy.random import random
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from warnings import catch_warnings
from warnings import simplefilter
from matplotlib import pyplot

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


# surrogate or approximation for the objective function
def surrogate(model, X):
	# catch any warning generated when making a prediction
	with catch_warnings():
		# ignore generated warnings
		simplefilter("ignore")
		return model.predict(X, return_std=True)

def plot(X, y, model):
    # scatter plot of inputs and real objective function
    pyplot.scatter(X, y)
    # line plot of surrogate function across domain
    Xsamples = asarray(arange(0, 105, 5))
    Xsamples = Xsamples.reshape(-1,1)
    #Xsamples = Xsamples.reshape(len(Xsamples), 1)
    ysamples, _ = surrogate(model, Xsamples)
    pyplot.plot(Xsamples, ysamples)
    # show the plot
    pyplot.show()
    
"""
ValueError: Expected 2D array, got 1D array instead:
array=[ 0  5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95].
Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a 
single sample.
"""

def opt_acquisition(X, y, model):
    # random search, generate random samples
    Xsamples = random(100)
    Xsamples = Xsamples.reshape(-1,1)
    #Xsamples = Xsamples.reshape(len(Xsamples), 1)
    # calculate the acquisition function for each sample
    scores = acquisition(X, Xsamples, model)
    # locate the index of the largest scores
    ix = argmax(scores)
    return Xsamples[ix, 0]

def acquisition(X, Xsamples, model):
	# calculate the best surrogate score found so far
	yhat, _ = surrogate(model, X)
	best = max(yhat)
	# calculate mean and stdev via surrogate function
	mu, std = surrogate(model, Xsamples)
	mu = mu[:, 0]
	# calculate the probability of improvement
	probs = norm.cdf((mu - best) / (std+1E-9))
	return probs

if __name__=='__main__':
    X = arange(0, 105, 5)
    print("X = ", X)
    # sample the domain without noise
    y = asarray([objective(x, 0) for x in X])
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

    #surrogate
    model = GaussianProcessRegressor()
    model.fit(X, y)
    yhat = model.predict(X, return_std=True)


    objective()

    # perform the optimization process
for i in range(100):
	# select the next point to sample
	x = opt_acquisition(X, y, model)
	# sample the point
	actual = objective(x)
	# summarize the finding for our own reporting
	est, _ = surrogate(model, [[x]])
	print('>x=%.3f, f()=%3f, actual=%.3f' % (x, est, actual))
	# add the data to the dataset
	X = vstack((X, [[x]]))
	y = vstack((y, [[actual]]))
	# update the model
	model.fit(X, y)
    #m.run()