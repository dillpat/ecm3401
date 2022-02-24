# from os import kill
from pyagame import maze, agent
import numpy as np
from Coin import Coin

m=maze(10,10)
m.CreateMaze(loopPercent=60)
a=agent(m, footprints=True)
m.tracePath({a:m.path})


# #m.enableWASD(a)
# #m.enableArrowKey(a)
#print("Path length: ", len(m.path))


# def addCoins(m=None, number=8):
#     coin_list = []
#     for i in range(number):
#         position = np.random.randint(low=1, high=11, size=2)
#         print(position)
#         coin = Coin(m, x=position[0], y=position[1])
#         coin_list.append(coin)
#     #print(coin_list)
#     return coin_list

#addCoins()
#coin_coords1 = np.random.rand(8,2)*2
#print(coin_coords1)
m.run()