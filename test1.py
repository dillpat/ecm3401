from os import kill
from pyamaze import maze, agent


m=maze(10,10)
m.CreateMaze(loopPercent=60)
#a=agent(m)
#m.tracePath({a:m.path})
#m.enableWASD(a)
#m.enableArrowKey(a)
print("Path length: ", len(m.path))



#m.run()