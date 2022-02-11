from os import kill
from pyamaze import maze, agent


m=maze(5,5)
m.CreateMaze(loopPercent=60)
a=agent(m, footprints=True)
m.tracePath({a:m.path}, kill=False)
#m.enableWASD(a)
#m.enableArrowKey(a)

m.run()