class Enemy:

    def __init__(self, parentMaze,  x=None, y=None, value=10, health=10):
        self._parentMaze=parentMaze
        self.x=x
        self.y=y
        self._value=value
        self._parentMaze._coins.append(self)
        self.position=(self.x,self.y)
        self._health=health
        return

    """
    How health would work, player go to defuser, then to enemy, runs function which
    calculates 50/50 odds to kill enemy, plyaer has 8 tries before it has to go back
    to defuser to get more health.
    """
    @property
    def x(self):
        return self._x

    @x.setter
    def x(self,newX):
        self._x=newX

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self,newY):
        self._y=newY

    # Gets the cell position of the coin in the maze
    def cellPosition(self):
        return self.position

    @property
    def collected(self):
        return (self._collected)

    @collected.setter
    def collected(self, newState):
        self._collected = newState
        
    @property
    def value(self):
        return (self._value)

    @value.setter
    def value(self, newValue):
        self.value = newValue