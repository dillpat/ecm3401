class Enemy:

    def __init__(self, parentMaze, x=None, y=None, health=10):
        self._parentMaze=parentMaze
        self.x=x
        self.y=y
        self._parentMaze._enemy.append(self)
        self.position=(self.x,self.y)
        self._health=health
        self.defeated=False
        return

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
    def health(self):
        return (self._health)

    @health.setter
    def health(self, newHealth):
        self.health = newHealth