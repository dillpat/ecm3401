class Quadrant:

    def __init__(self, parentMaze, boundaries, probability = 0.25):
        self._parentMaze = parentMaze
        self.boundaries = boundaries
        self.proability = probability
        self.base = boundaries[0]

    