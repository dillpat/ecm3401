class Quadrant:

    def __init__(self, parentMaze, boundaries, probability = 0.25):
        self._parentMaze = parentMaze
        self.boundaries = boundaries
        self.probability = probability
        self.base = boundaries[0]
        self.coin_positions = {}
        self.enemy_positions = {}
