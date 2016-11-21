from __future__ import print_function

class Node(object):
    def __init__(self, coord, value):
        """Create a node.

        Args:
            coord (np.array): Coordinate of node
            value (float): Value at coordinate
        """
        #: int: dimensionality of node
        self.dim = coord.size
        #: np.array: coordinate of node
        self.coord = coord
        #: float: value of node
        self.value = value
