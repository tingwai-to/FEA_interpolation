from __future__ import print_function

class Node(object):
    def __init__(self, coord, value):
        self.dim = coord.size
        self.coord = coord
        self.value = value
