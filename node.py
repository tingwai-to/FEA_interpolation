from __future__ import print_function

class Node(object):
    def __init__(self, coord, value):
        self.dim = coord.size
        self.coord = coord
        self.value = value


class Node3D(Node):
    def __init__(self, coord, value):
        super(Node3D, self).__init__(coord, value)


class Node2D(Node):
    def __init__(self, coord, value):
        super(Node2D, self).__init__(coord, value)
