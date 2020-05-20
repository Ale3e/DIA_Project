from enum import Enum


class Status(Enum):
    seed = 0
    susceptible = 1
    active = 2
    inactive = 3


class Node:

    id = 0

    def __init__(self, cost):
        self.id = Node.id
        Node.id += 1
        self.cost = cost
        self.status = Status(1)
        self.adjacency_list = []
        self.adjacency_weights = []



    def setSeed(self):
        self.status = Status(0)

    def setSusceptible(self):
        self.status = Status(1)

    def setActive(self):
        self.status = Status(2)

    def setInactive(self):
        self.status = Status(3)



# PROVA CLASSE Node
# node_1 = Node(10)
# node_2 = Node(100)
#
#
#
# print(node_1.cost)
# print(node_1.status)
# node_1.activate()
# print(node_1.status)
# node_1.deactivate()
# print(node_1.status)
# print(node_1.status.value)

