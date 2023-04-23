# ------------------------------------------------------
# A program for calculating statistics for mazes of all
# topologies of predefined heuristics
#
# (C) 2023 Iavna Lev, Moscow, Russia
# email iavna.le@phystech.edu
# -------------------------------------------------------
import random
from typing import Union

class PlaneMatrix:
    """A class for a maze on a two-dimensional plane"""
    def __init__(self, data):
        self.data = data
        self.len_x = len(data)
        self.len_y = len(data[0])

    def __getitem__(self, key):
        x, y = self.get_true_index(key)
        return self.data[x][y]

    def __setitem__(self, key, value) -> None:
        x, y = self.get_true_index(key)
        self.data[x][y] = value

    def __iter__(self):
        for x in range(self.len_x):
            for y in range(self.len_y):
                yield self[(x, y)]

    def get_true_index(self, key):
        """By the input index of the element, it returns the correct index of the element for this topology"""
        x, y = key
        if not (0 <= x < self.len_x and 0 <= y < self.len_y):
            raise IndexError("Index out of range")
        return x, y


class ToroidalMatrix:
    """A class for a maze on the surface of a two-dimensional torus"""
    def __init__(self, data):
        self.data = data
        self.len_x = len(data)
        self.len_y = len(data[0])

    def __getitem__(self, key):
        x, y = self.get_true_index(key)
        return self.data[x][y]

    def __setitem__(self, key, value) -> None:
        x, y = self.get_true_index(key)
        self.data[x][y] = value

    def __iter__(self):
        for x in range(self.len_x):
            for y in range(self.len_y):
                yield self[(x, y)]

    def get_true_index(self, key):
        """By the input index of the element, it returns the correct index of the element for this topology"""
        x, y = key
        return x % self.len_x, y % self.len_y


class CylindricalMatrix:
    """A class for a maze on the surface of a two-dimensional cylinder"""
    def __init__(self, data):
        self.data = data
        self.len_x = len(data)
        self.len_y = len(data[0])

    def __getitem__(self, key):
        x, y = self.get_true_index(key)
        return self.data[x][y]

    def __setitem__(self, key, value) -> None:
        x, y = self.get_true_index(key)
        self.data[x][y] = value

    def __iter__(self):
        for x in range(self.len_x):
            for y in range(self.len_y):
                yield self[(x, y)]

    def get_true_index(self, key):
        """By the input index of the element, it returns the correct index of the element for this topology"""
        x, y = key
        if not 0 <= y < self.len_y:
            raise IndexError("Index out of range")
        return x % self.len_x, y


class SphericalMatrix:
    """A class for a maze on the surface of a two-dimensional sphere"""
    def __init__(self, data):
        self.data = data
        self.len_x = len(data)  # len_x should be equal to len_y
        self.len_y = len(data[0])

    def __getitem__(self, key):
        x, y = self.get_true_index(key)
        return self.data[x][y]

    def __setitem__(self, key, value) -> None:
        x, y = self.get_true_index(key)
        self.data[x][y] = value

    def __iter__(self):
        for x in range(self.len_x):
            for y in range(self.len_y):
                yield self[(x, y)]

    def get_true_index(self, key):
        """By the input index of the element, it returns the correct index of the element for this topology"""
        x, y = key
        if x == -1:
            return y, 0
        elif x == self.len_x:
            return y, self.len_y - 1
        elif y == -1:
            return 0, x
        elif y == self.len_y:
            return self.len_x - 1, x
        else:
            return x, y


class MobiusBandMatrix:
    """A class for a maze on the surface of a two-dimensional Möbius band"""
    def __init__(self, data):
        self.data = data
        self.len_x = len(data)
        self.len_y = len(data[0])

    def __getitem__(self, key):
        x, y = self.get_true_index(key)
        return self.data[x][y]

    def __setitem__(self, key, value) -> None:
        x, y = self.get_true_index(key)
        self.data[x][y] = value

    def __iter__(self):
        for x in range(self.len_x):
            for y in range(self.len_y):
                yield self[(x, y)]

    def get_true_index(self, key):
        """By the input index of the element, it returns the correct index of the element for this topology"""
        x, y = key
        if not 0 <= y < self.len_y:
            raise IndexError("Index out of range")
        if x == -1:
            return self.len_x - 1, self.len_y - 1 - y
        elif x == self.len_x:
            return 0, self.len_y - 1 - y
        else:
            return x, y


class KleinBottleMatrix:
    """A class for a maze on the surface of a two-dimensional Klein bottle"""
    def __init__(self, data):
        self.data = data
        self.len_x = len(data)
        self.len_y = len(data[0])

    def __getitem__(self, key):
        x, y = self.get_true_index(key)
        return self.data[x][y]

    def __setitem__(self, key, value) -> None:
        x, y = self.get_true_index(key)
        self.data[x][y] = value

    def __iter__(self):
        for x in range(self.len_x):
            for y in range(self.len_y):
                yield self[(x, y)]

    def get_true_index(self, key):
        """By the input index of the element, it returns the correct index of the element for this topology"""
        x, y = key
        if x == -1:
            return self.len_x - 1, self.len_y - 1 - y
        elif x == self.len_x:
            return 0, self.len_y - 1 - y
        else:
            return x % self.len_x, y % self.len_y


class ProjectivePlaneMatrix:
    """A class for a maze on the surface of a two-dimensional projective plane"""
    def __init__(self, data):
        self.data = data
        self.len_x = len(data)
        self.len_y = len(data[0])

    def __getitem__(self, key):
        x, y = self.get_true_index(key)
        return self.data[x][y]

    def __setitem__(self, key, value) -> None:
        x, y = self.get_true_index(key)
        self.data[x][y] = value

    def __iter__(self):
        for x in range(self.len_x):
            for y in range(self.len_y):
                yield self[(x, y)]

    def get_true_index(self, key):
        """By the input index of the element, it returns the correct index of the element for this topology"""
        x, y = key
        if x == -1:
            return self.len_x - 1, self.len_y - 1 - y
        elif x == self.len_x:
            return 0, self.len_y - 1 - y
        elif y == -1:
            return self.len_x - 1 - x, self.len_y - 1
        elif y == self.len_y:
            return self.len_x - 1 - x, 0
        else:
            return x, y

# Creating a type for data organization classes for various topologies
MazeMatrix = Union[
    PlaneMatrix,
    ToroidalMatrix,
    CylindricalMatrix,
    SphericalMatrix,
    MobiusBandMatrix,
    KleinBottleMatrix,
    ProjectivePlaneMatrix
]

# Initialization the number of nodes in the maze along the corresponding axes
size_x = 20
size_y = 20


def start_point_generate():
    """The function of generating the coordinates of the starting node in the maze"""
    if random.choice([True, False]):
        if random.choice([True, False]):
            s = (0, random.randint(0, size_y - 1))
        else:
            s = (size_x - 1, random.randint(0, size_y - 1))
    else:
        if random.choice([True, False]):
            s = (random.randint(0, size_x - 1), 0)
        else:
            s = (random.randint(0, size_x - 1), size_y - 1)
    return s


def finish_point_generate(start):
    """The function of generating the coordinates of the finishing node in the maze"""
    # return size_x - 1 - start[0], size_y - 1 - start[1]
    f = random.choice(range(0, size_x)), random.choice(range(0, size_y))
    while f == start:
        f = random.choice(range(0, size_x)), random.choice(range(0, size_y))
    return f


def count_components(v_walls, h_walls, matrix: MazeMatrix) -> int:
    """Converts the newly generated maze walls into an adjacency list of cells and returns the number of
    connectivity components of the graph corresponding to this adjacency list."""
    if not matrix:
        return 0

    # Initializing the adjacency list
    adjacency_list = {(i, j): [] for j in range(size_y) for i in range(size_x)}

    # Traversing all nodes and adding the corresponding available neighbors for each of them
    for i in range(size_x):
        for j in range(size_y):
            up, down, left, right = (i, j - 1), (i, j + 1), (i - 1, j), (i + 1, j)
            if not h_walls[i][j]:
                adjacency_list[(i, j)].append(matrix.get_true_index(up))
            if not h_walls[i][j + 1]:
                adjacency_list[(i, j)].append(matrix.get_true_index(down))
            if not v_walls[i][j]:
                adjacency_list[(i, j)].append(matrix.get_true_index(left))
            if not v_walls[i + 1][j]:
                adjacency_list[(i, j)].append(matrix.get_true_index(right))

    # Depth-first search function for connecting graph vertices into connectivity components
    def dfs(f_node) -> None:
        visited.add(f_node)
        for neighbor in adjacency_list[f_node]:
            if neighbor not in visited:
                dfs(neighbor)

    # Initializing a set of visited nodes
    visited = set()

    # Counting connectivity components
    num_components = 0
    for node in adjacency_list:
        if node not in visited:
            num_components += 1
            dfs(node)

    return num_components


def generate_maze(topology: str):
    """The function of generating a maze taking into account the selected topology of the maze.
    The function guarantees the connectivity of the maze graph."""

    def backtrack(x, y):
        # Marking the node visited
        visited[(x, y)] = True

        # Available directions of movement from the node
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        # Shuffling the list of directions
        random.shuffle(directions)

        """We cyclically move to neighboring nodes and, provided that they are not visited 
        and the movement is correct, we break the corresponding walls. 
        Each topology has its own conditions for the correctness of movement and walls to be destroyed. 
        After destroying the walls, we recursively call the backtrack function for a new node."""
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if topology == 'ℝ²':
                if 0 <= nx < size_x and 0 <= ny < size_y and not visited[(nx, ny)]:
                    if dx == 0:
                        h_walls[x][min(y, ny) + 1] = False
                    else:
                        v_walls[min(x, nx) + 1][y] = False
                    backtrack(*visited.get_true_index((nx, ny)))
            elif topology == 'T²':
                if not visited[(nx, ny)]:
                    if dx == 0:
                        if ny == size_y or ny == -1:
                            h_walls[x][0] = h_walls[x][-1] = False
                        else:
                            h_walls[x][min(y, ny) + 1] = False
                    else:
                        if nx == size_x or nx == -1:
                            v_walls[0][y] = v_walls[-1][y] = False
                        else:
                            v_walls[min(x, nx) + 1][y] = False
                    backtrack(*visited.get_true_index((nx, ny)))
            elif topology == 'S²':
                if not visited[(nx, ny)]:
                    if dx == 0:
                        if ny == -1:
                            h_walls[x][0] = v_walls[0][x] = False
                        elif ny == size_y:
                            h_walls[x][-1] = v_walls[-1][x] = False
                        else:
                            h_walls[x][min(y, ny) + 1] = False
                    else:
                        if nx == -1:
                            h_walls[0][y] = v_walls[y][0] = False
                        elif nx == size_x:
                            h_walls[-1][y] = v_walls[y][-1] = False
                        else:
                            v_walls[min(x, nx) + 1][y] = False
                    backtrack(*visited.get_true_index((nx, ny)))
            elif topology == 'ℝ¹×S¹':
                if 0 <= ny < size_y and not visited[(nx, ny)]:
                    if dx == 0:
                        h_walls[x][min(y, ny) + 1] = False
                    else:
                        if nx == size_x or nx == -1:
                            v_walls[0][y] = v_walls[-1][y] = False
                        else:
                            v_walls[min(x, nx) + 1][y] = False
                    backtrack(*visited.get_true_index((nx, ny)))
            elif topology == 'M²':
                if 0 <= ny < size_y and not visited[(nx, ny)]:
                    if dx == 0:
                        h_walls[x][min(y, ny) + 1] = False
                    else:
                        if nx == -1:
                            v_walls[0][y] = v_walls[-1][size_y - 1 - y] = False
                        elif nx == size_x:
                            v_walls[-1][y] = v_walls[0][size_y - 1 - y] = False
                        else:
                            v_walls[min(x, nx) + 1][y] = False
                    backtrack(*visited.get_true_index((nx, ny)))
            elif topology == 'K²':
                if not visited[(nx, ny)]:
                    if dx == 0:
                        if ny == size_y or ny == -1:
                            h_walls[x][0] = h_walls[x][-1] = False
                        else:
                            h_walls[x][min(y, ny) + 1] = False
                    else:
                        if nx == -1:
                            v_walls[0][y] = v_walls[-1][size_y - 1 - y] = False
                        elif nx == size_x:
                            v_walls[-1][y] = v_walls[0][size_y - 1 - y] = False
                        else:
                            v_walls[min(x, nx) + 1][y] = False
                    backtrack(*visited.get_true_index((nx, ny)))
            elif topology == 'ℝP²':
                if not visited[(nx, ny)]:
                    if dx == 0:
                        if ny == -1:
                            h_walls[x][0] = h_walls[size_x - 1 - x][-1] = False
                        elif ny == size_y:
                            h_walls[x][-1] = h_walls[size_x - 1 - x][0] = False
                        else:
                            h_walls[x][min(y, ny) + 1] = False
                    else:
                        if nx == -1:
                            v_walls[0][y] = v_walls[-1][size_y - 1 - y] = False
                        elif nx == size_x:
                            v_walls[-1][y] = v_walls[0][size_y - 1 - y] = False
                        else:
                            v_walls[min(x, nx) + 1][y] = False
                    backtrack(*visited.get_true_index((nx, ny)))

    # Initializing the visited object with the value None
    visited = None
    v_walls = None
    h_walls = None

    """As long as the number of connectivity components in the generated maze is not equal to 1, we will 
    re-generate the maze. In fact, the problem of the incoherence of the maze graph arises only in the case of 
    a spherical topology due to its cardinal difference from other topologies due to the way the boundaries are 
    glued together."""
    while count_components(v_walls, h_walls, visited) != 1:

        # Creating a two-dimensional boolean matrix for node visit flags when generating a maze
        matrix = [[False for _ in range(size_y)] for _ in range(size_x)]

        # Creating a data storage object taking into account the corresponding topology
        if topology == 'ℝ²':
            visited = PlaneMatrix(matrix)
        elif topology == 'T²':
            visited = ToroidalMatrix(matrix)
        elif topology == 'S²':
            visited = SphericalMatrix(matrix)
        elif topology == 'ℝ¹×S¹':
            visited = CylindricalMatrix(matrix)
        elif topology == 'M²':
            visited = MobiusBandMatrix(matrix)
        elif topology == 'K²':
            visited = KleinBottleMatrix(matrix)
        elif topology == 'ℝP²':
            visited = ProjectivePlaneMatrix(matrix)

        # Initializing lists of vertical and horizontal walls (all walls are installed by default)
        v_walls = [[True for _ in range(size_y)] for _ in range(size_x + 1)]
        h_walls = [[True for _ in range(size_y + 1)] for _ in range(size_x)]

        # Starting a recursive function from coordinate (0, 0)
        backtrack(0, 0)

        # Checking for graph connectivity is required only for the topology of the sphere
        if topology != 'S²':
            break

    # Generation of the start point
    start = start_point_generate()

    # Generation of the finish point
    finish = finish_point_generate(start)

    return v_walls, h_walls, start, finish


class Chromosome:
    """A chromosome class for implementing a genetic algorithm"""
    def __init__(self, alpha: float, beta: float, gamma: float, estimation: float = 0.0):
        # Initialization of chromosome attributes
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # Evaluation of the optimality of the heuristic chromosome
        self.estimation = estimation


def general_heuristic_function(x: int, y: int, topology: str, chromosome: Chromosome,
                               finish) -> float:
    """A heuristic function common to all topologies. Accepts the coordinates of the current node, topology,
    and heuristic chromosome. Returns a heuristic estimate of the distance from the current node to the final one."""

    # Unpacking heuristic parameters and coordinates of the final point
    alpha, beta, gamma = chromosome.alpha, chromosome.beta, chromosome.gamma
    fx, fy = finish[0], finish[1]

    # Calculation of heuristics for each corresponding topology
    if topology == 'ℝ²':
        return gamma * ((abs(fx - x) ** alpha + abs(fy - y) ** alpha) ** beta)
    elif topology == 'T²':
        return gamma * (((min(abs(fx - x), size_x - abs(fx - x))) ** alpha +
                         (min(abs(fy - y), size_y - abs(fy - y))) ** alpha) ** beta)
    elif topology == 'S²':
        return gamma * min(
            (abs(fx - x) ** alpha + abs(fy - y) ** alpha) ** beta,
            (abs(fy - x) ** alpha + (fx + y) ** alpha) ** beta,
            (abs(fx - y) ** alpha + (fy + x) ** alpha) ** beta,
            ((2 * size_x - 1 - fx - y) ** alpha + abs(fy - x) ** alpha) ** beta,
            ((2 * size_y - 1 - fy - x) ** alpha + abs(fx - y) ** alpha) ** beta
        )
    elif topology == 'ℝ¹×S¹':
        return gamma * (((min(abs(fx - x), size_x - abs(fx - x))) ** alpha + abs(fy - y) ** alpha) ** beta)
    elif topology == 'M²':
        return gamma * min(
            (abs(fx - x) ** alpha + abs(fy - y) ** alpha) ** beta,
            ((size_x - abs(fx - x)) ** alpha + abs(size_y - 1 - fy - y) ** alpha) ** beta
        )
    elif topology == 'K²':
        return gamma * min(
            (abs(fx - x) ** alpha + abs(fy - y) ** alpha) ** beta,
            ((size_x - abs(fx - x)) ** alpha + abs(size_y - 1 - fy - y) ** alpha) ** beta,
            (abs(fx - x) ** alpha + (min(abs(fy - y), size_y - abs(fy - y))) ** alpha) ** beta

        )
    elif topology == 'ℝP²':
        return gamma * min(
            (abs(fx - x) ** alpha + abs(fy - y) ** alpha) ** beta,
            ((size_x - abs(fx - x)) ** alpha + abs(size_y - 1 - fy - y) ** alpha) ** beta,
            (abs(size_x - 1 - fx - x) ** alpha + (size_y - abs(fy - y)) ** alpha) ** beta
        )


class Node:
    """Class for the node object in the maze"""
    def __init__(self, x: int, y: int, topology: str, chromosome: Chromosome, finish):
        # Node coordinates
        self.x = x
        self.y = y

        # The minimum number of steps to reach a given node from the starting point (infinite by default)
        self.g = float('inf')

        # Node attendance flag
        self.visited = False

        # Initializing a heuristic function
        self.h = general_heuristic_function(self.x, self.y, topology, chromosome, finish)

        # Link to the previous node
        self.prev = None

        # Node belonging to the final path
        self.path = False

        # Active node at the pathfinding iteration
        self.current = False

    def f(self) -> float:
        """Path priority function"""
        return self.g + self.h


class Solution:
    """A class implementing a pathfinding algorithm"""
    def __init__(self, topology, chromosome, v_walls, h_walls, start, finish):
        # Initialization of topology and heuristics
        self.topology = topology
        self.chromosome = chromosome

        # Initializing the maze
        self.h_walls = h_walls
        self.v_walls = v_walls
        self.start = start
        self.finish = finish

        # Initialization of a variable storing the number of nodes visited by the algorithm (inf by default)
        self.checked = float('inf')

        # The total length of the path from start to finish
        self.path_len = 0

        # Initializing nodes
        nodes = [[Node(i, j, topology=geometry, chromosome=self.chromosome, finish=self.finish)
                  for j in range(size_y)] for i in range(size_x)]

        # Initialization of the data structure depending on the topology
        if self.topology == 'ℝ²':
            self.nodes = PlaneMatrix(nodes)
        elif self.topology == 'T²':
            self.nodes = ToroidalMatrix(nodes)
        elif self.topology == 'S²':
            self.nodes = SphericalMatrix(nodes)
        elif self.topology == 'ℝ¹×S¹':
            self.nodes = CylindricalMatrix(nodes)
        elif self.topology == 'M²':
            self.nodes = MobiusBandMatrix(nodes)
        elif self.topology == 'K²':
            self.nodes = KleinBottleMatrix(nodes)
        elif self.topology == 'ℝP²':
            self.nodes = ProjectivePlaneMatrix(nodes)

        # The number of steps from the starting point to the starting point is zero
        self.nodes[self.start].g = 0

    def next_node(self) -> Node:
        """Returns the node with the minimum value of the f-function"""
        return min(self.nodes, key=lambda n: (n.f() if not n.visited else float('inf')))

    def open(self):
        """Launching the search algorithm"""
        while True:
            # ==================== A* Algorithm ====================
            cur = self.next_node()
            cur.visited = True
            # Calculate Top
            if not ((cur.y == 0 and self.topology in ['ℝ²', 'ℝ¹×S¹', 'M²']) or self.h_walls[cur.x][cur.y]):
                if self.nodes[(cur.x, cur.y - 1)].g > cur.g + 1:
                    self.nodes[(cur.x, cur.y - 1)].g = cur.g + 1
                    self.nodes[(cur.x, cur.y - 1)].prev = cur

            # Calculate Bottom
            if not ((cur.y == size_y - 1 and self.topology in ['ℝ²', 'ℝ¹×S¹', 'M²'])
                    or self.h_walls[cur.x][cur.y + 1]):
                if self.nodes[(cur.x, cur.y + 1)].g > cur.g + 1:
                    self.nodes[(cur.x, cur.y + 1)].g = cur.g + 1
                    self.nodes[(cur.x, cur.y + 1)].prev = cur

            # Calculate Left
            if not ((cur.x == 0 and self.topology == 'ℝ²') or self.v_walls[cur.x][cur.y]):
                if self.nodes[(cur.x - 1, cur.y)].g > cur.g + 1:
                    self.nodes[(cur.x - 1, cur.y)].g = cur.g + 1
                    self.nodes[(cur.x - 1, cur.y)].prev = cur

            # Calculate Right
            if not ((cur.x == size_x - 1 and self.topology == 'ℝ²') or self.v_walls[cur.x + 1][cur.y]):
                if self.nodes[(cur.x + 1, cur.y)].g > cur.g + 1:
                    self.nodes[(cur.x + 1, cur.y)].g = cur.g + 1
                    self.nodes[(cur.x + 1, cur.y)].prev = cur

            cur.current = False
            cur = self.next_node()
            cur.current = True
            if (cur.x, cur.y) == self.finish:
                while cur.g != 0:
                    cur.path = True
                    cur = cur.prev
                    self.path_len += 1
                self.checked = sum([1 if self.nodes[(i, j)].visited else 0 for i in range(size_x)
                                    for j in range(size_y)])
                break
            # ======================================================


def avg_round(lst: list, accuracy: int = 2) -> float:
    """Returns the average value in the list with the specified accuracy"""
    return round(sum(lst) / len(lst), accuracy) if len(lst) else 0.0


if __name__ == '__main__':
    MAZE = 1000  # Sample size of the mazes

    # The investigated heuristic chromosome
    # empirical_candidate = Chromosome(alpha=1.7, beta=1.4, gamma=1.6)
    empirical_candidate = Chromosome(alpha=4, beta=4, gamma=4)

    # Initialization of a convenient dictionary of topologies and selection of the topology under study
    topology_list = {'R': 'ℝ²', 'S': 'S²', 'T': 'T²', 'C': 'ℝ¹×S¹', 'M': 'M²', 'K': 'K²', 'P': 'ℝP²'}

    for key, geometry in topology_list.items():
        # List for collecting statistics
        statistics = []

        # Cyclic execution of the algorithm for all mazes
        for maze in range(MAZE):
            maze_basis = generate_maze(geometry)  # Maze Generation

            # Initialization of the class for the given topology, the generated maze and the chromosome
            S = Solution(geometry, empirical_candidate, *maze_basis)
            S.open()

            # Updating the accuracy estimate of the heuristic
            empirical_candidate.estimation = round(S.path_len / S.checked, 2)
            statistics.append(empirical_candidate.estimation)

        # Output of results to the console
        print(f"Topology: {key}; [{empirical_candidate.alpha}, {empirical_candidate.beta}, "
              f"{empirical_candidate.gamma}] Estimation: {avg_round(statistics)}", flush=True)
