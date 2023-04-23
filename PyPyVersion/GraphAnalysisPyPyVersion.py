# -------------------------------------------------------------------
# Code for counting the number of connectivity components from a
# given JSON file of vertical and horizontal walls in a maze
#
# (C) 2023 Iavna Lev, Moscow, Russia
# email iavna.le@phystech.edu
# -------------------------------------------------------------------
import json
from tkinter.filedialog import askopenfile


class PlaneMatrix:
    def __init__(self, data):
        self.data = data
        self.len_x = len(data)
        self.len_y = len(data[0])

    def get_true_index(self, key):
        x, y = key
        if not (0 <= x < self.len_x and 0 <= y < self.len_y):
            raise IndexError("Index out of range")
        return x, y


class ToroidalMatrix:
    def __init__(self, data):
        self.data = data
        self.len_x = len(data)
        self.len_y = len(data[0])

    def get_true_index(self, key):
        x, y = key
        return x % self.len_x, y % self.len_y


class CylindricalMatrix:
    def __init__(self, data):
        self.data = data
        self.len_x = len(data)
        self.len_y = len(data[0])

    def get_true_index(self, key):
        x, y = key
        if not 0 <= y < self.len_y:
            raise IndexError("Index out of range")
        return x % self.len_x, y


class SphericalMatrix:
    def __init__(self, data):
        self.data = data
        self.len_x = len(data)  # len_x should be equal to len_y
        self.len_y = len(data[0])

    def get_true_index(self, key):
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
    def __init__(self, data):
        self.data = data
        self.len_x = len(data)
        self.len_y = len(data[0])

    def get_true_index(self, key):
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
    def __init__(self, data):
        self.data = data
        self.len_x = len(data)
        self.len_y = len(data[0])

    def get_true_index(self, key):
        x, y = key
        if x == -1:
            return self.len_x - 1, self.len_y - 1 - y
        elif x == self.len_x:
            return 0, self.len_y - 1 - y
        else:
            return x % self.len_x, y % self.len_y


class ProjectivePlaneMatrix:
    def __init__(self, data):
        self.data = data
        self.len_x = len(data)
        self.len_y = len(data[0])

    def get_true_index(self, key):
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


def load_maze():
    """Loading a maze from a JSON file"""
    # Loading data from a JSON file with an exception for an error when choosing a file to upload
    try:
        with askopenfile(mode='r', filetypes=(('JSON Files', '*.json'),)) as file:
            data = json.load(file)
    except AttributeError:
        raise Exception("AttributeError: Failed to upload file")

    # Assigning loaded values to the corresponding variables
    v_walls = data['v_walls']
    h_walls = data['h_walls']

    return v_walls, h_walls


def count_components() -> int:
    """Based on the given lists of vertical and horizontal walls, builds an adjacency list for the nodes of the maze
    and returns the number of connectivity components in it"""
    v_walls, h_walls = load_maze()
    size_x = len(h_walls)
    size_y = len(v_walls[0])

    matrix = [[False for _ in range(size_y)] for _ in range(size_x)]
    if topology == 'ℝ²':
        matrix = PlaneMatrix(matrix)
    elif topology == 'T²':
        matrix = ToroidalMatrix(matrix)
    elif topology == 'S²':
        matrix = SphericalMatrix(matrix)
    elif topology == 'ℝ¹×S¹':
        matrix = CylindricalMatrix(matrix)
    elif topology == 'M²':
        matrix = MobiusBandMatrix(matrix)
    elif topology == 'K²':
        matrix = KleinBottleMatrix(matrix)
    elif topology == 'ℝP²':
        matrix = ProjectivePlaneMatrix(matrix)

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


if __name__ == '__main__':
    topology_list = {'R': 'ℝ²', 'S': 'S²', 'T': 'T²', 'C': 'ℝ¹×S¹', 'M': 'M²', 'K': 'K²', 'P': 'ℝP²'}
    topology = topology_list['S']
    k = count_components()
    print(f'Maze topology: {topology}')
    print(f'The number of connectivity components in the maze: {k}')
    if k != 1:
        print('The existence of a path in a maze is not guaranteed due to the incoherence of its graph')
    else:
        print('The existence of a path in a maze is guaranteed for any starting and finishing points')