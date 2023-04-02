# -------------------------------------------------------------------
# Code for counting the number of connectivity components from a
# given JSON file of vertical and horizontal walls in a maze
#
# (C) 2023 Iavna Lev, Moscow, Russia
# email iavna.le@phystech.edu
# -------------------------------------------------------------------
import json
from tkinter.filedialog import askopenfile
from MazeDataStructures import PlaneMatrix, ToroidalMatrix, CylindricalMatrix, SphericalMatrix, MobiusBandMatrix, \
    KleinBottleMatrix, ProjectivePlaneMatrix


def load_maze() -> tuple[list[list[bool]], list[list[bool]]]:
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
    def dfs(f_node: tuple[int, int]) -> None:
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
