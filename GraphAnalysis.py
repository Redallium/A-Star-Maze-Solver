import json
from tkinter.filedialog import askopenfile
from MazeDataStructures import PlaneMatrix, ToroidalMatrix, CylindricalMatrix, SphericalMatrix, MobiusBandMatrix, \
    KleinBottleMatrix, ProjectivePlaneMatrix


def load_maze():
    try:
        with askopenfile(mode='r', filetypes=(('JSON Files', '*.json'),)) as file:
            data = json.load(file)
    except AttributeError:
        return
    v_walls = data['v_walls']
    h_walls = data['h_walls']
    start = tuple(data['start'])
    finish = tuple(data['finish'])
    return v_walls, h_walls, start, finish


def count_components():
    v_walls, h_walls, start, finish = load_maze()
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

    adjacency_list = {(i, j): [] for j in range(size_y) for i in range(size_x)}

    for i in range(size_x):
        for j in range(size_y):
            up, down, left, right = (i, j-1), (i, j+1), (i-1, j), (i+1, j)
            if not h_walls[i][j]:
                adjacency_list[(i, j)].append(matrix.get_true_index(up))
            if not h_walls[i][j+1]:
                adjacency_list[(i, j)].append(matrix.get_true_index(down))
            if not v_walls[i][j]:
                adjacency_list[(i, j)].append(matrix.get_true_index(left))
            if not v_walls[i+1][j]:
                adjacency_list[(i, j)].append(matrix.get_true_index(right))

    # for key in adjacency_list:
    #     print(f'{key}: {adjacency_list[key]}')

    def dfs(node):
        visited.add(node)
        for neighbor in adjacency_list[node]:
            if neighbor not in visited:
                dfs(neighbor)

    visited = set()
    num_components = 0
    for node in adjacency_list:
        if node not in visited:
            num_components += 1
            dfs(node)
    return num_components


if __name__ == '__main__':
    guide_dict = {
        'ℝ²': 'Plane',
        'T²': 'Toroidal',
        'S²': 'Spherical',
        'ℝ¹×S¹': 'Cylindrical',
        'M²': 'MobiusBand',
        'K²': 'KleinBottle',
        'ℝP²': 'ProjectivePlane'
    }
    topology = 'S²'
    print(count_components())
