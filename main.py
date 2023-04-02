# -------------------------------------------------------------------
# GUI for modeling algorithms for finding paths in mazes on surfaces
# of different topologies using various heuristic functions
#
# (C) 2023 Iavna Lev, Moscow, Russia
# email iavna.le@phystech.edu
# -------------------------------------------------------------------
from tkinter import *
from tkinter.ttk import OptionMenu
from tkinter.filedialog import asksaveasfilename, askopenfile
import random
import json
import typing
from MazeDataStructures import PlaneMatrix, ToroidalMatrix, CylindricalMatrix, SphericalMatrix, MobiusBandMatrix, \
    KleinBottleMatrix, ProjectivePlaneMatrix

# Creating a type for data organization classes for various topologies
MazeMatrix = typing.Union[
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

# Initialization of node sizes and walls between them in pixels
box_px = 30
wall_px = 4

# GUI window parameters
WIDTH = (box_px + wall_px) * size_x + 3 * wall_px + 176
HEIGHT = (box_px + wall_px) * size_y + 3 * wall_px + 2

# Initialization of default values for the starting and finishing points of the maze
start = (0, 0)
finish = (size_x - 1, size_y - 1)

# Alias for the infinity value
inf = float('inf')


def general_heuristic_function(x: int, y: int, topology: str, h_chromosome: tuple[float, float, float]) -> float:
    """A heuristic function common to all topologies. Accepts the coordinates of the current node, topology,
    and heuristic chromosome. Returns a heuristic estimate of the distance from the current node to the final one."""

    # Unpacking heuristic parameters and coordinates of the final point
    alpha, beta, gamma = h_chromosome
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


def start_point_generate() -> tuple[int, int]:
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


def finish_point_generate() -> tuple[int, int]:
    """The function of generating the coordinates of the finishing node in the maze"""
    # return size_x - 1 - start[0], size_y - 1 - start[1]
    f = random.choice(range(0, size_x)), random.choice(range(0, size_y))
    while f == start:
        f = random.choice(range(0, size_x)), random.choice(range(0, size_y))
    return f


class Node:
    """Class for the node object in the maze"""
    def __init__(self, x, y, topology='ℝ²', algorithm='A*', h_func='manhattan'):
        # Node coordinates
        self.x = x
        self.y = y

        # The minimum number of steps to reach a given node from the starting point (infinite by default)
        self.g = inf

        # Node attendance flag
        self.visited = False

        # Current topology and search algorithm
        self.topology = topology
        self.algorithm = algorithm

        # Initializing a heuristic function
        self.h = 0
        self.set_h_func(h_func)

        # Link to the previous node
        self.prev = None

        # Node belonging to the final path
        self.path = False

        # Active node at the pathfinding iteration
        self.current = False

    def f(self) -> float:
        """Path priority function"""
        return self.g + self.h

    def set_h_func(self, h_func: str) -> None:
        """Function to call `general_heuristic_function` with a heuristic chromosome
        corresponding to the selected heuristic"""
        if self.algorithm == 'A*':
            if h_func == 'manhattan':
                self.h = general_heuristic_function(self.x, self.y, self.topology, (1, 1, 1))
            if h_func == 'euclidean':
                self.h = general_heuristic_function(self.x, self.y, self.topology, (2, 0.5, 1))
            if h_func == 'squared_euclidean':
                self.h = general_heuristic_function(self.x, self.y, self.topology, (2, 1, 1))
            if h_func == 'cubic_euclidean':
                self.h = general_heuristic_function(self.x, self.y, self.topology, (2, 1.5, 1))
            if h_func == 'custom':
                custom_chromosome = (1.7, 1.4, 1.6)
                self.h = general_heuristic_function(self.x, self.y, self.topology, custom_chromosome)
        else:
            self.h = 0


class App(Tk):
    """The main class that provides GUI operation"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Window settings
        self.minsize(width=WIDTH, height=HEIGHT)
        self.maxsize(width=WIDTH, height=HEIGHT)
        self.resizable(width=False, height=False)
        self.protocol("WM_DELETE_WINDOW", self.quit)
        self.title("Labyrinth Solver")

        # Sidebar
        sidebar = Frame(self, bg='lightgray')

        # Pack properties
        pack_dict = {'side': TOP, 'padx': 10, 'pady': 10}

        # <Run Algorithm> button
        Button(sidebar, text='Run Algorithm', command=self.run_algorithm, height=2, font=25).pack(**pack_dict)

        # Topology option menu
        self.topology = StringVar()
        self.topology.set('ℝ²')
        OptionMenu(sidebar, self.topology, 'ℝ²', *['ℝ²', 'S²', 'T²', 'ℝ¹×S¹', 'M²', 'K²', 'ℝP²']).pack(**pack_dict)

        # Algorithm option menu
        self.algorithm = StringVar()
        self.algorithm.set('A*')
        OptionMenu(sidebar, self.algorithm, 'A*', *['A*', 'Dijkstra'], command=self.update_options).pack(**pack_dict)

        # Heuristic option menu
        self.h_func = StringVar()
        self.h_func.set('manhattan')
        self.heuristic_option_menu = OptionMenu(sidebar, self.h_func, 'manhattan',
                                                *['manhattan', 'euclidean', 'squared_euclidean', 'cubic_euclidean',
                                                  'custom'])
        self.heuristic_option_menu.pack(**pack_dict)

        # <Generate Maze> button
        Button(sidebar, text='Generate Maze', command=self.generate_maze, height=2, font=25).pack(**pack_dict)

        # <Generate Walls> button
        Button(sidebar, text='Generate Walls', command=self.generate_walls, height=2, font=25).pack(**pack_dict)

        # <Load Maze> button
        Button(sidebar, text='Load Maze', command=self.load_maze, height=2, font=25).pack(**pack_dict)

        # <Save Maze> button
        Button(sidebar, text='Save Maze', command=self.save_maze, height=2, font=25).pack(**pack_dict)

        # <Reset Maze> button
        Button(sidebar, text='Reset Maze', command=self.reset_maze, height=2, font=25).pack(**pack_dict)

        # <Display Weights> boolean variable
        self.display_weights = BooleanVar()
        self.display_weights.set(False)
        Checkbutton(sidebar, text='Display Weights', variable=self.display_weights).pack(**pack_dict)

        sidebar.pack(side=LEFT, fill=Y)

        # Labyrinth initialization
        self.v_walls = None
        self.h_walls = None
        self.nodes = None
        self.reset_maze()

        # Canvas initialization
        self.canvas = Canvas(self, bg='white')
        self.canvas.pack(side=LEFT, expand=True, fill=BOTH)
        self.update_canvas()

        # Support for continuous GUI operation
        self.RUNNING = True

        # Algorithm startup variable
        self.ALGORITHM = False

    def update_options(self, choice) -> None:
        """Locks the heuristic selection options menu for Dijkstra's algorithm and unlocks for A*"""
        if choice != 'A*':
            self.heuristic_option_menu.configure(state="disabled")
        else:
            self.heuristic_option_menu.configure(state="enabled")

    def count_components(self, matrix: MazeMatrix) -> int:
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
                if not self.h_walls[i][j]:
                    adjacency_list[(i, j)].append(matrix.get_true_index(up))
                if not self.h_walls[i][j + 1]:
                    adjacency_list[(i, j)].append(matrix.get_true_index(down))
                if not self.v_walls[i][j]:
                    adjacency_list[(i, j)].append(matrix.get_true_index(left))
                if not self.v_walls[i + 1][j]:
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

    def generate_maze(self) -> None:
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
                match self.topology.get():
                    case 'ℝ²':
                        if 0 <= nx < size_x and 0 <= ny < size_y and not visited[(nx, ny)]:
                            if dx == 0:
                                h_walls[x][min(y, ny) + 1] = False
                            else:
                                v_walls[min(x, nx) + 1][y] = False
                            backtrack(*visited.get_true_index((nx, ny)))
                    case 'T²':
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
                    case 'S²':
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
                    case 'ℝ¹×S¹':
                        if 0 <= ny < size_y and not visited[(nx, ny)]:
                            if dx == 0:
                                h_walls[x][min(y, ny) + 1] = False
                            else:
                                if nx == size_x or nx == -1:
                                    v_walls[0][y] = v_walls[-1][y] = False
                                else:
                                    v_walls[min(x, nx) + 1][y] = False
                            backtrack(*visited.get_true_index((nx, ny)))
                    case 'M²':
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
                    case 'K²':
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
                    case 'ℝP²':
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

        """As long as the number of connectivity components in the generated maze is not equal to 1, we will 
        re-generate the maze. In fact, the problem of the incoherence of the maze graph arises only in the case of 
        a spherical topology due to its cardinal difference from other topologies due to the way the boundaries are 
        glued together."""
        while self.count_components(visited) != 1:

            # Creating a two-dimensional boolean matrix for node visit flags when generating a maze
            matrix = [[False for _ in range(size_y)] for _ in range(size_x)]

            # Creating a data storage object taking into account the corresponding topology
            match self.topology.get():
                case 'ℝ²':
                    visited = PlaneMatrix(matrix)
                case 'T²':
                    visited = ToroidalMatrix(matrix)
                case 'S²':
                    visited = SphericalMatrix(matrix)
                case 'ℝ¹×S¹':
                    visited = CylindricalMatrix(matrix)
                case 'M²':
                    visited = MobiusBandMatrix(matrix)
                case 'K²':
                    visited = KleinBottleMatrix(matrix)
                case 'ℝP²':
                    visited = ProjectivePlaneMatrix(matrix)

            # Initializing lists of vertical and horizontal walls (all walls are installed by default)
            v_walls = [[True for _ in range(size_y)] for _ in range(size_x + 1)]
            h_walls = [[True for _ in range(size_y + 1)] for _ in range(size_x)]

            # Starting a recursive function from coordinate (0, 0)
            backtrack(0, 0)

            # Assigning newly generated wall lists to class attributes
            self.v_walls = v_walls
            self.h_walls = h_walls

        # Generation of the start point
        global start
        start = start_point_generate()

        # Generation of the finish point
        global finish
        finish = finish_point_generate()

        # Soft reset of the maze state and canvas update
        self.soft_reset()
        self.update_canvas()
        self.update()

    def generate_walls(self) -> None:
        """Function for generating clusters of obstacles"""
        # Initializing cluster parameters
        num_obstacle_clusters, cluster_size_min, cluster_size_max = 16, 2, 5

        # Creating borders
        v_walls = [[not i % size_x for _ in range(size_y)] for i in range(size_x + 1)]
        h_walls = [[not i % size_y for i in range(size_y + 1)] for _ in range(size_x)]

        # Create obstacle clusters
        for i in range(num_obstacle_clusters):
            # Choose random cluster size
            cluster_size_x = random.randint(cluster_size_min, cluster_size_max)
            cluster_size_y = random.randint(cluster_size_min, cluster_size_max)

            # Choose random cluster position
            cluster_x = random.randint(0, size_x - cluster_size_x - 1)
            cluster_y = random.randint(0, size_y - cluster_size_y - 1)

            # Set obstacles in cluster
            for x in range(cluster_x, cluster_x + cluster_size_x):
                for y in range(cluster_y, cluster_y + cluster_size_y):
                    if x != size_y - 1:
                        h_walls[x][y + 1] = True
                    if y != 0:
                        h_walls[x][y] = True
                    if y != size_x - 1:
                        v_walls[x + 1][y] = True
                    if x != 0:
                        v_walls[x][y] = True

        # Assigning newly generated wall lists to class attributes
        self.v_walls = v_walls
        self.h_walls = h_walls

        # Generating the starting and finishing points in such a way that they do not fall into clusters
        global start
        start = start_point_generate()
        while v_walls[start[0]][start[1]] and v_walls[start[0]+1][start[1]] and \
                h_walls[start[0]][start[1]] and h_walls[start[0]][start[1]+1]:
            start = start_point_generate()

        global finish
        finish = finish_point_generate()
        while v_walls[finish[0]][finish[1]] and v_walls[finish[0]+1][finish[1]] and \
                h_walls[finish[0]][finish[1]] and h_walls[finish[0]][finish[1]+1] or start == finish:
            finish = finish_point_generate()

        # Soft reset of the maze state and canvas update
        self.soft_reset()
        self.update_canvas()
        self.update()

    def load_maze(self) -> None:
        """Loading a maze from a JSON file"""
        # Loading data from a JSON file with an exception for an error when choosing a file to upload
        try:
            with askopenfile(mode='r', filetypes=(('JSON Files', '*.json'),)) as file:
                data = json.load(file)
        except AttributeError:
            return

        # Assigning loaded values to the corresponding variables
        self.v_walls = data['v_walls']
        self.h_walls = data['h_walls']
        global start
        start = tuple(data['start'])
        global finish
        finish = tuple(data['finish'])

        # If the loaded maze is inconsistent with the size variables predefined in the program,
        # a corresponding error is issued
        if len(self.v_walls) != size_x + 1 or len(self.v_walls[0]) != size_y:
            raise Exception("WrongInputVWalls")
        if len(self.h_walls) != size_x or len(self.h_walls[0]) != size_y + 1:
            raise Exception("WrongInputHWalls")
        if not (0 <= start[0] < size_x and 0 <= start[1] < size_y):
            raise Exception("WrongStartCoordinates")
        if not (0 <= finish[0] < size_x and 0 <= finish[1] < size_y):
            raise Exception("WrongFinishCoordinates")

        # Soft reset of the maze state and canvas update
        self.soft_reset()
        self.update_canvas()
        self.update()

    def save_maze(self) -> None:
        """Saving the current maze to a JSON file"""
        try:
            filename = asksaveasfilename(defaultextension='.json', title="Save Maze As...",
                                         filetypes=(('JSON Files', '*.json'),))
            with open(filename, 'w', encoding='utf-8') as file:
                json.dump(
                    {'v_walls': self.v_walls, 'h_walls': self.h_walls, 'start': list(start), 'finish': list(finish)},
                    file)
        except FileNotFoundError:
            pass

    def next_node(self) -> Node:
        """Returns the node with the minimum value of the f-function"""
        return min(self.nodes, key=lambda n: (n.f() if not n.visited else inf))

    def update_canvas(self) -> None:
        """Canvas status update"""
        # Clearing the current canvas state
        self.canvas.delete(ALL)

        # Print borders
        # max_x = size_x * (wall_px + box_px) + 3*wall_px
        # max_y = size_y * (wall_px + box_px) + 3*wall_px
        # self.canvas.create_line(wall_px/2, wall_px/2, wall_px/2, max_y, width=2*wall_px, fill='black') #le
        # self.canvas.create_line(wall_px/2, wall_px/2, max_x, wall_px/2, width=2*wall_px, fill='black') #up
        # self.canvas.create_line(max_x-wall_px/2, wall_px/2, max_x-wall_px/2, max_y, width=2*wall_px, fill='black') #ri
        # self.canvas.create_line(wall_px/2, max_y-wall_px/2, max_x, max_y-wall_px/2, width=2*wall_px, fill='black') #lo

        # Printing nodes in the color corresponding to their current state
        weights = self.display_weights.get()
        for i in range(size_x):
            for j in range(size_y):
                if i == start[0] and j == start[1]:
                    col = 'red'
                elif (i == finish[0] and j == finish[1]) or self.nodes[(i, j)].path:
                    col = 'lightgreen'
                elif self.nodes[(i, j)].g == inf:
                    col = 'white'
                elif self.nodes[(i, j)].visited:
                    col = 'yellow'
                else:
                    col = 'orange'
                if self.nodes[(i, j)].current:
                    col = 'red'

                x0 = wall_px + i * (wall_px + box_px) + wall_px / 2
                y0 = wall_px + j * (wall_px + box_px) + wall_px / 2
                x1 = (i + 1) * (wall_px + box_px) + wall_px + wall_px / 2
                y1 = (j + 1) * (wall_px + box_px) + wall_px + wall_px / 2

                self.canvas.create_rectangle(x0, y0, x1, y1, fill=col, outline='lightgray')
                if weights and self.nodes[(i, j)].f() != float('inf'):
                    self.canvas.create_text(x0 + 4, y0 + 5, text='{:.4}'.format(str(self.nodes[(i, j)].f())),
                                            anchor=NW, font='Arial 10 bold', fill='black')

        # Print vertical walls
        for i, row in enumerate(self.v_walls):
            for j, wall in enumerate(row):
                if wall:
                    x0 = (wall_px + box_px) * i + wall_px / 2 + wall_px / 2
                    y0 = (wall_px + box_px) * j + wall_px / 2 + wall_px / 2 + wall_px / 2
                    x1 = x0 + wall_px
                    y1 = y0 + box_px + wall_px
                    self.canvas.create_rectangle(x0, y0, x1, y1, fill='black')

        # Print horizontal walls
        for i, row in enumerate(self.h_walls):
            for j, wall in enumerate(row):
                if wall:
                    y0 = (wall_px + box_px) * j + wall_px / 2 + wall_px / 2
                    x0 = (wall_px + box_px) * i + wall_px / 2 + wall_px / 2 + wall_px / 2
                    x1 = x0 + box_px + wall_px
                    y1 = y0 + wall_px
                    self.canvas.create_rectangle(x0, y0, x1, y1, fill='black')

    def reset_maze(self) -> None:
        """Reset the maze to the default state"""
        # Setting default values for start and finish points
        global start
        start = (0, 0)
        global finish
        finish = (size_x - 1, size_y - 1)

        # Soft maze reset
        self.soft_reset()

        # Setting default values for vertical and horizontal walls (borders only)
        self.v_walls = [[not i % size_x for _ in range(size_y)] for i in range(size_x + 1)]
        self.h_walls = [[not i % size_y for i in range(size_y + 1)] for _ in range(size_x)]

        # Setting the default value for the algorithm startup variable
        self.ALGORITHM = False

    def soft_reset(self) -> None:
        """Soft reset of the maze state"""
        # Getting the appropriate values for the current heuristic function, search algorithm, and maze topology
        h_func = self.h_func.get()
        algorithm = self.algorithm.get()
        topology = self.topology.get()

        # Creating a list of nodes with the appropriate parameters
        nodes = [[Node(i, j, topology=topology, algorithm=algorithm, h_func=h_func)
                  for j in range(size_y)] for i in range(size_x)]

        # Assigning an appropriate data organization structure to the current list of nodes depending on the topology
        match topology:
            case 'ℝ²':
                self.nodes = PlaneMatrix(nodes)
            case 'T²':
                self.nodes = ToroidalMatrix(nodes)
            case 'S²':
                self.nodes = SphericalMatrix(nodes)
            case 'ℝ¹×S¹':
                self.nodes = CylindricalMatrix(nodes)
            case 'M²':
                self.nodes = MobiusBandMatrix(nodes)
            case 'K²':
                self.nodes = KleinBottleMatrix(nodes)
            case 'ℝP²':
                self.nodes = ProjectivePlaneMatrix(nodes)

        # The number of steps from the starting point to the starting point is zero
        self.nodes[start].g = 0

    def run_algorithm(self) -> None:
        """Launching the pathfinding algorithm"""
        # Soft maze reset
        self.soft_reset()

        # Setting the True value for the algorithm startup variable
        self.ALGORITHM = True

    def open(self) -> None:
        while self.RUNNING:

            # ==================== A* Algorithm ====================
            cur = self.next_node()
            if self.ALGORITHM:
                cur.visited = True
                # Calculate Top
                if not ((cur.y == 0 and self.topology.get() in ['ℝ²', 'ℝ¹×S¹', 'M²']) or self.h_walls[cur.x][cur.y]):
                    if self.nodes[(cur.x, cur.y - 1)].g > cur.g + 1:
                        self.nodes[(cur.x, cur.y - 1)].g = cur.g + 1
                        self.nodes[(cur.x, cur.y - 1)].prev = cur

                # Calculate Bottom
                if not ((cur.y == size_y - 1 and self.topology.get() in ['ℝ²', 'ℝ¹×S¹', 'M²'])
                        or self.h_walls[cur.x][cur.y + 1]):
                    if self.nodes[(cur.x, cur.y + 1)].g > cur.g + 1:
                        self.nodes[(cur.x, cur.y + 1)].g = cur.g + 1
                        self.nodes[(cur.x, cur.y + 1)].prev = cur

                # Calculate Left
                if not ((cur.x == 0 and self.topology.get() == 'ℝ²') or self.v_walls[cur.x][cur.y]):
                    if self.nodes[(cur.x - 1, cur.y)].g > cur.g + 1:
                        self.nodes[(cur.x - 1, cur.y)].g = cur.g + 1
                        self.nodes[(cur.x - 1, cur.y)].prev = cur

                # Calculate Right
                if not ((cur.x == size_x - 1 and self.topology.get() == 'ℝ²') or self.v_walls[cur.x + 1][cur.y]):
                    if self.nodes[(cur.x + 1, cur.y)].g > cur.g + 1:
                        self.nodes[(cur.x + 1, cur.y)].g = cur.g + 1
                        self.nodes[(cur.x + 1, cur.y)].prev = cur

                cur.current = False
                cur = self.next_node()
                cur.current = True
                if (cur.x, cur.y) == finish:
                    self.ALGORITHM = False
                    while cur.g != 0:
                        cur.path = True
                        cur = cur.prev

            # ======================================================

            self.update_canvas()
            self.update()

    def quit(self) -> None:
        self.RUNNING = False
        super().quit()
        self.destroy()


if __name__ == '__main__':
    app = App()
    app.open()
