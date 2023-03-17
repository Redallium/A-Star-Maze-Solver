from tkinter import *
from tkinter.ttk import OptionMenu
from tkinter.filedialog import asksaveasfilename, askopenfile
import random
import json
from DataStructures import PlaneMatrix, ToroidalMatrix, CylindricalMatrix, SphericalMatrix, MobiusBandMatrix, \
    KleinBottleMatrix, ProjectivePlaneMatrix


size_x = 20
size_y = 20
box_px = 30
wall_px = 4

WIDTH = (box_px + wall_px) * size_x + 3 * wall_px + 176
HEIGHT = (box_px + wall_px) * size_y + 3 * wall_px + 2

start = (0, 0)
finish = (size_x - 1, size_y - 1)

inf = float('inf')


def general_heuristic_function(x, y, topology='ℝ²', heuristic_chromosome=(1, 1, 1)):
    alpha, beta, gamma = heuristic_chromosome
    if topology == 'ℝ²':
        return gamma * ((abs(finish[0] - x) ** alpha + abs(finish[1] - y) ** alpha) ** beta)
    elif topology == 'T²':
        return gamma * (((min(abs(finish[0] - x), size_x - abs(finish[0] - x))) ** alpha +
                         (min(abs(finish[1] - y), size_y - abs(finish[1] - y))) ** alpha) ** beta)
    elif topology == 'S²':
        return gamma * min(
            (abs(finish[0] - x) ** alpha + abs(finish[1] - y) ** alpha) ** beta,
            (abs(finish[1] - x) ** alpha + (finish[0] + y) ** alpha) ** beta,
            (abs(finish[0] - y) ** alpha + (finish[1] + x) ** alpha) ** beta,
            ((2 * size_x - 1 - finish[0] - y) ** alpha + abs(finish[1] - x) ** alpha) ** beta,
            ((2 * size_y - 1 - finish[1] - x) ** alpha + abs(finish[0] - y) ** alpha) ** beta
        )
    elif topology == 'ℝ¹×S¹':
        return gamma * (((min(abs(finish[0] - x), size_x - abs(finish[0] - x))) ** alpha +
                         abs(finish[1] - y) ** alpha) ** beta)
    elif topology == 'M²':
        return gamma * min(
            (abs(finish[0] - x) ** alpha + abs(finish[1] - y) ** alpha) ** beta,
            ((size_x - abs(finish[0] - x)) ** alpha + abs(size_y - 1 - finish[1] - y) ** alpha) ** beta
        )
    elif topology == 'K²':
        return gamma * min(
            (abs(finish[0] - x) ** alpha + abs(finish[1] - y) ** alpha) ** beta,
            ((size_x - abs(finish[0] - x)) ** alpha + abs(size_y - 1 - finish[1] - y) ** alpha) ** beta,
            (abs(finish[0] - x) ** alpha + (min(abs(finish[1] - y), size_y - abs(finish[1] - y))) ** alpha) ** beta

        )
    elif topology == 'ℝP²':
        return gamma * min(
            (abs(finish[0] - x) ** alpha + abs(finish[1] - y) ** alpha) ** beta,
            ((size_x - abs(finish[0] - x)) ** alpha + abs(size_y - 1 - finish[1] - y) ** alpha) ** beta,
            (abs(size_x - 1 - finish[0] - x) ** alpha + (size_y - abs(finish[1] - y)) ** alpha) ** beta
            # (abs(size_x - 1 - finish[0] - x) ** alpha + abs(size_y - 1 - finish[1] - y) ** alpha) ** beta
        )


def start_point_generate():
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


def finish_point_generate():
    # return size_x - 1 - start[0], size_y - 1 - start[1]
    return random.choice(range(0, size_x)), random.choice(range(0, size_y))


class Node:
    def __init__(self, x, y, topology='ℝ²', algorithm='A*', h_func='manhattan'):
        self.visited = False
        self.x = x
        self.y = y
        self.g = inf

        self.topology = topology
        self.algorithm = algorithm
        self.h = 0
        self.set_h_func(h_func)

        self.prev = None
        self.path = False
        self.current = False

    def f(self):
        return self.g + self.h

    def set_h_func(self, h_func):
        if self.algorithm == 'A*':
            if h_func == 'manhattan':
                self.h = general_heuristic_function(self.x, self.y, self.topology, (1, 1, 1))
            if h_func == 'euclidean':
                self.h = general_heuristic_function(self.x, self.y, self.topology, (2, 0.5, 1))
            if h_func == 'squared_euclidean':
                self.h = general_heuristic_function(self.x, self.y, self.topology, (2, 1, 1))
            if h_func == 'custom':
                custom_chromosome = (2, 2, 2)
                self.h = general_heuristic_function(self.x, self.y, self.topology, custom_chromosome)
        else:
            self.h = 0

    def __str__(self):
        return '[' + str(self.x) + ',' + str(self.y) + '] ' \
               'g=' + str(self.g) + ' h=' + str(self.h) + ' f=' + str(self.f()) + '\n'


class App(Tk):
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

        Button(sidebar, text='Run Algorithm', command=self.run_algorithm, height=2, font=25).pack(side=TOP, padx=10,
                                                                                                  pady=10)

        self.topology = StringVar()
        self.topology.set('ℝ²')
        OptionMenu(sidebar, self.topology, 'ℝ²',
                   *['ℝ²', 'S²', 'T²', 'ℝ¹×S¹', 'M²', 'K²', 'ℝP²']).pack(side=TOP, padx=10, pady=10)

        self.algorithm = StringVar()
        self.algorithm.set('A*')
        OptionMenu(sidebar, self.algorithm, 'A*', *['A*', 'Dijkstra']).pack(side=TOP, padx=10, pady=10)

        self.h_func = StringVar()
        self.h_func.set('manhattan')
        OptionMenu(sidebar, self.h_func, 'manhattan', *['manhattan', 'euclidean', 'squared_euclidean', 'custom']).pack(
            side=TOP, padx=10, pady=10)

        Button(sidebar, text='Generate Maze', command=self.generate_maze, height=2, font=25).pack(side=TOP, padx=10,
                                                                                                  pady=10)

        Button(sidebar, text='Generate Walls', command=self.generate_walls, height=2, font=25).pack(side=TOP, padx=10,
                                                                                                    pady=10)

        Button(sidebar, text='Load Maze', command=self.load_maze, height=2, font=25).pack(side=TOP, padx=10, pady=10)

        Button(sidebar, text='Save Maze', command=self.save_maze, height=2, font=25).pack(side=TOP, padx=10, pady=10)

        Button(sidebar, text='Reset Maze', command=self.reset_maze, height=2, font=25).pack(side=TOP, padx=10, pady=10)

        self.display_weights = BooleanVar()
        self.display_weights.set(False)
        Checkbutton(sidebar, text='Display Weights', variable=self.display_weights).pack(side=TOP, padx=10, pady=10)

        # multiplier = Frame(sidebar)
        # Label(multiplier, text='h Function multiplier').pack(side=TOP, padx=10, pady=2)
        # radio = Frame(multiplier)
        # self.h_mul = DoubleVar()
        # self.h_mul.set(1)
        # Radiobutton(radio, text='1/4', variable=self.h_mul, value=0.25).pack(side=LEFT, padx=2, pady=5)
        # Radiobutton(radio, text='1/2', variable=self.h_mul, value=0.5).pack(side=LEFT, padx=2, pady=5)
        # Radiobutton(radio, text='1', variable=self.h_mul, value=1).pack(side=LEFT, padx=2, pady=5)
        # Radiobutton(radio, text='3', variable=self.h_mul, value=3).pack(side=LEFT, padx=2, pady=5)
        # Radiobutton(radio, text='15', variable=self.h_mul, value=15).pack(side=LEFT, padx=2, pady=5)
        # radio.pack(side=TOP, padx=2, pady=2)
        # multiplier.pack(side=TOP, padx=10, pady=10)

        sidebar.pack(side=LEFT, fill=Y)

        # Labyrinth
        self.v_walls = None
        self.h_walls = None
        self.nodes = None
        self.reset_maze()

        # Canvas
        self.canvas = Canvas(self, bg='white')
        # self.canvas.bind("<Button-1>", self.canvas_click)
        self.canvas.pack(side=LEFT, expand=True, fill=BOTH)
        self.update_canvas()

        self.RUNNING = True
        self.ALGORITHM = False

    def generate_maze(self):
        v_walls = [[True for _ in range(size_y)] for _ in range(size_x + 1)]
        h_walls = [[True for _ in range(size_y + 1)] for _ in range(size_x)]

        def backtrack(x, y):
            visited[(x, y)] = True
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            random.shuffle(directions)
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

        # Initialize visited array for recursive backtracking algorithm
        matrix = [[False for _ in range(size_y)] for _ in range(size_x)]
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
        backtrack(0, 0)

        self.v_walls = v_walls
        self.h_walls = h_walls

        global start
        start = start_point_generate()

        global finish
        finish = finish_point_generate()

        self.soft_reset()
        self.update_canvas()
        self.update()

    def generate_walls(self):
        # num_obstacles = size_x * size_y // 3

        # v_walls = [[True for j in range(size_y)] for i in range(size_x - 1)]
        # h_walls = [[True for j in range(size_y - 1)] for i in range(size_x)]

        # obstacles = [[False for j in range(size_y)] for i in range(size_x)]
        # for i in range(num_obstacles):
        #     x, y = random.randint(0, size_x-1), random.randint(0, size_y-1)
        #     obstacles[x][y] = True

        # for i in range(size_x - 1):
        #     for j in range(size_y):
        #         if obstacles[i][j] or obstacles[i+1][j]:
        #             v_walls[i][j] = True
        #         else:
        #             v_walls[i][j] = False

        # for i in range(size_x):
        #     for j in range(size_y - 1):
        #         if obstacles[i][j] or obstacles[i][j+1]:
        #             h_walls[i][j] = True
        #         else:
        #             h_walls[i][j] = False
        num_obstacle_clusters, cluster_size_min, cluster_size_max = 16, 2, 5
        # Create empty walls
        v_walls = [[i == 0 or i == size_x for _ in range(size_y)] for i in range(size_x + 1)]
        h_walls = [[i == 0 or i == size_y for i in range(size_y + 1)] for _ in range(size_x)]
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
        self.v_walls = v_walls
        self.h_walls = h_walls

        global start
        start = start_point_generate()

        global finish
        finish = finish_point_generate()

        self.soft_reset()
        self.update_canvas()
        self.update()

    def load_maze(self):
        try:
            with askopenfile(mode='r', filetypes=(('JSON Files', '*.json'),)) as file:
                data = json.load(file)
        except AttributeError:
            return
        self.v_walls = data['v_walls']
        self.h_walls = data['h_walls']
        global start
        start = tuple(data['start'])
        global finish
        finish = tuple(data['finish'])
        if len(self.v_walls) != size_x + 1 or len(self.v_walls[0]) != size_y:
            raise Exception("WrongInputVWalls")
        if len(self.h_walls) != size_x or len(self.h_walls[0]) != size_y + 1:
            raise Exception("WrongInputHWalls")
        if not (0 <= start[0] < size_x and 0 <= start[1] < size_y):
            raise Exception("WrongStartCoordinates")
        if not (0 <= finish[0] < size_x and 0 <= finish[1] < size_y):
            raise Exception("WrongFinishCoordinates")
        self.soft_reset()
        self.update_canvas()

    def save_maze(self):
        try:
            filename = asksaveasfilename(defaultextension='.json', title="Save Maze As...",
                                         filetypes=(('JSON Files', '*.json'),))
            with open(filename, 'w', encoding='utf-8') as file:
                json.dump(
                    {'v_walls': self.v_walls, 'h_walls': self.h_walls, 'start': list(start), 'finish': list(finish)},
                    file)
        except FileNotFoundError:
            pass

    def next_node(self):
        return min(self.nodes, key=lambda n: (n.f() if not n.visited else inf))

    def update_canvas(self):
        self.canvas.delete(ALL)

        # Print borders
        # max_x = size_x * (wall_px + box_px) + 3*wall_px
        # max_y = size_y * (wall_px + box_px) + 3*wall_px
        # self.canvas.create_line(wall_px/2, wall_px/2, wall_px/2, max_y, width=2*wall_px, fill='black') #le
        # self.canvas.create_line(wall_px/2, wall_px/2, max_x, wall_px/2, width=2*wall_px, fill='black') #up
        # self.canvas.create_line(max_x-wall_px/2, wall_px/2, max_x-wall_px/2, max_y, width=2*wall_px, fill='black') #ri
        # self.canvas.create_line(wall_px/2, max_y-wall_px/2, max_x, max_y-wall_px/2, width=2*wall_px, fill='black') #lo

        # Print squares
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

    def reset_maze(self):
        global start
        start = (0, 0)
        global finish
        finish = (size_x - 1, size_y - 1)
        self.soft_reset()
        self.v_walls = [[i == 0 or i == size_x for _ in range(size_y)] for i in range(size_x + 1)]
        self.h_walls = [[i == 0 or i == size_y for i in range(size_y + 1)] for _ in range(size_x)]
        self.ALGORITHM = False

    def soft_reset(self):
        h_func = self.h_func.get()
        algorithm = self.algorithm.get()
        topology = self.topology.get()
        # h_mul = self.h_mul.get()
        nodes = [[Node(i, j, topology=topology, algorithm=algorithm, h_func=h_func)
                  for j in range(size_y)] for i in range(size_x)]
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

        self.nodes[start].g = 0

    def run_algorithm(self):
        self.soft_reset()
        self.ALGORITHM = True

    def open(self):
        while self.RUNNING:

            # ==================== A* Algorithm ====================
            cur = self.next_node()
            if self.ALGORITHM:
                # print(cur)
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
                    # print(sum([1 if self.nodes[(i, j)].visited else 0 for i in range(size_x) for j in range(size_y)]))

            # ======================================================

            self.update_canvas()
            self.update()
        self.cleanup()

    def cleanup(self):
        pass

    def quit(self):
        self.RUNNING = False
        super().quit()
        self.destroy()


if __name__ == '__main__':
    app = App()
    app.open()
