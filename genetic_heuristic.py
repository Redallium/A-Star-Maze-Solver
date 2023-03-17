import random
import json
# import sys

size_x = 20
size_y = 20


class CircularList:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, key):
        return self.data[key % len(self.data)]

    def __setitem__(self, key, value):
        self.data[key % len(self.data)] = value

    def __iter__(self):
        for key in range(len(self.data)):
            yield self[key]


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


def generate_maze(f_topology='ℝ²'):
    f_v_walls = [[True for _ in range(size_y)] for _ in range(size_x + 1)]
    f_h_walls = [[True for _ in range(size_y + 1)] for _ in range(size_x)]

    if f_topology == 'ℝ²':
        # Recursive backtracking algorithm to generate the maze
        def backtrack(x, y):
            visited[x][y] = True
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            random.shuffle(directions)
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < size_x and 0 <= ny < size_y and not visited[nx][ny]:
                    if dx == 0:
                        f_h_walls[x][min(y, ny) + 1] = False
                    else:
                        f_v_walls[min(x, nx) + 1][y] = False
                    backtrack(nx, ny)

    elif f_topology == 'T²':
        # Recursive backtracking algorithm to generate the maze
        def backtrack(x, y):
            visited[x % size_x][y % size_y] = True
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            random.shuffle(directions)
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if not visited[nx % size_x][ny % size_y]:  # 0 <= nx < size_x and 0 <= ny < size_y
                    if dx == 0:
                        if ny == size_y or ny == -1:
                            f_h_walls[x][0] = h_walls[x][-1] = False
                        else:
                            f_h_walls[x][min(y, ny) + 1] = False
                    else:
                        if nx == size_x or nx == -1:
                            f_v_walls[0][y] = v_walls[-1][y] = False
                        else:
                            f_v_walls[min(x, nx) + 1][y] = False
                    backtrack(nx % size_x, ny % size_y)

    # Initialize visited array for recursive backtracking algorithm
    visited = [[False for _ in range(size_y)] for _ in range(size_x)]
    backtrack(0, 0)
    f_start = start_point_generate()
    f_finish = finish_point_generate()
    return f_v_walls, f_h_walls, f_start, f_finish


class Chromosome:
    def __init__(self, alpha, beta, gamma, estimation=0):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.estimation = estimation

    def __add__(self, other):
        child_alpha = round((self.alpha + other.alpha) / 2, 1)
        child_beta = round((self.beta + other.beta) / 2, 1)
        child_gamma = round((self.gamma + other.gamma) / 2, 1)
        return Chromosome(alpha=child_alpha, beta=child_beta, gamma=child_gamma)

    def mutation(self):
        self.alpha = round(self.alpha * random.choice([1.1, 0.9, 1.5, 0.5]), 1)
        self.beta = round(self.beta * random.choice([1.1, 0.9, 1.5, 0.5]), 1)
        self.gamma = round(self.gamma * random.choice([1.1, 0.9, 1.5, 0.5]), 1)

    def __str__(self):
        return '[' + str(self.alpha) + ', ' + str(self.beta) + ', ' + str(self.gamma) + ', ' + str(
            self.estimation) + ']\n'


def general_heuristic_function(x, y, f_topology, f_chromosome, f_finish):
    alpha, beta, gamma = f_chromosome.alpha, f_chromosome.beta, f_chromosome.gamma
    if f_topology == 'ℝ²':
        return gamma * ((abs(f_finish[0] - x) ** alpha + abs(f_finish[1] - y) ** alpha) ** beta)
    elif f_topology == 'T²':
        return gamma * (((min(abs(f_finish[0] - x), size_x - abs(f_finish[0] - x))) ** alpha +
                         (min(abs(f_finish[1] - y), size_y - abs(f_finish[1] - y))) ** alpha) ** beta)


class Node:
    def __init__(self, x, y, f_topology, f_chromosome, f_finish):
        self.visited = False
        self.x = x
        self.y = y
        self.g = float('inf')

        self.h = general_heuristic_function(self.x, self.y, f_topology, f_chromosome, f_finish)

        self.prev = None
        self.path = False
        self.current = False

    def f(self):
        return self.g + self.h


class Solution:
    def __init__(self, f_topology, f_chromosome, f_v_walls, f_h_walls, f_start, f_finish):
        self.topology = f_topology
        self.chromosome = f_chromosome
        self.h_walls = f_h_walls
        self.v_walls = f_v_walls
        self.start = f_start
        self.finish = f_finish
        self.checked = float('inf')
        self.nodes = CircularList(
            [CircularList([Node(f_i, f_j, f_topology=self.topology, f_chromosome=self.chromosome, f_finish=self.finish)
                           for f_j in range(size_y)]) for f_i in range(size_x)])
        self.nodes[self.start[0]][self.start[1]].g = 0

    def next_node(self):
        temp = [min(idx, key=lambda n: (n.f() if not n.visited else float('inf'))) for idx in self.nodes]
        return min(temp, key=lambda n: (n.f() if not n.visited else float('inf')))

    def open(self):
        while True:
            # ==================== A* Algorithm ====================
            cur = self.next_node()
            cur.visited = True
            # Calculate Top
            if not ((cur.y == 0 and self.topology == 'ℝ²') or self.h_walls[cur.x][cur.y]):
                if self.nodes[cur.x][cur.y - 1].g > cur.g + 1:
                    self.nodes[cur.x][cur.y - 1].g = cur.g + 1
                    self.nodes[cur.x][cur.y - 1].prev = cur

            # Calculate Bottom
            if not ((cur.y == size_y - 1 and self.topology == 'ℝ²') or self.h_walls[cur.x][cur.y + 1]):
                if self.nodes[cur.x][cur.y + 1].g > cur.g + 1:
                    self.nodes[cur.x][cur.y + 1].g = cur.g + 1
                    self.nodes[cur.x][cur.y + 1].prev = cur

            # Calculate Left
            if not ((cur.x == 0 and self.topology == 'ℝ²') or self.v_walls[cur.x][cur.y]):
                if self.nodes[cur.x - 1][cur.y].g > cur.g + 1:
                    self.nodes[cur.x - 1][cur.y].g = cur.g + 1
                    self.nodes[cur.x - 1][cur.y].prev = cur

            # Calculate Right
            if not ((cur.x == size_x - 1 and self.topology == 'ℝ²') or self.v_walls[cur.x + 1][cur.y]):
                if self.nodes[cur.x + 1][cur.y].g > cur.g + 1:
                    self.nodes[cur.x + 1][cur.y].g = cur.g + 1
                    self.nodes[cur.x + 1][cur.y].prev = cur

            cur.current = False
            cur = self.next_node()
            cur.current = True
            if (cur.x, cur.y) == self.finish:
                while cur.g != 0:
                    cur.path = True
                    cur = cur.prev
                self.checked = sum([1 if self.nodes[f_i][f_j].visited else 0 for f_i in range(size_x)
                                    for f_j in range(size_y)])
                break
            # ======================================================


def form_first_generation(n=64):
    f_generation = list()
    f_generation.append(Chromosome(alpha=1, beta=1, gamma=1))  # manhattan
    f_generation.append(Chromosome(alpha=2, beta=0.5, gamma=1))  # euclidean
    f_generation.append(Chromosome(alpha=2, beta=1, gamma=1))  # squared euqlidean
    for _ in range(n - 3):
        alpha = round(random.randint(5, 30) / 10, 1)
        beta = round(random.randint(5, 30) / 10, 1)
        gamma = round(random.randint(1, 50) / 10, 1)
        f_generation.append(Chromosome(alpha, beta, gamma))
    return f_generation


def form_next_generation(f_generation):
    f_generation.sort(key=lambda c: c.estimation)
    n = len(f_generation)
    half_best = f_generation[:n // 2]
    next_gen = [] + half_best
    # random.shuffle(half_best)
    crossed = []
    for f_i in range(n // 4):
        new_chromosome = half_best[2 * f_i] + half_best[2 * f_i + 1]
        new_chromosome.mutation()
        crossed.append(new_chromosome)
    random.shuffle(half_best)
    for f_i in range(n // 4):
        new_chromosome = half_best[2 * f_i] + half_best[2 * f_i + 1]
        new_chromosome.mutation()
        crossed.append(new_chromosome)
    # crossed = [(generation[2*i] + generation[2*i+1]).mutation for i in range(N//2)]
    next_gen += crossed
    # print(len(next_gen))
    return next_gen


def save_maze(f_v_walls, f_h_walls, f_start, f_finish, filename):
    # filename = 'genetic_maze.json'
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump({'v_walls': f_v_walls, 'h_walls': f_h_walls, 'start': list(f_start), 'finish': list(f_finish)}, file)


if __name__ == '__main__':
    N = 64
    EPOCH = 20
    MAZE = 50
    topology = 'ℝ²'
    # topology = 'T²'
    avg_manhattan, avg_euclidean, avg_squared_euclidean, avg_custom = [], [], [], []
    for maze in range(1, MAZE + 1):
        print(f'maze #{maze}:', flush=True)
        v_walls, h_walls, start, finish = generate_maze()
        save_maze(v_walls, h_walls, start, finish, filename='gen_maze_' + str(maze) + '.json')
        generation = form_first_generation(N)
        for i, chromosome in enumerate(generation):
            S = Solution(topology, chromosome, v_walls, h_walls, start, finish)
            S.open()
            chromosome.estimation = S.checked
            if i == 0:
                print(f'Manhattan: {chromosome}', flush=True)
                avg_manhattan.append(chromosome.estimation)
            elif i == 1:
                print(f'Euclidean: {chromosome}', flush=True)
                avg_euclidean.append(chromosome.estimation)
            elif i == 2:
                print(f'Squared Euclidean: {chromosome}', flush=True)
                avg_squared_euclidean.append(chromosome.estimation)
        print(f'epoch #0: {min(generation, key=lambda c: c.estimation)}', flush=True)
        for epoch in range(1, EPOCH + 1):
            generation = form_next_generation(generation)
            for chromosome in generation[N // 2:]:
                S = Solution(topology, chromosome, v_walls, h_walls, start, finish)
                S.open()
                chromosome.estimation = S.checked
            # for c in generation:
            #     print(c)
            # print(f'epoch #{epoch}: {min(generation, key=lambda c: c.estimation)}')
        print(f'epoch #{EPOCH}: {min(generation, key=lambda c: c.estimation)}', flush=True)
        avg_custom.append(min(generation, key=lambda c: c.estimation))
    print(f'Average Manhattan Estimation: {round(sum(avg_manhattan) / len(avg_manhattan))}')
    print(f'Average Euclidean Estimation: {round(sum(avg_euclidean) / len(avg_euclidean))}')
    print(f'Average Squared Euclidean Estimation: {round(sum(avg_squared_euclidean) / len(avg_squared_euclidean))}')
    print(f'Average Custom Estimation: {round(sum([c.estimation for c in avg_custom]) / len(avg_custom))}')
    print(f'Average Custom Alpha: {round(sum([c.alpha for c in avg_custom]) / len(avg_custom), 1)}')
    print(f'Average Custom Beta: {round(sum([c.beta for c in avg_custom]) / len(avg_custom), 1)}')
    print(f'Average Custom Gamma: {round(sum([c.gamma for c in avg_custom]) / len(avg_custom), 1)}')
