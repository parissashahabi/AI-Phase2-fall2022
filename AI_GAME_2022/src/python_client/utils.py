import numpy as np
import pandas as pd
from collections import deque as queue


# def get_probabilities(sheet_name):
#     df_dict = pd.read_excel('probabilities/1.xlsx', sheet_name=sheet_name, usecols='B:K', header=None, skiprows=1)
#     return df_dict.to_numpy()


def grid_to_float_convertor(grid, num_rows, num_cols):
    for r in range(num_rows):
        for c in range(num_cols):
            if grid[r, c] == 'E' or grid[r, c] == 'EA':
                grid[r, c] = 0
            elif grid[r, c] == '1':
                grid[r, c] = 1
            elif grid[r, c] == '2':
                grid[r, c] = 2
            elif grid[r, c] == '3':
                grid[r, c] = 3
            elif grid[r, c] == '4':
                grid[r, c] = 4
            elif grid[r, c] == 'W':
                grid[r, c] = 5
            elif grid[r, c] == 'G':
                grid[r, c] = 6
            elif grid[r, c] == 'R':
                grid[r, c] = 7
            elif grid[r, c] == 'Y':
                grid[r, c] = 8
            elif grid[r, c] == 'g':
                grid[r, c] = 9
            elif grid[r, c] == 'r':
                grid[r, c] = 10
            elif grid[r, c] == 'y':
                grid[r, c] = 11
            elif grid[r, c] == '*':
                grid[r, c] = 12
            elif grid[r, c] == 'T':
                grid[r, c] = 13
    return np.array(grid, float)


def calculate_diagonal_distance(source, destination):
    dx = abs(source[0] - destination[0])
    dy = abs(source[1] - destination[1])
    return 2 * min(dx, dy) + (max(dx, dy) - min(dx, dy))


def create_grid(initial_grid, height, width):
    grid = []
    for i in range(height):
        grid.append([])
        for j in range(width):
            grid[-1].append(0)
            grid[i][j] = Node(i, j)
            grid[i][j].type = initial_grid[i][j]
    return grid


class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.type = ''


class Coloring:
    def __init__(self, initial_grid, height, width):
        self.dRow = [0, 1, 1, 1, 0, -1, -1, -1]
        self.dCol = [-1, -1, 0, 1, 1, 1, 0, -1]
        self.vis = [[False for i in range(width)] for i in range(height)]
        self.grid = create_grid(initial_grid, height, width)
        self.height = height
        self.width = width
        self.available_cells = []

    def is_valid(self, row, col):
        if row < 0 or col < 0 or row >= self.height or col >= self.width:
            return False
        if self.vis[row][col]:
            return False
        return True

    def bfs(self, row, col):
        q = queue()
        q.append((row, col))
        self.vis[row][col] = True
        while len(q) > 0:
            cell = q.popleft()
            x = cell[0]
            y = cell[1]
            self.available_cells.append(self.grid[x][y])
            for i in range(8):
                adj_x = x + self.dRow[i]
                adj_y = y + self.dCol[i]
                if self.is_valid(adj_x, adj_y) and not self.grid[adj_x][adj_y].type == "W":
                    q.append((adj_x, adj_y))
                    self.vis[adj_x][adj_y] = True

    def contains(self, tup):
        for cell in self.available_cells:
            if cell.x == tup[0] and cell.y == tup[1]:
                return True
        return False
