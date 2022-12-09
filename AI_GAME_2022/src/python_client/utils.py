import numpy as np
import pandas as pd


def get_probabilities(sheet_name):
    df_dict = pd.read_excel('probabilities/1.xlsx', sheet_name=sheet_name, usecols='B:K', header=None, skiprows=1)
    return df_dict.to_numpy()


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
    return np.array(grid, float)



