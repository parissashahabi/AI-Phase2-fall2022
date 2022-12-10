import pandas as pd
import numpy as np
from mdp_solver import MDPSolver
from base import BaseAgent, Action
from time import time
from utils import calculate_diagonal_distance, Coloring

GEM_SEQUENCE_SCORE = [
    [50, 0, 0, 0],
    [50, 200, 100, 0],
    [100, 50, 200, 100],
    [50, 100, 50, 200],
    [250, 50, 100, 50]
]
INITIAL_REWARD = {0: -1.0, 1: 50.0, 2: 50.0, 3: 50.0, 4: 50.0, 5: np.NAN, 6: np.NAN,
                  7: np.NAN, 8: np.NAN, 9: 25.0, 10: 25.0, 11: 25.0, 12: -20.0, 13: 0.0}


class Agent(BaseAgent):
    def __init__(self):
        super(Agent, self).__init__()
        self.num_states = self.grid_width * self.grid_height
        self.policy = None
        self.finished = False
        self.last_gem = 0
        self.agent = (0, 0)
        self.gems_locations = []
        self.keys_locations = []
        self.keys = set()
        self.initial_grid = None
        self.normal_cells_probabilities = pd.DataFrame(self.probabilities['normal']).transpose().to_numpy()
        self.slider_cells_probabilities = pd.DataFrame(self.probabilities['slider']).transpose().to_numpy()
        self.barbed_cells_probabilities = pd.DataFrame(self.probabilities['barbed']).transpose().to_numpy()
        self.teleport_cells_probabilities = pd.DataFrame(self.probabilities['teleport']).transpose().to_numpy()

    def get_reward(self):
        # TODO -> E, keys (9, 10, 11), teleport (13)
        reward = {0: -1.0, 5: np.NAN, 12: -20.0, 13: 0.0}
        for gem_type in [1, 2, 3, 4]:
            reward[gem_type] = float(GEM_SEQUENCE_SCORE[self.last_gem][gem_type - 1])
        for door_type in [6, 7, 8]:
            if door_type in self.keys:
                reward[door_type] = -1.0  # TODO -> E
                continue
            reward[door_type] = np.NAN
        for key_type in [9, 10, 11]:
            reward[key_type] = 25.0
        print(f'reward: {reward}')
        return reward

    def find_longest_distance_between_gems(self):
        coloring = Coloring(self.grid, self.grid_height, self.grid_width)
        coloring.bfs(0, 0)
        self.find_sliders()
        longest_distance = 0
        for gem_s in self.gems_locations:
            for gem_d in self.gems_locations:
                if coloring.contains(gem_s) and coloring.contains(gem_d):
                    distance = calculate_diagonal_distance(gem_s, gem_d)
                    if distance > longest_distance:
                        longest_distance = distance
        return longest_distance

    def get_state_from_pos(self, pos):
        return pos[0] * self.grid_width + pos[1]

    def find_sliders(self):
        keys, gems = [], []
        for x in range(self.grid_height):
            for y in range(self.grid_width):
                if self.grid[x][y] in ['1', '2', '3', '4']:
                    gems.append((x, y))
                elif self.grid[x][y] in ['g', 'r', 'y']:
                    keys.append((x, y))
        self.keys_locations = keys
        self.gems_locations = gems

    def get_action(self, state):
        action = self.policy[state]
        if action == 0:
            return Action.UP
        elif action == 1:  # Down
            return Action.DOWN
        elif action == 2:  # Left
            return Action.LEFT
        elif action == 3:  # Right
            return Action.RIGHT
        elif action == 4:  # Up Right
            return Action.UP_RIGHT
        elif action == 5:  # Up Left
            return Action.UP_LEFT
        elif action == 6:  # Down Right
            return Action.DOWN_RIGHT
        elif action == 7:  # Down Left
            return Action.DOWN_LEFT
        elif action == 8:  # NOOP
            return Action.NOOP

    def get_policy(self, reward):
        mdp_solver = MDPSolver(self.grid, reward, self.turn_count, self.normal_cells_probabilities,
                               self.slider_cells_probabilities, self.barbed_cells_probabilities, self.teleport_cells_probabilities)
        mdp_solver.train()
        mdp_solver.visualize_value_policy()
        self.policy = mdp_solver.get_policy()

    def generate_actions(self):
        # self.get_policy(INITIAL_REWARD)
        self.get_policy(self.get_reward())

    def get_agent_location(self):
        for x in range(self.grid_height):
            for y in range(self.grid_width):
                if 'A' in self.grid[x][y]:
                    self.agent = (x, y)

    def update_last_gem(self):
        last_gem_index = self.gems_locations.index(self.agent)
        x, y = self.gems_locations[last_gem_index]
        self.last_gem = int(self.initial_grid[x][y])

    def update_reached_keys_list(self):
        key_index = self.keys_locations.index(self.agent)
        x, y = self.keys_locations[key_index]
        if self.initial_grid[x][y] == 'g':
            self.keys.add(6)
        elif self.initial_grid[x][y] == 'r':
            self.keys.add(7)
        elif self.initial_grid[x][y] == 'y':
            self.keys.add(8)

    def do_turn(self) -> Action:
        if self.turn_count == 1:
            self.initial_grid = self.grid

        start_time = int(round(time() * 1000))
        self.get_agent_location()
        if self.agent in self.gems_locations or self.agent in self.keys_locations or self.turn_count == 1:
            if self.turn_count != 1:
                if self.agent in self.gems_locations:
                    self.update_last_gem()
                else:
                    self.update_reached_keys_list()
            print(f'-----------------turn count {self.turn_count}-----------------')
            print(f'factor: {((self.grid_height * self.grid_width) / self.max_turn_count) * self.find_longest_distance_between_gems()}')
            print(f'agent reached a gem: {self.agent in self.gems_locations}')
            self.find_sliders()
            print(f'gems location: {self.gems_locations}')
            print(f'last gem {self.last_gem}')
            print(f'reached keys {list(self.keys)}')
            self.generate_actions()
            print(f'policy: {self.policy}')
            print(f'agent location: {self.get_state_from_pos(self.agent)}')
            print(f'action: {self.get_action(self.get_state_from_pos(self.agent))}')

        state = self.get_state_from_pos(self.agent)
        action = self.get_action(state)
        cur_time = int(round(time() * 1000)) - start_time
        print(f'time {cur_time}')
        return action


if __name__ == '__main__':
    data = Agent().play()
    print("FINISH : ", data)
