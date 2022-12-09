import random
import numpy as np
from mdp_solver import MDPSolver
from base import BaseAgent, Action

GEM_SEQUENCE_SCORE = [
    [50, 0, 0, 0],
    [50, 200, 100, 0],
    [100, 50, 200, 100],
    [50, 100, 50, 200],
    [250, 50, 100, 50]
]
REWARDS = {0: {0: -1.0, 1: 50.0, 2: 50.0, 3: 50.0, 4: 50.0, 5: np.NAN, 6: np.NAN, 7: np.NAN, 8: np.NAN, 9: 25.0,
                       10: 25.0, 11: 25.0, 12: -20.0, 13: 10.0}}


class Agent(BaseAgent):
    def __init__(self):
        super(Agent, self).__init__()
        self.num_states = self.grid_width * self.grid_height
        self.policy = None
        self.finished = False
        self.last_gem = 0
        self.agent = (0, 0)
        self.gems_locations = []

    def get_state_from_pos(self, pos):
        return pos[0] * self.grid_width + pos[1]

    def find_gems(self):
        gems_list = []
        for x in range(self.grid_height):
            for y in range(self.grid_width):
                if self.grid[x][y] in ['1', '2', '3', '4']:
                    # gem_type = self.grid[x][y]
                    gems_list.append((x, y))
        self.gems_locations =  gems_list

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

    def get_policy(self, reward):
        mdp_solver = MDPSolver(self.grid, reward, self.turn_count)
        mdp_solver.train()
        mdp_solver.visualize_value_policy()
        self.policy = mdp_solver.get_policy()

    def generate_actions(self):
        self.get_policy(REWARDS[self.last_gem])

    def get_agent_location(self):
        for x in range(self.grid_height):
            for y in range(self.grid_width):
                if 'A' in self.grid[x][y]:
                    self.agent = (x, y)

    def do_turn(self) -> Action:
        self.get_agent_location()
        if self.agent in self.gems_locations or self.turn_count == 1:
            print(f'turn count: {self.turn_count}')
            self.find_gems()
            self.generate_actions()
        state = self.get_state_from_pos(self.agent)
        action = self.get_action(state)
        print(f'{self.turn_count}: {action}')
        return action


if __name__ == '__main__':
    data = Agent().play()
    print("FINISH : ", data)
