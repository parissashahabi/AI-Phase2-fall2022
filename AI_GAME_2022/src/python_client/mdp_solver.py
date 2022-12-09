from mdp import MDP
from value_iteration import ValueIteration


class MDPSolver:
    def __init__(self, grid, reward, count, normal_cells_probabilities, slider_cells_probabilities,
                 barbed_cells_probabilities, teleport_cells_probabilities):
        self.problem = MDP(grid, reward, count, normal_cells_probabilities, slider_cells_probabilities,
                           barbed_cells_probabilities, teleport_cells_probabilities)
        self.solver = ValueIteration(self.problem.reward_function, self.problem.transition_model, gamma=0.9)

    def train(self):
        self.solver.train()

    def visualize_value_policy(self):
        self.problem.visualize_value_policy(policy=self.solver.policy, values=self.solver.values)

    def get_policy(self):
        return self.solver.policy


# problem = MDP('data/1.txt', reward={0: -1.0, 1: 50.0, 2: 50.0, 3: 50.0, 4: 50.0, 5: np.NAN, 6: np.NAN,
#                                     7: np.NAN, 8: np.NAN, 9: 25.0, 10: 25.0, 11: 25.0, 12: -20.0, 13: 10.0})

# solver = ValueIteration(problem.reward_function, problem.transition_model, gamma=0.9)
# solver.train()

# print(solver.policy)
# problem.visualize_value_policy(policy=solver.policy, values=solver.values)
# problem.random_start_policy(policy=solver.policy, start_pos=(2, 0), n=1000)
