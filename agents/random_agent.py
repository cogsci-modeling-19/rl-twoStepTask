import numpy as np

class RandomAgent:
    """
    A random agent that selects actions uniformly at random.
    """
    def __init__(self, action_space, state_space):
        # the state space can be infered but here it is given for simplicity
        self.action_space = action_space
        self.state_space = state_space

    def policy(self, state, method=None):
        return np.random.choice(self.action_space)

    def update_beliefs(self, state, action, reward, next_state, terminal):
        return

    def get_action_probabilities(self, state):
        return np.ones(len(self.action_space)) / len(self.action_space)

    def get_action_probabilities(self, state):
        return np.random.uniform(0, 1, len(self.action_space))
    
    def reset(self):
        return
    
    def __str__(self):
        discribtion = "Random Agent\n"
        discribtion += "Action Space: " + str(self.action_space) + "\n"
        discribtion += "State Space: " + str(self.state_space) + "\n"
        return discribtion