import numpy as np


# environment of the experiment
class TwoStepEnv:
    """
    An environment that simulates a two-step task. It consists of two stages, where the agent can take actions
    and receive rewards based on the state and action taken. The environment has a state space, action space, transition probabilities,
    reward probabilities, reward disrtibution(sampled from the reward probabilities) and an optionional reward scaler matrix for more involved task configurations.
    """

    action_space = [0, 1]
    state_space = [0, 1, 2]

    def __init__(self):
        self.state = 0
        self.transition_prob = 0.7
        self.terminal = False
        self.info = {}
        self.reward = 1
        # matrix of reward scalers for each state-action pair for more complex reward configurations 
        # self.reward_scaler_matrix = np.array([[0, 0],
        #                                       [1, 1],
        #                                       [1, 1]])

        # matrix of transition probabilities
        # 0(action left) -> [0(stay in 0), p(go to 1), 1-p(go to 2)]
        # 1(action right) -> [0(stay in 0), 1-p(go to 1), p(go to 2)]
        self.stage_1_transition_matrix = np.array(
            [[0, self.transition_prob, 1 - self.transition_prob],  # action left
             [0, 1 - self.transition_prob, self.transition_prob]])  # action right

        self.min_reward_prob = 0.25
        self.max_reward_prob = 0.75
        # matrix of reward probabilities
        # 0(state 0) -> [0 (left), 0(right)]
        # 1(state 1) -> [p1 (left), p2(right)]
        # 2(state 2) -> [p3 (left), p4(right)]
        p_1_0 = np.random.uniform(self.min_reward_prob, self.max_reward_prob)
        p_1_1 = np.random.uniform(self.min_reward_prob, self.max_reward_prob)
        p_2_0 = np.random.uniform(self.min_reward_prob, self.max_reward_prob)
        p_2_1 = np.random.uniform(self.min_reward_prob, self.max_reward_prob)

        self.reward_prob_matrix = np.array(
            [[0, 0],  # first stage (state 0) for both actions
             [p_1_0, p_1_1],  # second stage (state 1) for both actions
             [p_2_0, p_2_1]])  # second stage (state 2) for both actions

        # 1 -> fixed reward prob.
        # 0 -> reward prob. can be changed a long the trials
        self.fixed_reward_prob_matrix = np.array([[1, 1],
                                                  [0, 0],
                                                  [0, 0]])
        
        # disribution of rewards according to the reward probabilities
        self.reward_distribution = np.zeros_like(self.reward_prob_matrix)
        self.update_reward_distribution()
  
    def reset(self):
        """
        Resets the environment to its initial state.

        Returns:
            int: The initial state of the environment.
        """
        self.state = 0
        self.terminal = False
        self.info = {}
        return self.state

    def step(self, action):
        """
        Takes an action in the environment and returns the next state, reward, terminal flag, and additional information.

        Args:
            action (int): The action to be taken in the environment.

        Returns:
            tuple: A tuple containing the next state, reward, terminal flag, and additional information.
        """
        if self.terminal:
            raise ValueError("Episode has already terminated")
        if action not in self.action_space:
            raise ValueError(
                f"The action: {action} is not valid, action space: {self.action_space}")

        # if in stage 1
        if self.state == 0:
            reward = self.reward_function(self.state, action)  # reward will be 0
            self.state = np.random.choice(self.state_space,
                                          p=self.stage_1_transition_matrix[action])

            # update the info
            # self.info["reward_stage_1"] = reward > 0
            self.info["common_transition"] = self.is_common_state(self.state, action)
            self.info["state_transition_to"] = self.state
            self.info["stepOneChoice"] = action

        # if in stage 2
        elif self.state in [1, 2]:
            reward = self.reward_function(self.state, action)
            self.terminal = True
            # update the info
            self.info["reward"] = reward > 0
            self.info["stepTwoChoice"] = action
            # [2:] -> take the reward probabilities for the second stage only
            self.info["rewardProbabilities"] = self.reward_prob_matrix.flatten()[2:]
            self.info["rewardDistribution"] = self.reward_distribution.flatten()[2:]


        else:
            raise ValueError(
                f"state:{self.state} is an invalid state, state space: {self.state_space}")

        return self.state, reward, self.terminal, self.info

    def reward_function(self, state, action):
        """
        Calculates the reward based on the state and action taken.

        Args:
            state (int): The current state of the environment.
            action (int): The action taken in the environment.

        Returns:
            int: The reward value.
        """
        if action not in self.action_space:
            raise ValueError(
                f"The action: {action} is not valid, action space: {self.action_space}")
        if state not in self.state_space:
            raise ValueError(
                f"state:{state} is an invalid state, state space: {self.state_space}")

        self.update_reward_distribution() # get reward through the reward distribution for possible further analysis of the task
        # give a reward according to the probability of getting a reward
        # for the action taken in the state ( state-action pair )
        reward = self.reward_distribution[state][action]
        # reward = np.random.uniform() < self.reward_prob_matrix[state][action]
        # scale the reward for a costume reward value equal to self.reward
        # makes no difference in case self.reward = 1
        # reward = int(reward) * self.reward_scaler_matrix[state][action]
        reward = int(reward) * self.reward
        return reward

    def state_transition_function(self, state, action):
        """
        Calculates the next state and terminal flag based on the current state and action taken.

        Args:
            state (int): The current state of the environment.
            action (int): The action taken in the environment.

        Returns:
            tuple: A tuple containing the next state and terminal flag.
        """
        if action not in self.action_space:
            raise ValueError(
                f"The action: {action} is not valid, action space: {self.action_space}")

        new_state = None
        terminal = False
        if state == 0:
            new_state = np.random.choice(self.state_space,
                                         p=self.stage_1_transition_matrix[action])
        elif state in [1, 2]:
            terminal = True
        else:
            raise ValueError(
                f"state:{state} is an invalid state, state space: {self.state_space}")

        return new_state, terminal

    def is_common_state(self, state, action):
        """
        Checks if the current state is a common state based on the action taken.

        Args:
            state (int): The current state of the environment.
            action (int): The action taken in the environment.

        Returns:
            bool: True if the state is a common state, False otherwise.
        """
        if action not in self.action_space:
            raise ValueError(
                f"The action: {action} is not valid, action space: {self.action_space}")
        if state not in self.state_space:
            raise ValueError(
                f"state:{state} is an invalid state, state space: {self.state_space}")

        # return self.stage_1_transition_matrix[action, state] >= 0.5
        return self.stage_1_transition_matrix[action, state] == np.max(
            self.stage_1_transition_matrix[action])

    def update_reward_distribution(self):
        """
        Updates the reward distribution based on the reward probabilities.

        Returns:
            numpy.ndarray: The updated reward distribution.
        """
        self.reward_distribution = np.random.uniform(size=self.reward_prob_matrix.shape) < self.reward_prob_matrix
        self.reward_distribution = self.reward_distribution.astype(float)
        return self.reward_distribution
                
    def set_reward_probabilities(self, reward_prob_matrix):
        """
        Sets the reward probabilities to the given matrix.

        Args:
            reward_prob_matrix (numpy.ndarray): The matrix of reward probabilities.

        Returns:
            numpy.ndarray: The updated reward probabilities matrix.
        """
        if reward_prob_matrix.shape != self.reward_prob_matrix.shape:
            raise ValueError(
                f"reward_prob_matrix shape: {reward_prob_matrix.shape} is not valid, shape should be {self.reward_prob_matrix.shape}")
        # clip the reward probabilities to be between min_reward_prob and max_reward_prob
        reward_prob_matrix = np.clip(reward_prob_matrix, self.min_reward_prob,
                                     self.max_reward_prob)

        # update the reward_prob_matrix
        # if the reward_prob_matrix is fixed -> do not update it, else update it with from the new reward_prob_matrix
        self.reward_prob_matrix = np.where(self.fixed_reward_prob_matrix,
                                           self.reward_prob_matrix, reward_prob_matrix)
        return self.reward_prob_matrix

    def set_reward_distribution(self, reward_distribution):
        """
        Sets the reward distribution to the given matrix.

        Args:
            reward_distribution (numpy.ndarray): The matrix of reward distribution.

        Returns:
            numpy.ndarray: The updated reward distribution.
        """
        if reward_distribution.shape != self.reward_distribution.shape:
            raise ValueError(
                f"reward_distribution shape: {reward_distribution.shape} is not valid, shape should be {self.reward_distribution.shape}")
        self.reward_distribution = reward_distribution
        return self.reward_distribution

    def __str__(self):
        """
        Returns a string representation of the environment.

        Returns:
            str: A string representation of the environment.
        """
        discription = f"Two Step Task Environment:\n"
        discription += f"state space: \n{self.state_space}\n"
        discription += f"action space: \n{self.action_space}\n"
        discription += f"transition probability: \n{self.transition_prob}\n"
        discription += f"stage 1 transition matrix: \n{self.stage_1_transition_matrix}\n"
        discription += f"initial reward probability matrix: \n{self.reward_prob_matrix}\n"
        discription += f"fixed reward probability matrix: \n{self.fixed_reward_prob_matrix}\n"
        discription += f"reward distribution (based on the current reward probability matrix): \n{self.reward_distribution}\n"
        # repr += f"reward scaler matrix: {self.reward_scaler_matrix}\n"
        discription += f"reward scaler: \n{self.reward}\n"

        # print("Two Step Task Environment:")
        # print(f"state space: {self.state_space}")
        # print(f"action space: {self.action_space}")
        # print(f"transition probabilities: {self.transition_prob}")
        # print(f"stage 1 transition matrix: {self.stage_1_transition_matrix}")
        # print(f"initial reward probability matrix: {self.reward_prob_matrix}")
        # print(f"fixed reward probability matrix: {self.fixed_reward_prob_matrix}")
        # print(f"reward distribution: {self.reward_distribution}")
        # # print(f"reward scaler matrix: {self.reward_scaler_matrix}")
        # print(f"reward scaler: {self.reward}")
        return discription
    
    def set_seed(self, seed):
        """
        Sets the seed for random number generation.

        Args:
            seed (int): The seed value.
        """
        pass

    def plot(self):
        """
        Plots the environment.
        """
        pass