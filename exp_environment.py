import pandas


class TwoStepExperimentEnv:
    action_space = [0, 1]
    state_space = [0, 1, 2]

    def __init__(self, experiment_data_file):
        self.state = 0
        self.reward = 1
        self.terminal = False
        self.info = {}
        self.count = 0

        self.data = pandas.read_csv(experiment_data_file)
        self.row = self.data.iloc[self.count]

    def reset(self):
        if self.count > len(self.data):
            raise ValueError("All trials have been completed, please create environment again")
        self.state = 0
        self.terminal = False
        self.info = {}
        return self.state

    def step(self, action):
        self.row = self.data.iloc[self.count]
        reward = 0
        step_one_two_param = self.row['stepOneTwo_Param']
        step_two_two_param = self.row['stepTwoTwo_Param']
        self.info["common_transition"] = [self.row['isHighProbOne'], self.row['isHighProbTwo']][action]
        if self.terminal:
            raise ValueError("Episode has already terminated")
        if self.state == 0:
            # if step_one_two_param is [0, 1] or [1, 0] then the next state is 1. otherwise, the next state is 2
            next_states = [1 if step_one_two_param in ([0, 1], [1, 0]) else 2,
                           1 if step_two_two_param in ([0, 1], [1, 0]) else 2]
            self.state = next_states[action]
            self.info["state_transition_to"] = self.state
            self.info["stepOneChoice"] = action
        elif self.state in [1, 2]:
            reward = self.reward_function(self.state, action)
            self.terminal = True
            self.info["reward"] = reward > 0
            self.info["stepTwoChoice"] = action - 2 if self.state == 2 else action
            self.info["rewardProbabilities"] = self.row['rewardProbabilities']
            self.count += 1
        else:
            raise ValueError(f"state:{self.state} is an invalid state, state space: {self.state_space}")
        return self.state, reward, self.terminal, self.info

    def string_to_bool_list(self, value):
        # "[true, true, false, true}" (str) -> [True, True, False, True] (Bool list)
        items = value.strip('[]').split(',')
        boolean_list = [item.lower() == 'true' for item in items]
        return boolean_list

    def reward_function(self, state, action):
        if state == 0:
            return 0
        elif state == 2:
            action += 2
        rewards = self.string_to_bool_list(self.row['rewards_Param'])
        return rewards[action]
