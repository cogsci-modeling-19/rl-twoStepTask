import numpy as np


class HybridAgent:
    def __init__(self, action_space, state_space, alpha_1=0.54, alpha_2=0.42, beta_1=5.19, beta_2=3.69, _lambda=0.57,
                 w=0.39, p=0.11):
        """
        Hybrid agent similar to the one used in Daw et al. (2011)
        Initialize hybrid agent
        :param action_space: Action space of the environment
        :param state_space: State space of the environment
        :param alpha_1: Learning rate for first step
        :param alpha_2: Learning rate for second step
        :param beta_1: Temperature parameter for softmax policy for first step
        :param beta_2: Temperature parameter for softmax policy for second step
        :param w: Weight of model-based values (0 for model-free, 1 for model-based)
        """
        self.action_space = action_space
        self.state_space = state_space
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self._lambda = _lambda
        self.w = w
        self.p = p

        self.q_td = np.zeros((len(state_space), len(action_space)))  # Q-table for model-free
        self.q_mb = np.zeros((len(state_space), len(action_space)))  # Q-table for model-based
        self.q_net = np.zeros((len(state_space), len(action_space)))  # Q-table for hybrid
        self.q_table = self.q_net

        # Initialize transition model as a 3D numpy array
        # Dimensions: [current_state, action, next_state]
        # For simplicity, initializing all transitions as equally likely
        self.transition_model = np.zeros((len(state_space), len(action_space), len(state_space)))
        self.transition_counts = np.zeros((len(state_space), len(action_space), len(state_space)))

        self.eligibility_traces = np.zeros((len(state_space), len(action_space)))

        self.previous_action = None

    def calculate_rpe(self, state, action, reward, next_state, next_action, terminal):
        """
        Calculate the reward prediction error
        :param state: current state
        :param action: current action
        :param reward: reward received in the current state
        :param next_state: next state
        :param next_action: next action
        :param terminal: whether this state is terminal
        :return:
        """
        if terminal:
            return reward - self.q_td[state, action]
        else:
            return reward + self.q_td[next_state, next_action] - self.q_td[state, action]

    def update_q_td(self, rpe):
        """
        Update the Q-table for model-free agent
        :param state: Current state
        :param action: Current action
        :param rpe: Reward prediction error
        :return:
        """
        # Apply the TD update to all state-action pairs using eligibility traces
        # Trace will be 1 for the current state-action pair. On stage 2, it will be lambda for the previous pair
        for s in range(len(self.state_space)):
            for a in range(len(self.action_space)):
                alpha = self.alpha_1 if s == 0 else self.alpha_2
                self.q_td[s, a] += alpha * rpe * self.eligibility_traces[s, a]

    def update_transition_model(self, current_state, action, next_state, terminal):
        if terminal:
            return

        # Increment the count for the observed transition
        self.transition_counts[current_state, action, next_state] += 1

        # Check if the transition is common or rare
        is_common_transition = self.transition_counts[current_state, action, next_state] == np.max(
            self.transition_counts[current_state, action])
        # Update the transition probabilities
        self.transition_model[current_state, action, next_state] = 0.7 if is_common_transition else 0.3
        other_states = np.array([state for state in self.state_space[1:] if state != next_state])
        for other_state in other_states:
            self.transition_model[current_state, action, other_state] = 0.3 if is_common_transition else 0.7

    def update_q_mb(self):
        # Update MB table for all state-action pairs so that first stage considers updated TD values
        for state in self.state_space:
            for action in self.action_space:
                if state == 0:
                    self.q_mb[state, action] = np.sum(
                        [self.transition_model[state, action, i] * np.max(self.q_td[i, :]) for i in
                         self.state_space[1:]])
                else:
                    self.q_mb[state, action] = self.q_td[state, action]

    def update_q_net(self):
        # Update MB table for all state-action pairs so that first stage considers updated TD values
        for state in self.state_space:
            for action in self.action_space:
                if state == 0:
                    self.q_net[state, action] = self.w * self.q_mb[state, action] + (1 - self.w) * self.q_td[
                        state, action]
                else:
                    self.q_net[state, action] = self.q_td[state, action]

    def update_beliefs(self, current_state, action, reward, next_state, terminal):
        # Decay all eligibility traces. This effectively sets trace for previous stage to lambda, if it's stage 2
        self.eligibility_traces *= self._lambda
        # Set eligibility trace = 1 for the current state-action pair
        self.eligibility_traces[current_state, action] = 1

        next_action = self.policy(next_state)
        rpe = self.calculate_rpe(current_state, action, reward, next_state, next_action, terminal)
        self.update_q_td(rpe)
        self.update_transition_model(current_state, action, next_state, terminal)
        self.update_q_mb()
        self.update_q_net()

        if terminal:
            # Clear all eligibility traces at the end of the episode
            self.eligibility_traces = np.zeros((len(self.state_space), len(self.action_space)))
        else:
            self.previous_action = action

    def get_action_probabilities(self, state):
        beta = self.beta_1 if state == 0 else self.beta_2
        # Calculate the rep(a) based on the current state and previous action
        rep_a = np.zeros_like(self.q_net[state, :])
        if state == 0 and self.previous_action is not None:
            rep_a[self.previous_action] = 1
        return self.softmax(self.q_net[state, :], beta, self.p, rep_a)

    def policy(self, state):
        return np.random.choice(self.action_space, p=self.get_action_probabilities(state))

    def softmax(self, q_values, beta, p, rep_a):
        adjusted_q_values = (q_values + p * rep_a - np.max(q_values))
        # Calculate the exponentiated weighted Q values with the repetition component
        exp_values = np.exp(beta * adjusted_q_values)

        # Ensure no division by zero and no 0 probabilities
        exp_values += 1e-8
        exp_values_sum = np.sum(exp_values, axis=0)

        # Compute the softmax probabilities
        probabilities = exp_values / exp_values_sum

        return probabilities

    def reset(self):
        self.q_td = np.zeros((len(self.state_space), len(self.action_space)))
        self.q_mb = np.zeros((len(self.state_space), len(self.action_space)))
        self.q_net = np.zeros((len(self.state_space), len(self.action_space)))
        self.q_table = self.q_net
        self.transition_model = np.zeros((len(self.state_space), len(self.action_space), len(self.state_space)))
        self.transition_counts = np.zeros((len(self.state_space), len(self.action_space), len(self.state_space)))
        self.eligibility_traces = np.zeros((len(self.state_space), len(self.action_space)))
        self.previous_action = None

    def __str__(self):
        discribtion = "Hybrid Agent\n"
        discribtion += "Action Space: " + str(self.action_space) + "\n"
        discribtion += "State Space: " + str(self.state_space) + "\n"
        discribtion += "Alpha 1: " + str(self.alpha_1) + "\n"
        discribtion += "Alpha 2: " + str(self.alpha_2) + "\n"
        discribtion += "Beta 1: " + str(self.beta_1) + "\n"
        discribtion += "Beta 2: " + str(self.beta_2) + "\n"
        discribtion += "Lambda: " + str(self._lambda) + "\n"
        discribtion += "W: " + str(self.w) + "\n"
        discribtion += "P: " + str(self.p) + "\n"
        return discribtion