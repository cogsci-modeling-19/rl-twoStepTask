import numpy as np


class AgentModelBased:
    """
    Model Based Agent that uses TD leaning (SARSA(1)) combined with a transition model to update its Q-table.
    Initialize Model Based Agent
    :param action_space: List of possible actions
    :param state_space: List of possible states
    :param alpha: Learning rate
    :param beta: Inverse temperature parameter for softmax policy
    :param gamma: Discount factor
    """
    def __init__(self, action_space, state_space, alpha=0.1, beta=1.0, gamma=0.9):
        self.action_space = action_space
        self.state_space = state_space
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.beta = beta  # Temperature parameter for softmax policy
        self.q_table = np.zeros((len(state_space), len(action_space)))

        # Initialize transition model as a 3D numpy array
        # Dimensions: [current_state, action, next_state]
        # For simplicity, initializing all transitions as equally likely
        self.transition_model = np.zeros((len(state_space),
                                          len(action_space),
                                          len(state_space)))
        self.transition_counts = np.zeros((len(state_space),
                                           len(action_space),
                                           len(state_space)))

    def softmax(self, arr, beta):
        e_x = np.exp(
            beta * (arr - np.max(arr)))  # subtract max value to prevent overflow
        return e_x / e_x.sum(
            axis=0)  # axis=0 for column-wise operation if arr is 2D, otherwise it's not needed

    def policy(self, state, beta=None):
        q_values = self.q_table[state, :]
        beta = self.beta if beta is None else beta
        # calculate the probability of each action in the state with softmax
        action_probabilities = self.softmax(q_values, beta)
        action = np.random.choice(self.action_space, p=action_probabilities)

        return action

    def update_transition_model(self, current_state, action, next_state, terminal):
        # Simple counting method to update transition probabilities
        # TODO - Implement more sophisticated methods like Bayesian update ?
        if terminal:
            return
        # update transition model only for stage 1 (state 0) 
        if current_state == 0:
            # Increment the count for the observed transition
            self.transition_counts[current_state, action, next_state] += 1
            # Normalize the transition probabilities for the current state-action pair
            total_transitions = self.transition_counts[current_state, action, :].sum()

            # incremental update
            # insure no zero probabilities with +1 in the numerator (good for exploration in the beginning)
            # self.transition_model[current_state, action, :] = (self.transition_counts[
            #                                     current_state, action, :] + 1
            #                                     ) / total_transitions          
                
            # TODO integrate high-low update below with parameterized P_COMMON
            # # high-low update
            P_COMMON = 0.7
            # self.transition_model[current_state, action, 0] = 0
            # what state got vitited more sofar
            most_visited_state = self.transition_counts[current_state, action, 1] >= total_transitions / 2  
            most_visited_state = int(most_visited_state)
            self.transition_model[current_state, action, 1:][1-most_visited_state] = P_COMMON
            self.transition_model[current_state, action, 1:][most_visited_state] = 1 - P_COMMON  
        

    def update_q_table(self, state, action, reward, next_state, terminal):
        # Update Q-table using the transition model
        if terminal:  # -> second stage -> update with TD
            self.q_table[state, action] += self.alpha * self.reward_prediction_error(
                state, action, reward, next_state, terminal)

        else:  # -> first stage -> update with transition model following the Bellman equation
            # iterate over all possible states
            # we can do that since the transition model will regulate which states
            # will be considered for the update
            # -> trasition propabilities are 0 for states that are not possible
            # initialize the sum for the Q-table update
            q_value_sum = 0
            for possible_state in self.state_space:
                # a list to hold Q-values for all actions from the possible state
                q_values = []
                # iterate over all possible actions in the next state
                for next_action in self.action_space:
                    # get the reward prediction error for the next state
                    next_terminal = possible_state in [1, 2]
                    reward_pred_error = self.reward_prediction_error(state, next_action, reward, next_state, next_terminal)
                    # Q-value for this action in the possible state
                    q_value = self.q_table[possible_state, next_action] + self.alpha * reward_pred_error
                    q_values.append(q_value)
                
                # take the maximum Q-value among all actions for the possible state
                max_q_value = np.max(q_values)
                # scale the max Q-value by the transition probability from current state-action pair to the possible state
                weighted_q_value = self.transition_model[state, action, possible_state] * max_q_value
                # add the weighted Q-value to the sum
                q_value_sum += weighted_q_value

            # update the Q-table entry for the state-action pair 
            self.q_table[state, action] = q_value_sum


    def reward_prediction_error(self, state, action, reward, next_state, terminal):
        if terminal:
            return reward - self.q_table[state, action]

        next_action = self.policy(next_state)
        return reward + self.gamma * self.q_table[next_state, next_action] - \
            self.q_table[state, action]

    def update_beliefs(self, state, action, reward, next_state, terminal):
        self.update_transition_model(state, action, next_state, terminal)
        self.update_q_table(state, action, reward, next_state, terminal)

    def get_action_probabilities(self, state):
        q_values = self.q_table[state, :]
        action_probabilities = self.softmax(q_values, self.beta)
        return action_probabilities
    
    def reset(self):
        self.q_table = np.zeros((len(self.state_space), len(self.action_space)))
        self.transition_model = np.zeros((len(self.state_space),
                                          len(self.action_space),
                                          len(self.state_space)))
        self.transition_counts = np.zeros((len(self.state_space),
                                           len(self.action_space),
                                           len(self.state_space)))
        # return self.q_table, self.transition_model, self.transition_counts

    def __str__(self):
        discribtion = "Model Based Agent\n"
        discribtion += "Action Space: " + str(self.action_space) + "\n"
        discribtion += "State Space: " + str(self.state_space) + "\n"
        discribtion += "Alpha: " + str(self.alpha) + "\n"
        discribtion += "Beta: " + str(self.beta) + "\n"
        discribtion += "Gamma: " + str(self.gamma) + "\n"
        return discribtion
    