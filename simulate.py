from agents.model_based import AgentModelBased
from agents.model_free import AgentModelFree
from agents.hybrid import HybridAgent
from agents.random_agent import RandomAgent
from environment import TwoStepEnv
from utils import random_walk_gaussian
import pandas as pd
import numpy as np


def simulate(agent_type='random', trials=200, seed=None, verbose=False, params:dict={}, from_data:pd.DataFrame=None, use_reward_distribution=False):
    """
    Simulate the two-step task using the environment and the given agent
    :param agent_type: ['random', 'model_free', 'model_based', 'hybrid']
    :param trials: number of trials to simulate
    :param seed: random seed
    :param verbose: print details if True
    :param params: parameters for the agent
    :param from_data: use reward probabilities from the given data
    :param use_reward_distribution: use the same sampled reward distribution if True
    :return: simulated task data as a dataframe and the agent
    """
    if verbose:
        print(f"Simulating {agent_type} agent, {trials} trials.")
        print(f"Agent parameters: {params if params else 'default'}")
    # set a random seed
    np.random.seed(seed)

    # simulate the task
    action_space = TwoStepEnv.action_space
    state_space = TwoStepEnv.state_space

    if agent_type == 'model_based':
        agent = AgentModelBased(action_space, state_space, **params)
    elif agent_type == 'model_free':
        agent = AgentModelFree(action_space, state_space, **params)
    elif agent_type == 'hybrid' or agent_type.startswith('hybrid'):
        agent = HybridAgent(action_space, state_space, **params)
    else:
        agent = RandomAgent(action_space, state_space, **params)
    env = TwoStepEnv()
    task_data = simulate_two_step_task(env, agent, trials=trials, from_data=from_data, use_reward_distribution=use_reward_distribution)

    # convert the data to a dataframe
    task_df = pd.DataFrame.from_dict(task_data, orient='index')

    # unset the random seed
    np.random.seed(None)
    return task_df, agent

def simulate_two_step_task(env: TwoStepEnv, agent, trials=200,
                           from_data:pd.DataFrame=None, use_reward_distribution=True):
    """
    Simulate with given agent
    :param env: environment
    :param agent: given agent
    :param trials: number of trials
    :param policy_method: method for action selection ['softmax', 'epsilon-greedy']
    :param from_data: use reward probabilities from the given data
    :param use_reward_distribution: use reward distribution if True
    :return: simulated task data
    """
    if from_data is not None:
        if use_reward_distribution:
            reward_distribution = from_data['rewardDistribution'].iloc[0]
            # convert the reward distribution to float
            reward_probabilities = reward_distribution.astype(float)
        else:
            reward_probabilities = from_data['rewardProbabilities'].iloc[0]
        # reshape the reward probabilities to the correct shape, with zeros for the first stage
        reward_probabilities = np.array([0, 0, *reward_probabilities])
        # reshape
        reward_probabilities = reward_probabilities.reshape((3, 2))
        env.set_reward_probabilities(reward_probabilities)

    task_data = {}

    sd_for_random_walk = 0.025
    time_step = 0
    while time_step < trials:
        # first stage choice
        terminal = False
        while not terminal:
            current_state = env.state
            action = agent.policy(env.state)

            next_state, reward, terminal, info = env.step(action)

            if agent:
                agent.update_beliefs(current_state, action, reward, next_state,
                                     terminal)

        info['trial_index'] = int(time_step)
        task_data[time_step] = info
        env.reset()
        # update the reward probabilities for the next trial
        # 3 cases:
        # 1. free simulation -> simulate a random walk in the reward probabilities
        # 2. simulate according to the reward probabilities from data and sample reward distribution in the environment 
        #                                                            -> use the same reward probabilities as the data
        # 3. simulate according to the reward distribution from data -> use the same reward distribution as the data
        if from_data is not None and time_step < trials - 1:
            if use_reward_distribution:
                reward_distribution = from_data['rewardDistribution'].iloc[time_step + 1]
                # convert the reward distribution to float
                new_reward_prob_matrix = reward_distribution.astype(float)
            else:
                new_reward_prob_matrix = from_data['rewardProbabilities'].iloc[time_step + 1]
            # include zeros for the first stage
            new_reward_prob_matrix = np.array([0, 0, *new_reward_prob_matrix])
            # reshape to the epxpected shape (3, 2): state-action
            new_reward_prob_matrix = new_reward_prob_matrix.reshape((3, 2))
        else:
            # simulate a random walk in the reward probabilities
            new_reward_prob_matrix = random_walk_gaussian(env.reward_prob_matrix,
                                                      sd_for_random_walk)
        env.set_reward_probabilities(new_reward_prob_matrix)
        time_step += 1

    return task_data
