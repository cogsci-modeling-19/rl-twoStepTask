import pandas as pd
import numpy as np

from agents.model_based import AgentModelBased
from agents.model_free import AgentModelFree
from agents.hybrid import HybridAgent
from agents.random_agent import RandomAgent
from environment import TwoStepEnv
from exp_environment import TwoStepExperimentEnv
from utils import random_walk_gaussian


def simulate(agent_type='random', trials=200, seed=0, verbose=False, params:dict={}):
    data_filename = 'data/participants/experiment_data_mj.csv'
    if verbose:
        print(f"Simulating {agent_type} agent, {trials} trials.")
        print(f"Agent parameters: {params if params else 'default'}")
    # set a random seed
    np.random.seed(seed)

    # simulate the task
    action_space = TwoStepExperimentEnv.action_space
    state_space = TwoStepExperimentEnv.state_space

    if agent_type == 'model_based':
        agent = AgentModelBased(action_space, state_space, **params)
    elif agent_type == 'model_free':
        agent = AgentModelFree(action_space, state_space, **params)
    elif agent_type == 'hybrid':
        agent = HybridAgent(action_space, state_space, **params)
    else:
        agent = RandomAgent(action_space, state_space, **params)
    env = TwoStepExperimentEnv(data_filename)
    task_data = simulate_two_step_task(env, agent, trials=trials)

    # convert the data to a dataframe
    task_df = pd.DataFrame.from_dict(task_data, orient='index')

    return task_df, agent

def simulate_two_step_task(env, agent=None, trials=200,
                           policy_method="softmax"):
    env.reset()
    task_data = {}

    sd_for_random_walk = 0.025
    time_step = 0
    while time_step < trials:
        # first stage choice
        terminal = False
        while not terminal:
            current_state = env.state
            if agent:
                action = agent.policy(env.state, method=policy_method)
            else:  # if no agent is given -> random action
                action = np.random.choice(env.action_space)

            next_state, reward, terminal, info = env.step(action)

            if agent:
                agent.update_beliefs(current_state, action, reward, next_state,
                                     terminal)

        info['trial_index'] = int(time_step)
        task_data[time_step] = info
        env.reset()
        # new_reward_prob_matrix = random_walk_gaussian(env.reward_prob_matrix,
        #                                               sd_for_random_walk)
        # env.set_reward_probabilities(new_reward_prob_matrix)
        time_step += 1

    return task_data
