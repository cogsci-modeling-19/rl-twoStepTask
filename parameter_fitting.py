from simulate import simulate
from agents.random_agent import RandomAgent
from agents.model_free import AgentModelFree
from agents.model_based import AgentModelBased
from agents.hybrid import HybridAgent
from environment import TwoStepEnv
from sklearn.model_selection import ParameterSampler
from itertools import product
from scipy.optimize import minimize
from scipy.stats import uniform
from datetime import datetime
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.interpolate import griddata

# from tqdm import tqdm
# import the tqdm library combpatibilitle with jupyter notebook
from tqdm.notebook import tqdm


# calculate the log likelihood of the human data given the model
def get_action_probs(agent=None, state=None):
    """
    get the action probabilities for the given state using the agent's policy
    :param agent: the agent
    :param state: the state
    :return: action probabilities for all actions as a numpy array
    """
    if agent is None:
        return np.random.uniform(size=2)

    return agent.get_action_probabilities(state)


def apply_choices(agent=None, stage_1=None, stage_2=None, recieved_reward=None):
    """
    Apply the choices to the agent and update the beliefs
    :param agent: the agent
    :param stage_1: tuple of the state and action of the first stage
    :param stage_2: tuple of the state and action of the second stage
    :param recieved_reward: the reward recieved
    :return: None
    """
    if agent is None:
        return

    state_1 = stage_1[0]
    action_1 = stage_1[1]
    state_2 = stage_2[0]
    action_2 = stage_2[1]
    # state, action, reward, next_state, terminal
    agent.update_beliefs(state_1, action_1, 0, state_2, False)
    agent.update_beliefs(state_2, action_2, recieved_reward, state_2, True)


def log_likelihood(agent, data, consider_both_stages=True, verbose=False, show_progress=False):
    """
    Calculate the log likelihood of the data given the model
    :param agent: the agent
    :param data: the data
    :param consider_both_stages: consider the choices of both stages for the log likelihood
    :param verbose: print everything for debugging
    :param show_progress: show the progress bar
    :return: the log likelihood
    """
    LogLikelihood_sum = 0
    for idx, trial_data in tqdm(
        data.iterrows(),
        desc="log_likelihood:",
        total=len(data),
        disable=not show_progress,
        leave=False,
    ):
        chosen_action_1 = trial_data["stepOneChoice"]
        chosen_action_2 = trial_data["stepTwoChoice"]
        stage_1_state = 0
        stage_2_state = trial_data["state_transition_to"]
        recieved_reward = int(trial_data["reward"])

        action_probs_stage_1 = get_action_probs(agent, stage_1_state)
        chosen_action_1_prob = action_probs_stage_1[chosen_action_1]
        agent.update_beliefs(stage_1_state, chosen_action_1, 0, stage_2_state, False)

        action_probs_stage_2 = get_action_probs(agent, stage_2_state)
        chosen_action_2_prob = action_probs_stage_2[chosen_action_2]
        agent.update_beliefs(stage_2_state, chosen_action_2, recieved_reward, stage_2_state, True)

        # calculate the log likelihood
        # based on only the first stage stage
        LogLikelihood_sum += np.log(chosen_action_1_prob)

        if consider_both_stages:
            # based on both stages
            # assuming the actions are independent following the Markov property
            # P(a1, a2) = P(a1) * P(a2)
            # -> log(P(a1, a2)) = log(P(a1)) + log(P(a2))
            LogLikelihood_sum += np.log(chosen_action_2_prob)

        # print everything for debugging
        if verbose:
            print(f"reward recieved: {recieved_reward}")
            print(f"chosen_action_1: {chosen_action_1}")
            print(f"chosen_action_2: {chosen_action_2}")
            print(f"stage_2_state: {stage_2_state}")
            print(f"action_probs_stage_1: {action_probs_stage_1}")
            print(f"action_probs_stage_2: {action_probs_stage_2}")
            print(f"q_table: {agent.q_table}")
            if hasattr(agent, "transition_model"):
                print(f"transition_matrix: {agent.transition_model[0]}")
            print("#" * 50)

    return LogLikelihood_sum


def fit_with_MCMC(
    parammeter_space: dict,
    data,
    agent_type,
    consider_both_stages=True,
    num_chains=1,
    num_samples=1000,
    num_burn_in=200,
    base_sampling_sd_ratio=0.1,
    show_progress=False,
):
    """
    Fit the Model to data using MCMC Sampling, using the Metropolis Hastings algorithm with simple scaled noraml proposal distribution to account for different scales of the parameter spaces
    :param parammeter_space: the parameter space, a dictionary of parameter names and their bounds
    :param data: the data
    :param agent_type: the model type
    :param consider_both_stages: consider the choices of both stages for the log likelihood
    :param num_chains: the number of chains
    :param num_samples: the number of samples
    :param num_burn_in: the number of burn in samples
    :param show_progress: show the progress bar
    :return: the best parameters, the best log likelihood, and the results dataframe of all samples
    """
    num_total_samples_per_chain = num_samples + num_burn_in
    param_names = list(parammeter_space.keys())
    param_bounds = [(np.min(parammeter_space[param]), np.max(parammeter_space[param])) for param in param_names]
    # calculate the sampling standard deviations
    sampling_sds = [base_sampling_sd_ratio * np.abs(bound[1] - bound[0]) for bound in param_bounds]

    # define the objective function (log likelihood function)
    def log_likelihood_function(params):
        # create the parameter dictionary
        params = dict(zip(param_names, params))
        agent = create_agent(agent_type, params)
        return log_likelihood(agent, data, consider_both_stages)

    samples = np.zeros((num_chains * num_total_samples_per_chain, len(param_names)))
    log_likelihood_values = []
    burn_in_colunm = np.zeros(num_chains * num_total_samples_per_chain, dtype=bool)
    # run the MCMC algorithm for the given number of chains
    for chain in tqdm(
        range(num_chains),
        desc="MCMC chains:",
        total=num_chains,
        disable=not show_progress,
        leave=True,
    ):
        # initial guess sampled from a uniform distribution within the parameter space
        initial_guess = [np.random.uniform(bound[0], bound[1]) for bound in param_bounds]
        current_params = np.array(initial_guess)
        parametr_array_shape = current_params.shape
        is_burn_in = True
        for i in tqdm(
            range(num_total_samples_per_chain),
            desc=f"Chain {chain + 1} :",
            total=num_total_samples_per_chain,
            disable=not show_progress,
            leave=True,
        ):
            # sample new parameters
            if i >= num_burn_in:
                is_burn_in = False
            new_params = current_params + np.random.normal(0, scale=1, size=parametr_array_shape) * sampling_sds
            # bound the parameters to the parameter space
            new_params = np.clip(
                new_params,
                a_min=[bound[0] for bound in param_bounds],
                a_max=[bound[1] for bound in param_bounds],
            )
            # calculate the log likelihood for the new parameters
            log_likelihood_current = log_likelihood_function(current_params)
            log_likelihood_new = log_likelihood_function(new_params)
            # calculate the acceptance probability
            accept_prob = np.exp(log_likelihood_new - log_likelihood_current)
            # accept or reject the new parameters
            if log_likelihood_new > log_likelihood_current or np.random.rand() < accept_prob:
                current_params = new_params
                log_likelihood_values.append(log_likelihood_new)
            else:
                log_likelihood_values.append(log_likelihood_current)
            # store the parameters
            # offset the index according to the chain number
            sample_idx = i + (chain * num_total_samples_per_chain)
            samples[sample_idx] = current_params
            burn_in_colunm[sample_idx] = is_burn_in

    # Convert to DataFrame
    results_df = pd.DataFrame(samples, columns=param_names)
    results_df["log_likelihood"] = log_likelihood_values
    # add a column for the chain number, starting from 1
    results_df["chain"] = np.repeat(np.arange(1, num_chains + 1), num_total_samples_per_chain)
    # add a column for the burn in samples
    results_df["burn_in"] = burn_in_colunm
    # get the best parameters and the best LL
    # exclude the burn in samples from each chain for the best parameters and LL
    best_params, best_LL = get_best_params_and_ll(results_df[results_df["burn_in"] == False])
    return best_params, best_LL, results_df


def fit_with_random_search(
    param_space: dict,
    data,
    agent_type,
    num_iterations=100,
    consider_both_stages=True,
    seed=0,
    show_progress=False,
):
    """
    Fit the Model to data using Random Search
    :param param_space: the parameter space, a dictionary of parameter names and their bounds
    :param data: the data
    :param agent_type: the model type
    :param num_iterations: the number of iterations
    :param consider_both_stages: consider the choices of both stages for the log likelihood
    :param seed: the random seed
    :param show_progress: show the progress bar
    :return: the best parameters, the best log likelihood, and the results dataframe of all samples
    """
    # convert the parameter space to a format that can be used by ParameterSampler
    # scipy.stats.uniform ->  (loc, loc + scale)
    param_distribution = {
        param: uniform(
            np.min(param_space[param]),
            np.max(param_space[param]) - np.min(param_space[param]),
        )
        for param in param_space.keys()
    }
    # Generate parameter samples
    param_list = list(ParameterSampler(param_distribution, n_iter=num_iterations, random_state=seed))

    # Initialize an empty list for the results
    sampled_results = []

    # Evaluate log likelihood for sampled parameter sets
    for params in tqdm(
        param_list,
        desc="random_search:",
        total=len(param_list),
        disable=not show_progress,
        leave=True,
    ):
        agent = create_agent(agent_type, params)
        log_likelihood_value = log_likelihood(agent, data, consider_both_stages)
        sampled_results.append({**params, "log_likelihood": log_likelihood_value})

    # Convert to DataFrame
    results_df = pd.DataFrame(sampled_results)
    # get the best parameters and the best LL
    best_params, best_LL = get_best_params_and_ll(results_df)
    return best_params, best_LL, results_df


def fit_with_grid_search(
    parammeter_space: dict,
    data,
    agent_type,
    consider_both_stages=True,
    show_progress=False,
):
    """
    Fit the Model to data using Grid Search
    :param parammeter_space: the parameter space, a dictionary of parameter names and their values
    :param data: the data
    :param agent_type: the model type
    :param consider_both_stages: consider the choices of both stages for the log likelihood
    :param show_progress: show the progress bar
    :return: the best parameters, the best log likelihood, and the results dataframe of all samples
    """
    param_names = list(parammeter_space.keys())
    param_combinations = list(product(*parammeter_space.values()))
    # Initialize an empty list for the results
    sampled_results = []
    for idx, param_vals in tqdm(
        enumerate(param_combinations),
        desc="grid_search:",
        total=len(param_combinations),
        disable=not show_progress,
        leave=True,
    ):
        params = dict(zip(param_names, param_vals))
        agent = create_agent(agent_type, params)
        log_likelihood_value = log_likelihood(agent, data, consider_both_stages)
        sampled_results.append({**params, "log_likelihood": log_likelihood_value})

    # convert sampled results to Dataframe
    results_df = pd.DataFrame(sampled_results)

    # get the best parameters and the best LL
    best_params, best_LL = get_best_params_and_ll(results_df)
    return best_params, best_LL, results_df


def get_best_params_and_ll(results_df):
    # Sort the dataframe and get first row
    sorted_df = results_df.sort_values(by="log_likelihood", ascending=False)
    best_parameters_row = sorted_df.iloc[0]

    # Extract the best parameters and log likelihood
    best_params = best_parameters_row.drop("log_likelihood").to_dict()
    best_log_likelihood = best_parameters_row["log_likelihood"]

    return best_params, best_log_likelihood


def create_agent(agent_type, params):
    """
    Create an agent of the given type with the given parameters
    :param agent_type: the agent type
    :param params: the parameters
    :return: initialized agent
    """
    if agent_type == "model_free":
        agent = AgentModelFree(TwoStepEnv.action_space, TwoStepEnv.state_space, **params)
    elif agent_type == "model_based":
        agent = AgentModelBased(TwoStepEnv.action_space, TwoStepEnv.state_space, **params)
    elif agent_type == "hybrid" or agent_type.startswith("hybrid"):
        agent = HybridAgent(TwoStepEnv.action_space, TwoStepEnv.state_space, **params)
    else:
        agent = RandomAgent(TwoStepEnv.action_space, TwoStepEnv.state_space, **params)
    return agent


def fit_to_average_cumulative_reward(parammeter_space: dict, agent_type, num_iterations=100, seed=None):
    """
    Fit the Model to the average cumulative reward using random search
    :param parammeter_space: the parameter space, a dictionary of parameter names and their values
    :param agent_type: the model type
    :param seed: the random seed
    :return: the best parameters, the best average cumulative reward, and the results dataframe of all samples
    """
    param_names = [name for name in list(parammeter_space.keys())]
    # use randome search to sample the parameter space
    param_distribution = {
        param: uniform(
            np.min(parammeter_space[param]),
            np.max(parammeter_space[param]) - np.min(parammeter_space[param]),
        )
        for param in parammeter_space.keys()
    }
    # Generate parameter samples
    param_list = list(ParameterSampler(param_distribution, n_iter=num_iterations, random_state=seed))
    sampled_results = []
    for idx, params in enumerate(param_list):
        data, _ = simulate(agent_type=agent_type, seed=seed, params=params)
        avg_cumulative_reward = data["reward"].mean()
        sampled_results.append({**params, "avg_cumulative_reward": avg_cumulative_reward})

    # Convert to DataFrame
    results_df = pd.DataFrame(sampled_results)
    # get the best parameters and the best average cumulative reward
    # manually here
    best_parameters_row = results_df.iloc[results_df["avg_cumulative_reward"].idxmax()]
    best_params = best_parameters_row.drop("avg_cumulative_reward").to_dict()
    best_avg_cumulative_reward = best_parameters_row["avg_cumulative_reward"]
    return best_params, best_avg_cumulative_reward, results_df


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# plotting functions


# plot the log likelihoods
def plot_fit_results(
    results_df: pd.DataFrame,
    parameter_space: dict,
    title="",
    iterpolation_steps=50,
    save=False,
    filename="plots/fit_results.png",
):
    num_params = len(parameter_space)
    if num_params == 1:
        plot_fit_results_1d(
            results_df,
            parameter_space,
            title=title,
            iterpolation_steps=iterpolation_steps,
            save=save,
            filename=filename,
        )

    elif num_params == 2:
        plot_fit_results_2d(
            results_df,
            parameter_space,
            title=title,
            iterpolation_steps=iterpolation_steps,
            save=save,
            filename=filename,
        )

    elif num_params > 2:
        plot_heatmap_slices(
            results_df,
            parameter_space,
            title=title,
            iterpolation_steps=iterpolation_steps,
            save=save,
            filename=filename,
        )

    else:
        print(f"parameter space should be of type dict, provided: {type(parameter_space)}")


def plot_fit_results_1d(
    results_df: pd.DataFrame,
    parameter_space: dict,
    title="",
    iterpolation_steps=50,
    save=False,
    filename="plots/fit_results.png",
):
    param_name, param_space = list(parameter_space.items())[0]
    x = np.linspace(param_space.min(), param_space.max(), iterpolation_steps)

    min_value = np.min(results_df["log_likelihood"])
    # set the fill value to a value slightly lower than the minimum log likelihood
    fill_value = min_value - np.abs(min_value * 0.1)
    values = griddata(
        points=results_df[param_name],
        values=results_df["log_likelihood"],
        xi=x,
        fill_value=fill_value,
        method="linear",
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(title)
    ax.plot(x, values)
    ax.set_xlabel(param_name)
    ax.set_ylabel("Log Likelihood")
    ax.set_title("Log Likelihood of human data given model")
    plt.tight_layout()
    plt.show()
    if save:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        # add timestamp to filename
        timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
        name, ext = os.path.splitext(filename)
        filename = f"{name}_{timestamp}{ext}"
        fig.savefig(filename)
        print(f"Plot saved to {filename}")


def plot_fit_results_2d(
    results_df: pd.DataFrame,
    parameter_space: dict,
    title="",
    iterpolation_steps=50,
    save=False,
    filename="plots/fit_results.png",
    ax=None,
):

    # Select two parameters to plot
    param_x, param_y = list(parameter_space.keys())

    # Create a grid
    x = np.linspace(
        np.min(parameter_space[param_x]),
        np.max(parameter_space[param_x]),
        iterpolation_steps,
    )
    y = np.linspace(
        np.min(parameter_space[param_y]),
        np.max(parameter_space[param_y]),
        iterpolation_steps,
    )
    x_grid, y_grid = np.meshgrid(x, y)

    # Interpolate log likelihood values
    min_value = np.min(results_df["log_likelihood"])
    # set the fill value to a value slightly lower than the minimum log likelihood
    fill_value = min_value - np.abs(min_value * 0.1)
    values_grid = griddata(
        points=results_df[[param_x, param_y]],
        values=results_df["log_likelihood"],
        xi=(x_grid, y_grid),
        fill_value=fill_value,
        rescale=True,
        method="linear",
    )
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle(title)
        ax.set_title("Log Likelihoods")
    else:
        fig = ax.figure
        ax.set_title(title)

    x_ticks = np.round(x, 3)
    y_ticks = np.round(y, 3)
    visible_ticks_x = [x_ticks[0], x_ticks[len(x_ticks) // 2], x_ticks[-1]]
    visible_ticks_y = [y_ticks[0], y_ticks[len(y_ticks) // 2], y_ticks[-1]]

    im = ax.imshow(values_grid, origin="lower", cmap="viridis")

    ax.set_xlabel(param_x)
    ax.set_xticks([0, len(x_ticks) // 2, len(x_ticks) - 1])
    ax.set_xticklabels(visible_ticks_x)
    ax.set_ylabel(param_y)
    ax.set_yticks([0, len(y_ticks) // 2, len(y_ticks) - 1])
    ax.set_yticklabels(visible_ticks_y)

    fig.colorbar(im, ax=ax, label="Log Likelihood")

    # Annotate the maximum log likelihood on the plot
    max_ll_idx = np.unravel_index(np.argmax(values_grid, axis=None), values_grid.shape)
    max_ll = np.round(values_grid[max_ll_idx], 3)
    max_param_1 = np.round(x[max_ll_idx[0]], 3)
    max_param_2 = np.round(y[max_ll_idx[1]], 3)

    ax.annotate(
        f"Max LL={max_ll}\n{param_x}={max_param_1}, {param_y}={max_param_2}",
        xy=(max_ll_idx[1], max_ll_idx[0]),
        xytext=(max_ll_idx[1] + 1, max_ll_idx[0] + 1),
        arrowprops=dict(facecolor="red", shrink=0.005),
        color="red",
        zorder=10,
        textcoords="offset points",
    )

    # fig.tight_layout()
    if ax is None:
        plt.show()
        if save:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            # add timestamp to filename
            timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
            name, ext = os.path.splitext(filename)
            filename = f"{name}_{timestamp}{ext}"
            fig.savefig(filename)
            print(f"Plot saved to {filename}")


def plot_heatmap_slices(
    results_df: pd.DataFrame,
    parameter_space: dict,
    title="",
    iterpolation_steps=100,
    save=False,
    filename="plots/fit_results.png",
):

    # num_params = len(parameter_space)
    # sns.kdeplot(results_df[param], label=param)
    param_names = []
    param_spaces = []
    param_ranges = []
    # execlude fixed parameters for plotting the 2D slices
    for param, space in parameter_space.items():
        min_value = np.min(space)
        max_value = np.max(space)
        if min_value == max_value:
            print(f"Parameter {param} is fixed at {min_value}")
            continue
        param_names.append(param)
        param_spaces.append(space)
        param_ranges.append((min_value, max_value))

    num_params = len(param_names)
    num_data_points = len(results_df)

    best_params_row = results_df.iloc[results_df["log_likelihood"].idxmax()]
    talorance_percent_of_range = 0.1
    # calculate the min and max values for the best parameters
    best_params_ranges = []
    for i in range(num_params):
        talorance = talorance_percent_of_range * (param_ranges[i][1] - param_ranges[i][0])
        best_params_ranges.append(
            (
                best_params_row[param_names[i]] - talorance,
                best_params_row[param_names[i]] + talorance,
            )
        )

    # plot the 2D slices for all pairs of parameters from the results_df
    num_all_2d_slices = (num_params * (num_params - 1)) // 2  # number of all possible pairs of parameters

    # prepare a figure to hold all heat maps
    grid_size = np.math.ceil(np.sqrt(num_all_2d_slices))  # Determine grid size to ensure a square figure
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(7 * grid_size, 5 * grid_size))
    fig.suptitle(title)
    if num_all_2d_slices == 1:
        axes = [axes]
    axes = axes.flatten()  # Flatten the array of axes for easy indexing
    current_plot = 0

    slices_progress = tqdm(
        total=num_all_2d_slices, desc="2D slices:", leave=True
    )  # progress bar for the 2D slices plots
    for i in range(num_params):
        for j in range(i + 1, num_params):
            # take only the values where all other parameters are at their maximum LL
            # initialize a mask for the slice
            mask = np.ones(num_data_points, dtype=bool)
            # loop over all parameters and consider only the values where the other parameters are around their maximum LL
            for k in range(num_params):
                if k != i and k != j:
                    # update the mask
                    mask = mask & (
                        (results_df[param_names[k]] > best_params_ranges[k][0])
                        & (results_df[param_names[k]] < best_params_ranges[k][1])
                    )
            # apply the mask
            slice_ij = results_df.loc[mask, [param_names[i], param_names[j], "log_likelihood"]]
            # define the 2d parameter space and and title to pass them to plot_fit_results_2d
            param_space_ij = {
                param_names[i]: param_spaces[i],
                param_names[j]: param_spaces[j],
            }

            # include the range of the other parameters in the title
            plot_title = ""
            for k in range(num_params):
                if k != i and k != j:
                    plot_title += f" {param_names[k]}={np.round(best_params_ranges[k], 3)}"
            ax = axes[current_plot]
            plot_fit_results_2d(
                slice_ij,
                param_space_ij,
                plot_title,
                iterpolation_steps,
                save,
                filename,
                ax=ax,
            )
            current_plot += 1

            slices_progress.update(1)  # update the progress bar
    slices_progress.close()  # close the progress bar
    # remove the empty plots
    for i in range(current_plot, len(axes)):
        fig.delaxes(axes[i])
    fig.tight_layout()
    plt.show()
    if save:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        # add timestamp to filename
        timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
        name, ext = os.path.splitext(filename)
        filename = f"{name}_{timestamp}{ext}"
        fig.savefig(filename)
        print(f"Plot saved to {filename}")


def plot_MCMC_chain_convergence(
    results_df,
    title="",
    max_plots_per_row=4,
    save=False,
    filename="plots/MCMC_chain_convergence.png",
):
    """
    Plot the convergence of the MCMC chains for each parameter
    :param results_df: the results dataframe
    :param title: the title of the plot
    :param save: save the plot
    :param filename: the filename to save the plot
    :return: None
    """
    mcmc_samples = results_df.copy()
    # sns.set(style="whitegrid")  # Set the seaborn style for the plot

    # Exclude 'chain' and 'log_likelihood' from parameters to plot
    parameters = [col for col in mcmc_samples.columns if col not in ["chain", "burn_in", "log_likelihood"]]
    # exclude parameters with fixed values
    parameters = [param for param in parameters if len(mcmc_samples[param].unique()) > 1]
    # add iteration column to align the samples of all chains
    mcmc_samples["iteration"] = mcmc_samples.groupby("chain").cumcount()
    # get the burn in offset, burn_in column is a boolean column
    burn_in_offset = mcmc_samples.groupby("chain")["burn_in"].sum().max()

    # Determine the layout of the subplots
    num_parameters = len(parameters)
    cols = min(
        max_plots_per_row, num_parameters
    )  # Number of columns is the smaller of max_plots_per_row or num_parameters
    rows = np.ceil(num_parameters / cols).astype(int)  # Calculate rows needed

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    fig.suptitle(title)

    if num_parameters == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, param in enumerate(parameters):
        ax = axes[i]

        sns.lineplot(
            x="iteration",
            y=param,
            hue="chain",
            data=mcmc_samples,
            ax=ax,
            palette="tab10",
            alpha=0.5,
        )
        ax.set_title(param)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(param)
        # Add vertical lines to indicate burn-in
        ax.axvline(x=burn_in_offset, color="red", linestyle="--", label="Burn-in offset")
        ax.legend(loc="upper right")

    # Remove empty subplots
    for i in range(num_parameters, len(axes)):
        fig.delaxes(axes[i])

    # Adjust layout
    fig.tight_layout()
    plt.show()

    if save:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
        name, ext = os.path.splitext(filename)
        filename = f"{name}_{timestamp}{ext}"
        fig.savefig(filename)
        print(f"Plot saved to {filename}")


def plot_samples_distribution(
    results_df,
    title="",
    max_plots_per_row=4,
    save=False,
    filename="plots/samples_distribution.png",
):
    """
    Plot the distribution of the samples of each parameter
    :param results_df: the results dataframe
    :param title: the title of the plot
    :max_plots_per_row: the maximum number of plots per row
    :param save: save the plot
    :param filename: the filename to save the plot
    :return: None
    """
    # sns.set(style="whitegrid")
    samples_ds = results_df.copy()
    # Exclude 'chain' and 'log_likelihood' from parameters to plot
    parameters = [col for col in samples_ds.columns if col not in ["chain", "burn_in", "log_likelihood"]]
    # exclude parameters with fixed values
    parameters = [param for param in parameters if len(samples_ds[param].unique()) > 1]
    # Determine the layout of the subplots
    num_parameters = len(parameters)
    cols = min(
        max_plots_per_row, num_parameters
    )  # Number of columns is the smaller of max_plots_per_row or num_parameters
    rows = np.ceil(num_parameters / cols).astype(int)  # Calculate rows needed

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    fig.suptitle(title)

    if num_parameters == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, param in enumerate(parameters):
        ax = axes[i]
        sns.histplot(samples_ds[param], kde=True, ax=ax)
        ax.set_title(param)
        ax.set_xlabel(param)
        ax.set_ylabel("Frequency")

    # Remove empty subplots
    for i in range(num_parameters, len(axes)):
        fig.delaxes(axes[i])

    # Adjust layout
    fig.tight_layout()
    plt.show()

    if save:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
        name, ext = os.path.splitext(filename)
        filename = f"{name}_{timestamp}{ext}"
        fig.savefig(filename)
        print(f"Plot saved to {filename}")
