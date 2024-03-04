from parameter_fitting import fit_with_MCMC, fit_with_random_search
from utils import calculate_bic
from  simulate import simulate
import numpy as np
import pandas as pd
from scipy.stats import uniform, expon
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
from datetime import datetime
# from tqdm import tqdm
# import the tqdm library combpatibilitle with jupyter notebook
from tqdm.notebook import tqdm

# Function to perform model recovery
def model_recovery(models_priors:dict, num_simulations=10, fit_method="random_search", seed=None, show_progress=True, **kwargs_for_fitting_functions):
    """
    Simulates data from each model and fits all models to the simulated data to recover the true model
    :param models_priors: A dictionary with the priors for each model
    :param num_simulations: Number of simulations to perform for each model
    :param fit_method: The method to use for fitting the models, either 'random_search' or 'MCMC'
    :param seed: Random seed for reproducibility
    :param show_progress: Whether to show the progress bar
    :param kwargs_for_fitting_functions: Additional arguments to pass to the fitting functions
    :return: A confusion matrix and an inversion matrix of the model recovery results
    """
    # set random seed for reproducibility
    np.random.seed(seed)
    # initialize lists to store the true and predicted model labels
    true_model_labels = []
    predicted_model_labels = []
    models = list(models_priors.keys())
    # get the parameter range for each model
    all_params_range = {model: get_param_range(param_dists) for model, param_dists in models_priors.items()}
    all_free_params = {model: get_free_params_names(param_range) for model, param_range in all_params_range.items()}
    # get the number of free parameters for each model
    num_free_params = {model: len(free_params) for model, free_params in all_free_params.items()}
    models_progress = tqdm(models, desc='models:', total=len(models), disable=not show_progress, leave=True, position=1)
    # print the number of free parameters for each model
    print('Number of free parameters for each model:')
    for model, num_params in num_free_params.items():
        print(f'{model}: {num_params}')
        
    models_progress.refresh()
    for true_model in models_progress:
        simulation_progress = tqdm(range(num_simulations), desc=f'simulating model recovery for: {true_model}', total=num_simulations, disable=not show_progress, leave=False, position=2)
        simulation_progress.refresh()
        for _ in simulation_progress:
            # Sample parameters from the true model's priors
            try:
                sampling_space = list(models_priors[true_model].items())
                params = {param: dist.rvs() for param, dist in sampling_space}
                # params = {param: dist.rvs() for param, dist in models_priors[true_model].items()}
            except AttributeError:
                params = {param: np.random.uniform(np.min(dist), np.max(dist)) for param, dist in models_priors[true_model].items()}
            # simulated_data, _ = simulate(agent_type=true_model, params=params, seed=seed)
            simulated_data, _ = simulate(agent_type=true_model, params=params)
            best_BIC = np.inf
            best_fit_model = None
            inner_model_comparison = tqdm(models, desc='models:', total=len(models), disable=not show_progress, leave=False, position=3)
            inner_model_comparison.refresh()
            for model in inner_model_comparison:
                # fit the model and compute the BIC
                params_range = all_params_range[model]
                if fit_method == "random_search":
                    fitted_params, best_LL, sampled_results_df = fit_with_random_search(params_range, simulated_data, agent_type=model,
                                                                                    num_iterations=kwargs_for_fitting_functions.get('num_iterations', 1000))
                elif fit_method == "MCMC":
                    fitted_params, best_LL, sampled_results_df = fit_with_MCMC(params_range, simulated_data, agent_type=model,
                                                                                num_samples=kwargs_for_fitting_functions.get('num_samples', 200),
                                                                                num_burn_in=kwargs_for_fitting_functions.get('num_burn_in', 50),
                                                                                num_chains=kwargs_for_fitting_functions.get('num_chains', 4))
                else:
                    raise ValueError(f"Unrecognized fitting method: {fit_method}, use either 'random_search' or 'MCMC'")
                
                num_params = num_free_params[model]
                num_data_points = len(simulated_data)
                BIC = calculate_bic(num_params, num_data_points, best_LL)
                if BIC < best_BIC:
                    best_BIC = BIC
                    best_fit_model = model
                
                # refresh the inner model comparison progress bar
                inner_model_comparison.refresh()
            
            true_model_labels.append(true_model)
            predicted_model_labels.append(best_fit_model)

            # refresh the simulation progress bar
            simulation_progress.refresh()

        # refresh the models progress bar
        models_progress.refresh()

    # compute confusion matrix
    conf_matrix = confusion_matrix(true_model_labels, predicted_model_labels, labels=models)
    conf_matrix_sum = conf_matrix.sum(axis=1).reshape(-1, 1)  # ensure it's a column vector
    conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix_sum

    # inversion matrix:
    # P(true_model | best_fit_model) = P(best_fit_model | true_model) * P(true_model) / P(best_fit_model)
    # under the assumption uniform prior over models: P(true_model) = 1 / num_models
    # the inversion matrix simplifies to:
    # P(true_model | best_fit_model) ∝ P(best_fit_model | true_model)
    # -> normalize each column of the confusion matrix results in the inversion matrix
    inversion_matrix = conf_matrix_normalized / conf_matrix_normalized.sum(axis=0)

    conf_matrix_normalized_df = pd.DataFrame(conf_matrix_normalized, index=models, columns=models)
    inversion_matrix_df = pd.DataFrame(inversion_matrix, index=models, columns=models)
    
    # unset the random seed
    np.random.seed(None)
    return conf_matrix_normalized_df, inversion_matrix_df

def get_free_params_names(param_range:dict):
    # exclude the fixed parameters -> min range == max range
    return {param for param, range in param_range.items() if range[0] != range[1]}

def get_param_range(param_dists: dict):
    param_ranges = {}
    for param, dist in param_dists.items():
        try:
            if dist.args[1] == 0:  # If 'scale' is 0 -> parameter is fixed
                param_ranges[param] = (dist.args[0], dist.args[0])
            else:
                # param_ranges[param] = (dist.ppf(0), dist.ppf(1))
                param_ranges[param] = (dist.args[0], dist.args[0] + dist.args[1])
        except (AttributeError, NotImplementedError):
            # Handle cases where dist is not a scipy.stats distribution or does not support PPF
            try:
                param_ranges[param] = (np.min(dist), np.max(dist))
            except (TypeError, ValueError) as e:
                # Fallback for unrecognized distribution types
                raise e("Unrecognized distribution type for parameter: {}".format(param))
    return param_ranges

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# plotting functions

def plot_confusion_and_inversion_matrix(conf_matrix:pd.DataFrame, inversion_matrix:pd.DataFrame, title:str, save=False, filename:str='plots/model_recovery.png', cmap='plasma'):
    # cmap = 'Blues'
    # cmap = 'viridis'
    # cmap = 'magma'
    # cmap = 'plasma'
    # cmap = 'cividis'

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle(title)
    sns.heatmap(conf_matrix, annot=True, cmap=cmap, cbar=False, ax=ax1)
    ax1.set_title('Confusion Matrix')
    ax1.set_xlabel('Predicted Model')
    ax1.set_ylabel('True Model')
    sns.heatmap(inversion_matrix, annot=True, cmap=cmap, cbar=False, ax=ax2)
    ax2.set_title('Inversion Matrix')
    ax2.set_xlabel('Predicted Model')
    ax2.set_ylabel('True Model')
    fig.suptitle(title)

    fig.tight_layout()
    plt.show()
    if save:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        # add timestamp to filename
        timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
        name, ext = os.path.splitext(filename)
        filename = f"{name}_{timestamp}{ext}"
        fig.savefig(filename)
        print(f'Plot saved to {filename}')