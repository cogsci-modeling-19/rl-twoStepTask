import numpy as np
import pandas as pd
import os
from datetime import datetime

def softmax(arr, beta):
    """
    Softmax function for action selection
    :param arr: The array of action values
    :param beta: Inverse temparature parameter for the softmax policy (higher beta -> more deterministic)
    :return: The probabilities of each action, sums up to 1
    """
    e_x = np.exp(beta * (arr - np.max(arr)))  # Subtract max value to prevent overflow
    return e_x / e_x.sum(axis=0)

def random_walk_gaussian(prob, sd, min_prob=0, max_prob=1):
    """
    Simulate a random walk in the reward probabilities using Gaussian noise
    :param prob: initial reward probabilities
    :param sd: standard deviation of the noise
    :param min_prob: minimum range
    :param max_prob: maximum range
    :return: new reward probabilities with added noise
    """
    new_prob = prob + np.random.normal(scale=sd, size=np.shape(prob))
    new_prob = np.clip(new_prob, min_prob, max_prob)
    return new_prob

def load_files_from_folder(folder_path, max_files=None, extension='.csv'):
    """
    Load CSV files from a specified folder.

    :param folder_path: Path to the folder containing CSV files.
    :param max_files: Maximum number of CSV files to load. If None, all files are loaded.
    :return: A list of pandas DataFrames.
    """
    csv_files = [f for f in os.listdir(folder_path) if f.endswith(extension)]
    
    # Sort files alphabetically to ensure consistent order
    csv_files.sort()
    
    # Limit the number of files to load, if max_files is specified
    if max_files is not None:
        csv_files = csv_files[:max_files]
    
    dataframes = []
    for filename in csv_files:
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        dataframes.append(df)
    
    return dataframes

# Load latest simulated data from csv
def load_latest_simulated_data(agent_type):
    data_folder = os.path.join("data", "simulated", agent_type)
    timestamped_folders = os.listdir(data_folder)
    timestamped_folders.sort()
    latest_folder = timestamped_folders[-1]
    filename = os.path.join(data_folder, latest_folder, "simulated_data.csv")
    print("Loading data from", filename)
    task_df = pd.read_csv(filename)
    return task_df

def save_simulated_data(task_df: pd.DataFrame, agent_type: str):
    """
    Save the simulated data to a csv file
    :param task_df: data as a dataframe
    :param agent_type: ['model_free', 'model_based', 'hybrid'] (used in path)
    :return: 
    """
    # save the data to a csv file
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    file_path = os.path.join("data", "simulated", agent_type, timestamp)
    # Create folder if it does not exist
    os.makedirs(file_path, exist_ok=True)
    filename = os.path.join(file_path, "simulated_data.csv")
    task_df.to_csv(filename, index=False)
    print("Data saved to", filename)

def convert_1d_numeric_string_array_to_array(string_array: str) -> np.array:
    return np.array([float(i) for i in string_array.strip('[]').replace(',', ' ').split()])

def detect_and_convert_1d_string_array(string_array: str):
    """
    Helper function to detect the type of the elements in a string array and convert them to the appropriate type
    Used for converting the rewardProbabilities and rewardDistribution columns from string to array
    :param string_array: array as a string, taken from experiment data
    :return: string converted to actual array
    """
    # Remove the outer brackets and split by ',' to handle both 1D and multidimensional arrays
    elements = string_array.strip('[]').replace(' ', '').split(',')
    # Attempt to determine the type of each element
    converted_elements = []
    for element in elements:
        if element.lower() in ['true', 'false']:
            # Convert string to boolean
            converted_elements.append(element.lower() == 'true')
        else:
            try:
                # Attempt to convert string to float
                converted_elements.append(float(element))
            except ValueError:
                # Handle the case where the conversion is not possible
                raise ValueError(f"Element {element} is neither a recognizable number nor a boolean value.")
    # Determine if the array is boolean or numeric based on the types of converted elements
    if all(isinstance(el, bool) for el in converted_elements):
        result_array = np.array(converted_elements, dtype=bool)
    elif all(isinstance(el, (int, float)) for el in converted_elements):
        # Convert elements to float if mixed types (e.g., boolean and numbers) or all numbers
        result_array = np.array(converted_elements, dtype=float)
    else:
        print(f"Array contains mixed types: {converted_elements}")
        result_array = np.array(converted_elements)
        print(f"Array converted to default type: {result_array.dtype}")
    return result_array

def preprocess_human_data(data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the human data from the experiments
    :param data_df: experiment data as a dataframe
    :return: preprocessed data
    """
    data = data_df.copy()
    
    # rename column reward_Param to rewardDistribution
    data.rename(columns={'rewards_Param': 'rewardDistribution'}, inplace=True)
    # infer common transition from the action taken in stage 1 and isHighProbOne/Two
    data['common_transition'] = np.where(data['stepOneChoice'] == 0,
                                            data['isHighProbOne'],
                                            data['isHighProbTwo'])

    # infer the state transition to from the action taken in stage 2
    data['state_transition_to'] = (data['stepTwoChoice'] // 2) + 1  # 1 if choice is 0 or 1. 2 if choice is 2 or 3

    # convert the rewardProbabilities from string to a array
    data['rewardProbabilities'] = data['rewardProbabilities'].apply(detect_and_convert_1d_string_array)
    
    # convert the rewardDistribution from string a array
    data['rewardDistribution'] = data['rewardDistribution'].apply(detect_and_convert_1d_string_array)
    
    # convert stepTwoChoice from range 0-3 to 0-1
    data['stepTwoChoice'] = data['stepTwoChoice'] % 2

    return data

def calculate_bic(num_params, num_data_points, ll):
    """
    Calculates Bayesian Information Criterion to be used in model comparison
    :param num_params: Number of free parameters that the model has
    :param num_data_points: Number of data points the model has been fitted to
    :param ll: Maximum log likelihood estimation for the model given data
    :return: BIC value
    """
    return num_params * np.log(num_data_points) - 2 * ll

def calculate_aic(num_params, ll):
    """
    Calculates Akaike Information Criterion to be used in model comparison
    :param num_params: Number of free parameters that the model has
    :param ll: Maximum log likelihood estimation for the model given data
    :return:
    """
    return 2 * num_params - 2 * ll