from utils import preprocess_human_data
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from datetime import datetime
from IPython.display import display
from tqdm.notebook import tqdm

def calc_plot_stay_probabilities(dfs: list[pd.DataFrame], labels: list[str]=None, title='', plot=True, return_df=True, max_plots_per_row=4, save=False, filename="plots/stay_probabilities.png"):
    """
    Calculate and plot the stay probabilities for the given dataframes
    :param dfs: List of dataframes
    :param labels: Labels for the dataframes
    :param title: Title for the plot
    :param plot: Plot the data if True
    :param return_df: Return the dataframes if True
    :param max_plots_per_row: Maximum number of plots per row
    :param save: Save the plot if True
    :param filename: Filename for the plot (if save=True)
    :return: dictioanry of stay probabilities for each dataframe and label
    """
    if isinstance(dfs, pd.DataFrame) or not isinstance(dfs, list):
        dfs = [dfs]
    if isinstance(labels, str) or not isinstance(labels, list):
        labels = [labels]
    if not labels or len(labels) != len(dfs):
        print("Labels not provided or not matching the number of dataframes, using default labels")
        labels = [f"data {i+1}" for i in range(len(dfs))]
        print('generated labels:', labels)
    
    # calculate stay probabilities
    stay_probabilities = [calculate_stay_probability(df) for df in dfs]
    # plot the stay probabilities
    if plot:
        plot_stay_probabilities(dfs=stay_probabilities, labels=labels, title=title, max_plots_per_row=max_plots_per_row, save=save, filename=filename)
    
    stay_probabilities = dict(zip(labels, stay_probabilities))
    
    if return_df:
        return stay_probabilities

def calc_plot_stay_probabilities_blocks(dfs: list[pd.DataFrame], labels: list[str]=None, title='', num_blocks=4, plot=True, return_df=True, max_plots_per_row=4, save=False, filename="plots/stay_probabilities_blocks.png"):
    
    if isinstance(dfs, pd.DataFrame) or not isinstance(dfs, list):
        dfs = [dfs]
    if isinstance(labels, str) or not isinstance(labels, list):
        labels = [labels]
    if not labels or len(labels) != len(dfs):
        print("Labels not provided or not matching the number of dataframes, using default labels")
        labels = [f"data {i+1}" for i in range(len(dfs))]
        print('generated labels:', labels)

    # calculate stay probabilities
    stay_probabilities_blocks = [calculate_stay_probability_blocks(df, num_blocks=num_blocks) for df in dfs]
    # plot the stay probabilities
    if plot:
        plot_stay_probabilities_progression(dfs=stay_probabilities_blocks, labels=labels, title=title,
                                            max_plots_per_row=max_plots_per_row, save=save, filename=filename)
    if return_df:
        return stay_probabilities_blocks

def calc_plot_stay_probabilities_moving_average(dfs: list[pd.DataFrame], labels: list[str]=None, title='', window_size=50, plot=True, return_df=True, max_plots_per_row=4, save=False, filename="plots/stay_probabilities_ma.png"):
    if isinstance(dfs, pd.DataFrame) or not isinstance(dfs, list):
        dfs = [dfs]
    if isinstance(labels, str) or not isinstance(labels, list):
        labels = [labels]
    if not labels or len(labels) != len(dfs):
        print("Labels not provided or not matching the number of dataframes, using default labels")
        labels = [f"data {i+1}" for i in range(len(dfs))]
        print('generated labels:', labels)

    # calculate stay probabilities
    stay_probabilities_moving_average = [calculate_stay_probability_moving_average(df, window_size=window_size)
                                        for df in dfs]
    # plot the stay probabilities
    if plot:
        plot_stay_probabilities_progression(dfs=stay_probabilities_moving_average, labels=labels, title=title,
                                            max_plots_per_row=max_plots_per_row, save=save, filename=filename)
    if return_df:
        return stay_probabilities_moving_average

def calc_plot_running_stay_probabilities(dfs: list[pd.DataFrame], labels: list[str]=None, title='', window_size=50, plot=True, return_df=True, max_plots_per_row=4, save=False, filename="plots/running_stay_probabilities.png"):
    if isinstance(dfs, pd.DataFrame) or not isinstance(dfs, list):
        dfs = [dfs]
    if isinstance(labels, str) or not isinstance(labels, list):
        labels = [labels]
    if not labels or len(labels) != len(dfs):
        print("Labels not provided or not matching the number of dataframes, using default labels")
        labels = [f"data {i+1}" for i in range(len(dfs))]
        print('generated labels:', labels)
    
    # calculate stay probabilities
    running_stay_probabilities = [calculate_running_stay_probabilities(df) for df in dfs]
    # plot the stay probabilities
    if plot:
        plot_running_stay_probabilities(running_stay_probabilities, labels=labels, title=title, window_size=window_size,
                                        max_plots_per_row=max_plots_per_row, save=save, filename=filename)
    if return_df:        
        return running_stay_probabilities

def calc_plot_running_average_cumulative_reward(dfs: list[pd.DataFrame], labels: list[str]=None, title='', window_size=50, plot=True, return_df=True, max_plots_per_row=4, save=False, filename="plots/average_cumulative_reward.png"):
    if isinstance(dfs, pd.DataFrame) or not isinstance(dfs, list):
        dfs = [dfs]
    if isinstance(labels, str) or not isinstance(labels, list):
        labels = [labels]
    if not labels or len(labels) != len(dfs):
        print("Labels not provided or not matching the number of dataframes, using default labels")
        labels = [f"data {i+1}" for i in range(len(dfs))]
        print('generated labels:', labels)
    
    # calculate the moving average of the reward
    running_average_cumulative_reward = [calculate_average_cumulative_reward_moveing_average(df, window_size=window_size)
                                        for df in dfs]
    # plot the moving average of the reward
    if plot:
        plot_running_average_cumulative_reward(running_average_cumulative_reward, labels=labels, title=title, max_plots_per_row=max_plots_per_row, save=save, filename=filename)
    if return_df:
        return running_average_cumulative_reward

def calc_plot_stay_probability_paired_diffs(sampled_data_lists, model_titles=None, title='', plot=True, return_df=True, max_plots_per_row=3, save=False, filename='plots/stay_prob_diffs.png'):
    n_plots = len(sampled_data_lists)
    if model_titles is None:
        model_titles = [f'Model {i}' for i in range(len(sampled_data_lists))]
    mean_diffs_all_models = calculate_stay_probability_paired_diffs(sampled_data_lists, model_titles)
    if plot:
        plot_stay_prob_paired_diffs(mean_diffs_all_models, model_titles, title=title, max_plots_per_row=max_plots_per_row, save=save, filename=filename)
    if return_df:
        return mean_diffs_all_models

def print_simple_task_summary(data: pd.DataFrame, title="", full=False):
    """
    Print some simple statistics about the data
    :param data: pandas dataframe
    :param title: string
    :param full: print full statistics if True
    :return: 
    """
    # print some statistics 
    # print_simple_statistics(task_df)
    task_df = data.copy()
    print("###", title)

    print("common transitions percentage in the task:", np.mean(task_df["common_transition"])*100, "%")
    print("rewarded trails percentage:", np.mean(task_df["reward"] > 0)*100, "%")

    if full:
        print("transition percentage from state 0 action 0 to state 1:", np.mean(task_df[task_df["stepOneChoice"] == 0]["state_transition_to"] == 1)*100, "%")
        print("transition percentage from state 0 action 1 to state 2:", np.mean(task_df[task_df["stepOneChoice"] == 1]["state_transition_to"] == 2)*100, "%")
        # get the counts of state transitions and action in stage 1
        counts_stage_2_action_1 = pd.DataFrame({
            'State Counts': task_df['state_transition_to'].value_counts(),
            'Action Stage 1 Counts': task_df['stepOneChoice'].value_counts()
        })
        display(counts_stage_2_action_1)

        mean_reward = task_df.groupby('state_transition_to')['reward'].mean().reset_index()
        mean_reward.columns = ['state_transition_to', 'mean_reward']
        display(mean_reward)

        # get the reward probability distributions for the final stage (2)
        def index_reward_probabilities(row):
            try:
                return row['rewardProbabilities'][row['state_transition_to'] + row['stepTwoChoice']]
            except:
                print(row)

        task_df_tmp = task_df.copy()
        task_df_tmp['mean_reward_prob_stepTwoChoice'] = task_df.apply(index_reward_probabilities, axis=1)
        reward_probabilities = task_df_tmp.groupby(['state_transition_to', 'stepTwoChoice'])['mean_reward_prob_stepTwoChoice'].mean().reset_index()
        display(reward_probabilities)

def calculate_repated_action(data: pd.DataFrame) -> pd.DataFrame:
    # get a copy of the data
    tmp_df = data.copy()

    # flag for the repeated action (stage 1), same action as the previous trial
    tmp_df['repeated_stepOneAction'] = tmp_df['stepOneChoice'].shift(1) == tmp_df[
        'stepOneChoice']
    
    # flag for the repeated action (stage 1), same action as the next trial (to calculate the stay probability)
    tmp_df['repeated_stepOneAction_next'] = tmp_df['repeated_stepOneAction'].shift(-1)
    
    # discard last trial (no next trial to compare with)
    tmp_df = tmp_df.iloc[:-1]

    return tmp_df

def calculate_stay_probability(data: pd.DataFrame) -> pd.DataFrame:
    # get a copy of the data
    tmp_df = data.copy()
    tmp_df = calculate_repated_action(tmp_df)

    # stay probabilities based on conditions
    # 2 factors:
    #       rewarded trail ( whether the reward in stage 2 is greater than )
    #       common_transition ( whether the transition from stage 1 to stage 2 is common or rare)
    results = tmp_df.groupby(['reward', 'common_transition'])[
        'repeated_stepOneAction_next'].mean().reset_index()

    # rename columns for clarity
    results.rename(
        columns={'repeated_stepOneAction_next': 'Stay Probability', 'reward': 'Rewarded',
                 'common_transition': 'Common'}, inplace=True)

    conditions = {
        (True, True): 'rewarded_common',
        (True, False): 'rewarded_rare',
        (False, True): 'unrewarded_common',
        (False, False): 'unrewarded_rare'
    }
    results['Condition'] = results.apply(
        lambda row: conditions[(row['Rewarded'], row['Common'])], axis=1)

    # rounding the stay probabilities
    results['Stay Probability'] = results['Stay Probability'].apply(
        lambda x: np.round(x, 3))

    return results

def calculate_stay_probability_blocks(data: pd.DataFrame, num_blocks=4) -> pd.DataFrame:
    
    tmp_df = calculate_repated_action(data) # returns a df with one less row
    total_trials = len(tmp_df) 

    if num_blocks < 2 or num_blocks > total_trials//2:
        raise ValueError(f"num_blocks must be a positive integer less than half the number of trials {total_trials + 1}, but got {num_blocks}")
        
    trials_per_block = total_trials // num_blocks
    additional_trials = total_trials % num_blocks  # excess trials that don't evenly divide

    stay_prob_blocks = pd.DataFrame(index=range(num_blocks),
                                    columns=['rewarded_common', 'rewarded_rare', 'unrewarded_common', 'unrewarded_rare'])
    
    for block in range(num_blocks):
        start_index = block * trials_per_block + min(block, additional_trials)  # adjust start index for uneven blocks
        if block < additional_trials:  # distribute the excess trials evenly across the first blocks
            end_index = start_index + trials_per_block + 1
        else:
            end_index = start_index + trials_per_block
        
        block_df = tmp_df.iloc[start_index:end_index]
        block_stay_prob = calculate_stay_probability(block_df)
        
        # ensure a value is set even if a condition is missing in the block
        for condition in stay_prob_blocks.columns:
            condition_df = block_stay_prob[block_stay_prob['Condition'] == condition]
            if not condition_df.empty:
                stay_prob_blocks.at[block, condition] = condition_df['Stay Probability'].values[0]
            else:
                stay_prob_blocks.at[block, condition] = 0  # Or a default value, e.g., 0 or np.nan
    
    return stay_prob_blocks

def calculate_stay_probability_moving_average(data: pd.DataFrame, window_size=10) -> pd.DataFrame:
    # Ensure the repeated actions are calculated
    tmp_df = calculate_repated_action(data)

    # Define conditions
    conditions = ['rewarded_common', 'rewarded_rare', 'unrewarded_common', 'unrewarded_rare']
    condition_columns = {condition: f'{condition}' for condition in conditions}

    # Initialize columns for the moving averages of each condition
    for column in condition_columns.values():
        tmp_df[column] = np.nan

    # Define conditions based on 'reward' and 'common_transition' columns
    tmp_df['Condition'] = np.where((tmp_df['reward'] == True) & (tmp_df['common_transition'] == True), 'rewarded_common',
                                   np.where((tmp_df['reward'] == True) & (tmp_df['common_transition'] == False), 'rewarded_rare',
                                            np.where((tmp_df['reward'] == False) & (tmp_df['common_transition'] == True), 'unrewarded_common',
                                                     'unrewarded_rare')))

    # Calculate the moving average for each condition within the window
    for index, row in tmp_df.iterrows():
        start_index = max(0, index - window_size + 1)
        window_df = tmp_df[start_index:index+1]  # Select rows for the current window
        for condition in conditions:
            condition_df = window_df[window_df['Condition'] == condition]  # Filter rows for the current condition
            if not condition_df.empty:
                # Calculate and assign the moving average for the current condition and trial
                tmp_df.at[index, condition_columns[condition]] = condition_df['repeated_stepOneAction_next'].mean()

    return tmp_df

def calculate_running_stay_probabilities(data):
    task_df = data.copy()
    # Initialize columns for stay decisions and running probabilities
    task_df['stay_decision'] = False
    task_df['common_rewarded_prob'] = 0.0
    task_df['common_unrewarded_prob'] = 0.0
    task_df['rare_rewarded_prob'] = 0.0
    task_df['rare_unrewarded_prob'] = 0.0

    # Trackers for calculating running probabilities
    stay_counts = {'common_rewarded': 0, 'common_unrewarded': 0, 'rare_rewarded': 0, 'rare_unrewarded': 0}
    total_counts = {'common_rewarded': 0, 'common_unrewarded': 0, 'rare_rewarded': 0, 'rare_unrewarded': 0}

    task_df = calculate_repated_action(task_df)

    for i in range(0, len(task_df)):
        current = task_df.iloc[i]
        # prev = task_df.iloc[i-1]

        # # Check if the participant stayed with the same choice
        # if current['stepOneChoice'] == prev['stepOneChoice']:
        if current['repeated_stepOneAction_next']:
            task_df.loc[i, 'stay_decision'] = True

        condition = ('common_' if current['common_transition'] else 'rare_') + ('rewarded' if current['reward'] else 'unrewarded')

        # Update counts
        if task_df.loc[i, 'stay_decision']:
            stay_counts[condition] += 1
        total_counts[condition] += 1

        # Calculate running probabilities
        for key in stay_counts:
            if total_counts[key] > 0:
                task_df.loc[i, key + '_prob'] = stay_counts[key] / total_counts[key]
        
    # rename columns for consistency
    task_df.rename(columns={'common_rewarded_prob': 'rewarded_common', 'common_unrewarded_prob': 'unrewarded_common',
                            'rare_rewarded_prob': 'rewarded_rare', 'rare_unrewarded_prob': 'unrewarded_rare'}, inplace=True)

    return task_df

def calculate_stay_probability_paired_diffs(sampled_data_lists, model_titles):
    ## Initialize a dictionary to hold the mean differences for each model type
    mean_diffs_all_models = {}
    
    # Iterate over each list of sampled data DataFrames and their corresponding title
    for sampled_data, title in zip(sampled_data_lists, model_titles):
        diffs = []
        # Calculate differences for each DataFrame in the current list
        for stay_prob_df in sampled_data:
            diff = {}
            rewarded_common = stay_prob_df.loc[(stay_prob_df['Rewarded']==True) & (stay_prob_df['Common']==True), 'Stay Probability'].values[0]
            rewarded_rare = stay_prob_df.loc[(stay_prob_df['Rewarded']==True) & (stay_prob_df['Common']==False), 'Stay Probability'].values[0]
            unrewarded_common = stay_prob_df.loc[(stay_prob_df['Rewarded']==False) & (stay_prob_df['Common']==True), 'Stay Probability'].values[0]
            unrewarded_rare = stay_prob_df.loc[(stay_prob_df['Rewarded']==False) & (stay_prob_df['Common']==False), 'Stay Probability'].values[0]
            diff['rewarded_common'] = rewarded_common
            diff['diff_rewarded_rare'] = rewarded_rare - rewarded_common
            diff['unrewarded_common'] = unrewarded_common
            diff['diff_unrewarded_rare'] = unrewarded_rare - unrewarded_common
            diff['diff_rewarded_unrewarded'] = (rewarded_common - unrewarded_common) + (rewarded_rare - unrewarded_rare)
            diffs.append(diff)
        
        # Calculate mean differences for the current model type
        mean_diffs = {key: np.mean([d[key] for d in diffs]) for key in diffs[0].keys()}
        mean_diffs_all_models[title] = mean_diffs
    
    return mean_diffs_all_models

def calculate_average_cumulative_reward_moveing_average(data: pd.DataFrame, window_size=10) -> pd.DataFrame:
    # calculate the moving average of the reward
    tmp_df = data.copy()
    tmp_df['avg_cumolative_reward'] = tmp_df['reward'].rolling(window=window_size, min_periods=1).mean()
    return tmp_df

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# plotting functions

def plot_running_average_cumulative_reward(dfs: list[pd.DataFrame], labels: list[str]=None, title='', max_plots_per_row=4, save=False, filename="plots/average_cumulative_reward.png"):
    # sns.set_style("whitegrid")
    
    if isinstance(dfs, pd.DataFrame) or not isinstance(dfs, list):
        dfs = [dfs]  # Wrap the single DataFrame in a list
    
    if labels is not None and not isinstance(labels, list):
        labels = [labels]

    n_plots = len(dfs)
    # Calculate the number of rows and columns for the subplot grid
    rows = (n_plots - 1) // max_plots_per_row + 1  # Ensure at least one row
    cols = min(n_plots, max_plots_per_row)  # Max of 4 columns
    
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(6*cols, 6*rows), sharey=True, sharex=True)
    fig.suptitle(title)
    
    # If there's only one subplot, axes won't be an array, so we wrap it in a list for consistency
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    if labels is None:
        labels = [f"Plot {i}" for i in range(len(dfs))]
    if len(labels) < len(dfs):
        labels = labels + [f"Plot {i}" for i in range(len(dfs) - len(labels))]
    
    for i, data in enumerate(dfs):
        ax = axes[i]
        df = data.copy()
        # Create the line plot
        ax.plot(df.index, df['avg_cumolative_reward'], label='Average Cumulative Reward', linestyle='-', color='blue')

        ax.set_title(labels[i], fontsize=20)

        # Set the size of the x and y ticks labels
        ax.tick_params(labelsize=12)

        if i >= (n_plots - max_plots_per_row):
            ax.set_xlabel('Trial', fontsize=15)
        else:
            # Hide the x-axis label
            ax.set_xlabel('', visible=False)
        if i%max_plots_per_row == 0:
            ax.set_ylabel('Average Cumulative Reward', fontsize=15)
        else:
            # Hide the y-axis label
            ax.set_ylabel('', visible=False)
        
        # plot a line at y=0.5
        ax.axhline(y=0.5, color='r', linestyle='--', label='Chance Level')

    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    plt.show()
    if save:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        # add timestamp to filename
        timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
        name, ext = os.path.splitext(filename)
        filename = f"{name}_{timestamp}{ext}"
        fig.savefig(filename)
        print(f'Plot saved to {filename}')

def plot_stay_probabilities(dfs: list[pd.DataFrame], labels: list[str]=None, title='', max_plots_per_row=4, save=False, filename="plots/stay_probabilities.png"):
    # sns.set_style("whitegrid")
    
    if isinstance(dfs, pd.DataFrame) or not isinstance(dfs, list):
        dfs = [dfs]  # Wrap the single DataFrame in a list
    
    if labels is not None and not isinstance(labels, list):
        labels = [labels]

    n_plots = len(dfs)
    # Calculate the number of rows and columns for the subplot grid
    rows = (n_plots - 1) // max_plots_per_row + 1  # Ensure at least one row
    cols = min(n_plots, max_plots_per_row)  # Max of 4 columns
    
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(6*cols, 6*rows), sharey=True, sharex=True)
    fig.suptitle(title)

    # If there's only one subplot, axes won't be an array, so we wrap it in a list for consistency
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    if labels is None:
        labels = [f"Plot {i}" for i in range(len(dfs))]
    if len(labels) < len(dfs):
        labels = labels + [f"Plot {i}" for i in range(len(dfs) - len(labels))]
    
    min_stay_prob = np.min([data['Stay Probability'].min() for data in dfs])
    y_limit_min = 0.5 if min_stay_prob > 0.5 else min_stay_prob - 0.1

    for i, data in enumerate(dfs):
        ax = axes[i]
        df = data.copy()
        # Convert 'Rewarded' to a string type for clear plotting
        df['Rewarded'] = df['Rewarded'].map({True: 'Rewarded', False: 'Unrewarded'})
        df['Common'] = df['Common'].map({True: 'Common', False: 'Rare'})
        # Create the bar plot
        bar = sns.barplot(x='Rewarded', y='Stay Probability', hue='Common',
                        data=df, ax=ax,
                        order=['Rewarded', 'Unrewarded'],
                        hue_order=['Common', 'Rare'])

        # Set the y-axis limit
        ax.set_ylim(y_limit_min, 1)

        ax.set_title(labels[i], fontsize=20)

        # Set the size of the legend and the title of the legend
        ax.legend(title_fontsize='13', fontsize='12')

        # Set the size of the x and y ticks labels
        ax.tick_params(labelsize=12)

        # Add percentages on top of each bar
        for p in bar.patches:
            bar.annotate(format(p.get_height(), '.2f'),
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center',
                        xytext=(0, 10),
                        textcoords='offset points', fontsize=12)
            
        if i >= (n_plots - max_plots_per_row):
            ax.set_xlabel('Reward', fontsize=15)
        else:
            # Hide the x-axis label
            ax.set_xlabel('', visible=False)
        if i%max_plots_per_row == 0:
            ax.set_ylabel('Stay Probability', fontsize=15)
        else:
            # Hide the y-axis label
            ax.set_ylabel('', visible=False)

    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    plt.show()
    if save:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        # add timestamp to filename
        timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
        name, ext = os.path.splitext(filename)
        filename = f"{name}_{timestamp}{ext}"
        fig.savefig(filename)
        print(f'Plot saved to {filename}')


def plot_stay_probabilities_progression(dfs: list[pd.DataFrame], title='', labels: list[str]=None, max_plots_per_row=4, save=False, filename="plots/stay_probabilities_ma.png"):
    
    if isinstance(dfs, pd.DataFrame) or not isinstance(dfs, list):
        dfs = [dfs]
    if labels is not None and not isinstance(labels, list):
        labels = [labels]
    
    if labels is None:
        labels = [f"Plot {i}" for i in range(len(dfs))]
    if len(labels) < len(dfs):
        labels = labels + [f"Plot {i}" for i in range(len(dfs) - len(labels))]

    n_plots = len(dfs)
    # Calculate the number of rows and columns for the subplot grid
    rows = (n_plots - 1) // max_plots_per_row + 1  # Ensure at least one row
    cols = min(n_plots, max_plots_per_row)  # Max of 4 columns

    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(6*cols, 6*rows), sharey=True, sharex=True)
    fig.suptitle(title)

    # If there's only one subplot, axes won't be an array, so we wrap it in a list for consistency
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # plot the moving averages for all 4 conditions over the trials
    for i, ax in enumerate(axes):
        if i < n_plots:
            df = dfs[i]

            ax.plot(df.index, df['rewarded_common'], label='Rewarded Common', linestyle='-', color='blue')
            ax.plot(df.index, df['rewarded_rare'], label='Rewarded Rare', linestyle='-', color='orange')
            ax.plot(df.index, df['unrewarded_common'], label='Unrewarded Common', linestyle='--', color='blue')
            ax.plot(df.index, df['unrewarded_rare'], label='Unrewarded Rare', linestyle='--', color='orange')
            
            ax.set_title(labels[i])
            ax.set_xlabel('Trial')
            ax.set_ylabel('Stay Probability MA')
            ax.legend(loc='best')

        else:
            ax.axis('off')  # Hide unused subplots

    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    plt.show()
    if save:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        # add timestamp to filename
        timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
        name, ext = os.path.splitext(filename)
        filename = f"{name}_{timestamp}{ext}"
        fig.savefig(filename)
        print(f'Plot saved to {filename}')

def plot_running_stay_probabilities(dfs:list, labels:list=None,window_size=1, max_plots_per_row=3, title='', save=False, filename="plots/running_step_probabilities.png"):
    
    if isinstance(dfs, pd.DataFrame) or not isinstance(dfs, list):
        dfs = [dfs]
    if labels is not None and not isinstance(labels, list):
        labels = [labels]

    # Create a copy of the DataFrame to avoid modifying the original
    n_plots = len(dfs)
    # Calculate the number of rows and columns for the subplot grid
    rows = (n_plots - 1) // max_plots_per_row + 1  # Ensure at least one row
    cols = min(n_plots, max_plots_per_row)  # Max of 4 columns
    
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(6*cols, 6*rows), sharey=True, sharex=True)
    fig.suptitle(title)
    
    # If there's only one subplot, axes won't be an array, so we wrap it in a list for consistency
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    if labels is None:
        labels = [f"Plot {i}" for i in range(len(dfs))]
    if len(labels) < len(dfs):
        labels = labels + [f"Plot {i}" for i in range(len(dfs) - len(labels))]
    
    for i, data in enumerate(dfs):
        ax = axes[i]
        df_copy = data.copy()

        # Calculate moving averages on the copy
        df_copy['common_rewarded_prob_ma'] = df_copy['rewarded_common'].rolling(window=window_size, min_periods=1).mean()
        df_copy['rare_rewarded_prob_ma'] = df_copy['rewarded_rare'].rolling(window=window_size, min_periods=1).mean()
        df_copy['common_unrewarded_prob_ma'] = df_copy['unrewarded_common'].rolling(window=window_size, min_periods=1).mean()
        df_copy['rare_unrewarded_prob_ma'] = df_copy['unrewarded_rare'].rolling(window=window_size, min_periods=1).mean()

        # Plot each condition's moving average from the copied DataFrame
        ax.plot(df_copy['trial_index'], df_copy['common_rewarded_prob_ma'], label='Rewarded Common (MA)', linestyle='-', color='b')
        ax.plot(df_copy['trial_index'], df_copy['rare_rewarded_prob_ma'], label='Rewarded Rare (MA)', linestyle='-', color='orange')
        ax.plot(df_copy['trial_index'], df_copy['common_unrewarded_prob_ma'], label='Unrewarded Common (MA)', linestyle='--', color='b')
        ax.plot(df_copy['trial_index'], df_copy['rare_unrewarded_prob_ma'], label='Unrewarded Rare (MA)', linestyle='--', color='orange')

        ax.set_title(labels[i])
        ax.legend()
        ax.grid(True)

        if i >= (n_plots - max_plots_per_row):
            ax.set_xlabel('Trial Index', fontsize=15)
        else:
            # Hide the x-axis label
            ax.set_xlabel('', visible=False)
        if i%max_plots_per_row == 0:
            ax.set_ylabel('Running Stay Probability (MA)', fontsize=15)
        else:
            # Hide the y-axis label
            ax.set_ylabel('', visible=False)

    plt.show()
    if save:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        # add timestamp to filename
        timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
        name, ext = os.path.splitext(filename)
        filename = f"{name}_{timestamp}{ext}"
        fig.savefig(filename)
        print(f'Plot saved to {filename}')

def plot_stay_prob_paired_diffs(mean_diffs_all_models, model_titles, title='', max_plots_per_row=3, save=False, filename='plots/stay_prob_diffs.png'):
    n_plots = len(model_titles)
    if n_plots != len(model_titles):
        raise ValueError('sampled_data_lists and model_titles must have the same length')
    
    # Calculate the number of rows and columns for the subplot grid
    rows = (n_plots - 1) // max_plots_per_row + 1  # Ensure at least one row
    cols = min(n_plots, max_plots_per_row) 
    # Plot the differences for each model type
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows), sharey=True)
    fig.suptitle(title)
    if n_plots == 1:  # If there's only one model, ensure axes is iterable
        axes = np.array([axes])
    axes = axes.flatten()

    # plot the differences for each model
    for ax, title in zip(axes, model_titles):
        mean_diffs = mean_diffs_all_models[title]
        # execlude the rewarded_common and unrewarded_common
        mean_diffs.pop('rewarded_common', None)
        mean_diffs.pop('unrewarded_common', None)
        ax.bar(mean_diffs.keys(), mean_diffs.values())
        ax.set_title(title)
        xticks_labels = ax.get_xticklabels()
        ax.set_xticklabels(xticks_labels, rotation=45)
        ax.axhline(0, color='grey', linewidth=0.8)

    plt.tight_layout()
    plt.show()
    if save:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        # add timestamp to filename
        timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
        name, ext = os.path.splitext(filename)
        filename = f"{name}_{timestamp}{ext}"
        fig.savefig(filename)
        print(f'Plot saved to {filename}')

if __name__ == "__main__":
    # test the function
    # load and inspect some human data
    file_names = ['experiment_data_mj.csv', 'experiment_data_andrei.csv', 
                'experiment_data_SeEun.csv', 'experiment_data.csv', 'two_steps_experiment_data_muhip.csv']
    stay_prob_list = []
    human_data_list = [] 
    for file_name in file_names:
        file_name = os.path.join("data", "participants", file_name)
        human_data = pd.read_csv(file_name)
        human_data = preprocess_human_data(human_data)
        human_data_list.append(human_data)
        
        stay_probability_h, _ = calculate_stay_probability(human_data)
        stay_prob_list.append(stay_probability_h)

    stay_prob_list_1 = stay_prob_list.copy()
    calc_plot_stay_probability_paired_diffs([stay_prob_list, stay_prob_list_1], max_plots_per_row=3)