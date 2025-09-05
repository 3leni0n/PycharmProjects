import os
from pathlib import Path
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import ssm


# Define a function to parse the data for GLM-HMM
def parse_glmhmm(df, at=False):
    """
    Parse the data for GLM-HMM.
    :param df: DataFrame containing the data (one or many sessions concatenated).
    :return: inputs, choices
    """

    inputs = []
    choices = []

    for session_id, df_session in df.groupby('Session'):
        n_trials = len(df_session)
        stim_vals = df_session.ILD.values
        stim_vals = stim_vals / abs(df.ILD.max())  # Normalize ILD to [-1, 1]
        bias = np.ones(n_trials)

        if at:
            action_trace = get_action_trace(df_session)
            session_input = np.column_stack((stim_vals, bias, action_trace))
        else:
            session_input = np.column_stack((stim_vals, bias))

        session_choices = df_session.Choice.values.astype(int)[:, None]
        inputs.append(session_input)
        choices.append(session_choices)

    return inputs, choices


def get_action_trace(df, max_trial_lag=10, tau=2):
    """
    Computes the action trace for each trial in the DataFrame. The action trace is an exponentially weighted sum of past choices, where more recent choices have a greater influence. The weights decay exponentially with a time constant tau. Output is normalized between -1 and +1.
    :param df: DataFrame containing the data with a 'Choice' column (0=left;1=right)
    :param max_trial_lag: Number of past trials to consider
    :param tau: Decay constant
    :return: List of action trace values for each trial
    """

    lags = np.arange(1, max_trial_lag + 1)  # Lags from 1 to k
    weights = np.exp(-lags / tau)  # Exponential decay weights
    Z = np.sum(weights)  # Fixed normalizer

    action_trace = []
    for t in range(len(df)):
        past_choices = df.loc[max(0, t-max_trial_lag):t-1, 'Choice']
        past_signed_choices = 2 * past_choices.to_numpy() - 1  # map 0→-1, 1→+1
        effective_weights = weights[:len(past_signed_choices)]
        weighted_sum = np.sum(past_signed_choices * effective_weights)
        # action_trace.append(weighted_sum / np.sum(effective_weights))
        action_trace.append(weighted_sum / Z)

    return action_trace


def compute_window(data, win_length=20):
    """
    Computes a rolling average with a length of window samples.
    """

    roll_avg = []
    for i in range(len(data)):
        if i < win_length:
            roll_avg.append(round(np.nanmean(data[0:i + 1]), 2))
        else:
            roll_avg.append(round(np.nanmean(data[i - win_length + 1:i + 1]), 2))
    return roll_avg


def interpret_states(weights):
    """
    Assigns labels to GLM-HMM states based on their weights.
    :param weights: GLM-HMM weights of shape (n_states, obs_dim, input_dim)
    :return: Dictionary mapping state index to label
    """

    stim_weights = weights[:, 0, 0]
    bias_weights = weights[:, 0, 1]

    # Engaged = max |stim|
    engaged = np.argmax(np.abs(stim_weights))

    # The other two states
    others = [s for s in range(len(weights)) if s != engaged]
    biased_left = others[np.argmin(bias_weights[others])]
    biased_right = others[np.argmax(bias_weights[others])]

    # Labels dictionary (original indexing)
    state_labels = {engaged: 'engaged',
                    biased_left: 'left bias',
                    biased_right: 'right bias'}

    remap = np.array([engaged, biased_left, biased_right])

    # Print state labels and weights
    for state, label in state_labels.items():
        print(f'State {state + 0}: {label} '
              f'(stim. weight: {stim_weights[state]:.2f}, '
              f'bias weight: {bias_weights[state]:.2f})')

    return state_labels, remap


def apply_remap(remap, weights, trans_mat, posterior_probs, state_labels):
    """
    Remap posterior probabilities and most likely states to consistent labeling.

    :param posterior_probs: N sessions list of np.ndarray of shape (n_trials, n_states)
    :param state_max_posterior: np.ndarray of shape (n_trials,), argmax state indices
    :param remap: np.ndarray of shape (n_states,), mapping old -> new
    :param state_labels: dict, mapping old state idx -> label
    :return: posterior_probs_remapped, state_max_posterior_remapped, ordered_labels
    """

    weights = weights[remap]  # Remap weights
    trans_mat = trans_mat[remap][:, remap]  # Remap transition matrix
    posterior_probs = [p[:, remap] for p in posterior_probs]  # Remap posterior probabilities
    # posterior_probs_concat = posterior_probs_concat[:, remap]
    # state_max_posterior = remap[state_max_posterior]  # Remap most likely state
    state_labels = [state_labels[old] for old in remap]  # Remap labels

    return weights, trans_mat, posterior_probs, state_labels


def filter_behavior(df):
    """
    Filter the behavior DataFrame for one subject.
    :param df: DataFrame containing the data
    :return: Filtered DataFrame
    """

    # Filters for groups 1-3
    # df = df[df.Stage == 4].reset_index(drop=True)
    # df = df[df.Motor == 4].reset_index(drop=True)
    # df = df[df.StimDur == 1].reset_index(drop=True)
    df = df[df.P > 0].reset_index(drop=True)

    # Drop misses (Choice == NaN)
    df = df.dropna(subset=['Choice']).reset_index(drop=True)

    return df


def fit_glm_hmm(df, save=False):
    """
    Fit GLM-HMM to the data of one subject.
    :param df: DataFrame containing the data of one subject
    :return: DataFrame with added 'State' column indicating most likely state per trial
    """

    # Set GLM-HMM parameters
    n_states = 3  # Number of discrete states (from Ashwood et al. 2020)
    obs_dim = 1  # Number of observed dimensions (1 for binary choice)
    n_categories = 2  # Number of categories for output (2 for binary choice)
    input_dim = 2  # Input dimensions (stimulus and bias)

    # Initialize GLM-HMM
    glmhmm = ssm.HMM(n_states, obs_dim, input_dim, observations='input_driven_obs',
                     observation_kwargs=dict(C=n_categories), transitions='standard')

    # Filter data
    df = filter_behavior(df)

    # Parse data
    inputs, choices = parse_glmhmm(df, at=False)

    # Set fitting parameters
    method = 'em'  # Expectation Maximization method
    num_iters = 200  # Max number of EM iterations
    tolerance = 1e-4  # Tolerance for stopping criterion

    # Fit GLM-HMM
    fit_ll = glmhmm.fit(choices, inputs=inputs, method=method, num_iters=num_iters, tolerance=tolerance)

    # Retrieve parameters
    weights = glmhmm.observations.params
    weights = -weights  # Flip sign of weights
    trans_mat = glmhmm.transitions.transition_matrix  # Need to remap this

    # Get posterior probabilities and most likely states
    posterior_probs = [glmhmm.expected_states(data=data, input=input)[0]
                       for data, input in zip(choices, inputs)]
    posterior_probs_concat = np.concatenate(posterior_probs)
    # state_max_posterior = np.argmax(posterior_probs_concat, axis=1)

    # Interpret states and remap
    state_labels, remap = interpret_states(weights)
    weights = weights[remap]
    trans_mat = trans_mat[remap][:, remap]
    state_labels = [state_labels[old] for old in remap]
    posterior_probs = [p[:, remap] for p in posterior_probs]
    posterior_probs_concat = posterior_probs_concat[:, remap]
    # state_max_posterior = remap[state_max_posterior]
    state_max_posterior = np.argmax(posterior_probs_concat, axis=1)

    # Add model outputs to DataFrame
    df['State'] = state_max_posterior
    df['StateLabel'] = df['State'].map({i: label for i, label in enumerate(state_labels)})
    df['p0'] = posterior_probs_concat[:, 0]
    df['p1'] = posterior_probs_concat[:, 1]
    df['p2'] = posterior_probs_concat[:, 2]
    df['Weights'] = [weights[s] for s in df['State']]

    if save:
        # Select the output folder and create it if it doesn't exist
        experiment = df.Experiment.unique()[0]
        folder_out = Path.home() / 'PycharmProjects' / 'glmhmm' / experiment
        if not os.path.exists(folder_out):
            folder_out.mkdir(parents=True, exist_ok=True)
        subject = str(df.Subject.unique()[0])
        df.to_csv((folder_out / subject).with_suffix('.csv'), index=False)

    return df, weights, trans_mat, state_labels


def main(experiment):
    """
    Fit GLM-HMM to all subjects of one group and save the results to a CSV file.
    :return: None
    """

    folder = Path.home() / 'PycharmProjects' / 'glue_sessions' / experiment

    animals = os.listdir(folder)  # List animals
    animals.sort()  # Sort them by name
    animals = [x for x in animals if not 'corrupted_sessions' in x]  # Get rid of the corrupted sessions csv files

    for animal in animals:
        print(f'Fitting the GLM-HMM for subject {animal[:3]}...')
        path = folder / animal
        df = pd.read_csv(path)

        try:
            df = fit_glm_hmm(df, save=True)
        except Exception as e:
            print(f'Error fitting GLM-HMM for subject {animal}: {e}')
            continue
        print('\n')
