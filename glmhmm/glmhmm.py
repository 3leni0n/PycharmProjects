import os
from pathlib import Path
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import ssm


# From my_fun and kernels_tools, but can import them because would need to install libraries in this environment
def get_ild(stim_set=6):
    # Load sounds
    if stim_set == 1:
        sounds_path = Path.home() / 'PycharmProjects' / 'create_sounds' / 'sounds.csv'
    if stim_set == 2:
        sounds_path = Path.home() / 'PycharmProjects' / 'create_sounds' / 'sounds_2.csv'
    elif stim_set == 6:
        sounds_path = Path.home() / 'PycharmProjects' / 'create_sounds' / 'sounds_6.1.csv'

    sounds = pd.read_csv(sounds_path)
    n_frames = sounds.n_frames.unique()[0]

    # Left frames
    left_frames_column_names = [f'EL{n:01}' for n in range(n_frames)]
    frames_left = sounds[left_frames_column_names].values

    # Right frames
    right_frames_column_names = [f'ER{n:01}' for n in range(n_frames)]
    frames_right = sounds[right_frames_column_names].values

    # Frames ILD (elementwise substraction)
    frames_ild = frames_right - frames_left
    frames_ild = pd.DataFrame(frames_ild)
    frames_ild.insert(0, column='filename', value=sounds.filename)  # Insert behavior_filenames in first column

    return frames_ild


def make_frames_dm(df, stim_set=6, residuals=True, zscore=False):

    # Load sounds
    if stim_set == 1:
        sounds_path = Path.home() / 'PycharmProjects' / 'create_sounds' / 'sounds.csv'
    if stim_set == 2:
        sounds_path = Path.home() / 'PycharmProjects' / 'create_sounds' / 'sounds_2.csv'
    elif stim_set == 6:
        sounds_path = Path.home() / 'PycharmProjects' / 'create_sounds' / 'sounds_6.1.csv'

    sounds = pd.read_csv(sounds_path)
    n_frames = sounds.n_frames.unique()[0]
    frames_ild = get_ild(stim_set=stim_set)

    # Residuals (https://www-nature-com.sire.ub.edu/articles/nature08275)
    if residuals:
        sounds_ild = sounds.ILD
        first_frame = frames_ild[0]
        first_frame = first_frame.copy()
        first_frame.iloc[0] = 0  # Set to 0 to avoid artifact of net ILD 70 having 0 weight
        first_frame.iloc[-1] = 0  # Set to 0 to avoid artifact of net ILD 70 having 0 weight
        if stim_set == 6:
            frames_ild = frames_ild.drop(['filename', 0], axis=1).sub(sounds_ild, axis='rows')
            frames_ild.insert(0, column='filename', value=sounds.filename)  # Insert back filenames in 1st column
            frames_ild.insert(1, column=0, value=first_frame)  # Insert back first_frame in 2nd column
        else:
            frames_ild = frames_ild.drop('filename', axis=1).sub(sounds_ild, axis='rows')
            frames_ild.insert(0, column='filename', value=sounds.filename)  # Insert back filenames in 1st column

    filenames = df.Filename.tolist()

    # Get frames per trial
    stim_strength = frames_ild.loc[
        [np.where(sounds.filename == np.array(filenames[i]))[0][0] for i in range(len(filenames))]].drop(
        columns=['filename'])
    stim_strength.reset_index(drop=True, inplace=True)  # Indices must match for modeling

    # Zscore
    if not residuals:  # To not do both (otherwise I'd be subtracting the mean twice)
        if zscore:
            stim_strength = pd.DataFrame(stats.zscore(stim_strength, axis=0))  # Z-score the ILDs (along axis 0 or None
            # returns same result, but not axis 1). 0 along trials that's what I want to do :)

    design_matrix = stim_strength

    return design_matrix, n_frames


def make_net_ild_dm(df):
    """
    Make a design matrix with the net ILDs. There is a column for each absolute, unique ILD value (except 0). It
    transforms the nominal ILD into ILD net magnitude (2, 4, 8 , 70 dB) that take values +1, 0 or -1. In each trial,
    only one of these regressors is non-zero.
    When separating the stimuli S_k =  nominal_ILD + residuals and give a separate beta for the nominal and for each
    residual frame, you are somehow assuming that the impact of the nominal ILD grows linearly with ILD. But this is
    probably not the case. Particularly if spanning a range from ILD 0 to 70 dB. One simple way to not assume anything
    about how the impact of the stimuli grows with ILD is to define separate regressors for each absolute value of the
    ILD, that is 2, 4, 8 and 70. Each of this ILDs will define a regressor e.g. ILD_8 =  +1 (if ILD was +8 dB), -1 (if
    ILD was -8 db) and 0 (if ILD was other than +- 8dB). This way, you should be able to include ALL stimuli in the
    analysis (maximum evidence too).
    :param df:
    :return: Design matrix
    """
    ilds = df.ILD.astype('int')
    net_ilds = np.sort(df.ILD.abs().unique().astype('int'))[1:]
    design_matrix = np.zeros((len(df), len(net_ilds)), dtype=int)
    # columns = [str(_) for _ in net_ilds]
    design_matrix = pd.DataFrame(design_matrix, columns=net_ilds)
    for i, ild in enumerate(ilds):
        if ild != 0:
            design_matrix.loc[i, abs(ild)] = np.sign(ild)
    return design_matrix


def make_session_index_dm(df, column='Date'):
    """
    # Make a design matrix in which there are as many columns as unique dates. Then, for each column, there is a 1 if
    the trial belongs to that session and a 0 otherwise
    :param df: Input DataFrame
    :param column: Column of the DataFrame that contains the dates
    :return: Design matrix
    """
    dates = df[column].unique()
    design_matrix = np.zeros((len(df), len(dates)), dtype=int)
    for i, date in enumerate(dates):
        design_matrix[df[column] == date, i] = 1
    design_matrix = pd.DataFrame(design_matrix)
    return design_matrix

########################################################################################################################

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


# Define a function to parse the data for GLM-HMM
def parse_glmhmm_full(df, at=False):
    """
    Parse the data for GLM-HMM.
    :param df: DataFrame containing the data (one or many sessions concatenated).
    :return: inputs, choices
    """

    dm_session_index = make_session_index_dm(df)  # Add bias (constant) per session

    experiment = df.Experiment.unique()[0]

    # Set stimuli set
    if experiment == '2AFC_6':
        stim_set = 6
    elif experiment == '2AFC':
        stim_set = 1
    else:
        stim_set = 2

    inputs = []
    choices = []

    for session_id, df_session in df.groupby('Session'):

        n_trials = len(df_session)

        # Make stimulus strength design matrix
        stim_strength, n_frames = make_frames_dm(df_session, stim_set=stim_set, residuals=True, zscore=False)
        stim_strength = stim_strength / stim_strength.values.max()  # Normalize ILD to [-1, 1]

        # Make net ILD design matrix
        dm_net_ild = make_net_ild_dm(df_session)

        # stim_vals = df_session.ILD.values
        # stim_vals = stim_vals / abs(df.ILD.max())  # Normalize ILD to [-1, 1]

        # bias = np.ones(n_trials)
        dm_session_index_sess = dm_session_index.iloc[df_session.index.values, :]

        if at:
            action_trace = get_action_trace(df_session)
            session_input = np.column_stack((
                # stim_vals,
                # stim_strength,
                dm_net_ild,
                action_trace,
                # bias,
                dm_session_index_sess
            ))
        else:
            session_input = np.column_stack((
                # stim_vals,
                # stim_strength,
                dm_net_ild,
                # bias,
                dm_session_index_sess
            ))

        session_choices = df_session.Choice.values.astype(int)[:, None]
        inputs.append(session_input)
        choices.append(session_choices)

    return inputs, choices


def get_action_trace(df, max_trial_lag=10, tau=2):
    """
    Computes the action trace for each trial in the DataFrame. The action trace is an exponentially weighted sum of past
    choices, where more recent choices have a greater influence. The weights decay exponentially with a time constant
    tau. Output is normalized between -1 and +1.
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
