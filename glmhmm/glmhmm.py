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


def make_choice_history_dm(df, k=10):
    """
    Make a design matrix with the choice history. There is a column for each previous trial (up to k). In each trial,
    only one of these regressors is non-zero.
    :param df: DataFrame with hit and choice data
    :param k: Number of trials to look back
    :return: Design matrix
    """
    def get_choice_history(df, k):
        """
        Get the choice history for a trial number k.
        :param df: DataFrame with hit and choice data
        :param k: Number of trials to look back
        :return:  r_minus, r_plus
        """
        r_minus = []
        r_plus = []
        for _ in range(len(df)):
            if _ < k:
                r_minus.append(np.nan)
                r_plus.append(np.nan)
            else:
                # r-(t-k): error right = +1, error left = -1, no error (correct) = 0
                if df.Hit[_ - k] == 0 and df.Choice[_ - k] == 1:
                    r_minus.append(1)
                elif df.Hit[_ - k] == 0 and df.Choice[_ - k] == 0:
                    r_minus.append(-1)
                elif df.Hit[_ - k] == 1:
                    r_minus.append(0)
                # r+(t-k): correct right = +1, correct left = -1, no correct (error) = 0
                if df.Hit[_ - k] == 1 and df.Choice[_ - k] == 1:
                    r_plus.append(1)
                elif df.Hit[_ - k] == 1 and df.Choice[_ - k] == 0:
                    r_plus.append(-1)
                elif df.Hit[_ - k] == 0:
                    r_plus.append(0)

        return r_minus, r_plus

    design_matrix = pd.DataFrame()  # Create empty DataFrame to store previous choices

    for _ in reversed(range(1, k + 1)):
        print(f'Getting choice history of trial lag {_}')
        r_minus, r_plus = get_choice_history(df, _)
        design_matrix['Rminus' + str(_)] = r_minus
        design_matrix['Rplus' + str(_)] = r_plus

    # Reorder exog columns, so I can split later in half the params for plotting r+ or r-
    r_minus_columns = ['Rminus' + str(_) for _ in reversed(range(1, k + 1))]
    r_plus_columns = ['Rplus' + str(_) for _ in reversed(range(1, k + 1))]
    design_matrix = design_matrix[r_minus_columns + r_plus_columns]

    return design_matrix


def get_experiment(experiment=None, path_session='glue_sessions'):
    """
    Get experiment
    :param experiment: If not None, experiment=experiment. Else, show possible experiments and ask for user input.
    :param path_session: if glue_sessions look for individual sessions, elif intersession look for intersessions
    :return: experiment, path_experiment: experiment (user input), path to the experiment folder
    """

    if experiment is None:

        path_experiment = Path.home() / 'PycharmProjects' / path_session  # Where the data for all animals is
        experiments = list(path_experiment.iterdir())  # List experiments
        experiments.sort()  # Sort them by name
        experiments = [x.name for x in path_experiment.iterdir() if x.is_dir()]  # Get rid of non folders

        try:
            experiments.remove('__pycache__')  # Pycharm's file
        except ValueError:
            pass

        print('Experiments:\n ' + str(experiments)[1:-1])  # Remove square brackets
        experiment = input('Enter experiment name')
        path_experiment = Path.home() / 'PycharmProjects' / path_session / experiment  # Where the data for the animal is

    else:
        path_experiment = Path.home() / 'PycharmProjects' / path_session / experiment

    return experiment, path_experiment


def get_animal(experiment=None, path_session='glue_sessions', animal=None):
    """
    Get animal
    :param experiment: If not None, experiment=experiment. Else, show possible experiments and ask for user input
    :param path_session: if glue_sessions look for individual sessions, elif intersession look for intersessions
    :param animal: If not None, animal=animal. Else, show possible animals and ask for user input
    :return: animal
    """

    if experiment is None:
        experiment, folder_in = get_experiment(experiment, path_session)

    if animal is None:
        animals = list(folder_in.iterdir())  # List animals
        # animals = os.listdir(folder_in)  # List animals
        animals = [x.name for x in animals]  # Get rid of non folders
        animals.sort()  # Sort them by name
        animals = [x[:-4] for x in animals]  # Get rid of .csv extension
        animals = [i for i in animals if '_corrupted_sessions' not in i]  # Remove '_corrupted_sessions'.csv files

        print('Animals: ' + str(animals))  # Remove square brackets
        animal = input('Enter animal')  # Ask user to input animal to glue sessions from

    return animal


########################################################################################################################

# Define a function to parse the data for GLM-HMM

def get_action_trace(df, max_trial_lag=10, tau=2):
    """
    Computes the action trace for each trial in the DataFrame. The action trace is an exponentially weighted sum of past
    choices, where more recent choices have a greater influence. The weights decay exponentially with a time constant
    tau. Output is normalized between -1 and +1.
    :param df: DataFrame containing the data with a column of interest (0=left; 1=right)
    :param max_trial_lag: Number of past trials to consider
    :param tau: Decay constant
    :return: List of action trace values for each trial for choices, errors and correct trials
    """

    lags = np.arange(1, max_trial_lag + 1)  # Lags from 1 to k
    weights = np.exp(-lags / tau)  # Exponential decay weights
    Z = np.sum(weights)  # Fixed normalizer

    # Precompute signed choice
    signed_choice = 2 * df['Choice'].to_numpy() - 1  # Map 0→-1, 1→+1
    r_minus = signed_choice * df['Punish'].to_numpy()
    r_plus = signed_choice * df['Hit'].to_numpy()

    at_choice = []  # Action trace for choices
    at_error = []  # Action trace for error choices
    at_correct = []  # Action trace for correct choices

    for t in range(len(df)):
        past_choice = signed_choice[max(0, t - max_trial_lag):t]
        past_rminus = r_minus[max(0, t - max_trial_lag):t]
        past_rplus = r_plus[max(0, t - max_trial_lag):t]

        effective_weights = weights[:len(past_choice)]

        at_choice.append(np.sum(past_choice * effective_weights) / Z)
        at_error.append(np.sum(past_rminus * effective_weights) / Z)
        at_correct.append(np.sum(past_rplus * effective_weights) / Z)

    return at_choice, at_error, at_correct


def parse_glmhmm(df, covariates=None):
    """
    Parse the data for GLM-HMM with flexible covariates.
    :param df: DataFrame containing the data (one or many sessions concatenated)
    :param covariates: List of covariates to include. Options:
        'stim_vals',
        'stim_strength',
        'net_ild',
        'action_trace',
        'action_trace_error',
        'action_trace_correct',
        'bias',
        'session_index'
    :return: inputs, choices
    """

    accepted_covariates = ['stim_vals', 'stim_strength', 'net_ild', 'at_choice', 'at_error', 'at_correct', 'bias',
                           'session_index']
    if covariates is None:
        covariates = ['net_ild', 'bias', 'at_choice']  # Default model
    else:
        for cov in covariates:
            if cov not in accepted_covariates:
                raise ValueError(f'Covariate {cov} not recognized. Accepted covariates are: {accepted_covariates}')

    df.reset_index(drop=True)  # Reset index to ensure consistent slicing
    dm_session_index = make_session_index_dm(df)  # Add bias (constant) per session

    # Set stimuli set
    experiment = df.Experiment.unique()[0]
    if experiment == '2AFC_6':
        stim_set = 6
    elif experiment == '2AFC':
        stim_set = 1
    else:
        stim_set = 2

    # inputs and choices must be lists of arrays, one per session
    inputs = []
    choices = []

    for session_id, df_session in df.groupby('Session'):

        n_trials = len(df_session)
        session_cols = []

        if 'stim_vals' in covariates:
            stim_vals = df_session.ILD.values
            stim_vals = stim_vals / abs(df.ILD.max())  # Normalize ILD to [-1, 1]
            session_cols.append(stim_vals)

        if 'stim_strength' in covariates:
            stim_strength, n_frames = make_frames_dm(df_session, stim_set=stim_set, residuals=True, zscore=False)
            stim_strength = stim_strength / stim_strength.values.max()  # Normalize ILD to [-1, 1]
            session_cols.append(stim_strength)

        if 'net_ild' in covariates:
            dm_net_ild = make_net_ild_dm(df_session)
            session_cols.append(dm_net_ild)

        if 'bias' in covariates:
            session_cols.append(np.ones(n_trials))

        if 'session_index' in covariates:
            dm_session_index_sess = dm_session_index.iloc[df_session.index.values, :]
            session_cols.append(dm_session_index_sess)

        if any(x in covariates for x in ['at_choice', 'at_error', 'at_correct']):
            at_choice, at_error, at_correct = get_action_trace(df_session)
            if 'at_choice' in covariates:
                session_cols.append(np.array(at_choice))
            if 'at_error' in covariates:
                session_cols.append(np.array(at_error))
            if 'at_correct' in covariates:
                session_cols.append(np.array(at_correct))

        # Combine selected covariates
        session_input = np.column_stack(session_cols)
        session_choices = df_session.Choice.values.astype(int)[:, None]

        inputs.append(session_input)
        input_dim = inputs[0].shape[1]
        assert all(sess.shape[1] == input_dim for sess in inputs), 'Not all sessions have the same number of inputs'
        choices.append(session_choices)

    return inputs, choices


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


########################################################################################################################


def test_full_model(experiments=['2AFC_2', '2AFC_3']):

    weights = []

    for experiment in experiments:

        if experiment == '2AFC_2':
            animals = ['325', '327', '329', '330', '332', '333', '335', '337']
        elif experiment == '2AFC_3':
            animals = ['419', '420', '422', '616', '619', '623']

        for i, animal in enumerate(animals):

            # Load behavioral data
            experiment, folder_in = get_experiment(experiment, path_session='glue_sessions')
            print(f'Getting GLM-HMM {animal} ({i + 1}/{len(animals)} of Experiment {experiment})')
            folder_in = Path(folder_in / animal).with_suffix('.csv')
            print(f'Loading data from {folder_in}')
            df = pd.read_csv(folder_in, low_memory=False)

            # Filters for groups 1-3
            # df = df[df.Stage == 4].reset_index(drop=True)
            # df = df[df.Motor == 4].reset_index(drop=True)
            # # df = df[df.StimDur == 1].reset_index(drop=True)
            df = df[df.P > 0].reset_index(drop=True)

            # Drop misses (Choice == NaN)
            df = df.dropna(subset=['Choice']).reset_index(drop=True)

            # Add session index per trial for plotting accuracy
            session_index = pd.factorize(df['Session'])[0]  # Get session index per trial
            loc = df.columns.get_loc('Session') + 1  # To the right of Session column
            df.insert(loc, 'SessionIndex', session_index)  # Add session index column

            # Remove bad sessions with few trials (and therefore missing at least one of the stimulus evidences)
            bad_sessions = []
            # Count number of trials per session
            for session_id, df_session in df.groupby('Session'):
                if len(df_session.ILD.abs().unique()) != len(df.ILD.abs().unique()):  # Should be 5 (including 0)
                    print(f'Session {session_id} does not have enough trials, skipping...')
                    bad_sessions.append(session_id)
            # Remove bad sessions from df
            df = df[~df.Session.isin(bad_sessions)].reset_index(drop=True)

            # Parse the data
            inputs, choices = parse_glmhmm(df, covariates=['net_ild', 'bias', 'at_choice'])
            # inputs, choices = parse_glmhmm(df, covariates=['net_ild', 'bias', 'at_error', 'at_correct'])

            # Set the parameters of the GLM-HMM
            n_states = 2  # Number of discrete states
            obs_dim = 1  # Number of observed dimensions (1 for binary choice)
            n_categories = 2  # Number of categories for output (2 for binary choice)
            input_dim = inputs[0].shape[1]

            glmhmm = ssm.HMM(n_states, obs_dim, input_dim, observations='input_driven_obs',
                             observation_kwargs=dict(C=n_categories), transitions='standard')

            # Fitting with stop earlier if increase in LL is below tolerance specified by tolerance parameter
            method = 'em'  # Expectation Maximization method
            num_iters = 200  # Max number of EM iterations
            tolerance = 1e-4  # tolerance for stopping criterion
            fit_ll = glmhmm.fit(choices, inputs=inputs, method='em', num_iters=num_iters, tolerance=tolerance)

            weights_subject = glmhmm.observations.params
            weights_subject = -weights_subject  # Flip sign of weights
            weights.append(weights_subject)

    return weights


def interpret_weights(weights, cov_index=3):
    """
    Interpret the HMM latent states (zt) of one or several subjects based on the GLM weights of its covariates.
    Assign engaged (larger) and disengaged (smaller) depending on the weight of the stimulus covariate (cov_index).
    Work for 2 states only.
    :param weights: GLM weights of shape (n_states, obs_dim, input_dim)
    :param cov_index: Index of the covariate to use for interpretation
    :return: remapped_weights, remap_indices
    """

    def _interpret_single(weights):
        if weights.shape[0] != 2:
            raise ValueError('Currently only supports 2 states')
        cov = weights[:, 0, cov_index]
        disengaged_index = np.argmin(cov)
        engaged_index = np.argmax(cov)
        remap_indices = [disengaged_index, engaged_index]
        remapped_weights = weights[remap_indices]
        print(f"Remapped weights (dis., eng.): {remapped_weights[:, 0, cov_index]}")
        return remapped_weights, remap_indices

    # Check if input is a list (multiple subjects) or a single array
    if isinstance(weights, list):
        remapped_weights_subjects = []
        remap_indices_subjects = []
        for w in weights:
            remapped_weights, remap_indices = _interpret_single(w)
            remapped_weights_subjects.append(remapped_weights)
            remap_indices_subjects.append(remap_indices)
        return remapped_weights_subjects, remap_indices_subjects
    else:
        return _interpret_single(weights)


def plot_GLMHMM_kernel(remapped_weights, **kwargs):
    """
    Plot GLM remapped weights for one subject for each state. Requires weight remapping first according to
    interpretation.
    :param weights_remapped: np.array with GLM weights of shape (n_states, obs_dim, input_dim) already remapped
    (0 = disengaged, 1 = engaged)
    :param kwargs: Additional keyword arguments for plt.plot(). E.g. alpha=0.5
    :return: None
    """

    weights_disengaged = remapped_weights[0, 0, :]
    weights_engaged = remapped_weights[1, 0, :]

    bias_index = 4
    weights_disengaged[bias_index] = abs(weights_disengaged[bias_index])
    weights_engaged[bias_index] = abs(weights_engaged[bias_index])

    plt.plot(weights_disengaged, color='tab:gray', marker='o', **kwargs)
    plt.plot(weights_engaged, color='tab:blue', marker='o', **kwargs)
    plt.axhline(0, color='black', linestyle='--')
    cov_names = ['|2|', '|4|', '|8|', '|70|', 'bias', '$A_t$']
    # cov_names = ['|2|', '|4|', '|8|', '|70|', 'bias', '$A_{t^-}$', '$A_{t^+}$']
    plt.xticks(np.arange(len(cov_names)), cov_names)
    plt.xlabel('Covariate')
    plt.ylabel(f'Weight')
    plt.title(f'GLM-HMM kernel')
    sns.despine()


def plot_mean_GLMHMM_kernel(remapped_weights_subjects):
    """
    Plot GLM remapped weights for all subjects for each state. Requires weight remapping first according to
    interpretation.
    :param weights_remapped: List GLM weights per subject of shape (n_states, obs_dim, input_dim) already remapped
    (0 = disengaged, 1 = engaged)
    :return: None
    """

    plt.figure(constrained_layout=True)

    mean_weights_engaged = []
    mean_weights_disengaged = []
    for i, w in enumerate(remapped_weights_subjects):
        weights_disengaged = w[0, 0, :]
        weights_engaged = w[1, 0, :]

        bias_index = 4
        weights_disengaged[bias_index] = abs(weights_disengaged[bias_index])
        weights_engaged[bias_index] = abs(weights_engaged[bias_index])

        # if weights_disengaged[3] > 10 or weights_engaged[3] > 10:
        #     print(f'Skipping animal {animals[i]} with weights {weights_disengaged[3]}, {weights_engaged[3]}')
        #     continue

        mean_weights_disengaged.append(weights_disengaged)
        mean_weights_engaged.append(weights_engaged)
        plot_GLMHMM_kernel(remapped_weights_subjects[i], alpha=0.1)

    # convert to arrays
    mean_weights_disengaged = np.array(mean_weights_disengaged)
    mean_weights_engaged = np.array(mean_weights_engaged)

    # Compute the mean across animals
    mean_weights_disengaged = np.mean(mean_weights_disengaged, axis=0)
    mean_weights_engaged = np.mean(mean_weights_engaged, axis=0)

    plt.plot(mean_weights_disengaged, color='tab:gray', marker='o', label='Disengaged')
    plt.plot(mean_weights_engaged, color='tab:blue', marker='o', label='Engaged')
    cov_names = ['|2|', '|4|', '|8|', '|70|', 'bias', '$A_t$']
    # cov_names = ['|2|', '|4|', '|8|', '|70|', 'bias', '$A_{t^-}$', '$A_{t^+}$']
    plt.xticks(np.arange(len(cov_names)), cov_names)
    plt.xlabel('Covariate')
    plt.ylabel(f'Weight')
    plt.title(f'GLM-HMM kernel')
    plt.legend(frameon=False)
    sns.despine()


def plot_paired_boxplot_GLMHMM_kernel(remapped_weights_subjects, animals):
    """
    Plot paired boxplots for engaged vs disengaged weights across all subjects for each covariate.
    """

    data = []
    for animal_id, w in zip(animals, remapped_weights_subjects):
        for state_idx in range(w.shape[0]):
            for cov_idx in range(w.shape[2]):
                data.append({
                    'Animal': animal_id,
                    'State': state_idx,
                    # 'Label': 'Disengaged' if state_idx == 0 else 'Engaged',
                    'Covariate': cov_idx,
                    'Weight': w[state_idx, 0, cov_idx]
                })
    data = pd.DataFrame(data)
    data.loc[data['Covariate'] == 4, 'Weight'] = data.loc[data['Covariate'] == 4, 'Weight'].abs()  # Absolute  bias

    plt.figure(constrained_layout=True)
    ax = sns.boxplot(x='Covariate', y='Weight', hue='State', data=data,
                palette={0: 'tab:gray', 1: 'tab:blue'}, showfliers=False)
    plt.axhline(0, color='black', linestyle='--')
    cov_names = ['|2|', '|4|', '|8|', '|70|', 'bias', '$A_t$']
    # cov_names = ['|2|', '|4|', '|8|', '|70|', 'bias', '$A_{t^-}$', '$A_{t^+}$']
    plt.xticks(np.arange(len(cov_names)), cov_names)
    plt.title(f'GLM-HMM kernel')
    handles, labels = ax.get_legend_handles_labels()  # Get legend
    ax.legend(handles, ['Disengaged', 'Engaged'], frameon=False, title='State')  # Rename legend labels
    sns.despine()

    # Draw paired lines of subjects between boxes (states)
    for cov in data['Covariate'].unique():
        for animal in data['Animal'].unique():
            subset = data[(data['Covariate'] == cov) & (data['Animal'] == animal)]
            x0 = cov - 0.2  # disengaged box (gray)
            x1 = cov + 0.2  # engaged box (blue)
            y0 = subset[subset['State'] == 0]['Weight'].values[0]
            y1 = subset[subset['State'] == 1]['Weight'].values[0]
            ax.plot([x0, x1], [y0, y1], color='k', alpha=0.1)


weights_subjects = test_full_model(experiments=['2AFC_2', '2AFC_3'])
remapped_weights_subjects, remap_indices_subjects = interpret_weights(weights_subjects, cov_index=3)
animals = ['325', '327', '329', '330', '332', '333', '335', '337', '419', '420', '422', '616', '619', '623']
plot_paired_boxplot_GLMHMM_kernel(remapped_weights_subjects, animals)