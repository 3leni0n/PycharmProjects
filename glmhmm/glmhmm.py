import os
import seaborn as sns
import ssm
from pathlib import Path
import pickle

from my_fun import get_experiment
from cherry.cherry import *
from kernels.kernels_tools import *


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
        covariates = ['stim_vals', 'bias', 'at_choice']  # Default model
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


def fit_all(experiment):
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


def test_full_model(experiments=['2AFC_2', '2AFC_3', '2AFC_4']):

    all_animals = []
    weights = []
    cherries = main(experiments)  # Get good subjects from cherry

    for experiment in experiments:
        animals = cherries[experiment]
        all_animals.extend(animals)

        # if experiment == '2AFC_2':
        #     animals = ['325', '327', '329', '330', '332', '333', '335', '337']
        # elif experiment == '2AFC_3':
        #     animals = ['419', '420', '422', '616', '619', '623']

        for i, animal in enumerate(animals):

            # Load behavioral data
            experiment, folder_in = get_experiment(experiment, path_session='glue_sessions')
            print(f'Fitting GLM-HMM for subject {animal} ({i + 1}/{len(animals)} of Experiment {experiment})')
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
            inputs, choices = parse_glmhmm(df, covariates=['stim_vals', 'bias', 'at_choice'])
            # inputs, choices = parse_glmhmm(df, covariates=['net_ild', 'bias', 'at_choice'])
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

    return weights, all_animals


def interpret_weights(weights, cov_index=0):
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

    bias_index = 1
    weights_disengaged[bias_index] = abs(weights_disengaged[bias_index])
    weights_engaged[bias_index] = abs(weights_engaged[bias_index])

    plt.plot(weights_disengaged, color='tab:gray', marker='o', **kwargs)
    plt.plot(weights_engaged, color='tab:blue', marker='o', **kwargs)
    plt.axhline(0, color='black', linestyle='--')
    cov_names = ['stim.', '|bias|', '$A_t$']
    # cov_names = ['|2|', '|4|', '|8|', '|70|', 'bias', '$A_t$']
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

        bias_index = 1
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
    cov_names = ['stim.', '|bias|', '$A_t$']
    # cov_names = ['|2|', '|4|', '|8|', '|70|', 'bias', '$A_t$']
    # cov_names = ['|2|', '|4|', '|8|', '|70|', 'bias', '$A_{t^-}$', '$A_{t^+}$']
    plt.xticks(np.arange(len(cov_names)), cov_names)
    plt.xlabel('Covariate')
    plt.ylabel(f'Weight')
    plt.title(f'GLM-HMM kernel')
    plt.legend(frameon=False)
    sns.despine()


def plot_paired_boxplot_GLMHMM_kernel(remapped_weights_subjects, all_animals):
    """
    Plot paired boxplots for engaged vs disengaged weights across all subjects for each covariate.
    """

    data = []
    for animal_id, w in zip(all_animals, remapped_weights_subjects):
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
    data.loc[data['Covariate'] == 1, 'Weight'] = data.loc[data['Covariate'] == 1, 'Weight'].abs()  # Absolute  bias

    plt.figure(constrained_layout=True)
    ax = sns.boxplot(x='Covariate', y='Weight', hue='State', data=data,
                palette={0: 'tab:gray', 1: 'tab:blue'}, showfliers=False)
    plt.axhline(0, color='black', linestyle='--')
    cov_names = ['stim.', '|bias|', '$A_t$']
    # cov_names = ['|2|', '|4|', '|8|', '|70|', 'bias', '$A_t$']
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


# cherries = main()  # Absolute import due to another main function in this script
# weights_subjects, all_animals = test_full_model()
# remapped_weights_subjects, remap_indices_subjects = interpret_weights(weights_subjects)
# animals = ['325', '327', '329', '330', '332', '333', '335', '337', '419', '420', '422', '616', '619', '623']
# plot_paired_boxplot_GLMHMM_kernel(remapped_weights_subjects, all_animals)

# path = Path.home() / 'PycharmProjects' / 'glmhmm' / 'remapped_weights_subjects.pkl'
# # Save list of arrays with the weights
# # with open(path, 'wb') as f:
# #     pickle.dump(remapped_weights_subjects, f)
# #     print(f'Saved weights to {path}')
#
# # Load list of arrays with the weights
# with open(path, 'rb') as f:
#     remapped_weights_subjects = pickle.load(f)
#     print(f'Loaded weights from {path}')
#
#
# path = Path.home() / 'PycharmProjects' / 'glmhmm' / 'all_animals.pkl'
# # Save list of all animals
# # with open(path, 'wb') as f:
# #     pickle.dump(all_animals, f)
# #     print(f'Saved weights to {path}')
#
# # Load list of arrays with the weights
# with open(path, 'rb') as f:
#     all_animals = pickle.load(f)
#     print(f'Loaded weights from {path}')