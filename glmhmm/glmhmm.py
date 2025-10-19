import os
from scipy.stats import ttest_rel
import pickle
import ssm
from matplotlib import cm
from my_fun import get_experiment, save_notebook_files, add_star_between
from cherry.cherry import *
from kernels.kernels_tools import *
from plotting_style import *


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


def test_full_model(experiments=['2AFC_2', '2AFC_3', '2AFC_4'], drug=None, interpret=True):

    all_animals = []
    fit_ll = []
    weights = []
    trans_mat = []
    posterior_probs = []
    log_likelihood = []
    # log_probability = []

    cherries = main(experiments)  # Get good subjects from cherry

    for experiment in experiments:
        animals = cherries[experiment]
        all_animals.extend(animals)

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

            # # Remove bad sessions with few trials (and therefore missing at least one of the stimulus evidences)
            # bad_sessions = []
            # # Count number of trials per session
            # for session_id, df_session in df.groupby('Session'):
            #     if len(df_session.ILD.abs().unique()) != len(df.ILD.abs().unique()):  # Should be 5 (including 0)
            #         print(f'Session {session_id} does not have enough trials, skipping...')
            #         bad_sessions.append(session_id)
            # # Remove bad sessions from df
            # df = df[~df.Session.isin(bad_sessions)].reset_index(drop=True)

            # Drug sessions
            if drug is None:
                df = df[df.Drug.isnull()].reset_index(drop=True)
            elif drug in [0, 1] and experiment == '2AFC_6':
                df = df[df.Drug == drug].reset_index(drop=True)

            # Parse the data
            inputs, choices = parse_glmhmm(df, covariates=['stim_vals', 'bias', 'at_choice'])
            # inputs, choices = parse_glmhmm(df, covariates=['net_ild', 'bias', 'at_choice'])
            # inputs, choices = parse_glmhmm(df, covariates=['net_ild', 'bias', 'at_error', 'at_correct'])

            # Set the parameters of the GLM-HMM
            n_states = 2  # Number of discrete states
            obs_dim = 1  # Number of observed dimensions (1 for binary choice)
            n_categories = 2  # Number of categories for output (2 for binary choice)
            input_dim = inputs[0].shape[1]

            # Initialize GLM-HMM
            glmhmm = ssm.HMM(n_states, obs_dim, input_dim, observations='input_driven_obs',
                             observation_kwargs=dict(C=n_categories), transitions='standard')

            # Fitting with stop earlier if increase in LL is below tolerance specified by tolerance parameter
            method = 'em'  # Expectation Maximization method
            num_iters = 200  # Max number of EM iterations
            tolerance = 1e-4  # tolerance for stopping criterion
            fit_ll.append(glmhmm.fit(choices, inputs=inputs, method=method, num_iters=num_iters, tolerance=tolerance))

            weights.append(-glmhmm.observations.params)  # Flip sign of weights
            trans_mat.append(glmhmm.transitions.transition_matrix)

            # Get expected states
            posterior_probs.append([glmhmm.expected_states(data=data, input=input)[0]
                               for data, input in zip(choices, inputs)])

            log_likelihood.append(glmhmm.log_likelihood(choices, inputs=inputs))
            # log_probability.append(glmhmm.log_probabilities(choices, inputs=inputs))

    if interpret:
        weights, trans_mat, posterior_probs, remap_indices = interpret_weights(weights, trans_mat, posterior_probs, cov_index=0)

    results = {
        'all_animals': all_animals,
        'fit_ll': fit_ll,
        'weights': weights,
        'trans_mat': trans_mat,
        'posterior_probs': posterior_probs,
        'log_likelihood': log_likelihood,
        'remap_indices': remap_indices if interpret else None,
        'drug': drug,
        'interpret': interpret
    }

    return results


def interpret_weights(weights, trans_mat, posterior_probs, cov_index=0):
    """
    Interpret the HMM latent states (zt) of one or several subjects based on the GLM weights of its covariates.
    Assign engaged (larger) and disengaged (smaller) depending on the weight of the stimulus covariate (cov_index).
    Work for 2 states only.
    :param weights: GLM weights of shape (n_states, obs_dim, input_dim)
    :param cov_index: Index of the covariate to use for interpretation
    :return: remapped_weights, remap_indices
    """

    def _interpret_single(weights, trans_mat, posterior_probs):

        if weights.shape[0] != 2:
            raise ValueError('Currently only supports 2 states')

        cov = weights[:, 0, cov_index]
        disengaged_index = np.argmin(cov)
        engaged_index = np.argmax(cov)
        remap_indices = [disengaged_index, engaged_index]
        remapped_weights = weights[remap_indices]
        remapped_trans_mat = trans_mat[np.ix_(remap_indices, remap_indices)]
        remapped_posterior_probs = [p[:, remap_indices] for p in posterior_probs]
        print(f'Remapped weights (dis., eng.): {remapped_weights[:, 0, cov_index]}')

        return remapped_weights, remapped_trans_mat, remapped_posterior_probs, remap_indices

    # Check if input is a list (multiple subjects) or a single array
    if isinstance(weights, list) and isinstance(trans_mat, list) and isinstance(posterior_probs, list):

        remapped_weights_subjects = []
        remapped_trans_mat_subjects = []
        remapped_posterior_probs_subjects = []
        remap_indices_subjects = []

        for w, t, p in zip(weights, trans_mat, posterior_probs):
            remapped_weights, remapped_trans_mat, remapped_posterior_probs, remap_indices = _interpret_single(w, t, p)
            remapped_weights_subjects.append(remapped_weights)
            remapped_trans_mat_subjects.append(remapped_trans_mat)
            remapped_posterior_probs_subjects.append(remapped_posterior_probs)
            remap_indices_subjects.append(remap_indices)
        return remapped_weights_subjects, remapped_trans_mat_subjects, remapped_posterior_probs_subjects, remap_indices_subjects

    else:
        return _interpret_single(weights, trans_mat, posterior_probs)


# Plotting functions

def plot_GLMHMM_kernel(remapped_weights, **kwargs):
    """
    Plot GLM remapped weights for one subject for each state. Requires weight remapping first according to
    interpretation.
    :param weights_remapped: np.array with GLM weights of shape (n_states, obs_dim, input_dim) already remapped
    (0 = disengaged, 1 = engaged)
    :param kwargs: Additional keyword arguments for plt.plot()
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


def plot_mean_GLMHMM_kernel(remapped_weights, **kwargs):
    """
    Plot GLM remapped weights for all subjects for each state. Requires weight remapping first according to
    interpretation.
    :param remapped_weights: List GLM weights per subject of shape (n_states, obs_dim, input_dim) already remapped
    (0 = disengaged, 1 = engaged)
    :param kwargs: Additional keyword arguments for plt.plot()
    :return: None
    """

    plt.figure(**kwargs, constrained_layout=True)

    mean_weights_engaged = []
    mean_weights_disengaged = []
    for i, w in enumerate(remapped_weights):
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
        plot_GLMHMM_kernel(remapped_weights[i], alpha=0.1)

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


def plot_paired_boxplot_GLMHMM_kernel(remapped_weights, all_animals, **kwargs):
    """
    Plot paired boxplots for engaged vs disengaged weights across all subjects for each covariate.
    :param remapped_weights: List GLM weights per subject of shape (n_states, obs_dim, input_dim) already remapped
    (0 = disengaged, 1 = engaged)
    :param all_animals: List of animal IDs corresponding to remapped_weights
    :param kwargs: Additional keyword arguments for plt.plot()
    :return: None
    """

    data = []
    for animal_id, w in zip(all_animals, remapped_weights):
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

    plt.figure(constrained_layout=True, **kwargs)
    plt.axhline(0, color='black', linestyle='--')
    ax = sns.boxplot(x='Covariate', y='Weight', hue='State', data=data,
                palette={0: 'tab:gray', 1: 'tab:blue'}, showfliers=False)
    cov_names = ['stim.', '|bias|', '$A_t$']
    # cov_names = ['|2|', '|4|', '|8|', '|70|', 'bias', '$A_t$']
    # cov_names = ['|2|', '|4|', '|8|', '|70|', 'bias', '$A_{t^-}$', '$A_{t^+}$']
    plt.xticks(np.arange(len(cov_names)), cov_names)
    plt.title(f'GLM-HMM kernel')
    handles, labels = ax.get_legend_handles_labels()  # Get legend
    ax.legend(handles, ['Disengaged', 'Engaged'], frameon=False, title='State')  # Rename legend labels
    sns.despine()

    # Draw paired lines of subjects between boxes (states)
    for cov in sorted(data['Covariate'].unique()):
        for animal in sorted(data['Animal'].unique()):
            subset = data[(data['Covariate'] == cov) & (data['Animal'] == animal)]
            x0 = cov - 0.2  # disengaged box (gray)
            x1 = cov + 0.2  # engaged box (blue)
            y0 = subset[subset['State'] == 0]['Weight'].values[0]
            y1 = subset[subset['State'] == 1]['Weight'].values[0]
            ax.plot([x0, x1], [y0, y1], color='k', alpha=0.1)

    # Compute paired-samples t-tests for each covariate between states
    for cov in sorted(data['Covariate'].unique()):
        cov_disengaged = data[(data['Covariate'] == cov) & (data['State'] == 0)].sort_values('Animal')['Weight']
        cov_engaged = data[(data['Covariate'] == cov) & (data['State'] == 1)].sort_values('Animal')['Weight']
        t_stat, p_val = ttest_rel(cov_engaged, cov_disengaged)
        print(f'Covariate {cov}: t={t_stat:.3f}, p={p_val:.4f}')
        add_star_between(p_val, x1=cov - 0.2, x2=cov + 0.2)

    return ax


def plot_trans_mat(trans_mat, **kwargs):
    """
    Plot transition matrix of one or several subjects.
    :param trans_mat: np.array with transition matrix of shape (n_states, n_states)
    :param kwargs: Additional keyword arguments for plt.plot()
    :return: None
    """

    # If trans_mat is a list of arrays, average them
    if isinstance(trans_mat, list):
        trans_mat = np.mean(np.stack(trans_mat, axis=0), axis=0)  # Stack and average

    n_states = int(np.mean(trans_mat.shape))
    plt.figure(**kwargs, constrained_layout=True)

    # gen_trans_mat = np.exp(log_trans_mat)[0]
    plt.imshow(trans_mat, vmin=-1, vmax=1, cmap='bone', origin='lower')

    for i in range(trans_mat.shape[0]):
        for j in range(trans_mat.shape[1]):
            # text = str(np.around(trans_mat[i, j], decimals=2))
            text = f"{trans_mat[i, j]:.2f}".lstrip('0').replace('-0.', '-.').rstrip('0').rstrip('.')
            plt.text(j, i, text, ha='center', va='center', color='k')

    # plt.xlim(-0.5, n_states + 0.5)
    ticks = range(0, n_states)
    ticklabels = [str(i) for i in range(n_states)]
    plt.xticks(ticks, ticklabels)
    plt.yticks(ticks, ticklabels)
    # plt.ylim(n_states - 0.5, -0.5)
    plt.ylabel('state $t$')
    plt.xlabel('state $t+1$')
    # plt.title('Transition matrix')

    # Ensure all spines are visible, even if despine() was called outside
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(True)


def plot_occupancy(posterior_probs, **kwargs):
    """
    Plot state occupancies for one or several subjects based on posterior probabilities.
    :param posterior_probs: List of posterior probabilities (np.array of shape (n_trials, n_states)) per subject
    :param kwargs: Additional keyword arguments for plt.plot()
    :return: None
    """

    # Normalize input to list of lists of arrays
    if isinstance(posterior_probs[0], np.ndarray):
        posterior_probs = [posterior_probs]  # Single animal

    n_states = posterior_probs[0][0].shape[1]
    colors = ['tab:gray', 'tab:blue']
    labels = ['D.', 'E.']

    occupancies = []

    for p in posterior_probs:

        # Concatenate posterior probabilities across sessions
        posterior_probs_concat = np.concatenate(p)

        # Get state with maximum posterior probability at particular trial
        state_max_posterior = np.argmax(posterior_probs_concat, axis=1)

        # Obtain state fractional occupancies
        _, state_occupancies = np.unique(state_max_posterior, return_counts=True)
        state_occupancies = state_occupancies / np.sum(state_occupancies)
        occupancies.append(state_occupancies)

    occupancies = np.array(occupancies)
    mean_occupancy = np.mean(occupancies, axis=0)

    # Plot fractional occupancies
    plt.figure(**kwargs, constrained_layout=True)

    for z, occ in enumerate(mean_occupancy):
        print(f'State {z} occupancy: {occ:.2f}')
        plt.bar(z, occ, color=colors[z], edgecolor='k')

    plt.xticks(range(len(labels)), labels)
    plt.ylim((0, 1))
    plt.yticks([0, 0.5, 1], ['0', '0.5', '1'])
    plt.xlabel('State')
    plt.ylabel('Occ.')
    # plt.title('Occupancy')
    sns.despine()


def plot_occupancy_boxplot(posterior_probs, **kwargs):
    """
    Plot state occupancies across subjects as boxplots.

    :param posterior_probs: List of posterior probabilities (np.array of shape n_trials × n_states) per subject
    :param kwargs: Additional keyword arguments for plt.figure()
    """
    # Normalize input to list of subjects
    if isinstance(posterior_probs[0], np.ndarray) and posterior_probs[0].ndim == 2:
        posterior_probs = [posterior_probs]  # single animal

    occupancies = []
    for p in posterior_probs:
        posterior_concat = np.concatenate(p)  # combine sessions if multiple
        state_max = np.argmax(posterior_concat, axis=1)
        _, counts = np.unique(state_max, return_counts=True)
        counts = counts / np.sum(counts)  # fractional occupancy
        occupancies.append(counts)

    occupancies = np.array(occupancies)
    df = pd.DataFrame(occupancies, columns=[0, 1])
    df.rename(columns={0: 'Disengaged', 1: 'Engaged'}, inplace=True)

    # Melt for seaborn
    df_melt = df.melt(var_name='State', value_name='Occupancy')

    plt.figure(**kwargs, constrained_layout=True)
    sns.boxplot(x='State', y='Occupancy', data=df_melt,
                palette=['tab:gray', 'tab:blue'], showfliers=False)
    sns.stripplot(x='State', y='Occupancy', data=df_melt,
                  color='k', alpha=0.1)
    plt.xlabel('State')
    plt.ylim(0, 1)
    plt.ylabel('Fractional Occupancy')
    sns.despine()


def plot_log_likelihood(log_likelihood, posterior_probs, to_bits=True, **kwargs):
    """
    Plot log likelihood of one or several subjects.
    :param log_likelihood: List of log likelihoods (float) per subject
    :param posterior_probs: List of posterior probabilities (np.array of shape (n_trials, n_states)) per subject. Used
    to compute number of trials for normalization.
    :param to_bits: If True, normalize log likelihood by log(2) to convert to bits
    :param kwargs: Additional keyword arguments for plt.plot()
    :return:
    """

    plt.figure(**kwargs, constrained_layout=True)

    if to_bits:
        n_trials = [sum(p.shape[0] for p in pp) for pp in posterior_probs]

        log_likelihood = [
            (ll / n) / np.log(2) if to_bits else ll / n
            for ll, n in zip(log_likelihood, n_trials)
        ]
        print(f'Log likelihood (normalized by number of trials): '
              f'{np.mean(log_likelihood):.2f} ± {np.std(log_likelihood):.2f} (mean ± SD across subjects)')
        ylabel = 'LL per trial (bits)' if to_bits else 'LL per trial (nats)'
    else:
        print(f'Log likelihood: {np.mean(log_likelihood):.2f} ± {np.std(log_likelihood):.2f} (mean ± SD across subjects)')
        ylabel = 'LL (nats)'

    # Plot boxplot
    if len(log_likelihood) > 1:
        sns.boxplot(y=log_likelihood, color='tab:blue', showfliers=True)
        plt.scatter(np.zeros(len(log_likelihood)), log_likelihood, color='k', alpha=0.1)
    else:
        plt.bar(0, log_likelihood[0], color='tab:blue', edgecolor='k')

    # plt.title('Log Likelihood')
    plt.xlabel('GLM-HMM')
    plt.ylabel(ylabel)
    sns.despine()


def plot_trans_mat_box_plots(trans_mat, **kwargs):
    """
    Plot box plots of transition matrix probabilities across subjects.
    :param trans_mat: List of transition matrices (np.array of shape (n_states, n_states)) per subject
    :param kwargs: Additional keyword arguments for plt.plot()
    :return: None
    """

    if not isinstance(trans_mat, list):
        print('trans_mat must be a list')
        return

    trans_mat = np.array([m.flatten() for m in trans_mat])
    columns = ['D→D', 'D→E', 'E→D', 'E→E']
    df = pd.DataFrame(trans_mat, columns=columns)
    df = df[['D→D', 'E→E']]  # Keep only D→D and E→E (the other two are redundant)
    df_melt = df.melt(var_name='Transition', value_name='Probability')

    # Create a color palette based on mean probabilities
    means = df.mean().to_dict()  # Compute mean probability for each transition
    norm = plt.Normalize(vmin=-1, vmax=1)  # Normalize to 0–1 for colormap sampling
    cmap = cm.get_cmap('bone')
    palette = {k: cmap(norm(v)) for k, v in means.items()}  # Sample bone color according to mean probability

    plt.figure(**kwargs, constrained_layout=True)
    # plt.figure(**kwargs, constrained_layout=True)
    sns.boxplot(x='Transition', y='Probability', data=df_melt, palette=palette, showfliers=False)
    sns.stripplot(x='Transition', y='Probability', data=df_melt,
                  color='k', alpha=0.1, jitter=False)
    # plt.ylim(0, 1)
    plt.ylabel('Probability')
    plt.xlabel('Transition')
    plt.title('Matrix')
    sns.despine()


# # Save results
# path = Path.home() / 'PycharmProjects' / 'glmhmm' / 'results.pkl'
# with open(path, 'wb') as f:
#     pickle.dump(results, f)

# # Load results
# path = Path.home() / 'PycharmProjects' / 'glmhmm' / 'results.pkl'
# with open(path, 'rb') as f:
#     results = pickle.load(f)
#
# # Unpack results
# all_animals = results['all_animals']
# fit_ll = results['fit_ll']
# weights = results['weights']
# trans_mat = results['trans_mat']
# posterior_probs = results['posterior_probs']
# log_likelihood = results['log_likelihood']
# remap_indices = results['remap_indices']
# interpret = results['interpret']
