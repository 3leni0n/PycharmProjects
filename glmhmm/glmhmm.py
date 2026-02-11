import os
from scipy.stats import sem, ttest_1samp, ttest_rel
import pickle
import ssm
from matplotlib import cm
from typing_extensions import no_type_check

from my_fun import get_experiment, add_stars, add_star_between, filter_drug_sessions, filter_behavior, fig_size, timer
from cherry.cherry import *
from kernels.kernels_tools import *
import numpy as np
np.random.seed(42)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_action_trace(df, max_trial_lag=10, tau_choice=1.58, tau_error=2.22, tau_correct=0.95):
    """
    Computes the action trace for each trial in the DataFrame. The action trace is an exponentially weighted sum of past
    choices, where more recent choices have a greater influence. The weights decay exponentially with a time constant
    tau. Output is normalized between -1 and +1.
    :param df: DataFrame containing the data with a column of interest (0=left; 1=right)
    :param max_trial_lag: Number of past trials to consider
    :param tau_choice: Decay constant for choice action trace. Fitted from data (mean across subjects)
    :param tau_error: Decay constant for error action trace. Fitted from data (mean across subjects)
    :param tau_correct: Decay constant for correct action trace. Fitted from data (mean across subjects)
    :return: List of action trace values for each trial for choices, errors and correct trials
    """

    lags = np.arange(1, max_trial_lag + 1)  # Lags from 1 to k

    # Exponential decay weights
    weights_choice = np.exp(-lags / tau_choice)
    weights_error = np.exp(-lags / tau_error)
    weights_correct = np.exp(-lags / tau_correct)

    # Fixed normalizers
    Z_choice = np.sum(weights_choice)
    Z_error = np.sum(weights_error)
    Z_correct = np.sum(weights_correct)

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

        # Slice the weights to match available history and reverse
        w_choice = weights_choice[:len(past_choice)][::-1]
        w_error = weights_error[:len(past_rminus)][::-1]
        w_correct = weights_correct[:len(past_rplus)][::-1]

        at_choice.append(np.sum(past_choice * w_choice) / Z_choice)
        at_error.append(np.sum(past_rminus * w_error) / Z_error)
        at_correct.append(np.sum(past_rplus * w_correct) / Z_correct)

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
                           'session_index', 'prev_choice', 'wsls']
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

        if 'prev_choice' in covariates:
            prev_choice = df_session.Choice.shift(1).fillna(0).values
            session_cols.append(prev_choice)

        if 'wsls' in covariates:
            wsls = df_session.Side.shift(1).fillna(0).replace({0: -1, 1: 1}).values
            session_cols.append(wsls)

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
    return roll_avg[::-1]


def interpret_weights(weights, trans_mat, posterior_probs):
    """
    Interpret the HMM latent states (zt) of one or several subjects based on the GLM weights of its covariates.
    Assign engaged (larger) and disengaged (smaller) depending on the weight of the stimulus covariate (cov_index).
    :param weights: GLM weights of shape (n_states, obs_dim, input_dim)
    :param cov_index: Index of the covariate to use for interpretation
    :return: remapped_weights, remap_indices
    """

    n_states = weights.shape[0]
    stim_weights = weights[:, 0, 0]
    engaged_index = np.argmax(stim_weights)  # Engaged = max(stim)
    # engaged_index = np.argmax(np.abs(cov))

    if n_states == 2:
        disengaged_index = np.argmin(stim_weights)
        remap_indices = [disengaged_index, engaged_index]
    elif n_states == 3:
        others = [s for s in range(len(weights)) if s != engaged_index]
        bias_weights = weights[:, 0, 1]
        biased_left = others[np.argmin(bias_weights[others])]
        biased_right = others[np.argmax(bias_weights[others])]
        remap_indices = [engaged_index, biased_left, biased_right]
    elif n_states == 1 or n_states > 3:
        remap_indices = list(range(n_states))  # No interpretation — identity remap for testing
    else:
        raise ValueError('Only 2 or 3 states supported for interpretation.')

    remapped_weights = weights[remap_indices]
    remapped_trans_mat = trans_mat[np.ix_(remap_indices, remap_indices)]
    remapped_posterior_probs = [p[:, remap_indices] for p in posterior_probs]
    print(f'Remapped weights (states x covariates): {remapped_weights[:, 0, :]}')

    return remapped_weights, remapped_trans_mat, remapped_posterior_probs, remap_indices


def force_lapse_model(glmhmm, lapse_state=0, gamma_l=0.05, gamma_r=0.05):
    """
    Constrain a 2-state GLM-HMM to the classic lapse model:
      - lapse_state has zero stimulus weights and fixed bias
      - transitions are identical rows
    """

    # Zero stimulus weights for lapse state (everything except bias)
    glmhmm.observations.params[lapse_state, 0, 0] = 0.0  # stim weight

    # Set lapse state bias = -log(gamma_l / gamma_r)
    bias_index = 1  # Assuming bias is the second input
    glmhmm.observations.params[lapse_state, 0, bias_index] = -np.log(gamma_l / gamma_r)

    # Constrain transition matrix to have identical rows
    p_lapse = gamma_l + gamma_r
    p_engaged = 1 - p_lapse
    glmhmm.transitions.transition_matrix[:] = np.array([[p_engaged, p_lapse], [p_engaged, p_lapse]])


def force_tiffany_model(glmhmm, covariates=['stim','bias','at_choice']):
    """
    Force selected weights to zero in a 2-state GLM-HMM:
      - disengaged state: stim weight = 0
      - engaged state: action trace weight = 0
    Assumes covariates order matches the list.
    """
    disengaged_state = 0
    engaged_state = 1

    # Get indices of covariates to zero
    stim_idx = covariates.index('stim')
    action_trace_idx = covariates.index('at_choice')

    # Zero weights
    glmhmm.observations.params[disengaged_state, 0, stim_idx] = 0.0
    glmhmm.observations.params[engaged_state, 0, action_trace_idx] = 0.0


def fit_constrained_model(glmhmm, choices, inputs, constrain='lapse', method='em', num_iters=200, tolerance=1e-4):
    lls = []
    for i in range(num_iters):
        ll = glmhmm.fit(
            choices, inputs=inputs, method=method, num_iters=1, initialize=(i==0)
        )
        if constrain == 'lapse':
            force_lapse_model(glmhmm)
        elif constrain == 'tiffany':
            force_tiffany_model(glmhmm, covariates=['stim','bias','at_choice'])
        lls.append(ll[-1])
        if i > 1 and abs(lls[-1] - lls[-2]) < tolerance:
            break


def fit_glmhmm(df, n_states=2, covariates=None, constrain=None, drug=None, save=False):
    """
    Fit GLM-HMM to the data of one subject.
    :param df: DataFrame containing the data of one subject
    :param n_states: Number of discrete states
    :param covariates: List of covariates to include
    :param constrain: String with the contrain to enforce in the model. Options: 'lapse' or' tiffany'
    :param drug: If None, fit rest sessions (no drug nor saline); if 0, fit saline sessions; if 1, fit drug sessions.
    :param save: If True, save the fitted DataFrame to CSV.
    :return: DataFrame with added columns for model fitting results.
    """

    if n_states == 1:
        state_label_map = {0: 'State0'}
    if n_states == 2:
        state_label_map = {0: 'Disengaged', 1: 'Engaged'}
    elif n_states == 3:
        state_label_map = {0: 'Engaged', 1: 'BiasedLeft', 2: 'BiasedRight'}
    elif n_states >= 4:
        state_label_map = {i: f'State{i}' for i in range(n_states)}  # Testing only

    experiment = df.Experiment.unique()[0]

    # Filter data
    df = filter_behavior(df, clean_start=True, drop_miss=True, filter_drug=False)

    # Set output folder
    if n_states == 1 and covariates == ['bias']:  # Null model (weighted Bernoulli coin-flip)
        folder_out = Path.home() / 'PycharmProjects' / 'glmhmm' / 'TEST' / f'{n_states}s_null' / experiment
    elif n_states == 2 and covariates == ['stim_vals', 'bias'] and constrain == 'lapse':  # Classic lapse model
        folder_out = Path.home() / 'PycharmProjects' / 'glmhmm' / 'TEST' / f'{constrain}' / experiment
    elif n_states == 2 and covariates == ['stim_vals', 'bias', 'at_choice'] and constrain == 'tiffany':  # Tiffany et al. (2024)
        folder_out = Path.home() / 'PycharmProjects' / 'glmhmm' / 'TEST' / f'{constrain}' / experiment
    elif n_states == 3 and covariates == ['stim_vals', 'bias', 'prev_choice', 'wsls']:  # Ashwood et al. (2022)
        folder_out = Path.home() / 'PycharmProjects' / 'glmhmm' / 'TEST' / f'ashwood' / experiment
    else:
        folder_out = Path.home() / 'PycharmProjects' / 'glmhmm' / 'TEST' / f'{n_states}s_{len(covariates)}cov' / experiment

    # Drug sessions
    if experiment == '2AFC_6':  # Drug experiment
        if drug is None:
            print('Fitting rest sessions (no drug nor saline)')
            # Keep the sessions where drug is NaN (rest sessions, no saline nor drug)
            df = df[df.Drug.isnull()].reset_index(drop=True)
            condition = 'rest'
        elif drug == 'paired':
            print('Fitting paired drug and saline sessions')
            df = filter_drug_sessions(df)
            condition = 'paired_sessions'
        elif drug in [0, 1]:  # Slice saline (0) or drug (1) sessions
                df = filter_drug_sessions(df)
                df = df[df.Drug == drug].reset_index(drop=True)
                condition = 'saline' if drug == 0 else 'drug'
                print(f'Fitting only {condition} sessions')
        # Get summary df of paired sessions
        # summary = (
        #     df[['Subject', 'Date', 'Drug']]
        #     .drop_duplicates(['Subject', 'Date', 'Drug'])
        #     .sort_values(['Subject', 'Date', 'Drug'])
        #     .reset_index(drop=True)
        # )
        # summary
        folder_out = Path.home() / 'PycharmProjects' / 'glmhmm' / 'pharma' / condition / experiment

    # Parse the data
    inputs, choices = parse_glmhmm(df, covariates=covariates)

    # Set the parameters of the GLM-HMM
    obs_dim = 1  # Number of observed dimensions (1 for binary choice)
    n_categories = 2  # Number of categories for output (2 for binary choice)
    input_dim = inputs[0].shape[1]  # len(covariates)

    # Initialize GLM-HMM
    glmhmm = ssm.HMM(n_states, obs_dim, input_dim, observations='input_driven_obs',
                     observation_kwargs=dict(C=n_categories), transitions='standard')

    # Fit GLM-HMM
    # Stop before completing the number of iterations if LL is below tolerance
    method = 'em'  # Expectation Maximization method
    num_iters = 200 # Max number of EM iterations
    tolerance = 1e-4  # Tolerance for stopping criterion

    if constrain is not None:
        fit_constrained_model(glmhmm, choices, inputs, constrain=constrain, method=method, num_iters=num_iters, tolerance=tolerance)
    else:
        glmhmm.fit(choices, inputs=inputs, method=method, num_iters=num_iters, tolerance=tolerance)

    weights = -glmhmm.observations.params  # Flip sign of weights
    trans_mat = glmhmm.transitions.transition_matrix
    posterior_probs = [glmhmm.expected_states(data=data, input=input)[0]
                       for data, input in zip(choices, inputs)]
    log_likelihood = glmhmm.log_likelihood(choices, inputs=inputs)
    # log_probability.append(glmhmm.log_probabilities(choices, inputs=inputs))

    # Interpret states and remap
    weights, trans_mat, posterior_probs, remap_indices = interpret_weights(weights, trans_mat, posterior_probs)

    posterior_probs = np.concatenate(posterior_probs)
    state_max_posterior = np.argmax(posterior_probs, axis=1)

    # Parameters  (to reshape weights and transition matrix later from DataFrame)
    df['Nstates'] = n_states
    df['ObservedDimensions'] = obs_dim
    df['Ncategories'] = n_categories
    df['InputDimensions'] = input_dim

    # Fitted results
    weights = weights.flatten()
    trans_mat = trans_mat.flatten()
    df['Weights'] = [weights] * len(df)
    df['TransMat'] = [trans_mat] * len(df)
    df['State'] = state_max_posterior
    df['StateLabel'] = df['State'].map(state_label_map)
    df['Remap'] = [remap_indices] * len(df)
    df['LogLikelihood'] = [log_likelihood] * len(df)

    for i in range(n_states):
        df[f'p{i}'] = posterior_probs[:, i]

    if save:
        # Select the output folder and create it if it doesn't exist
        if not os.path.exists(folder_out):
            folder_out.mkdir(parents=True, exist_ok=True)
        subject = df['Subject'].astype(str).str.zfill(3).unique()[0]
        df.to_csv((folder_out / subject).with_suffix('.csv'), index=False)

    return df


@timer
def fit_all(experiments=['2AFC_2', '2AFC_3', '2AFC_4', '2AFC_6'], n_states=2, covariates=None, constrain=None,
            cherry=True, drug=None, save=True):
    """
    Fit GLM-HMM to all subjects of one group and save the results to a CSV file.
    :param experiments: List of experiments to fit
    :param n_states: Number of discrete states (2 or 3)
    :param covariates: List of covariates to include
    :param cherry: If True, cherrypick the best subjects
    :param drug: If None, fit rest sessions (no drug nor saline); if 0, fit saline sessions; if 1, fit drug sessions
    :param save: If True, save the fitted DataFrame to CSV
    :return: DataFrame with added columns for model fitting results for all subjects.
    """

    df_fit_all = pd.DataFrame()

    if cherry:
        cherries = main(experiments)  # Get good subjects from cherry
    else:
        cherries = {}  # All subjects
        for exp in experiments:
            exp, folder_in = get_experiment(exp, path_session='glue_sessions')
            subjects = os.listdir(folder_in)
            subjects = [s for s in subjects if len(s) <= 7]  # Filter only subject data
            subjects.sort()
            cherries[exp] = subjects

    for exp in experiments:
        subjects = cherries[exp]

        for i, subj in enumerate(subjects):
            # Load behavioral data
            exp, folder_in = get_experiment(exp, path_session='glue_sessions')
            print(f'Fitting GLM-HMM for subject {subj} ({i + 1}/{len(subjects)} of Experiment {exp})')
            folder_in = Path(folder_in / subj).with_suffix('.csv')
            print(f'Loading data from {folder_in}')
            df = pd.read_csv(folder_in, low_memory=False)
            try:
                df_fit = fit_glmhmm(df, n_states=n_states, covariates=covariates, constrain=constrain, drug=drug, save=save)
                df_fit_all = pd.concat([df_fit_all, df_fit], ignore_index=True)
            except Exception as e:
                print(f'Error fitting GLM-HMM for subject {subj}: {e}')
                continue
            print('\n')

    return df_fit_all


def get_str_mat(df, col_name):
    """
    Extract string matrix from dataframe and convert back to array
    (pandas stores them as strings when writing to csv). It can be the weights or the transition matrix.
    :params: df: DataFrame containing the data
    :params: col_name: Name of the column containing the string matrices
    """

    n_states = df.Nstates.unique()[0]
    input_dim = df.InputDimensions.unique()[0]
    matrix = df[col_name]

    if col_name == 'Weights':
        dim2 = input_dim
    elif col_name == 'TransMat':
        dim2 = n_states
    else:
        raise ValueError(f"Column name {col_name} not recognized. Use 'Weights' or 'TransMat'.")

    # Convert all strings to arrays
    matrix = np.array([
        np.fromstring(s.replace('\n', '').strip('[]'), sep=' ').reshape(n_states, dim2)
        for s in matrix
    ])

    flattened = matrix.reshape(matrix.shape[0], -1)  # Flatten each 2D array to 1D
    unique_flat = np.unique(flattened, axis=0)  # Keep only unique arrays
    matrix = unique_flat.reshape(-1, matrix.shape[1], matrix.shape[2])  # Reshape back to original 2D shape per array

    return matrix


def iqr_inliers(x):
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    return (x >= q1 - 1.5 * iqr) & (x <= q3 + 1.5 * iqr)


def remove_outliers(weights):
    # Remove outliers based on IQR for each covariate (for zooming in plots)
    n_subjects, n_states, n_cov = weights.shape
    mask = np.ones(n_subjects, dtype=bool)  # Initialize mask with all True
    for i_state in range(n_states):
        for j_cov in range(n_cov):
            inliers = iqr_inliers(weights[:, i_state, j_cov])
            mask &= inliers  # keep only subjects that are inliers for all covariates
    print(f'Kept {mask.sum()} / {n_subjects} subjects after column-wise IQR filtering')
    # weights = weights[mask]
    return mask


def plot_GLMHMM_kernel(df):

    weights = get_str_mat(df, col_name='Weights')
    mask = remove_outliers(weights)
    weights = weights[mask]

    n_animals, n_states, n_cov = weights.shape

    # Plot weights
    mean_weights = weights.mean(axis=0)
    sem_weights = sem(weights, axis=0)

    # Plot each state's mean kernel
    if n_states == 2 and n_cov == 3:
        state_labels = ['Disengaged', 'Engaged']
        colors = ['tab:gray', 'tab:green']
        cov_labels = ['Stim.', 'Bias', r'$A_t$']
        title = '2 states 3 covariates'
    elif n_states == 3 and n_cov == 2:
        state_labels = ['Engaged', 'Left bias', 'Right bias']
        colors = ['tab:green', 'tab:blue', 'tab:orange']
        cov_labels = ['Stim.', 'Bias']
        title = '3 states 2 covariates'

    x = np.arange(len(cov_labels))

    figsize = fig_size(n_cols=2)
    plt.figure(figsize=figsize, constrained_layout=True)
    plt.axhline(y=0, color='k', ls='--')

    # Individuals
    for w in weights:  # w shape: (n_states, n_cov)
        for s in range(n_states):
            plt.plot(x, w[s], color=colors[s % len(colors)], alpha=0.1)

    # Mean ± SEM
    for s in range(n_states):
        plt.errorbar(
            x, mean_weights[s], yerr=sem_weights[s],
            marker='o', lw=3, color=colors[s % len(colors)],
            label=state_labels[s]
        )

    plt.xticks(x, cov_labels)
    # plt.xlabel('Covariates')
    plt.ylabel('Weights')
    plt.title(title)
    plt.legend(frameon=False)
    sns.despine()


# Plotting functions


def results_2_df(df):

    all_animals = df.Subject.unique()
    experiment = df.groupby('Subject')['Experiment'].first().reindex(all_animals).values
    weights = get_str_mat(df, col_name='Weights')
    # mask = remove_outliers(weights)
    # weights = weights[mask]

    n_states = weights.shape[1]
    if n_states == 1:
        state_label_map = {0: 'State0'}
    if n_states == 2:
        state_label_map = {0: 'Disengaged', 1: 'Engaged'}
    elif n_states == 3:
        state_label_map = {0: 'Engaged', 1: 'BiasedLeft', 2: 'BiasedRight'}
    elif n_states >= 4:
        state_label_map = {i: f'State{i}' for i in range(n_states)}  # Testing only

    data = []
    for animal_id, w, exp in zip(all_animals, weights, experiment):
        for state_idx in range(w.shape[0]):
            for cov_idx in range(w.shape[1]):
                data.append({
                    'Animal': animal_id,
                    'Experiment': exp,
                    'State': state_idx,
                    'Label': state_label_map[state_idx],
                    'Covariate': cov_idx,
                    'Weight': w[state_idx, cov_idx]
                })
    data = pd.DataFrame(data)
    data.loc[data['Covariate'] == 1, 'Weight'] = data.loc[data['Covariate'] == 1, 'Weight'].abs()  # Absolute  bias

    return data


def plot_paired_boxplot_GLMHMM_kernel(data, drug=False, **kwargs):
    """
    Plot paired boxplots for engaged vs disengaged weights across all subjects for each covariate.
    :param remapped_weights: List GLM weights per subject of shape (n_states, obs_dim, input_dim) already remapped
    (0 = disengaged, 1 = engaged)
    :param all_animals: List of animal IDs corresponding to remapped_weights
    :param kwargs: Additional keyword arguments for plt.plot()
    :return: None
    """

    title = kwargs.pop('title', 'GLM-HMM kernel')
    loc = kwargs.pop('loc', None)
    bbox = kwargs.pop('bbox_to_anchor', None)
    # palette = kwargs.pop('palette', None)

    # Color scheme and legend labels
    if drug:
        palette = {0: 'tab:gray', 1: 'tab:pink'}
        labels = ['Saline', 'Drug']
        cov_names = ['stim.', '|bias|', '$A_t$']
        hue = 'Drug'
        n_states = data[hue].nunique()
    else:
        hue = 'State'
        n_states = data.State.nunique()
        if n_states == 2:
            palette = {0: 'tab:gray', 1: 'tab:green'}
            labels = ['D', 'E']
            cov_names = ['stim.', '|bias|', '$A_t$']
        elif n_states == 3:
            palette = {0: 'tab:green', 1: 'tab:blue', 2: 'tab:orange'}
            labels = ['E', 'L', 'R']
            cov_names = ['stim.', '|bias|']

    palette = kwargs.pop('palette', palette)

    plt.figure(constrained_layout=True, **kwargs)
    plt.axhline(0, color='tab:gray', linestyle='--')
    ax = sns.boxplot(x='Covariate', y='Weight', hue=hue, data=data, palette=palette, showfliers=False, showcaps=False,
                     fill=False)

    plt.xticks(np.arange(len(cov_names)), cov_names)
    plt.xlabel('')
    plt.title(title)
    handles, _ = ax.get_legend_handles_labels()  # Get legend
    ax.legend(handles, labels, loc=loc,  bbox_to_anchor=bbox, frameon=False)  # Rename legend labels
    sns.despine()

    # Get seaborn box width and compute positions for each state
    width = 0.8
    state_positions = [(i - (n_states - 1) / 2) * width / n_states for i in range(n_states)]

    # Draw paired lines of subjects between boxes (states)
    for cov in sorted(data['Covariate'].unique()):
        for animal in sorted(data['Animal'].unique()):
            subset = data[(data['Covariate'] == cov) & (data['Animal'] == animal)]
            for i in range(n_states - 1):
                x_start = cov + state_positions[i]
                x_end   = cov + state_positions[i + 1]
                y_start = subset[subset[hue] == i]['Weight'].values[0]
                y_end   = subset[subset[hue] == i + 1]['Weight'].values[0]
                ax.plot([x_start, x_end], [y_start, y_end], color='k', alpha=0.1)

    # Compute paired-samples t-tests for each covariate between states
    y_star = ax.get_ylim()[1]
    for cov in sorted(data['Covariate'].unique()):
        if n_states == 2:
            cov_disengaged = data[(data['Covariate'] == cov) & (data[hue] == 0)].sort_values('Animal')['Weight']
            cov_engaged = data[(data['Covariate'] == cov) & (data[hue] == 1)].sort_values('Animal')['Weight']
            t_stat, p_val = ttest_rel(cov_engaged, cov_disengaged)
            print(f'Covariate {cov}: t={t_stat:.3f}, p={p_val:.4f}')
            add_star_between(p_val, x1=cov - abs(state_positions[0]), x2=cov + abs(state_positions[1]), y=y_star)
        if n_states == 3:
            # Get weights for each state
            engaged = data[(data['Covariate'] == cov) & (data[hue] == 0)].sort_values('Animal')['Weight']
            left_bias = data[(data['Covariate'] == cov) & (data[hue] == 1)].sort_values('Animal')['Weight']
            right_bias = data[(data['Covariate'] == cov) & (data[hue] == 2)].sort_values('Animal')['Weight']

            # Engaged vs Left bias
            t_stat, p_val = ttest_rel(engaged, left_bias)
            print(f'Covariate {cov}, Eng-Lbias: t={t_stat:.3f}, p={p_val:.4f}')
            add_star_between(p_val, x1=cov + state_positions[0] + 0.05, x2=cov + state_positions[1] - 0.05, y=y_star)

            # Engaged vs Right bias
            t_stat, p_val = ttest_rel(engaged, right_bias)
            print(f'Covariate {cov}, Eng-Rbias: t={t_stat:.3f}, p={p_val:.4f}')
            add_star_between(p_val, x1=cov + state_positions[0], x2=cov + state_positions[2], y=y_star + y_star * 0.15)

            # Left bias vs Right bias
            t_stat, p_val = ttest_rel(left_bias, right_bias)
            print(f'Covariate {cov}, Lbias-Rbias: t={t_stat:.3f}, p={p_val:.4f}')
            add_star_between(p_val, x1=cov + state_positions[1] + 0.05, x2=cov + state_positions[2] - 0.05, y=y_star)

    # Collect all y positions of star lines or texts
    ax = plt.gca()
    ys = []
    for line in ax.lines:
        ys.extend(line.get_ydata())
    for text in ax.texts:
        ys.append(text.get_position()[1])

    # Include legend top in data coordinates
    legend = ax.get_legend()
    if legend is not None:
        ax.figure.canvas.draw()
        bbox = legend.get_window_extent().transformed(ax.transData.inverted())
        ys.append(bbox.y1)  # top of legend

    # Find highest y among them and expand slightly
    ymax = max(ys)
    ax.set_ylim(top=ymax * 1.1)

    return ax


def plot_trans_mat(trans_mat, **kwargs):
    """
    Plot transition matrix of one or several subjects.
    :param trans_mat: np.array with transition matrix of shape (n_states, n_states)
    :param kwargs: Additional keyword arguments for plt.plot()
    :return: None
    """

    if trans_mat.ndim == 2:  # Target dimension
        pass
    elif trans_mat.ndim == 3:  # Multiple subjects
        trans_mat = trans_mat.mean(axis=0)
    else:  # Invalid dimension
        raise ValueError('trans_mat must be 2D (n_states x n_states) or '
            '3D (n_subjects x n_states x n_states)'        )

    n_states = trans_mat.shape[0]
    if n_states == 2:
        ticklabels = ['D', 'E']  # Short labels
    elif n_states == 3:
        ticklabels = ['E', 'L', 'R']  # Short labels
    else:
        NotImplementedError('n_states must be 2 or 3')

    plt.figure(**kwargs, constrained_layout=True)
    plt.imshow(trans_mat, vmin=-1, vmax=1, cmap='bone', origin='lower')

    for i in range(trans_mat.shape[0]):
        for j in range(trans_mat.shape[1]):
            # text = str(np.around(trans_mat[i, j], decimals=2))
            text = f"{trans_mat[i, j]:.3f}".lstrip('0').replace('-0.', '-.').rstrip('0').rstrip('.')
            plt.text(j, i, text, ha='center', va='center', color='k')

    # plt.xlim(-0.5, n_states + 0.5)
    ticks = range(0, n_states)
    # ticklabels = [str(i) for i in range(n_states)]
    plt.xticks(ticks, ticklabels)
    plt.yticks(ticks, ticklabels)
    # plt.ylim(n_states - 0.5, -0.5)
    plt.ylabel('State $t$')
    plt.xlabel('State $t+1$')
    # plt.title('Transition matrix')

    # Ensure all spines are visible, even if despine() was called outside
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(True)

    return trans_mat


def plot_trans_mat_box_plots(trans_mat, **kwargs):
    """
    Plot box plots of transition matrix probabilities across subjects.
    :param trans_mat: List of transition matrices (np.array of shape (n_states, n_states)) per subject
    :param kwargs: Additional keyword arguments for plt.plot()
    :return: None
    """

    n_subjects, n_states, _ = trans_mat.shape

    if n_states == 2:
        labels = ['D', 'E']
    elif n_states == 3:
        labels = ['D', 'N', 'E']
    else:
        raise NotImplementedError('n_states must be 2 or 3')

    # Flatten each subject
    trans_mat = trans_mat.reshape(n_subjects, n_states * n_states)

    # Generate column names dynamically
    columns = [f'{labels[i]}→{labels[j]}' for i in range(n_states) for j in range(n_states)]

    # Keep only diagonal transitions
    diag_cols = [f'{labels[i]}→{labels[i]}' for i in range(n_states)]
    df = pd.DataFrame(trans_mat, columns=columns)[diag_cols]
    df_melt = df.melt(var_name='Transition', value_name='Probability')

    # Color palette based on mean probabilities
    means = df.mean().to_dict()
    norm = plt.Normalize(vmin=-1, vmax=1)
    cmap = cm.get_cmap('bone')
    palette = {k: cmap(norm(v)) for k, v in means.items()}

    plt.figure(**kwargs, constrained_layout=True)
    sns.boxplot(x='Transition', y='Probability', data=df_melt, palette=palette, showfliers=False)
    # sns.stripplot(x='Transition', y='Probability', data=df_melt, color='k', alpha=0.1, jitter=False)
    plt.ylim(None, 1)
    plt.ylabel('Probability')
    plt.xlabel('Transition')
    plt.title('Matrix')
    sns.despine()


def plot_occupancy(df, **kwargs):
    """
    Plot state occupancies for one or multiple subjects based on posterior probabilities.
    :param df: pd.DataFrame with columns p0, p1, ..., p(n_states)
    """

    # Posterior probability columns
    p_cols = [c for c in df.columns if c.startswith('p')]
    p_cols = sorted(p_cols, key=lambda x: int(x[1:]))

    n_states = df.Nstates.unique()[0]
    if n_states == 2:
        colors = kwargs.pop('color', ['tab:gray', 'tab:green'])
        labels = ['D', 'E']
    elif n_states == 3:
        colors = kwargs.pop('color', ['tab:green', 'tab:blue', 'tab:orange'])
        labels = ['E', 'L', 'R']
    else:
        raise NotImplementedError

    occupancies = []

    for _, g in df.groupby('Subject'):
        posterior_probs = g[p_cols].to_numpy()
        state_max_posterior = np.argmax(posterior_probs, axis=1)
        counts = np.bincount(state_max_posterior, minlength=n_states)
        occupancies.append(counts / counts.sum())

    occupancies = np.asarray(occupancies)
    mean_occupancy = occupancies.mean(axis=0)

    # Plot fractional occupancies
    plt.figure(**kwargs, constrained_layout=True)

    for _, occ in enumerate(mean_occupancy):
        print(f'State {labels[_]} occupancy: {occ:.2f}')
        plt.bar(_, occ, color=colors[_], edgecolor='k')

    plt.xticks(range(len(labels)), labels)
    plt.ylim((0, 1))
    plt.yticks([0, 0.5, 1], ['0', '0.5', '1'])
    plt.ylabel('Occ.')
    sns.despine()


def plot_occupancy_boxplot(df, **kwargs):
    """
    Plot state occupancies across subjects as boxplots.
    Expects columns 'Subject' and 'State' in df.
    """

    plt.figure(constrained_layout=True, **kwargs)

    occupancies = []
    for subject, sub_df in df.groupby('Subject'):
        state_counts = sub_df['State'].value_counts(normalize=True)
        if df['State'].nunique() == 2:
            disengaged = state_counts.get(0)
            engaged = state_counts.get(1)
            labels = ['D', 'E']
            occupancies.append({'Subject': subject, 'Disengaged': disengaged, 'Engaged': engaged})
            palette = kwargs.pop('palette', ['tab:gray', 'tab:green'])
            plt.axhline(0.5, color='tab:gray', ls='--')  # Reference line at 0.5
        elif df['State'].nunique() == 3:
            engaged = state_counts.get(0)
            biased_left = state_counts.get(1)
            biased_right = state_counts.get(2)
            labels = ['E', 'L', 'R']
            occupancies.append({'Subject': subject, 'Engaged': engaged, 'BiasedLeft': biased_left, 'BiasedRight': biased_right})
            palette = kwargs.pop('palette', ['tab:green', 'tab:blue', 'tab:orange'])
            plt.axhline(1/3, color='tab:gray', ls='--')  # Reference line at 0.5
        else:
            raise ValueError('This function only supports 2 or 3 states')

    df_occ = pd.DataFrame(occupancies)
    df_melt = df_occ.melt(id_vars='Subject', var_name='State', value_name='Occupancy')  # Melt for seaborn

    sns.boxplot(x='State', y='Occupancy', data=df_melt, palette=palette, showfliers=False, showcaps=False, fill=False)
    # sns.lineplot(data=df_melt,  # Paired lines per subject. Not needed because sums to 1
    #     x='State', y='Occupancy',
    #     units='Subject', estimator=None,
    #     alpha=0.1, legend=False, color='k'
    # )

    # Paired t-test between states (not needed because sums to 1)
    # t_stat, p_val = ttest_rel(df.Engaged, df.Disengaged)
    # print(f't = {t_stat:.3f}, p = {p_val:.3f}')
    # add_star_between(p_val)

    plt.xticks(np.arange(len(labels)), labels)
    plt.xlabel('')
    plt.ylim(0, 1)
    plt.yticks([0, 0.5, 1], ['0', '0.5', '1'])
    plt.ylabel('Occupancy')
    sns.despine()

    mean_occupancy = df_melt.groupby('State')['Occupancy'].mean()
    print(mean_occupancy)

    return df_occ


def norm_ll(ll, n_trials, ll_null=None, to_bits=True):
    """
    Normalize log likelihood of one or several subjects.
    """

    # Subtract Bernoulli baseline (Ashwood style)
    if ll_null is not None:
        ll_norm = [ll_i - ll_b for ll_i, ll_b in zip(ll, ll_null)]
    else:
        ll_norm = ll

    # Normalize per trial
    ll_norm = [ll_i / n for ll_i, n in zip(ll_norm, n_trials)]

    # Conver to bits/trial
    if to_bits:
        ll_norm = [ll_i / np.log(2) for ll_i in ll_norm]

    return ll_norm


def plot_ll(ll, n_trials, ll_null=None, to_bits=True, positions=[1], **kwargs):
    """
    Plot log likelihood of one or several subjects.
    :param log_likelihood: List of log likelihoods (float) per subject
    :param n_trials: List of number of trials (int) per subject
    :param log_likelihood_bernoulli: List of Bernoulli log likelihoods (float) per subject to subtract as baseline
    :param to_bits: If True, normalize log likelihood by log(2) to convert to bits
    :param kwargs: Additional keyword arguments for plt.plot()
    :return:
    """

    # plt.figure(**kwargs, constrained_layout=True)
    color = kwargs.pop('color', 'k')

    ll = norm_ll(ll, n_trials, ll_null, to_bits=to_bits)

    ylabel = f"LL {'(bits/trial)' if to_bits else '(nats/trial)'}"
    mean_ll = np.mean(ll)
    std_ll = np.std(ll, ddof=1)
    print(f'{mean_ll:.2f} ± {std_ll:.2f} {ylabel}')

    if len(ll) > 1:
        # sns.boxplot(y=log_likelihood, color=color, showfliers=True)
        plt.boxplot(ll, positions=positions, showfliers=False)
        # plt.scatter([positions]*len(ll), ll, color=color, alpha=0.1)
    else:
        plt.bar(positions, ll[0], color=color, edgecolor=color)

    # plt.title('Log Likelihood')
    plt.xlabel('GLM-HMM')
    plt.ylabel(ylabel)
    sns.despine()

    return ll


def plot_model_comparison(n_cov=2, ll_null=None, fit=False):

    if n_cov == 1:
        covariates = ['stim_vals']
        cov_labels = 'stim.'
    elif n_cov == 2:
        covariates = ['stim_vals', 'bias']
        cov_labels = 'stim., bias'
    elif n_cov == 3:
        covariates = ['stim_vals', 'bias', 'at_choice']
        cov_labels = r'stim., bias, $A_t$'
    elif n_cov == 4:
        covariates = ['stim_vals', 'bias', 'at_error', 'at_correct']
        cov_labels = r'stim., bias, $A_{t-}$, $A_{t+}$'

    lls = []
    inliers = []
    for n_states in range(1, 6):  # 1-5 states
        print(f'Loading results of model with {n_states} states and {n_cov} covariates')
        if fit:
            df = fit_all(experiments=['2AFC_2', '2AFC_3', '2AFC_4', '2AFC_6'], n_states=n_states, covariates=covariates, drug=None, save=True)
        else:
            df = glue_groups(experiments=['2AFC_2', '2AFC_3', '2AFC_4'], path_session=f'glmhmm/TEST/{n_states}s_{n_cov}cov')

        ll = df['LogLikelihood'].unique()
        n_trials = df.groupby('Subject').size()
        ll = plot_ll(ll, n_trials, ll_null, to_bits=True, positions=[n_states])
        ll = np.array(ll)
        lls.append(ll)
        inliers.append(iqr_inliers(ll))

    mask = np.logical_and.reduce(inliers)

    for i in range(len(lls) - 1):
        for y_prev, y_next in zip(lls[i][mask], lls[i + 1][mask]):
            plt.plot([i + 1, i + 2], [y_prev, y_next], color='k', alpha=0.1)
        t_stat, p_val = ttest_rel(lls[i][mask], lls[i + 1][mask])
        print(f't = {t_stat:.3f}, p = {p_val:.3f}')
        add_star_between(p_val, i + 1, i + 2)

    plt.xticks([1, 2, 3, 4, 5], ['1', '2', '3', '4', '5'])
    plt.xlabel('N states')
    plt.title(f'{n_cov} Covariates\n({cov_labels})')
    sns.despine()

    return lls


def plot_model_comparison_diffs(n_cov=2, ll_null=None, fit=False):

    if n_cov == 1:
        covariates = ['stim_vals']
        cov_labels = 'stim.'
    elif n_cov == 2:
        covariates = ['stim_vals', 'bias']
        cov_labels = 'stim., bias'
    elif n_cov == 3:
        covariates = ['stim_vals', 'bias', 'at_choice']
        cov_labels = r'stim., bias, $A_t$'
    elif n_cov == 4:
        covariates = ['stim_vals', 'bias', 'at_error', 'at_correct']
        cov_labels = r'stim., bias, $A_{t-}$, $A_{t+}$'

    lls = []
    inliers = []
    for n_states in range(1, 6):  # 1-5 states
        print(f'Loading results of model with {n_states} states and {n_cov} covariates')
        if fit:
            df = fit_all(experiments=['2AFC_2', '2AFC_3', '2AFC_4', '2AFC_6'], n_states=n_states, covariates=covariates, drug=None, save=True)
        else:
            df = glue_groups(experiments=['2AFC_2', '2AFC_3', '2AFC_4'], path_session=f'glmhmm/TEST/{n_states}s_{n_cov}cov')

        ll = df['LogLikelihood'].unique()
        n_trials = df.groupby('Subject').size()
        ll = norm_ll(ll, n_trials, ll_null, to_bits=True)
        ll = np.array(ll)
        lls.append(ll)
        inliers.append(iqr_inliers(ll))

    mask = np.logical_and.reduce(inliers)

   # Compute all consecutive differences
    diffs = [lls[i + 1][mask] - lls[i][mask] for i in range(len(lls) - 1)]

    # Prepare long-form DataFrame for all Δ LLs
    df_plot = pd.DataFrame({'diffs': np.concatenate(diffs),
    'comparison': np.repeat(['2s–1s', '3s–2s', '4s–3s', '5s–4s'],
                            [len(d) for d in diffs])})

    # Plotting Δ LL clouds
    cmap = cm.get_cmap('tab20c')
    colors = [cmap(18), cmap(17), cmap(16)]
    if n_cov ==2:
        color = colors[0]
    elif n_cov ==3:
        color = colors[1]
    elif n_cov ==4:
        color = colors[2]
    sns.stripplot(x='comparison', y='diffs', data=df_plot, color=color, zorder=1)
    y_max = plt.gca().get_ylim()[1]

    p_vals = []
    for i, diff in enumerate(diffs):
        mean_diff = diff.mean()
        ci_low, ci_high = np.percentile(diff, [2.5, 97.5])
        plt.plot([i, i], [ci_low, ci_high], color='k', zorder=2)
        plt.scatter(i, mean_diff, facecolor='white', edgecolor='k', zorder=3)

        # One-sample t-test against 0
        t_stat, p_val = ttest_1samp(diff, 0)
        p_vals.append(p_val)
        print(f'{i+1}→{i+2}: t = {t_stat:.3f}, p = {p_val:.3f}')

    p_vals = np.array(p_vals)
    # add_stars(p_vals, y_max)

    plt.axhline(0, color='k', ls='--')
    plt.xticks(range(4), ['2-1', '3-2', '4-3', '5-4'])
    plt.title(f'{n_cov} Covariates\n({cov_labels})')
    plt.xlabel('N States')
    plt.ylabel('Δ LL (bits/trial)')
    sns.despine()

    return diffs


"""
# Save results
path = Path.home() / 'PycharmProjects' / 'glmhmm' / 'results.pkl'
with open(path, 'wb') as f:
    pickle.dump(results, f)

# Load results
path = Path.home() / 'PycharmProjects' / 'glmhmm' / 'results.pkl'
with open(path, 'rb') as f:
    results = pickle.load(f)

# Unpack results
all_animals = results['all_animals']
fit_ll = results['fit_ll']
weights = results['weights']
trans_mat = results['trans_mat']
posterior_probs = results['posterior_probs']
log_likelihood = results['log_likelihood']
remap_indices = results['remap_indices']
interpret = results['interpret']

results_saline = test_full_model(experiments=['2AFC_6'], drug=0, interpret=True)
results_drug = test_full_model(experiments=['2AFC_6'], drug=1, interpret=True)

# Load results
path = Path.home() / 'PycharmProjects' / 'glmhmm' / 'results_saline.pkl'
with open(path, 'rb') as f:
    results_saline = pickle.load(f)
posterior_probs_saline = results_saline['posterior_probs']
trans_mat_saline = results_saline['trans_mat']
df_occ_saline = plot_occupancy_boxplot(posterior_probs_saline)

# Load results
path = Path.home() / 'PycharmProjects' / 'glmhmm' / 'results_drug.pkl'
with open(path, 'rb') as f:
    results_drug = pickle.load(f)
posterior_probs_drug = results_drug['posterior_probs']
trans_mat_drug = results_drug['trans_mat']
df_occ_drug = plot_occupancy_boxplot(posterior_probs_drug)
"""