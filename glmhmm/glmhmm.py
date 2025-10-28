import os
from scipy.stats import ttest_rel
import pickle
import ssm
from matplotlib import cm
from my_fun import get_experiment, add_star_between, filter_behavior, filter_drug_sessions
from cherry.cherry import *
from kernels.kernels_tools import *
from plotting_style import *
import numpy as np
np.random.seed(42)


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


# New interpret function for 2 states
def interpret_weights(weights, trans_mat, posterior_probs):
    """
    Interpret the HMM latent states (zt) of one or several subjects based on the GLM weights of its covariates.
    Assign engaged (larger) and disengaged (smaller) depending on the weight of the stimulus covariate (cov_index).
    Work for 2 states only.
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
    else:
        raise ValueError('Only 2 or 3 states supported for interpretation.')

    remapped_weights = weights[remap_indices]
    remapped_trans_mat = trans_mat[np.ix_(remap_indices, remap_indices)]
    remapped_posterior_probs = [p[:, remap_indices] for p in posterior_probs]
    print(f'Remapped weights (states x covariates): {remapped_weights[:, 0, :]}')

    return remapped_weights, remapped_trans_mat, remapped_posterior_probs, remap_indices


# New fitting function for 2 states (drug input, so slice sessions in saline vs drug)
def fit_glmhmm(df, n_states=2, covariates=None, drug=None, save=False):
    """
    Fit GLM-HMM to the data of one subject.
    :param df: DataFrame containing the data of one subject
    :param n_states: Number of discrete states (2 or 3)
    :param drug: If None, fit rest sessions (no drug nor saline); if 0, fit saline sessions; if 1, fit drug sessions.
    :param save: If True, save the fitted DataFrame to CSV.
    :return: DataFrame with added columns for model fitting results.
    """

    if n_states not in (2, 3):
        raise ValueError('n_states must be 2 or 3')

    if covariates is None:
        if n_states == 2:
            covariates = ['stim_vals', 'bias', 'at_choice']
            state_label_map = {0: 'Disengaged', 1: 'Engaged'}
        elif n_states == 3:
            covariates = ['stim_vals', 'bias']
            state_label_map = {0: 'Disengaged', 1: 'BiasedLeft', 2: 'BiasedRight'}
    else:
        covariates = covariates

    # Filter data
    # df = filter_behavior(df, drop_miss=True, clean_start=True, filter_drug=False)
    df = df[df.P > 0].reset_index(drop=True)
    df = df.dropna(subset=['Choice']).reset_index(drop=True)

    experiment = df.Experiment.unique()[0]

    # # Drug sessions
    # df = filter_drug_sessions(df)
    # if drug is None:
    #     # Keep the sessions where drug is NaN (rest sessions, no saline nor drug)
    #     df = df[df.Drug.isnull()].reset_index(drop=True)
    #     condition = 'rest'
    #     folder_out = Path.home() / 'PycharmProjects' / 'glmhmm' / '2_states' / experiment
    # elif drug in [0, 1] and experiment == '2AFC_6':
    #     # Slice saline (0) or drug (1) sessions
    #     # df = filter_drug_sessions(df)
    #     df = filter_behavior(df, drop_miss=True, clean_start=True, filter_drug=True)
    #     df = df[df.Drug == drug].reset_index(drop=True)
    #     condition = 'saline' if drug == 0 else 'drug'
    #     folder_out = Path.home() / 'PycharmProjects' / 'glmhmm' / '2_states' / condition / experiment
    folder_out = Path.home() / 'PycharmProjects' / 'glmhmm' / f'{n_states}_states_TEST_unpaired' / experiment

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


def fit_all(experiments=['2AFC_2', '2AFC_3', '2AFC_4', '2AFC_6'], n_states=2, drug=None, save=True):
    """
    Fit GLM-HMM to all subjects of one group and save the results to a CSV file.
    :return: None
    """
    df_fit_all = pd.DataFrame()
    cherries = main(experiments)  # Get good subjects from cherry

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
                df_fit = fit_glmhmm(df, n_states=n_states, covariates=None, drug=drug, save=save)
                df_fit_all = pd.concat([df_fit_all, df_fit], ignore_index=True)
            except Exception as e:
                print(f'Error fitting GLM-HMM for subject {subj}: {e}')
                continue
            print('\n')

    return df_fit_all


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


def results_2_df(all_animals, experiment, weights):
    data = []
    for animal_id, w, exp in zip(all_animals, weights, experiment):
        for state_idx in range(w.shape[0]):
            for cov_idx in range(w.shape[2]):
                data.append({
                    'Animal': animal_id,
                    'Experiment': exp,
                    'State': state_idx,
                    'Label': 'Disengaged' if state_idx == 0 else 'Engaged',
                    'Covariate': cov_idx,
                    'Weight': w[state_idx, 0, cov_idx]
                })
    data = pd.DataFrame(data)
    data.loc[data['Covariate'] == 1, 'Weight'] = data.loc[data['Covariate'] == 1, 'Weight'].abs()  # Absolute  bias
    return data


# def plot_paired_boxplot_GLMHMM_kernel(remapped_weights, all_animals, drug=False, **kwargs):
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

    # data = []
    # for animal_id, w in zip(all_animals, remapped_weights):
    #     for state_idx in range(w.shape[0]):
    #         for cov_idx in range(w.shape[2]):
    #             data.append({
    #                 'Animal': animal_id,
    #                 'State': state_idx,
    #                 # 'Label': 'Disengaged' if state_idx == 0 else 'Engaged',
    #                 'Covariate': cov_idx,
    #                 'Weight': w[state_idx, 0, cov_idx]
    #             })
    # data = pd.DataFrame(data)
    # data.loc[data['Covariate'] == 1, 'Weight'] = data.loc[data['Covariate'] == 1, 'Weight'].abs()  # Absolute  bias

    # Color scheme and legend labels
    if drug:
        palette = {0: 'tab:gray', 1: 'tab:pink'}
        labels = ['Saline', 'Drug']
        hue = 'Drug'
    else:
        palette = {0: 'tab:gray', 1: 'tab:blue'}
        labels = ['Disengaged', 'Engaged']
        hue = 'State'

    palette = kwargs.pop('palette', palette)

    plt.figure(constrained_layout=True, **kwargs)
    plt.axhline(0, color='black', linestyle='--')
    ax = sns.boxplot(x='Covariate', y='Weight', hue=hue, data=data,
                palette=palette, showfliers=False)
    cov_names = ['stim.', '|bias|', '$A_t$']
    # cov_names = ['|2|', '|4|', '|8|', '|70|', 'bias', '$A_t$']
    # cov_names = ['|2|', '|4|', '|8|', '|70|', 'bias', '$A_{t^-}$', '$A_{t^+}$']
    plt.xticks(np.arange(len(cov_names)), cov_names)
    plt.xlabel('')
    plt.title(title)
    handles, _ = ax.get_legend_handles_labels()  # Get legend
    ax.legend(handles, labels, loc=loc,  bbox_to_anchor=bbox, frameon=False)  # Rename legend labels
    sns.despine()

    # Draw paired lines of subjects between boxes (states)
    for cov in sorted(data['Covariate'].unique()):
        for animal in sorted(data['Animal'].unique()):
            subset = data[(data['Covariate'] == cov) & (data['Animal'] == animal)]
            x0 = cov - 0.2  # disengaged box (gray)
            x1 = cov + 0.2  # engaged box (blue)
            y0 = subset[subset[hue] == 0]['Weight'].values[0]
            y1 = subset[subset[hue] == 1]['Weight'].values[0]
            ax.plot([x0, x1], [y0, y1], color='k', alpha=0.1)

    # Compute paired-samples t-tests for each covariate between states
    y_star = ax.get_ylim()[1]
    for cov in sorted(data['Covariate'].unique()):
        cov_disengaged = data[(data['Covariate'] == cov) & (data[hue] == 0)].sort_values('Animal')['Weight']
        cov_engaged = data[(data['Covariate'] == cov) & (data[hue] == 1)].sort_values('Animal')['Weight']
        t_stat, p_val = ttest_rel(cov_engaged, cov_disengaged)
        print(f'Covariate {cov}: t={t_stat:.3f}, p={p_val:.4f}')
        add_star_between(p_val, x1=cov - 0.2, x2=cov + 0.2, y=y_star)

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
    ticklabels = ['D', 'E']  # Short labels
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
    colors = kwargs.pop('color', ['tab:gray', 'tab:blue'])
    labels = ['D', 'E']

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
    # plt.xlabel('State')
    plt.ylabel('Occ.')
    # plt.title('Occupancy')
    sns.despine()


def plot_occupancy_boxplot(posterior_probs, **kwargs):
    """
    Plot state occupancies across subjects as boxplots.

    :param posterior_probs: List of posterior probabilities (np.array of shape n_trials × n_states) per subject
    :param kwargs: Additional keyword arguments for plt.figure()
    """

    palette = kwargs.pop('palette', ['tab:gray', 'tab:blue'])

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
    columns = ['Disengaged', 'Engaged']
    df = pd.DataFrame(occupancies, columns=columns)
    df['Subject'] = df.index

    # Melt for seaborn
    df_melt = df.melt(id_vars='Subject', var_name='State', value_name='Occupancy')
    plt.figure(constrained_layout=True, **kwargs)
    sns.boxplot(x='State', y='Occupancy', data=df_melt,
                palette=palette, showfliers=False)

    # sns.lineplot(data=df_melt,  # Paired lines per subject. Not needed because sums to 1
    #     x='State', y='Occupancy',
    #     units='Subject', estimator=None,
    #     alpha=0.25, legend=False, color='k'
    # )

    # Paired t-test between states (not needed because sums to 1)
    # t_stat, p_val = ttest_rel(df.Engaged, df.Disengaged)
    # print(f't = {t_stat:.3f}, p = {p_val:.3f}')
    # add_star_between(p_val)

    # plt.xlabel('State')
    plt.xlabel('')
    plt.ylim(0, 1)
    plt.ylabel('Fractional occupancy')
    sns.despine()

    return df


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
#
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

# results_saline = test_full_model(experiments=['2AFC_6'], drug=0, interpret=True)
# results_drug = test_full_model(experiments=['2AFC_6'], drug=1, interpret=True)

# Load results
# path = Path.home() / 'PycharmProjects' / 'glmhmm' / 'results_saline.pkl'
# with open(path, 'rb') as f:
#     results_saline = pickle.load(f)
# posterior_probs_saline = results_saline['posterior_probs']
# trans_mat_saline = results_saline['trans_mat']
# df_occ_saline = plot_occupancy_boxplot(posterior_probs_saline)
#
# # Load results
# path = Path.home() / 'PycharmProjects' / 'glmhmm' / 'results_drug.pkl'
# with open(path, 'rb') as f:
#     results_drug = pickle.load(f)
# posterior_probs_drug = results_drug['posterior_probs']
# trans_mat_drug = results_drug['trans_mat']
# df_occ_drug = plot_occupancy_boxplot(posterior_probs_drug)
#
# # Combine occupancy dataframes by state. So one df_engaged and one df_disengaged
# df_occ_saline['Drug'] = 0
# df_occ_drug['Drug'] = 1
# df_occ = pd.concat([df_occ_saline, df_occ_drug], ignore_index=True)

# # Engaged only
# plt.figure()
# sns.boxplot(data=df_occ, x='Drug', y='Engaged', palette=['tab:gray','tab:pink'], showfliers=False)
# plt.xticks([0,1], ['Saline','Drug'])
# plt.ylabel('Occupancy (Engaged)')
# sns.despine()
#
# # Engaged only
# plt.figure()
# sns.boxplot(data=df_occ, x='Drug', y='Disengaged', palette=['tab:gray','tab:pink'], showfliers=False)
# plt.xticks([0,1], ['Saline','Drug'])
# plt.ylabel('Occupancy (Engaged)')
# sns.despine()

# trans_mat_saline = plot_trans_mat(trans_mat_saline)
# trans_mat_drug = plot_trans_mat(trans_mat_drug)
# trans_mat = np.mean([trans_mat_saline, trans_mat_drug], axis=0)
# plot_trans_mat(trans_mat)



