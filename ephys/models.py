from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import sem, ttest_1samp

from ephys.preprocessing import *
from ephys.analysis import *

from my_fun.my_fun import add_stars

sns.set_theme()
sns.set_style('ticks')
sns.set_context('poster')


ephys_ids = [
    '007_2024-06-22_10-48-57',
    '007_2024-06-23_12-46-55',
    '007_2024-06-24_17-47-22',
    '007_2024-06-27_15-06-28',
    '007_2024-07-09_12-10-57',
    '007_2024-07-10_12-03-35',
    '007_2024-07-11_12-39-21',
    '007_2024-07-12_13-29-26'
]

df = pd.DataFrame()

depth = 'cortex'

for i in range(len(ephys_ids)):

    df_ttl, df_behavior, n_trials, df_spikes, cluster_info, x, height, labels, y, width, left, ts_edges, events_edges = \
        preprocess(ephys_ids[i])

    # Select clusters based on depth
    if depth == 'cortex':
        clusters = cluster_info[cluster_info.depth <= 1500].cluster_id
    elif depth == 'deep':
        clusters = cluster_info[cluster_info.depth > 1500].cluster_id
    df_spikes = df_spikes[df_spikes.cluster.isin(clusters)]

    # Add lick data to behavior dataframe
    """
    Qs:
    What causes the variability in the lick RT?
    What causes the variability in the lick rate (ie inter-lick-interval)?
    What makes some correct responses  have 2-3 licks and some 8-10?
    Are all response selective neurons locked to the licks?
    """
    bin_size = 0.02
    time_win = [-2, 0]
    licks, n_licks, rt = get_rt(df_behavior)
    # licks = get_peri_stim_licks(df_behavior)
    bins, licks_psth = compute_psth(licks, time_win=[1, 2], bin_size=bin_size)
    df_behavior['RT'] = rt
    df_behavior['nLicks'] = n_licks
    df_behavior['LickRate'] = np.mean(licks_psth, axis=1) / bin_size

    # Add baseline FR and sync to behavior dataframe
    peri_stim_spikes = get_peri_stim_spikes(df_spikes, df_ttl, time_win=time_win)
    bins, baseline_FR = compute_psth(peri_stim_spikes, time_win=time_win, bin_size=bin_size)
    baseline_FR = np.mean(baseline_FR, axis=1)
    baseline_FR = baseline_FR/len(cluster_info)
    sync = get_sync(df_spikes, df_ttl, time_win=[-2, 3], bin_size=bin_size, method='anal', smooth=True)
    df_behavior['BaselineFR'] = baseline_FR
    df_behavior['Sync'] = sync

    df = pd.concat([df, df_behavior], ignore_index=True)

# Make new column called SessionIndex
df['SessionIndex'] = df.groupby('Session').ngroup()

session_index = pd.get_dummies(df.SessionIndex, dtype='int')
n_sessions = df.Session.nunique()  # Number of sessions
df = pd.concat([df, session_index], axis=1)  # Add session index to the dataframe

# Normalize regressors (per session)
normalize = 'zscore'
if normalize == 'zscore':
    function = lambda x: zscore(x)
elif normalize == 'max':
    function = lambda x: x / x.max()

df['zTrial'] = df.groupby('Session').Trial.transform(function)
df['zSync'] = df.groupby('Session').Sync.transform(function)
df['zBaseFR'] = df.groupby('Session').BaselineFR.transform(function)
df['zLickRate'] = df.groupby('Session').LickRate.transform(function)
df['zRT'] = df.groupby('Session').RT.transform(function)

########################################################################################################################


def run_GLM(df, session_index, type='Hit', plot=True):
    """
    Fit a GLM model to the data of a single session
    :param df: dataframe with the data
    :param session_index: index of the session
    :param type: type of GLM model to fit
    :param plot: plot the results
    :return: parameters, standard errors, p-values
    """

    indexes = df[df.SessionIndex == session_index].index.values  # Indexes of the current session

    endog = df[type]
    if type == 'Hit':
        exog = df[['zTrial', 'zBaseFR', 'zSync']]
        # exog = df[['zBaseFR', 'zSync']]
        color = 'tab:green'
        title = 'Accuracy'
    elif type == 'Miss':
        exog = df[['zTrial', 'zBaseFR', 'zSync']]
        # exog = df[['zBaseFR', 'zSync']]
        color = 'k'
        title = 'Miss'
    elif type == 'RepChoice':
        # exog = df[['zTrial', 'AfterHit', 'RepTrial', 'zBaseFR', 'zSync']]
        exog = df[['AfterHit', 'RepTrial', 'zBaseFR', 'zSync']]
        color = 'tab:brown'
        title = 'Rep. bias'
    elif type == 'zLickRate':
        # exog = df[['zTrial', 'Hit', 'zBaseFR', 'zSync']]
        exog = df[['Hit', 'zBaseFR', 'zSync']]
        color = 'cyan'
        title = 'Lick rate'
    elif type == 'zSync':
        exog = df[['zTrial', 'zBaseFR', 'Hit', 'Miss', 'RepChoice']]
        color = 'k'
        title = 'zSync'

    # Fit the model
    endog = endog.iloc[indexes]
    exog = sm.add_constant(exog)  # Add constant (not needed if adding one intercept per session)
    # exog = pd.concat([exog, session_index], axis=1)  # Add session intercepts
    exog = exog.iloc[indexes]
    model = sm.GLM(endog, exog, family=sm.families.Binomial(), missing='drop')  # GLM with Binomial family
    results = model.fit()
    params = results.params
    bse = results.bse
    p_values = results.pvalues
    summary = results.summary()
    print(summary)
    x = params.index.values
    y = params.values
    yerr = bse

    if plot:
        plt.figure(constrained_layout=True)
        if type == 'RepChoice' or type == 'zSync':
            plt.xticks(rotation=45)
        plt.plot(x, y, c=color, marker='o', ls='none', alpha=1)
        plt.errorbar(x, y, yerr=yerr, color=color, fmt='o', alpha=1)
        plt.axhline(0, color='tab:gray', linestyle='--')
        plt.title(f'{title} (session {session_index}, {len(endog)} trials)')
        plt.xlabel('Coefficients')
        plt.ylabel('Weight')
        plt.legend(frameon=False)
        sns.despine()
        add_stars(p_values, y)

    return params, bse, p_values, color, title


def mean_GLM(df, type='Miss', replace=False, plot=True):

    PARAMS = pd.DataFrame()
    BSE = pd.DataFrame()
    P_VALUES = pd.DataFrame()

    sessions = df.SessionIndex.unique()
    if replace:
        sessions = np.random.choice(sessions, len(sessions), replace=replace)

    for _ in range(df.Session.nunique()):

        print(f'Fitting session {sessions[_]}...')
        params, bse, p_values, color, title = run_GLM(df, sessions[_], type=type, plot=True)


        # Store the results
        PARAMS = pd.concat([PARAMS, pd.Series(params, name=_)], axis=1)
        BSE = pd.concat([BSE, pd.Series(bse, name=_)], axis=1)
        P_VALUES = pd.concat([P_VALUES, pd.Series(p_values, name=_)], axis=1)

    # Perform a one-sample t-test
    res = ttest_1samp(PARAMS.values, 0, axis=1)

    # color = 'tab:red'

    # Plot the average weights
    if plot:
        plt.figure(constrained_layout=True)

        # Plot individual sessions
        for _ in range(df.Session.nunique()):
            plt.plot(PARAMS.index.values, PARAMS[_], c=color, marker='o', ls='none', alpha=0.1)

        if type == 'RepChoice' or type == 'zSync':
            plt.xticks(rotation=45)

        x = PARAMS.index.values
        y = PARAMS.values.mean(axis=1)
        yerr = sem(PARAMS.values, axis=1)
        plt.errorbar(x, y, yerr=yerr, color=color, fmt='o')
        plt.axhline(0, color='tab:gray', linestyle='--')
        plt.title(f'{title} ({n_sessions} sessions, {len(df)} trials)')
        plt.ylabel('Weight')
        sns.despine()
        add_stars(res.pvalue, y)

        return PARAMS, BSE, P_VALUES, res


df_after_error = df[df.AfterHit == 0].reset_index(drop=True)
df_after_correct = df[df.AfterHit == 1].reset_index(drop=True)

# Make a new colum called AfterSync which is the sync value of the next trial
df['AfterSync'] = df.Sync.shift(-1)





# # Number of bootstrap samples
# n_bootstraps = 10
# bootstrap_means = []
#
# # Generate bootstrap samples and calculate the mean
# for _ in range(n_bootstraps):
#     y,PARAMS = plot_model(df, type='Hit', replace=True)
#     bootstrap_means.append(y)
#
# # Calculate 95% confidence interval
# lower_bound = np.percentile(bootstrap_means, 2.5, axis=0)
# upper_bound = np.percentile(bootstrap_means, 97.5, axis=0)
#
# # Plot the confidence interval
# plt.fill_between(PARAMS.index.values, lower_bound, upper_bound, color='tab:green', alpha=0.25, edgecolor='none',
#                  label='95% CI')




# # Plot intercepts
# plt.errorbar(np.repeat(len(params[:-n_sessions]), n_sessions), params.values[-n_sessions:], yerr=bse[-n_sessions:],
#              color=color, fmt='o')
# xticks = plt.xticks()[0] + [len(params[:-n_sessions])]
# xticklabels = [label.get_text() for label in plt.xticks()[1]] + ['Cons']
# plt.xticks(xticks, xticklabels)


# n_shuffles = 1000  # Number of shuffles
# shuffled_params = []
#
# for _ in range(n_shuffles):
#     # Shuffle the dependent variable
#     shuffled_endog = endog.sample(frac=1, random_state=None).reset_index(drop=True)
#
#     # Fit the GLM model with the shuffled data
#     model = sm.GLM(shuffled_endog, exog, family=sm.families.Binomial(), missing='drop')
#     results = model.fit()
#
#     # Store the coefficients (excluding session intercepts if present)
#     shuffled_params.append(results.params.values[:-n_sessions])
#
# # Convert to a NumPy array for easier manipulation
# shuffled_params = np.array(shuffled_params)
#
# lower_bound = np.percentile(shuffled_params, 2.5, axis=0)
# upper_bound = np.percentile(shuffled_params, 97.5, axis=0)
# plt.fill_between(x, lower_bound, upper_bound, color=color, alpha=0.25, edgecolor='none', label='Shuffle Confidence
# Band')


def plot_sync_hist(df, hue='Hit'):
    """
    Plot histogram of sync split by condition
    :param hue: 'Miss', 'RepChoice', 'Hit'
    :return:
    """

    if hue == 'Miss':
        palette = ['tab:gray', 'k']
        labels = ['Miss', 'Resp.']
    elif hue == 'RepChoice':
        palette = ['tab:purple', 'tab:brown']
        labels = ['Rep.', 'Alt.']
    elif hue == 'Hit':
        palette = ['tab:red', 'tab:green']
        labels = ['Hit', 'Error']

    # Make histogram of sync split by miss/response
    plt.figure(constrained_layout=True)
    sns.histplot(data=df, x='zSync', hue=hue, multiple='layer', kde=True, stat='density', bins='auto', common_norm=False,
                 palette=palette)
    plt.title(f'Sync dist. ({df.Session.nunique()} sessions, {len(df)} trials)')
    sns.despine()
    plt.legend(labels=labels, frameon=False, loc='upper right')


# Compute the mean sync per session
# sync = df.groupby('SessionIndex').Sync.mean()

# Compute the mean sync per session but only for sessions with indexes 1, 2, 3
# sync = df[df.SessionIndex.isin([1, 2, 3])].groupby('SessionIndex').Sync.mean()

# for i in df.SessionIndex.unique():
#     print(i)
#     plot_sync_hist(df[df.SessionIndex == i], hue='Hit')




# # Generalized Linear Mixed Effects Models
# endog = df.Hit
# exog = df[['zTrial', 'zBaseFR', 'zSync']]
# exog_vc = session_index
# ident = np.zeros(exog_vc.shape[1], dtype=int)
# model = sm.BinomialBayesMixedGLM(endog, exog, exog_vc, ident)
#
#
#
# random = {"a": '0 + C(SessionIndex)'}
# model = sm.BinomialBayesMixedGLM.from_formula(
#                'Miss ~ zTrial + zBaseFR + zSync', random, df)
# result = model.fit_vb()
# print(result.summary())
#
# plt.figure(constrained_layout=True)
# plt.errorbar(np.arange(4), result.fe_mean, result.fe_sd, color='k', fmt='o')
# plt.axhline(0, color='tab:gray', linestyle='--')