"""
To do:
- Choice history regressors to re-start every session
"""

import time
from pathlib import Path
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from matplotlib import pyplot as plt
from scipy import stats
import seaborn as sns
from collections import namedtuple
from my_fun.my_fun import get_experiment, get_animal, save_fig, timer
from kernels.kernels_tools import *
from plotting_style import *


# GLM weights of previously rewarded (r+) and previously unrewarded (r-) responses. These kernels quantify the
# influence on choice of the side (left vs. right) of previous responses.
# From 'Response outcomes gate the impact of expectations on perceptual decisions', Figure 4
# (https://www-nature-com.sire.ub.edu/articles/s41467-020-14824-w)

@timer
def get_hk(experiment=None, animal=None, residuals=True, zscore=False, drug=None, trial_lag=10, iterations=1000):
    """
    Get history kernels for a given animal. The history kernel is the GLM weight of previously rewarded (r+) and
    previously unrewarded (r-) responses (choices). These kernels quantify the influence on choice of the side (left vs.
    right) of previous responses.
    :param experiment: str, name of the experiment
    :param animal: str, name of the animal
    :param drug: bool, drug session to analyze. If None, all sessions are analyzed
    :param trial_lag: int, number of trials to consider in the past
    :param iterations: int, number of iterations to compute shuffles of the kernel
    :return: hk: namedtuple, history kernel
    """

    # Get the path to the data
    experiment, folder_in = get_experiment(experiment, path_session='glue_sessions')
    # experiment, folder_in = get_experiment(experiment, path_session='glmhmm')  # For engagement data
    animal = get_animal(experiment=experiment, animal=animal)
    folder_in = Path(folder_in / animal).with_suffix('.csv')

    ####################################################################################################################

    # Load behavioral data
    df = pd.read_csv(folder_in)

    ####################################################################################################################

    # Load intersession data
    # path_intersession = '/home/alexis/PycharmProjects/intersession/' + experiment + '/' + animal + '_intersession.csv'
    path_intersession = Path.home() / 'PycharmProjects' / 'intersession' / experiment / (animal + '_intersession.csv')
    df_intersession = pd.read_csv(path_intersession)

    # There are some short, corrupted sessions (dates) for which there is no intersession data because one of the values
    # for some of the columns is empty. Remove them from trial data
    dates_trials = df.Date.unique()
    dates_intersession = df_intersession.Dates.unique()
    dates_to_remove = [x for x in dates_trials if x not in dates_intersession]
    df = df[~df.Date.isin(dates_to_remove)]

    # Add intersession data to df. Needs to be done before filtering out trials so lengths match
    session_index = []
    accuracy = []
    accuracy_left = []
    accuracy_right = []
    for i in range(len(df_intersession)):
        session_index += [df_intersession.index.values[i]] * df_intersession.Trials[i]
        accuracy += [df_intersession.Accuracy[i]] * df_intersession.Trials[i]
        accuracy_left += [df_intersession.AccuracyLeft[i]] * df_intersession.Trials[i]
        accuracy_right += [df_intersession.AccuracyRight[i]] * df_intersession.Trials[i]
    df['SessionIndex'] = session_index
    df['Accuracy'] = accuracy
    df['AccuracyLeft'] = accuracy_left
    df['AccuracyRight'] = accuracy_right

    ####################################################################################################################

    # Filter trials
    df = df[df.Choice.notna()]  # Drop misses (nan in choices), otherwise the code crashes
    df = df[df.P>0]
    # ilds = np.sort(df.ILD.unique())
    # df = df[df.ILD.isin(target_ilds)]  # Select only trials with the desired ILDs
    # # df = df[df.Hit == 1]  # Only correct trials
    accuracy_threshold = 0.5
    df = df[(df.AccuracyLeft >= accuracy_threshold) & (df.AccuracyRight >= accuracy_threshold)]  # Select only trials
    # with accuracy >= threshold

    ####################################################################################################################

    # Drug sessions/trials
    if drug is not None:  # Select drug session trials
        df_drug = pd.read_csv(Path.home() / 'PycharmProjects' / 'drugs' / 'Mouse injections MK801.csv')  # Load drug data
        df = df[df.Drug.notnull()]
        df_intersession = df_intersession[df_intersession.Drug.notnull()]
        # drug_session_dates = df_intersession[df_intersession.Drug == drug].Dates

        # Get paired rest sessions prior to drug inyection
        if drug == 'rest':
            mk801_indexes = df_intersession[df_intersession.Drug == 'MK801'].Dates.index.values
            n_mk801_indexes = len(mk801_indexes)
            paired_rest_indexes = df_intersession[df_intersession.Drug == 'rest'].Dates.index.values
            paired_rest_indexes = np.random.choice(paired_rest_indexes, n_mk801_indexes, False)
            paired_rest_dates = df_intersession.Dates[paired_rest_indexes]
            df = df[df.Date.isin(paired_rest_dates)]

        # Get paired saline sessions prior to drug inyection
        if drug == 'saline':
            mk801_indexes = df_intersession[df_intersession.Drug == 'MK801'].Dates.index.values
            paired_saline_indexes = mk801_indexes - 1
            paired_saline_dates = df_intersession.Dates[paired_saline_indexes]
            df = df[df.Date.isin(paired_saline_dates)]

        df = df[df.Drug == drug]
    else:  # Don't select drug session trials
        try:
            df = df[df.Drug.isnull()]  # Remove drug experimental sessions
        except AttributeError:
            pass

    # Drop sessions in which the animal didn't do the task. This can be achieved by using an accuracy threshold of 0.5
    # df.drop(index=df[(df.Date == '2022-05-24') & (df.Setup == 337)].index, inplace=True)  # Left accuracy 4%
    # df.drop(index=df[(df.Date == '2022-05-25') & (df.Setup == 337)].index, inplace=True)  # Left accuracy 3%
    # df.drop(index=df[(df.Date == '2022-06-01') & (df.Setup == 337)].index, inplace=True)  # Left accuracy 43%
    # df.drop(index=df[(df.Date == '2022-05-26') & (df.Setup == 332)].index, inplace=True)  # Miss rate 50%
    # df.drop(index=df[(df.Date == '2022-05-27') & (df.Setup == 333)].index, inplace=True)  # Miss rate 83%

    ####################################################################################################################

    # Make design matrix
    #
    df = df.reset_index(drop=True)
    n_trials = len(df)

    # Make stimulus strength design matrix

    # Set stimuli set
    if experiment == '2AFC_6':
        stim_set = 6
    elif experiment == '2AFC':
        stim_set = 1
    else:
        stim_set = 2

    stim_strength, n_frames = make_frames_dm(df, stim_set=stim_set, residuals=residuals, zscore=zscore)

    dm_choice_history = make_choice_history_dm(df, trial_lag)
    dm_session_index = make_session_index_dm(df)
    dm_ild = make_net_ild_dm(df)
    exog = pd.concat([dm_choice_history, dm_session_index, dm_ild, stim_strength], axis=1)
    exog = exog.iloc[trial_lag:, :].reset_index(drop=True)  # To remove the nans

    ####################################################################################################################

    endog = df.Choice
    endog = endog.iloc[trial_lag:].reset_index(drop=True)  # To remove the nans
    model = sm.GLM(endog, exog, family=sm.families.Binomial(), missing='drop')  # GLM with Binomial family
    results = model.fit()
    params = results.params
    bse = results.bse
    p_values = results.pvalues
    summary = results.summary()
    print(summary)

    # Rpminus
    params_rminus = params.iloc[:trial_lag]  # From params
    shuffles_rminus = get_shuffles_GLM(endog, exog, iterations, kind='hk_rminus')
    shuffles_rminus = [shuffles_rminus[i].iloc[:trial_lag] for i in range(len(shuffles_rminus))]  # From shuffles
    bse_rminus = bse.iloc[:trial_lag]  # From bse
    p_values_rminus = p_values.iloc[:trial_lag]  # From p_values

    # Rplus
    params_rplus = params.iloc[trial_lag:trial_lag * 2]  # From params
    shuffles_rplus = get_shuffles_GLM(endog, exog, iterations, kind='hk_rplus')
    shuffles_rplus = [shuffles_rplus[i].iloc[trial_lag:trial_lag * 2] for i in range(len(shuffles_rplus))]  # From shuffles
    bse_rplus = bse.iloc[trial_lag:trial_lag * 2]  # From bse
    p_values_rplus = p_values.iloc[trial_lag:trial_lag * 2]  # From p_values

    # Session index
    params_session_index = params.iloc[trial_lag * 2:-4 - n_frames]
    bse_session_index = bse.iloc[trial_lag * 2:-4 - n_frames]
    p_values_session_index = p_values.iloc[trial_lag * 2:-4 - n_frames]
    shuffles_session_index = get_shuffles_GLM(endog, exog, iterations, kind='hk_session_index')
    shuffles_session_index = [shuffles_session_index[i].iloc[trial_lag * 2:-4 - n_frames] for i in range(len(shuffles_session_index))]  # From shuffles

    # Net ILD
    params_net_stim = params.iloc[-4 - n_frames:-n_frames]
    bse_net_stim = bse.iloc[-4 - n_frames:-n_frames]
    p_values_net_stim = p_values.iloc[-4 - n_frames:-n_frames]
    shuffles_net_stim = get_shuffles_GLM(endog, exog, iterations, kind='hk_net_stim')
    shuffles_net_stim = [shuffles_net_stim[i].iloc[-4 - n_frames:-n_frames] for i in range(len(shuffles_net_stim))]  # From shuffles

    # Stimulus strength
    params_frames = params[-n_frames:]
    bse_frames = bse[-n_frames:]
    p_values_frames = p_values[-n_frames:]
    shuffles_frames = get_shuffles_GLM(endog, exog, iterations, kind='pk_frames')
    shuffles_frames = [shuffles_frames[i].iloc[-n_frames:] for i in range(len(shuffles_frames))]  # From shuffles

    # Store results in a namedtuple
    HK = namedtuple('HK', [
        # params
        'params_rminus',
        'params_rplus',
        'params_session_index',
        'params_net_stim',
        'params_frames',

        # bse
        'std_err_rminus',
        'std_err_rplus',
        'std_err_session_index',
        'std_err_net_stim',
        'std_err_frames',

        # p_values
        'p_values_rminus',
        'p_values_rplus',
        'p_values_session_index',
        'p_values_net_stim',
        'p_values_frames',

        # shuffles
        'shuffles_rminus',
        'shuffles_rplus',
        'shuffles_session_index',
        'shuffles_net_stim',
        'shuffles_frames',

        # metadata
        'n_trials',
        'trial_lag',
        'experiment',
        'animal',
        'drug',
        'iterations',
        'n_frames'
    ])

    hk = HK(
        # params
        params_rminus=params_rminus,
        params_rplus=params_rplus,
        params_session_index=params_session_index,
        params_net_stim=params_net_stim,
        params_frames=params_frames,

        # bse
        std_err_rminus=bse_rminus,
        std_err_rplus=bse_rplus,
        std_err_session_index=bse_session_index,
        std_err_net_stim=bse_net_stim,
        std_err_frames=bse_frames,

        # p_values
        p_values_rminus=p_values_rminus,
        p_values_rplus=p_values_rplus,
        p_values_session_index=p_values_session_index,
        p_values_net_stim=p_values_net_stim,
        p_values_frames=p_values_frames,

        # shuffles
        shuffles_rminus=shuffles_rminus,
        shuffles_rplus=shuffles_rplus,
        shuffles_session_index=shuffles_session_index,
        shuffles_net_stim=shuffles_net_stim,
        shuffles_frames=shuffles_frames,

        # metadata
        n_trials=n_trials,
        trial_lag=trial_lag,
        experiment=experiment,
        animal=animal,
        drug=drug,
        iterations=iterations,
        n_frames=n_frames
    )

    return hk


def plot_hk(experiment=None, animal=None, drug=None, trial_lag=10, iterations=1000, save=False):
    """
    Plot the history kernel of a given animal.
    :param experiment: str, name of the experiment
    :param animal: str, name of the animal
    :param drug: bool, drug session to analyze. If None, all sessions are analyzed
    :param trial_lag: int, number of trials to consider in the past
    :param iterations: int, number of iterations to compute shuffles of the kernel
    :param save: bool, whether to save the figure or not
    :return:
    """

    if type(experiment) == list:
        hk = get_mean_hk(experiments=['2AFC_2', '2AFC_3'], animals=None, drug=None, trial_lag=trial_lag,
                         iterations=iterations)
        title = f'N={len(hk.animal)}, {hk.n_trials} trials'
        filename_prefix = 'mean_'
    else:
        hk = get_hk(experiment=experiment, animal=animal, drug=drug, trial_lag=trial_lag, iterations=iterations)
        title = f'Mouse {hk.animal}, {hk.n_trials} trials'
        filename_prefix = ''

    # Default plotting parameters
    color = 'k'
    color_upper_shuffle = 'tab:red'
    label = ''

    trial_lag = hk.trial_lag

    for _ in range(1, 3):

        if _ == 1:  # R+
            xlabel = 'Trial lag (' + '$r^{-}$)'
            # params_indexes = np.arange(trial_lag * _)
            filename = f'{hk.animal} r- HK, trial lag {hk.trial_lag}'
            y = hk.params_rminus
            yerr = hk.std_err_rminus
            shuffles = hk.shuffles_rminus
        elif _ == 2:  # R-
            xlabel = 'Trial lag (' + '$r^{+}$)'
            # params_indexes = np.arange(trial_lag, trial_lag * _)
            filename = f'{hk.animal} r+ HK, trial lag {hk.trial_lag}'
            y = hk.params_rplus
            yerr = hk.std_err_rplus
            shuffles = hk.shuffles_rplus

        ################################################################################################################

        # PLOT HISTORY KERNEL
        figsize = fig_size(n_cols=2)
        plt.figure(figsize=figsize, constrained_layout=True)
        x = np.arange(1, trial_lag + 1)
        # y = hk.params.iloc[params_indexes]
        # yerr = hk.std_err.iloc[params_indexes]
        plt.plot(x, y, color=color, marker='o', label=label)
        plt.errorbar(x, y, yerr=yerr, color=color, marker='o', fmt='none', mec='none', ms=0)
        plt.title(title)
        plt.xticks(x, x[::-1])
        plt.xlabel(xlabel)
        ylabel = 'GLM weight'
        plt.ylabel(ylabel)
        plt.legend(frameon=False)
        # yticks = plt.gca().get_yticks()  # Get current axis yticks for the significance annotations

        # if hk.p_values is not None:
        #     for i in range(n_trials_lag):
        #         if hk.p_values[i] <= 0.05:
        #             text = '*'
        #         else:
        #             # text = 'ns'
        #             text = ''
        #         plt.annotate(text, xy=(i + 1, yticks[1]), xytext=(i + 1 + int(residuals), yticks[1]),
        #                      color=color, va='center', ha='center', fontsize='medium')

        shuffles_mean = np.mean(shuffles, axis=0)  # Get the mean of all the shuffles
        percentiles2dot5 = np.percentile(shuffles, 2.5, axis=0)  # Get upper 5 percentile of the shuffled_var
        percentiles97dot5 = np.percentile(shuffles, 97.5, axis=0)  # Get upper 5 percentile of the shuffled_var
        plt.plot(x, shuffles_mean, color='tab:gray', ls='--', zorder=1.8)
        plt.plot(x, percentiles2dot5, color=color_upper_shuffle, ls=':', zorder=1.9)
        plt.plot(x, percentiles97dot5, color=color_upper_shuffle, ls=':', zorder=2)

        sns.despine()

        if save:
            filename = filename_prefix + 'prev_resp' + filename
            folder_out = Path.home() / 'Documentos' / 'kernels' / 'HK' / experiment
            save_fig(folder_out, filename)
            plt.close()

    ################################################################################################################

    # PLOT NET STIMULUS KERNEL
    figsize = fig_size(n_cols=2)
    plt.figure(figsize=figsize, constrained_layout=True)
    x = np.arange(len(hk.params_net_stim.index.astype(int).values))
    y = hk.params_net_stim
    yerr = hk.std_err_net_stim
    plt.plot(x, y, color=color, marker='o')
    plt.errorbar(x, y, yerr=yerr, color=color, marker='o', fmt='none', mec='none', ms=0)
    plt.title(title)
    plt.xlabel('Net stimuli')
    plt.ylabel('GLM weight')
    # plt.legend(frameon=False)
    plt.xticks(x, ['2', '4', '8', '70'])

    shuffles_mean = np.mean(hk.shuffles_net_stim, axis=0)  # Get the mean of all the shuffles
    percentiles95 = np.percentile(hk.shuffles_net_stim, 95, axis=0)  # Get upper 5 percentile of the shuffled_var
    plt.plot(x, shuffles_mean, color='tab:gray', ls='--', zorder=1.8)
    plt.plot(x, percentiles95, color='tab:red', ls=':', zorder=1.9)
    sns.despine()

    if save:
        filename = filename_prefix + 'net_stim' + filename
        folder_out = Path.home() / 'Documentos' / 'kernels' / 'HK' / experiment
        save_fig(folder_out, filename)
        plt.close()

    ####################################################################################################################

    # PLOT SESSION INDEX KERNEL

    if type(experiment) == str:  # Don't do it for the mean kernel

        figsize = (fig_size(n_cols=1)[0], fig_size(n_cols=2)[1])
        plt.figure(figsize=figsize, constrained_layout=True)
        x = hk.params_session_index.index.values
        y = hk.params_session_index
        yerr = hk.std_err_session_index
        plt.plot(x, y, color=color, marker='o')
        plt.errorbar(x, y, yerr=yerr, color=color, marker='o', fmt='none', mec='none', ms=0)
        plt.title(title)
        plt.xlabel('Session index')
        plt.ylabel('GLM weight')
        # plt.legend(frameon=False)
        # plt.xticks(x, ['2', '4', '8', '70'])
        shuffles_mean = np.mean(hk.shuffles_session_index, axis=0)  # Get the mean of all the shuffles
        percentiles2dot5 = np.percentile(hk.shuffles_session_index, 2.5,
                                         axis=0)  # Get upper 5 percentile of the shuffled_var
        percentiles97dot5 = np.percentile(hk.shuffles_session_index, 97.5,
                                          axis=0)  # Get upper 5 percentile of the shuffled_var
        plt.plot(x, shuffles_mean, color='tab:gray', ls='--', zorder=1.8)
        plt.plot(x, percentiles2dot5, color='tab:red', ls=':', zorder=1.85)
        plt.plot(x, percentiles97dot5, color='tab:red', ls=':', zorder=1.9)
        sns.despine()

        if save:
            filename = 'session_index' + filename
            folder_out = Path.home() / 'Documentos' / 'kernels' / 'HK' / experiment
            save_fig(folder_out, filename)
            plt.close()

    ####################################################################################################################

    # PLOT STIMULUS FRAMES KERNEL

    figsize = fig_size(n_cols=2)
    plt.figure(figsize=figsize, constrained_layout=True)

    n_frames = hk.n_frames
    ylabel = 'GLM weight (residuals)'

    x = np.arange(1, n_frames + 1)
    y = hk.params_frames
    yerr = hk.std_err_frames
    plt.plot(x, y, color=color, marker='o', label=label)
    plt.errorbar(x, y, yerr=yerr, color=color, marker='o', fmt='none', mec='none', ms=0)
    plt.title(title)
    plt.xlabel('Stimulus frame')
    plt.ylabel(ylabel)
    plt.legend(frameon=False)

    # if pk.p_values is not None:
    #     for i in range(n_frames):
    #         if pk.p_values[i] <= 0.05:
    #             text = '*'
    #         else:
    #             # text = 'ns'
    #             text = ''
    #         plt.annotate(text, xy=(i + 1 + int(residuals), yticks[1]), xytext=(i + 1 + int(residuals), yticks[1]),
    #                      color=color, va='center', ha='center', fontsize='medium')

    shuffles_mean = np.mean(hk.shuffles_frames, axis=0)  # Get the mean of all the shuffles
    percentiles95 = np.percentile(hk.shuffles_frames, 95, axis=0)  # Get upper 5 percentile of the shuffled_var
    plt.plot(x, shuffles_mean, color='tab:gray', ls='--', zorder=1.8)
    plt.plot(x, percentiles95, color=color_upper_shuffle, ls=':', zorder=1.9)

    sns.despine()

    if save:
        filename = f'{hk.animal}_PK_ILDs'
        filename = filename_prefix + filename
        folder_out = Path.home() / 'OneDrive' / 'Imágenes' / 'Figures' / 'kernels' / 'PK' / experiment
        if not folder_out.exists():
            folder_out.mkdir(parents=True, exist_ok=True)
        save_fig(folder_out, filename)
        plt.close()


def plot_hks(experiment='2AFC_2', animals=['325', '327', '329', '330', '332', '333', '335', '337'], drug=None,
             trial_lag=10, iterations=1000, save=False):
    """
    Plot the history kernels of all animals of a given batch.
    :param experiment: str, name of the experiment
    :param animal: str, name of the animal
    :param drug: bool, drug session to analyze. If None, all sessions are analyzed
    :param trial_lag: int, number of trials to consider in the past
    :param iterations: int, number of iterations to compute shuffles of the kernel
    :param save: bool, whether to save the figure or not
    :return:
    """

    experiment, folder_in = get_experiment(experiment, path_session='glue_sessions')
    for i in range(len(animals)):
        print(f'Plotting kernel of animal {animals[i]} ({i + 1}/{len(animals)})')
        plot_hk(experiment=experiment, animal=animals[i], drug=drug, trial_lag=trial_lag, iterations=iterations,
                save=save)


def get_mean_hk(experiments=['2AFC_2', '2AFC_3'], animals=None, drug=None, trial_lag=10, iterations=1000):
    """
    Get the mean peak of the history kernel of all animals of a given batch.
    :param experiments: list of str, name of the experiments
    :param animals: list of str, name of the animal
    :param drug: bool, drug session to analyze. If None, all sessions are analyzed
    :param trial_lag: int, number of trials to consider in the past
    :param iterations: int, number of iterations to compute shuffles of the kernel
    :return:
    """

    experiments_across_batches = []
    animals_across_batches = []

    params_rminus_across_animals = []
    params_rplus_across_animals = []
    params_session_index_across_animals = []
    params_net_stim_across_animals = []
    params_frames_across_animals = []

    shuffles_rminus_across_animals = []
    shuffles_rplus_across_animals = []
    shuffles_net_stim_across_animals = []
    shuffles_frames_across_animals = []

    n_trials_across_animals = []
    pks = []

    for j in range(len(experiments)):

        experiment, folder_in = get_experiment(experiments[j], path_session='glue_sessions')

        if experiments[j] == '2AFC_2':
            animals = ['325', '327', '329', '330', '332', '333', '335', '337']
            # animals = ['332', '333', '337']  # Drug experiments
        elif experiments[j] == '2AFC_3':
            animals = ['419', '420', '422', '616', '619', '623']

        for i in range(len(animals)):
            print(f'Getting kernel of animal {animals[i]} ({i + 1}/{len(animals)})')
            hk = get_hk(experiment=experiment, animal=animals[i], drug=drug, trial_lag=trial_lag, iterations=iterations)
            animals_across_batches.append(hk.animal)

            params_rminus_across_animals.append(hk.params_rminus)
            params_rplus_across_animals.append(hk.params_rplus)
            params_session_index_across_animals.append(hk.params_session_index)
            params_net_stim_across_animals.append(hk.params_net_stim)
            params_frames_across_animals.append(hk.params_frames)

            shuffles_rminus_across_animals.append(hk.shuffles_rminus)
            shuffles_rplus_across_animals.append(hk.shuffles_rplus)
            shuffles_net_stim_across_animals.append(hk.shuffles_net_stim)
            shuffles_frames_across_animals.append(hk.shuffles_frames)

            n_trials_across_animals.append(hk.n_trials)
            pks.append(hk)
        experiments_across_batches.append(experiment)

    n_trials = sum(n_trials_across_animals)
    n_frames = hk.n_frames

    # Get mean and sem of the R+ and R- across animals
    params_rminus_across_animals = np.array(params_rminus_across_animals)
    params_rminus_mean_across_animals = np.mean(params_rminus_across_animals, 0)
    params_rminus_sem_across_animals = stats.sem(params_rminus_across_animals, 0)
    params_rplus_across_animals = np.array(params_rplus_across_animals)
    params_rplus_mean_across_animals = np.mean(params_rplus_across_animals, 0)
    params_rplus_sem_across_animals = stats.sem(params_rplus_across_animals, 0)

    # Get the session index parameters across animals
    # params_session_index_across_animals = np.array(params_session_index_across_animals)
    # Pad session index arrays to the same length
    max_len = max(len(arr) for arr in params_session_index_across_animals)
    padded_arrays = [np.pad(arr, (0, max_len - len(arr)), mode='constant', constant_values=np.nan)
                     for arr in params_session_index_across_animals]
    params_session_index_across_animals = np.array(padded_arrays)
    params_session_index_mean_across_animals = np.nanmean(params_session_index_across_animals, axis=0)
    params_session_index_sem_across_animals = stats.sem(params_session_index_across_animals, axis=0, nan_policy='omit')

    # Get mean and sem of the net stimuli parameters across animals
    params_net_stim_across_animals = np.array(params_net_stim_across_animals)
    params_net_stim_mean_across_animals = np.mean(params_net_stim_across_animals, 0)
    params_net_stim_sem_across_animals = stats.sem(params_net_stim_across_animals, 0)

    # Get mean and sem of the stimulus frames parameters across animals
    params_frames_across_animals = np.array(params_frames_across_animals)
    params_frames_mean_across_animals = np.mean(params_frames_across_animals, 0)
    params_frames_sem_across_animals = stats.sem(params_frames_across_animals, 0)

    # Get the mean and percentile 95 of the shuffles across animals
    shuffles_rminus_across_animals = np.array(shuffles_rminus_across_animals)  # Convert list of lists to 3 dim array (animal x
    # iterations x params)
    shuffles_rminus_means_across_animals = np.mean(shuffles_rminus_across_animals, 0)
    shuffles_rplus_across_animals = np.array(shuffles_rplus_across_animals)  # Convert list of lists to 3 dim array (animal x
    # iterations x params)
    shuffles_rplus_means_across_animals = np.mean(shuffles_rplus_across_animals, 0)
    shuffles_net_stim_across_animals = np.array(shuffles_net_stim_across_animals)
    shuffles_net_stim_means_across_animals = np.mean(shuffles_net_stim_across_animals, 0)
    shuffles_frames_across_animals = np.array(shuffles_frames_across_animals)
    shuffles_frames_means_across_animals = np.mean(shuffles_frames_across_animals, 0)

    # Store results in a namedtuple
    params_rminus_mean_across_animals = pd.Series(params_rminus_mean_across_animals)
    params_rminus_sem_across_animals = pd.Series(params_rminus_sem_across_animals)
    params_rplus_mean_across_animals = pd.Series(params_rplus_mean_across_animals)
    params_rplus_sem_across_animals = pd.Series(params_rplus_sem_across_animals)
    params_rplus_sem_across_animals = pd.Series(params_rplus_sem_across_animals)
    params_net_stim_mean_across_animals = pd.Series(params_net_stim_mean_across_animals)
    params_net_stim_sem_across_animals = pd.Series(params_net_stim_sem_across_animals)

    # Transform suffles_means_across_animals into a list of pd.Series
    shuffles_rminus_means_across_animals = [pd.Series(shuffles_rminus_means_across_animals[i, :]) for i in
                                            range(len(shuffles_rminus_means_across_animals))]
    shuffles_rplus_means_across_animals = [pd.Series(shuffles_rplus_means_across_animals[i, :]) for i in
                                            range(len(shuffles_rplus_means_across_animals))]
    shuffles_net_stim_means_across_animals = [pd.Series(shuffles_net_stim_means_across_animals[i, :]) for i in
                                            range(len(shuffles_net_stim_means_across_animals))]

    # Rename the first element of each pd.Series as 'const' instead of '0'
    # shuffles_rminus_means_across_animals = [shuffles_rminus_means_across_animals[i].rename({0: 'const'}) for i in
    #                                  range(len(shuffles_rminus_means_across_animals))]

    # Store results in a namedtuple
    MeanHK = namedtuple('MeanPK', [
        'params_rminus',
        'params_rplus',
        'params_session_index',
        'params_net_stim',
        'params_frames',
        'std_err_rminus',
        'std_err_rplus',
        'std_err_session_index',
        'std_err_net_stim',
        'std_err_frames',
        'p_values',
        'shuffles_rminus',
        'shuffles_rplus',
        'shuffles_net_stim',
        'shuffles_frames',
        'n_trials',
        'experiment',
        'animal',
        'drug',
        'trial_lag',
        'iterations',
        'n_frames'
    ])

    mean_hk = MeanHK(
        params_rminus=params_rminus_mean_across_animals,
        params_rplus=params_rplus_mean_across_animals,
        params_session_index=params_session_index_mean_across_animals,
        params_net_stim=params_net_stim_mean_across_animals,
        params_frames=params_frames_mean_across_animals,
        std_err_rminus=params_rminus_sem_across_animals,
        std_err_rplus=params_rplus_sem_across_animals,
        std_err_session_index=params_session_index_sem_across_animals,
        std_err_net_stim=params_net_stim_sem_across_animals,
        std_err_frames=params_frames_sem_across_animals,
        p_values=None,
        shuffles_rminus=shuffles_rminus_means_across_animals,
        shuffles_rplus=shuffles_rplus_means_across_animals,
        shuffles_net_stim=shuffles_net_stim_means_across_animals,
        shuffles_frames=shuffles_frames_means_across_animals,
        n_trials=n_trials,
        experiment=experiments_across_batches,
        animal=animals_across_batches,
        drug=drug,
        trial_lag=trial_lag,
        iterations=iterations,
        n_frames=n_frames
    )

    return mean_hk


########################################################################################################################

# Debugging

# animals = ['325', '327', '329', '330', '332', '333', '335', '337']  # Bach 2 (with ILDs) -326, -334
# animals = ['419', '420', '422', '616', '619', '623']  # Batch 3 (with ILDs)  -617, -620
# animals = ['332', '333', '337']  # Drug experiments


# hk = get_hk(experiment=experiment, animal=animal, drug=drug, trial_lag=trial_lag, iterations=iterations)
# plot_hk(experiment=experiment, animal=animal, drug=drug, trial_lag=trial_lag, iterations=iterations, save=save)
# plot_hks(experiment=experiment, animals=animals, drug=drug, trial_lag=trial_lag, iterations=iterations, save=save)
# hk = get_mean_hk(experiments=experiments, animals=None, drug=None, trial_lag=trial_lag, iterations=iterations)

# plot_hks(experiment='2AFC_2', animals=['325', '327', '329', '330', '332', '333', '335', '337'], drug=None,
#              trial_lag=trial_lag, iterations=iterations, save=save)
# plot_hks(experiment='2AFC_3', animals=['419', '420', '422', '616', '619', '623'], drug=None,
#              trial_lag=trial_lag, iterations=iterations, save=save)


# Drug analyses
def plot_hk_drug():
    mean_hk_saline = get_mean_hk(experiments=experiments, animals=None, drug='saline', trial_lag=trial_lag, iterations=iterations)
    mean_hk_drug = get_mean_hk(experiments=experiments, animals=None, drug='MK801', trial_lag=trial_lag, iterations=iterations)

    title = f'N={len(mean_hk_drug.animal)}, {mean_hk_drug.n_trials + mean_hk_saline.n_trials} trials'

    # Default plotting parameters
    filename_prefix = 'drug_'
    color_saline = 'tab:gray'
    color_drug = 'tab:pink'

    trial_lag = mean_hk_saline.trial_lag

    for _ in range(1, 3):

        if _ == 1:  # R+
            xlabel = 'Trial lag (' + '$r^{-}$)'
            # params_indexes = np.arange(trial_lag * _)
            filename = f'{mean_hk_saline.animal} r- HK, trial lag {mean_hk_saline.trial_lag}'
            filename = filename_prefix + filename
            y_saline = mean_hk_saline.params_rminus
            yerr_saline = mean_hk_saline.std_err_rminus
            y_drug = mean_hk_drug.params_rminus
            yerr_drug = mean_hk_drug.std_err_rminus
            # shuffles = hk.shuffles_rminus
        elif _ == 2:  # R-
            xlabel = 'Trial lag (' + '$r^{+}$)'
            # params_indexes = np.arange(trial_lag, trial_lag * _)
            filename = f'{mean_hk_drug.animal} r+ HK, trial lag {mean_hk_drug.trial_lag}'
            filename = filename_prefix + filename
            y_saline = mean_hk_saline.params_rplus
            yerr_saline = mean_hk_saline.std_err_rplus
            y_drug = mean_hk_drug.params_rplus
            yerr_drug = mean_hk_drug.std_err_rplus
            # shuffles = mean_hk_drug.shuffles_rplus

        ################################################################################################################

        # PLOT HISTORY KERNEL

        plt.figure(constrained_layout=True)
        x = np.arange(1, trial_lag + 1)
        # y = hk.params.iloc[params_indexes]
        # yerr = hk.std_err.iloc[params_indexes]
        plt.plot(x, y_saline, color=color_saline, marker='o', label='saline')
        plt.errorbar(x, y_saline, yerr=yerr_saline, color=color_saline, marker='o', fmt='none', mec='none', ms=0)
        plt.plot(x, y_drug, color=color_drug, marker='o', label='MK801')
        plt.errorbar(x, y_drug, yerr=yerr_drug, color=color_drug, marker='o', fmt='none', mec='none', ms=0)
        plt.title(title)
        plt.xticks(x, x[::-1])
        plt.xlabel(xlabel)
        ylabel = 'GLM weight'
        plt.ylabel(ylabel)
        plt.legend(frameon=False)
        # yticks = plt.gca().get_yticks()  # Get current axis yticks for the significance annotations

        # if hk.p_values is not None:
        #     for i in range(n_trials_lag):
        #         if hk.p_values[i] <= 0.05:
        #             text = '*'
        #         else:
        #             # text = 'ns'
        #             text = ''
        #         plt.annotate(text, xy=(i + 1, yticks[1]), xytext=(i + 1 + int(residuals), yticks[1]),
        #                      color=color, va='center', ha='center', fontsize='medium')

        # shuffles_mean = np.mean(shuffles, axis=0)  # Get the mean of all the shuffles
        # percentiles2dot5 = np.percentile(shuffles, 2.5, axis=0)  # Get upper 5 percentile of the shuffled_var
        # percentiles97dot5 = np.percentile(shuffles, 97.5, axis=0)  # Get upper 5 percentile of the shuffled_var
        # plt.plot(x, shuffles_mean, color='tab:gray', ls='--', zorder=1.8)
        # plt.plot(x, percentiles2dot5, color=color_upper_shuffle, ls=':', zorder=1.9)
        # plt.plot(x, percentiles97dot5, color=color_upper_shuffle, ls=':', zorder=2)

        sns.despine()

        if save:
            filename = filename_prefix + 'prev_resp' + filename
            folder_out = Path.home() / 'Documentos' / 'kernels' / 'HK' / 'drug'
            save_fig(folder_out, filename)
            plt.close()

        ################################################################################################################

        # PLOT NET STIMULUS KERNEL

        plt.figure(constrained_layout=True)
        x = mean_hk_saline.params_net_stim.index.values
        x[-1] = 16  # Trick to zoom in
        x = [2, 4, 8, 16]
        y_saline = mean_hk_saline.params_net_stim
        yerr_saline = mean_hk_saline.std_err_net_stim
        y_drug = mean_hk_drug.params_net_stim
        yerr_drug = mean_hk_drug.std_err_net_stim

        plt.plot(x, y_saline, color=color_saline, marker='o', label='saline')
        plt.errorbar(x, y_saline, yerr=yerr_saline, color=color_saline, marker='o', fmt='none', mec='none', ms=0)

        plt.plot(x, y_drug, color=color_drug, marker='o', label='MK801')
        plt.errorbar(x, y_drug, yerr=yerr_drug, color=color_drug, marker='o', fmt='none', mec='none', ms=0)
        plt.title(title)
        plt.xlabel('Net stimuli')
        plt.ylabel('GLM weight')
        plt.legend(frameon=False)
        plt.xticks(x, ['2', '4', '8', '70'])

        # shuffles_mean = np.mean(hk.shuffles_net_stim, axis=0)  # Get the mean of all the shuffles
        # percentiles95 = np.percentile(hk.shuffles_net_stim, 95, axis=0)  # Get upper 5 percentile of the shuffled_var
        # plt.plot(x, shuffles_mean, color='tab:gray', ls='--', zorder=1.8)
        # plt.plot(x, percentiles95, color='tab:red', ls=':', zorder=1.9)
        sns.despine()

        if save:
            filename = filename_prefix + 'net_stim' + filename
            folder_out = Path.home() / 'Documentos' / 'kernels' / 'HK' / 'drug'
            save_fig(folder_out, filename)
            plt.close()