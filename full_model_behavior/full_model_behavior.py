import time
from pathlib import Path
import os
import pickle
import pandas as pd
import numpy as np
import statsmodels.api as sm
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy import stats
import seaborn as sns
from collections import namedtuple
from my_fun.my_fun import get_experiment, get_animal, save_fig, timer, filter_drug_sessions, filter_behavior
from cherry.cherry import *
from kernels.kernels_tools import *


# Create namedtuple objects at the module level for pickling purposes
FK = namedtuple('FK', [
    'params_rminus', 'params_rplus', 'params_session_index', 'params_net_stim', 'params_frames',  # params
    'std_err_rminus', 'std_err_rplus', 'std_err_session_index', 'std_err_net_stim', 'std_err_frames',  # bse
    'p_values_rminus', 'p_values_rplus', 'p_values_session_index', 'p_values_net_stim', 'p_values_frames',  # p_values
    'shuffles_rminus', 'shuffles_rplus', 'shuffles_session_index', 'shuffles_net_stim', 'shuffles_frames',  # shuffles
    'n_trials', 'trial_lag', 'experiment', 'animal', 'drug', 'iterations', 'n_frames'  # metadata
])

MeanFK = namedtuple('MeanFK', [
    'params_rminus', 'params_rplus', 'params_session_index', 'params_net_stim', 'params_frames',  # params
    'std_err_rminus', 'std_err_rplus', 'std_err_session_index', 'std_err_net_stim', 'std_err_frames',  # bse
    'p_values',  # p_values
    'shuffles_rminus', 'shuffles_rplus', 'shuffles_net_stim', 'shuffles_frames',  # shuffles
    'n_trials', 'experiment', 'animal', 'drug', 'trial_lag', 'iterations', 'n_frames'  # metadata
])


@timer
def get_fk(experiment=None, animal=None, residuals=True, zscore=False, drug=None, trial_lag=10, iterations=1000, save=False):
    """
    Get full kernel for a given animal. The history kernel is the GLM weight of previously rewarded (r+) and
    previously unrewarded (r-) responses (choices). These kernels quantify the influence on choice of the side (left vs.
    right) of previous responses.
    :param experiment: str, name of the experiment
    :param animal: str, name of the animal
    :param drug: bool, drug session to analyze. If None, all sessions are analyzed
    :param trial_lag: int, number of trials to consider in the past
    :param iterations: int, number of iterations to compute shuffles of the kernel
    :return: fk: namedtuple, history kernel
    """

    # Get the path to the data
    experiment, folder_in = get_experiment(experiment, path_session='glue_sessions')
    # experiment, folder_in = get_experiment(experiment, path_session='glmhmm')  # For engagement data
    animal = get_animal(experiment=experiment, animal=animal)
    folder_in = Path(folder_in / animal).with_suffix('.csv')

    # Load behavioral data
    df = pd.read_csv(folder_in)

    # Filter trials
    df = filter_behavior(df, clean_start=True, drop_miss=True, filter_drug=False)

    # Load intersession data
    path_intersession = Path.home() / 'PycharmProjects' / 'intersession' / experiment / (str(int(animal)) + '_intersession.csv')
    # str(int(animal)) to remove the 0 padding in ID
    df_intersession = pd.read_csv(path_intersession)
    threshold = 0.5  # Accuracy threshold to remove bad sessions
    # mask = df_intersession.Accuracy < threshold
    mask = (df_intersession.AccuracyLeft < threshold) | (df_intersession.AccuracyRight < threshold)

    # Remove bad sessions based on intersession data
    if drug is None:
        dates_to_remove = df_intersession[mask].Dates
        df = df[~df.Date.isin(dates_to_remove)].reset_index(drop=True)

    # # Engagement
    # engaged = [1 if state == 0 else 0 for state in df.State]
    # df['Engaged'] = engaged
    # df = df[df.Engaged == 1]  # Only engaged trials

    ####################################################################################################################

    # Drug sessions/trials
    if drug is not None and experiment == '2AFC_6':  # Select drug session/trials
        if drug in [0, 1]:
            print('Filtering in drug sessions...')
            # Filter out saline sessions that are not paired to drug sessions
            df = filter_drug_sessions(df)
            df = df[df.Drug == drug]

    else:  # Don't select drug session trials
        print('Filtering out drug sessions...')
        try:
            df = df[df.Drug.isnull()]  # Remove drug experimental sessions
        except AttributeError:
            pass

    ####################################################################################################################

    # Make design matrix

    df = df.reset_index(drop=True)
    n_trials = len(df)

    # Set stimuli set
    if experiment == '2AFC_6':
        stim_set = 6
    else:
        stim_set = 2

    stim_strength, n_frames = make_frames_dm(df, stim_set=stim_set, residuals=residuals, zscore=zscore)
    dm_choice_history = make_choice_history_dm(df, trial_lag)  # Make history design matrix
    dm_session_index = make_session_index_dm(df)  # Make session index design matrix
    dm_ild = make_net_ild_dm(df)  # Make net ILD design matrix

    exog = pd.concat([dm_choice_history, dm_session_index, dm_ild, stim_strength], axis=1)
    endog = df.Choice

    # Drop rows with NaNs in exog or endog
    valid_rows = ~(exog.isna().any(axis=1) | endog.isna())
    exog = exog.loc[valid_rows].reset_index(drop=True)
    endog = df.Choice.loc[valid_rows].reset_index(drop=True)

    ####################################################################################################################

    model = sm.GLM(endog, exog, family=sm.families.Binomial(), missing='drop')  # GLM with Binomial family
    results = model.fit()
    params = results.params
    bse = results.bse
    p_values = results.pvalues
    summary = results.summary()
    # print(summary)

    # Rpminus
    params_rminus = params.iloc[:trial_lag]  # From params
    shuffles_rminus = get_shuffles_GLM(endog, exog, iterations, kind='fk_rminus', stim_set=stim_set)
    shuffles_rminus = [shuffles_rminus[i].iloc[:trial_lag] for i in range(len(shuffles_rminus))]  # From shuffles
    bse_rminus = bse.iloc[:trial_lag]  # From bse
    p_values_rminus = p_values.iloc[:trial_lag]  # From p_values

    # Rplus
    params_rplus = params.iloc[trial_lag:trial_lag * 2]  # From params
    shuffles_rplus = get_shuffles_GLM(endog, exog, iterations, kind='fk_rplus', stim_set=stim_set)
    shuffles_rplus = [shuffles_rplus[i].iloc[trial_lag:trial_lag * 2] for i in range(len(shuffles_rplus))]  # From shuffles
    bse_rplus = bse.iloc[trial_lag:trial_lag * 2]  # From bse
    p_values_rplus = p_values.iloc[trial_lag:trial_lag * 2]  # From p_values

    # Session index
    params_session_index = params.iloc[trial_lag * 2:-4 - n_frames]
    bse_session_index = bse.iloc[trial_lag * 2:-4 - n_frames]
    p_values_session_index = p_values.iloc[trial_lag * 2:-4 - n_frames]
    shuffles_session_index = get_shuffles_GLM(endog, exog, iterations, kind='fk_session_index', stim_set=stim_set)
    shuffles_session_index = [shuffles_session_index[i].iloc[trial_lag * 2:-4 - n_frames] for i in range(len(shuffles_session_index))]  # From shuffles

    # Net ILD
    params_net_stim = params.iloc[-4-n_frames:-n_frames]
    bse_net_stim = bse.iloc[-4-n_frames:-n_frames]
    p_values_net_stim = p_values.iloc[-4-n_frames:-n_frames]
    shuffles_net_stim = get_shuffles_GLM(endog, exog, iterations, kind='fk_net_stim', stim_set=stim_set)
    shuffles_net_stim = [shuffles_net_stim[i].iloc[-4-n_frames:-n_frames] for i in range(len(shuffles_net_stim))]  # From shuffles

    # Stimulus strength
    params_frames = params[-n_frames:]
    bse_frames = bse[-n_frames:]
    p_values_frames = p_values[-n_frames:]
    shuffles_frames = get_shuffles_GLM(endog, exog, iterations, kind='fk_frames', stim_set=stim_set)
    shuffles_frames = [shuffles_frames[i].iloc[-n_frames:] for i in range(len(shuffles_frames))]  # From shuffles

    fk = FK(
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

    if save:
        filename = f'fk_{animal}'
        with open(filename, 'wb') as f:
            pickle.dump(fk, f)

    return fk


# def plot_fk(fk, experiment=None, animal=None, drug=None, trial_lag=10, iterations=1000, save=False, **kwargs):
def plot_fk(fk, save=False, **kwargs):
    """
    Plot the full kernel of a given animal.
    :param experiment: str, name of the experiment
    :param animal: str, name of the animal
    :param drug: bool, drug session to analyze. If None, all sessions are analyzed
    :param trial_lag: int, number of trials to consider in the past
    :param iterations: int, number of iterations to compute shuffles of the kernel
    :param save: bool, whether to save the figure or not
    :return:
    """

    experiment = fk.experiment

    if type(experiment) == list:
        # fk = get_mean_fk(experiments=experiment, animals=None, drug=drug, trial_lag=trial_lag,
        #                  iterations=iterations)
        title = f'N={len(fk.animal)}, {fk.n_trials} trials'
        filename_prefix = 'mean_'
    else:
        # fk = get_fk(experiment=experiment, animal=animal, drug=drug, trial_lag=trial_lag, iterations=iterations)
        # title = f'Mouse {fk.animal}, {fk.n_trials} trials'
        filename_prefix = ''

    # Default plotting parameters
    color = 'k'
    color_upper_shuffle = 'tab:red'
    label = ''

    ####################################################################################################################

    # PLOT STIMULUS FRAMES KERNEL

    plt.figure(**kwargs, constrained_layout=True)
    n_frames = fk.n_frames
    x = np.arange(1, n_frames + 1)

    shuffles_mean = np.mean(fk.shuffles_frames, axis=0)  # Get the mean of all the shuffles
    percentiles95 = np.percentile(fk.shuffles_frames, 95, axis=0)  # Get upper 5 percentile of the shuffled_var
    plt.plot(x, shuffles_mean, color='tab:gray', ls='--')
    plt.plot(x, percentiles95, color=color_upper_shuffle, ls=':')

    y = fk.params_frames
    yerr = fk.std_err_frames
    plt.plot(x, y, color=color, marker='o', label=label)
    plt.errorbar(x, y, yerr=yerr, color=color, marker='o', fmt='none', mec='none', ms=0)
    plt.xticks([2, 4, 6, 8, 10], ['2', '4', '6', '8', '10'])
    # plt.title(title)
    plt.xlabel('Stimulus frame')
    plt.ylabel('Weight')
    sns.despine()

    # if pk.p_values is not None:
    #     for i in range(n_frames):
    #         if pk.p_values[i] <= 0.05:
    #             text = '*'
    #         else:
    #             # text = 'ns'
    #             text = ''
    #         plt.annotate(text, xy=(i + 1 + int(residuals), yticks[1]), xytext=(i + 1 + int(residuals), yticks[1]),
    #                      color=color, va='center', ha='center', fontsize='medium')

    if save:
        filename = f'{fk.animal}_PK_ILDs'
        filename = filename_prefix + filename
        folder_out = Path.home() / 'OneDrive' / 'Imágenes' / 'Figures' / 'kernels' / 'PK' / experiment
        if not folder_out.exists():
            folder_out.mkdir(parents=True, exist_ok=True)
        save_fig(folder_out, filename)
        plt.close()

    ####################################################################################################################

    # PLOT NET STIMULUS KERNEL

    plt.figure(**kwargs, constrained_layout=True)
    x = np.arange(len(fk.params_net_stim.index.astype(int).values))

    # Plot shuffles
    shuffles_mean = np.mean(fk.shuffles_net_stim, axis=0)  # Get the mean of all the shuffles
    percentiles95 = np.percentile(fk.shuffles_net_stim, 95, axis=0)  # Get upper 5 percentile of the shuffled_var
    plt.plot(x, shuffles_mean, color='tab:gray', ls='--')
    plt.plot(x, percentiles95, color='tab:red', ls=':')
    sns.despine()

    # Plot kernel
    y = fk.params_net_stim
    yerr = fk.std_err_net_stim
    plt.plot(x, y, color=color, marker='o')
    plt.errorbar(x, y, yerr=yerr, color=color, marker='o', fmt='none', mec='none', ms=0)
    # plt.title(title)
    plt.xlabel('Net stimuli (dB)')
    plt.ylabel('Weight')
    plt.xticks(x, ['2', '4', '8', '70'])

    if save:
        filename = filename_prefix + 'net_stim' + filename
        folder_out = Path.home() / 'Documentos' / 'kernels' / 'FK' / experiment
        save_fig(folder_out, filename)
        plt.close()

    ####################################################################################################################

    # PLOT HISTORY KERNEL

    axes = []
    trial_lag = fk.trial_lag
    for _ in range(1, 3):
        if _ == 1:  # R+
            xlabel = 'Trial lag (' + '$r^{-}$)'
            # params_indexes = np.arange(trial_lag * _)
            filename = f'{fk.animal} r- HK, trial lag {fk.trial_lag}'
            y = fk.params_rminus
            yerr = fk.std_err_rminus
            shuffles = fk.shuffles_rminus
        elif _ == 2:  # R-
            xlabel = 'Trial lag (' + '$r^{+}$)'
            # params_indexes = np.arange(trial_lag, trial_lag * _)
            filename = f'{fk.animal} r+ HK, trial lag {fk.trial_lag}'
            y = fk.params_rplus
            yerr = fk.std_err_rplus
            shuffles = fk.shuffles_rplus

        plt.figure(**kwargs, constrained_layout=True)
        ax = plt.gca()
        axes.append(ax)

        x = np.arange(1, trial_lag + 1)

        # Plot shuffles
        shuffles_mean = np.mean(shuffles, axis=0)  # Get the mean of all the shuffles
        percentiles2dot5 = np.percentile(shuffles, 2.5, axis=0)  # Get upper 5 percentile of the shuffled_var
        percentiles97dot5 = np.percentile(shuffles, 97.5, axis=0)  # Get upper 5 percentile of the shuffled_var
        plt.plot(x, shuffles_mean, color='tab:gray', ls='--')
        plt.plot(x, percentiles2dot5, color=color_upper_shuffle, ls=':')
        plt.plot(x, percentiles97dot5, color=color_upper_shuffle, ls=':')

        # plot kernel
        plt.plot(x, y, color=color, marker='o', label=label)
        plt.errorbar(x, y, yerr=yerr, color=color, marker='o', fmt='none', mec='none', ms=0)
        # plt.title(title)
        # plt.xticks(x, x[::-1])  # Reverse ticks
        plt.xticks(x[::2], x[::-1][::2])  # Show every 2nd tick (reversed)
        # plt.xticks(x[::5], x[::-1][::5])
        # plt.xticks([2, 4, 6, 8, 10], ['2', '4', '6', '8', '10'])
        plt.xlabel(xlabel)
        ylabel = 'Weight'
        plt.ylabel(ylabel)
        sns.despine()

        # yticks = plt.gca().get_yticks()  # Get current axis yticks for the significance annotations
        # if fk.p_values is not None:
        #     for i in range(n_trials_lag):
        #         if fk.p_values[i] <= 0.05:
        #             text = '*'
        #         else:
        #             # text = 'ns'
        #             text = ''
        #         plt.annotate(text, xy=(i + 1, yticks[1]), xytext=(i + 1 + int(residuals), yticks[1]),
        #                      color=color, va='center', ha='center', fontsize='medium')

        if save:
            filename = filename_prefix + 'prev_resp' + filename
            folder_out = Path.home() / 'Documentos' / 'kernels' / 'FK' / experiment
            save_fig(folder_out, filename)
            plt.close()

    ymins, ymaxs = zip(*(ax.get_ylim() for ax in axes))
    ylim = (min(ymins), max(ymaxs))
    for ax in axes:
        ax.set_ylim(ylim)

    ####################################################################################################################

    # PLOT SESSION INDEX KERNEL

    if type(experiment) == str:  # Don't do it for the mean kernel

        kwargs['figsize'] = (2 * kwargs['figsize'][0], kwargs['figsize'][1])
        plt.figure(**kwargs, constrained_layout=True)
        x = fk.params_session_index.index.values

        # Plot shuffles
        shuffles_mean = np.mean(fk.shuffles_session_index, axis=0)  # Get the mean of all the shuffles
        percentiles2dot5 = np.percentile(fk.shuffles_session_index, 2.5,
                                         axis=0)  # Get upper 5 percentile of the shuffled_var
        percentiles97dot5 = np.percentile(fk.shuffles_session_index, 97.5,
                                          axis=0)  # Get upper 5 percentile of the shuffled_var
        plt.plot(x, shuffles_mean, color='tab:gray', ls='--')
        plt.plot(x, percentiles2dot5, color='tab:red', ls=':')
        plt.plot(x, percentiles97dot5, color='tab:red', ls=':')

        # Plot kernel
        y = fk.params_session_index
        yerr = fk.std_err_session_index
        plt.plot(x, y, color=color, marker='o')
        plt.errorbar(x, y, yerr=yerr, color=color, marker='o', fmt='none', mec='none', ms=0)
        # plt.title(title)
        ax = plt.gca()
        ax.set_xlim(x[0]-1, x[-1]+1)
        plt.xlabel('Session index')
        plt.ylabel('Weight')
        sns.despine()

        if save:
            filename = 'session_index' + filename
            folder_out = Path.home() / 'Documentos' / 'kernels' / 'FK' / experiment
            save_fig(folder_out, filename)
            plt.close()


def plot_fks(experiment='2AFC_2', animals=['325', '327', '329', '330', '332', '333', '335', '337'], drug=None,
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
        plot_fk(experiment=experiment, animal=animals[i], drug=drug, trial_lag=trial_lag, iterations=iterations,
                save=save)


def get_mean_fk(experiments=['2AFC_2', '2AFC_3', '2AFC_4'], cherry=True, drug=None, iterations=1000, save=False):
    """
    Get the mean peak of the full kernel of all animals of a given batch.
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
        animals = cherries[exp]
        exp, folder_in = get_experiment(exp, path_session='glue_sessions')

        for i, subj in enumerate(animals):
            print(f'Getting kernel of animal {animals[i]} ({i + 1}/{len(animals)})')
            fk = get_fk(experiment=exp, animal=animals[i], drug=drug, iterations=iterations)
            animals_across_batches.append(fk.animal)

            params_rminus_across_animals.append(fk.params_rminus)
            params_rplus_across_animals.append(fk.params_rplus)
            params_session_index_across_animals.append(fk.params_session_index)
            params_net_stim_across_animals.append(fk.params_net_stim)
            params_frames_across_animals.append(fk.params_frames)

            shuffles_rminus_across_animals.append(fk.shuffles_rminus)
            shuffles_rplus_across_animals.append(fk.shuffles_rplus)
            shuffles_net_stim_across_animals.append(fk.shuffles_net_stim)
            shuffles_frames_across_animals.append(fk.shuffles_frames)

            n_trials_across_animals.append(fk.n_trials)
            pks.append(fk)
        experiments_across_batches.append(exp)

    n_trials = sum(n_trials_across_animals)
    n_frames = fk.n_frames
    trial_lag = fk.trial_lag

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

    mean_fk = MeanFK(
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

    if save:
        if type (mean_fk.experiment) == str:
            filename = f'fk_mean_{mean_fk.experiment[0]}'
        elif type(mean_fk.experiment) == list:
            filename = 'fk_mean'
        with open(filename, 'wb') as f:
            pickle.dump(mean_fk, f)

    return mean_fk


def plot_drug_fk(experiment=['2AFC_6'], animal=None, drug=None, trial_lag=10, iterations=1000, save=False, **kwargs):

    if type(experiment) == list:
        pk_saline = get_mean_fk(experiments=experiment, animals=None, drug=0, trial_lag=trial_lag, iterations=iterations)
        pk_drug = get_mean_fk(experiments=experiment, animals=None, drug=1, trial_lag=trial_lag, iterations=iterations)
        title = f'N={len(pk_saline.animal)}, {pk_saline.n_trials + pk_drug.n_trials} trials'
        filename_prefix = 'mean_'
    else:
        pk_saline = get_fk(experiment=experiment, animal=animal, drug=0, trial_lag=trial_lag, iterations=iterations)
        pk_drug = get_fk(experiment=experiment, animal=animal, drug=1, trial_lag=trial_lag, iterations=iterations)
        title = f'Mouse {pk_saline.animal}, {pk_saline.n_trials} trials'
        filename_prefix = ''

    # Default plotting parameters
    color = 'k'
    color_saline = 'tab:gray'
    color_drug = 'tab:pink'
    color_upper_shuffle = 'tab:red'
    # label = ''
    ylabel = 'Weight'

    ####################################################################################################################

    # PLOT STIMULUS FRAMES KERNEL

    plt.figure(**kwargs, constrained_layout=True)
    n_frames = pk_saline.n_frames
    x = np.arange(1, n_frames + 1)

    y_saline = pk_saline.params_frames
    y_drug = pk_drug.params_frames
    yerr_saline = pk_saline.std_err_frames
    yerr_drug = pk_drug.std_err_frames
    plt.axhline(0, color='k', ls='--')
    # plt.plot(x, y_saline, color=color_saline, marker='o', label=label)
    plt.errorbar(x, y_saline, yerr=yerr_saline, color=color_saline, marker='o', label='saline')
    plt.errorbar(x, y_drug, yerr=yerr_drug, color=color_drug, marker='o', label='drug')
    # plt.title(title)
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel('Stimulus frame')
    plt.ylabel(ylabel)
    plt.legend(frameon=False)
    sns.despine()  # Despine axes triming the 0

    # # Annotate significance
    # if n_mean_frames is not None:  # If averaged frames, loop over the number of averaged frames instead
    #     n_frames = n_mean_frames

    # if pk.p_values is not None:
    #     for i in range(n_frames):
    #         if pk.p_values[i] <= 0.05:
    #             text = '*'
    #         else:
    #             # text = 'ns'
    #             text = ''
    #         plt.annotate(text, xy=(i + 1 + int(residuals), yticks[1]), xytext=(i + 1 + int(residuals), yticks[1]),
    #                      color=color, va='center', ha='center', fontsize='medium')

    # shuffles_mean = np.mean(pk.shuffles, axis=0)  # Get the mean of all the shuffles
    # percentiles95 = np.percentile(pk.shuffles, 95, axis=0)  # Get upper 5 percentile of the shuffled_var
    # plt.plot(x, shuffles_mean, color='tab:gray', ls='--', zorder=1.8)
    # plt.plot(x, percentiles95, color=color_upper_shuffle, ls=':', zorder=1.9)
    # xticks = np.arange(1, n_frames + 1, 2)
    # xticklabels = xticks + 1
    # plt.xticks(xticks, xticklabels)

    # if n_mean_frames == 2:
    #     plt.xticks([1, 2])  # Readjust xticks

    if save:
        filename = f'{fk.animal}_PK_ILDs_, {n_mean_frames} averaged frames'
        filename = filename_prefix + filename
        folder_out = Path.home() / 'OneDrive' / 'Imágenes' / 'Figures' / 'kernels' / 'PK' / experiment
        if not folder_out.exists():
            folder_out.mkdir(parents=True, exist_ok=True)
        save_fig(folder_out, filename)
        plt.close()

    ####################################################################################################################

    # PLOT NET STIMULUS KERNEL

    plt.figure(**kwargs, constrained_layout=True)
    x = np.arange(len(pk_saline.params_net_stim.index.astype(int).values))
    y_saline = pk_saline.params_net_stim
    y_drug = pk_drug.params_net_stim
    yerr_saline = pk_saline.std_err_net_stim
    yerr_drug = pk_drug.std_err_net_stim
    plt.axhline(0, color='k', ls='--')
    # plt.plot(x, y, color=color, marker='o')
    plt.errorbar(x, y_saline, yerr=yerr_saline, color=color_saline, marker='o')
    plt.errorbar(x, y_drug, yerr=yerr_drug, color=color_drug, marker='o')
    # plt.title(title)
    plt.xlabel('Net stimuli (dB)')
    plt.ylabel('Weight')
    plt.legend(frameon=False)
    plt.xticks(x, ['2', '4', '8', '70'])

    # shuffles_mean = np.mean(pk.net_stim_shuffles, axis=0)  # Get the mean of all the shuffles
    # percentiles95 = np.percentile(pk.net_stim_shuffles, 95, axis=0)  # Get upper 5 percentile of the shuffled_var
    # plt.plot(x, shuffles_mean, color='tab:gray', ls='--', zorder=1.8)
    # plt.plot(x, percentiles95, color='tab:red', ls=':', zorder=1.9)
    sns.despine()

    if save:
        filename = f'{fk.animal}_PK_net_stim_{target_ilds}, {n_mean_frames} averaged frames'
        filename = filename_prefix + filename
        folder_out = Path.home() / 'Documentos' / 'kernels' / 'PK' / experiment
        save_fig(folder_out, filename)
        plt.close()

####################################################################################################################

    # PLOT HISTORY KERNEL

    trial_lag = pk_saline.trial_lag
    for _ in range(1, 3):
        if _ == 1:  # R+
            xlabel = 'Trial lag (' + '$r^{-}$)'
            # params_indexes = np.arange(trial_lag * _)
            filename = f'{pk_saline.animal} r- HK, trial lag {pk_saline.trial_lag}'
            y_saline = pk_saline.params_rminus
            y_drug = pk_drug.params_rminus
            yerr_saline = pk_saline.std_err_rminus
            yerr_drug = pk_drug.std_err_rminus
            # shuffles = pk.shuffles_rminus
        elif _ == 2:  # R-
            xlabel = 'Trial lag (' + '$r^{+}$)'
            # params_indexes = np.arange(trial_lag, trial_lag * _)
            filename = f'{pk_saline.animal} r+ HK, trial lag {pk_saline.trial_lag}'
            y_saline = pk_saline.params_rplus
            y_drug = pk_drug.params_rplus
            yerr_saline = pk_saline.std_err_rplus
            yerr_drug = pk_drug.std_err_rplus
            # shuffles = pk.shuffles_rplus

        plt.figure(**kwargs, constrained_layout=True)
        x = np.arange(1, trial_lag + 1)

        # # Plot shuffles
        # shuffles_mean = np.mean(shuffles, axis=0)  # Get the mean of all the shuffles
        # percentiles2dot5 = np.percentile(shuffles, 2.5, axis=0)  # Get upper 5 percentile of the shuffled_var
        # percentiles97dot5 = np.percentile(shuffles, 97.5, axis=0)  # Get upper 5 percentile of the shuffled_var
        # plt.plot(x, shuffles_mean, color='tab:gray', ls='--')
        # plt.plot(x, percentiles2dot5, color=color_upper_shuffle, ls=':')
        # plt.plot(x, percentiles97dot5, color=color_upper_shuffle, ls=':')

        # plot kernel
        plt.axhline(0, color='k', ls='--')
        plt.plot(x, y_saline, color=color_saline, marker='o', label='Saline')
        plt.plot(x, y_drug, color=color_drug, marker='o', label='Saline')
        plt.errorbar(x, y_saline, yerr=yerr_saline, color=color_saline, marker='o', fmt='none', mec='none', ms=0)
        plt.errorbar(x, y_drug, yerr=yerr_drug, color=color_drug, marker='o', fmt='none', mec='none', ms=0)
        # plt.title(title)
        # plt.xticks(x, x[::-1])
        plt.xticks(x[::5], x[::-1][::5])
        plt.xlabel(xlabel)
        ylabel = 'Weight'
        plt.ylabel(ylabel)
        sns.despine()

        # yticks = plt.gca().get_yticks()  # Get current axis yticks for the significance annotations
        # if fk.p_values is not None:
        #     for i in range(n_trials_lag):
        #         if fk.p_values[i] <= 0.05:
        #             text = '*'
        #         else:
        #             # text = 'ns'
        #             text = ''
        #         plt.annotate(text, xy=(i + 1, yticks[1]), xytext=(i + 1 + int(residuals), yticks[1]),
        #                      color=color, va='center', ha='center', fontsize='medium')

        if save:
            filename = filename_prefix + 'prev_resp' + filename
            folder_out = Path.home() / 'Documentos' / 'kernels' / 'FK' / experiment
            save_fig(folder_out, filename)
            plt.close()


########################################################################################################################

# Debugging

# animals = ['325', '327', '329', '330', '332', '333', '335', '337']  # Bach 2 (with ILDs) -326, -334
# animals = ['419', '420', '422', '616', '619', '623']  # Batch 3 (with ILDs)  -617, -620
# animals = ['332', '333', '337']  # Drug experiments


# fk = get_fk(experiment=experiment, animal=animal, drug=drug, trial_lag=trial_lag, iterations=iterations)
# plot_fk(experiment=experiment, animal=animal, drug=drug, trial_lag=trial_lag, iterations=iterations, save=save)
# plot_fks(experiment=experiment, animals=animals, drug=drug, trial_lag=trial_lag, iterations=iterations, save=save)
# fk = get_mean_fk(experiments=experiments, animals=None, drug=None, trial_lag=trial_lag, iterations=iterations)

# plot_fks(experiment='2AFC_2', animals=['325', '327', '329', '330', '332', '333', '335', '337'], drug=None,
#              trial_lag=trial_lag, iterations=iterations, save=save)
# plot_fks(experiment='2AFC_3', animals=['419', '420', '422', '616', '619', '623'], drug=None,
#              trial_lag=trial_lag, iterations=iterations, save=save)


# Drug analyses
def plot_fk_drug():
    mean_fk_saline = get_mean_fk(experiments=experiments, animals=None, drug='saline', trial_lag=trial_lag, iterations=iterations)
    mean_fk_drug = get_mean_fk(experiments=experiments, animals=None, drug='MK801', trial_lag=trial_lag, iterations=iterations)

    title = f'N={len(mean_fk_drug.animal)}, {mean_fk_drug.n_trials + mean_fk_saline.n_trials} trials'

    # Default plotting parameters
    filename_prefix = 'drug_'
    color_saline = 'tab:gray'
    color_drug = 'tab:pink'

    trial_lag = mean_fk_saline.trial_lag

    for _ in range(1, 3):

        if _ == 1:  # R+
            xlabel = 'Trial lag (' + '$r^{-}$)'
            # params_indexes = np.arange(trial_lag * _)
            filename = f'{mean_fk_saline.animal} r- HK, trial lag {mean_fk_saline.trial_lag}'
            filename = filename_prefix + filename
            y_saline = mean_fk_saline.params_rminus
            yerr_saline = mean_fk_saline.std_err_rminus
            y_drug = mean_fk_drug.params_rminus
            yerr_drug = mean_fk_drug.std_err_rminus
            # shuffles = fk.shuffles_rminus
        elif _ == 2:  # R-
            xlabel = 'Trial lag (' + '$r^{+}$)'
            # params_indexes = np.arange(trial_lag, trial_lag * _)
            filename = f'{mean_fk_drug.animal} r+ HK, trial lag {mean_fk_drug.trial_lag}'
            filename = filename_prefix + filename
            y_saline = mean_fk_saline.params_rplus
            yerr_saline = mean_fk_saline.std_err_rplus
            y_drug = mean_fk_drug.params_rplus
            yerr_drug = mean_fk_drug.std_err_rplus
            # shuffles = mean_fk_drug.shuffles_rplus

        ################################################################################################################

        # PLOT HISTORY KERNEL

        plt.figure(constrained_layout=True)
        x = np.arange(1, trial_lag + 1)
        # y = fk.params.iloc[params_indexes]
        # yerr = fk.std_err.iloc[params_indexes]
        plt.plot(x, y_saline, color=color_saline, marker='o', label='saline')
        plt.errorbar(x, y_saline, yerr=yerr_saline, color=color_saline, marker='o', fmt='none', mec='none', ms=0)
        plt.plot(x, y_drug, color=color_drug, marker='o', label='MK801')
        plt.errorbar(x, y_drug, yerr=yerr_drug, color=color_drug, marker='o', fmt='none', mec='none', ms=0)
        # plt.title(title)
        plt.xticks(x, x[::-1])
        plt.xlabel(xlabel)
        ylabel = 'Weight'
        plt.ylabel(ylabel)
        plt.legend(frameon=False)
        # yticks = plt.gca().get_yticks()  # Get current axis yticks for the significance annotations

        # if fk.p_values is not None:
        #     for i in range(n_trials_lag):
        #         if fk.p_values[i] <= 0.05:
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
            folder_out = Path.home() / 'Documentos' / 'kernels' / 'FK' / 'drug'
            save_fig(folder_out, filename)
            plt.close()

        ################################################################################################################

        # PLOT NET STIMULUS KERNEL

        plt.figure(constrained_layout=True)
        x = mean_fk_saline.params_net_stim.index.values
        x[-1] = 16  # Trick to zoom in
        x = [2, 4, 8, 16]
        y_saline = mean_fk_saline.params_net_stim
        yerr_saline = mean_fk_saline.std_err_net_stim
        y_drug = mean_fk_drug.params_net_stim
        yerr_drug = mean_fk_drug.std_err_net_stim

        plt.plot(x, y_saline, color=color_saline, marker='o', label='saline')
        plt.errorbar(x, y_saline, yerr=yerr_saline, color=color_saline, marker='o', fmt='none', mec='none', ms=0)

        plt.plot(x, y_drug, color=color_drug, marker='o', label='MK801')
        plt.errorbar(x, y_drug, yerr=yerr_drug, color=color_drug, marker='o', fmt='none', mec='none', ms=0)
        # plt.title(title)
        plt.xlabel('Net stimuli')
        plt.ylabel('Weight')
        plt.legend(frameon=False)
        plt.xticks(x, ['2', '4', '8', '70'])

        # shuffles_mean = np.mean(fk.shuffles_net_stim, axis=0)  # Get the mean of all the shuffles
        # percentiles95 = np.percentile(fk.shuffles_net_stim, 95, axis=0)  # Get upper 5 percentile of the shuffled_var
        # plt.plot(x, shuffles_mean, color='tab:gray', ls='--', zorder=1.8)
        # plt.plot(x, percentiles95, color='tab:red', ls=':', zorder=1.9)
        sns.despine()

        if save:
            filename = filename_prefix + 'net_stim' + filename
            folder_out = Path.home() / 'Documentos' / 'kernels' / 'FK' / 'drug'
            save_fig(folder_out, filename)
            plt.close()