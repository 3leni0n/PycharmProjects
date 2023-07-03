import time
from pathlib import Path
import os
import pandas as pd
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import seaborn as sns
from my_fun.my_fun import get_experiment, get_animal, save_fig

# Plotting parameters
sns.set_theme()
sns.set_style('white')
sns.set_style('ticks')
sns.set_context('poster')


def get_RTs_subject(experiment=None, animal=None, hit=None, bin_size=100):

    time_start = time.time()

    ####################################################################################################################

    # Get the path to the data
    experiment = get_experiment(experiment)
    folder_in = Path.home() / 'PycharmProjects' / 'glue_sessions' / experiment
    animal = get_animal(experiment, animal)
    folder_in = Path(folder_in / animal).with_suffix('.csv')

    ####################################################################################################################

    # Load behavioral data
    df = pd.read_csv(folder_in)

    ####################################################################################################################

    # Load intersession data
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
    df = df[df.P > 0]  # Only trials/sessions with P > 0
    # if target_ilds is not None:
    #     df = df[df.ILD.isin(target_ilds)]  # Select only trials with the desired ILDs
    accuracy_threshold = 0.5
    df = df[(df.AccuracyLeft >= accuracy_threshold) & (df.AccuracyRight >= accuracy_threshold)]  # Select only trials
    # with accuracy >= threshold

    # Correct vs error trials
    if hit is not None:
        df = df[df.Hit == hit]  # Select only correct trials

    if df.Experiment.unique() == '2AFC_2':
        df = df[df.Drug.isnull()]  # Remove drug experimental sessions

    ####################################################################################################################

    n_trials = len(df)

    bins = range(0, df.Trial.max() + bin_size, 100)
    df['TrialBins'] = pd.cut(df['Trial'], bins=bins)
    RTs_bin_median = df.groupby(['TrialBins'])['RespWinLen'].median()
    RTs_bin_sem = df.groupby(['TrialBins'])['RespWinLen'].sem().values
    bin_indexes = [int(interval.right) for interval in RTs_bin_median.index]

    time_end = time.time()
    runtime = time_end - time_start
    print('The script took', round(runtime, 2), 'seconds to run')

    return RTs_bin_median, RTs_bin_sem, bin_indexes, n_trials


def get_RTs_experiment(experiment=None, hit=None, bin_size=100):

    time_start = time.time()

    experiment = get_experiment(experiment)
    folder_in = Path.home() / 'PycharmProjects' / 'glue_sessions' / experiment

    animals = os.listdir(folder_in)  # List animals
    animals.sort()  # Sort them by name
    animals = [x for x in animals if '_corrupted_sessions.csv' not in x]  # Remove'_corrupted_sessions' from animals:
    animals = [x[:-4] for x in animals]  # Remove extension '.csv' from animals

    # Curated list
    if experiment == '2AFC_2':
        animals = ['325', '327', '329', '330', '332', '333', '335', '337']
    elif experiment == '2AFC_3':
        animals = ['419', '420', '422', '616', '619', '623']

    RT_median = []
    RT_sem = []
    bin_indexes_max = []
    n_trials_sum = []
    # plt.figure(constrained_layout=True)
    for animal in animals:
        print(animal)
        RTs_bin_median, RTs_bin_sem, bin_indexes, n_trials = get_RTs_subject(experiment=experiment,
                                                                                           animal=animal, hit=hit,
                                                                                           bin_size=bin_size)
        RT_median.append(RTs_bin_median)
        RT_sem.append(RTs_bin_sem)
        bin_indexes_max.append(bin_indexes)
        n_trials_sum.append(n_trials)

    RT_median = pd.DataFrame(RT_median)
    RT_sem = pd.DataFrame(RT_median)
    bin_indexes_max = pd.DataFrame(bin_indexes_max)
    n_trials_sum = sum(n_trials_sum)

    return RT_median, RT_sem, bin_indexes_max, n_trials_sum


def get_RTs_median(experiments=['2AFC_2', '2AFC_3'], hit=None, bin_size=100):

    RTs_median = pd.DataFrame()
    n_trials_sum_sum = []
    for experiment in experiments:
        print(experiment)
        RT_median, RT_sem, bin_indexes_max, n_trials_sum = get_RTs_experiment(experiment=experiment, hit=hit, bin_size=bin_size)
        RTs_median = pd.concat([RTs_median, RT_median])
        n_trials_sum_sum.append(n_trials_sum)

    n_trials_sum_sum = sum(n_trials_sum_sum)

    return RTs_median, n_trials_sum_sum


def plot_RTs(experiments=['2AFC_2', '2AFC_3'], hit=None, bin_size=100, save=False):

    plt.figure(constrained_layout=True)

    if hit is None:
        RTs_median, n_trials_sum_sum = get_RTs_median(experiments=['2AFC_2', '2AFC_3'], hit=hit, bin_size=100)
        color = 'k'
        label = ''
        filename = f'median_RTs_experiments_{experiments}_{bin_size}_trials_bins'
        for _ in range(len(RTs_median)):
            print()
            x = RTs_median.iloc[_].index.right
            plt.plot(x, RTs_median.iloc[_], marker='o', color='tab:gray', alpha=0.25)

        RTs_median = RTs_median.iloc[:, :-1]  # Remove last column as there's only one value
        RTs_median_median = RTs_median.median()
        x = RTs_median_median.index.right
        RTs_median_sem = RTs_median.sem()
        plt.plot(x, RTs_median_median, marker='o', color=color, label=label)
        plt.errorbar(x, RTs_median_median, RTs_median_sem, color=color, marker='o', fmt='none', mec='none', ms=0)
        n_trials = n_trials_sum_sum

    else:
        n_trials = []
        for i in range(2):
            if i == 0:
                color = 'tab:red'
                label = 'error'
                filename = f'median_RTs_error_experiments_{experiments}_{bin_size}_trials_bins'
            elif i == 1:
                color = 'tab:green'
                label = 'correct'
                filename = f'median_RTs_correct_experiments_{experiments}_{bin_size}_trials_bins'


            RTs_median, n_trials_sum_sum = get_RTs_median(experiments=['2AFC_2', '2AFC_3'], hit=i, bin_size=100)
            n_trials.append(n_trials_sum_sum)
            RTs_median = RTs_median.iloc[:, :-1]  # Remove last column as there's only one value
            RTs_median_median = RTs_median.median()
            x = RTs_median_median.index.right
            RTs_median_sem = RTs_median.sem()
            plt.plot(x, RTs_median_median, marker='o', color=color, label=label)
            plt.errorbar(x, RTs_median_median, RTs_median_sem, color=color, marker='o', fmt='none', mec='none', ms=0)
        n_trials = sum(n_trials)

    plt.xlabel('Trial')
    plt.ylabel('RT (s)')
    plt.title(f'N={len(RTs_median)}, {n_trials} trials')
    plt.legend(frameon=False)
    sns.despine(trim=True)  # Despine axes triming the 0

    if save:
        folder_out = Path.home() / 'Documentos' / 'trial index' / 'RTs' / 'median'
        save_fig(folder_out, filename)



# Debugging
# experiment = '2AFC_2'
# animal = '333'
plot_RTs(experiments=['2AFC_2', '2AFC_3'], hit=None, bin_size=100, save=True)