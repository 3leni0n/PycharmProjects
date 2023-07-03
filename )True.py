# coding: utf-8
print('PyDev console: using IPython 8.8.0\n')

import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['C:\\Users\\alexi\\PycharmProjects'])
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
experiment = '2AFC_2'
animal = '333'
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
# df = df[df.Hit == 0]  # Select only error trials
df = df[df.Hit == 1]  # Select only correct trials

if df.Experiment.unique() == '2AFC_2':
    df = df[df.Drug.isnull()]  # Remove drug experimental sessions
trial_ranges = range(0, df['Trial'].max() + 100, 100)
df.Trial.max()
trial_ranges = range(0, df.Trial.max() + 100, 100)
trial_ranges = pd.cut(df['Trial'], bins=trial_ranges)
trial_ranges.max()
trial_ranges.max().index
2+2
df['TrialRange'] = pd.cut(df['Trial'], bins=trial_ranges)
df.TrialRange
trial_ranges = range(0, df['Trial'].max() + 100, 100)
df['TrialRange'] = pd.cut(df['Trial'], bins=trial_ranges)
yaxis = df.groupby(['TrialRange', 'Subject'])['RespWinLen'].mean()
mean_by_trial = yaxis.groupby('TrialRange').mean()
yaxis = df.groupby(['TrialRange'])['RespWinLen'].mean()
mean_by_trial = yaxis.groupby('TrialRange').mean()
yaxis
mean_by_trial
mean_by_trial == yaxis
yaxis = df.groupby(['TrialRange'])['RespWinLen'].mean()
mean_by_trial = yaxis.groupby('TrialRange').mean()
yazis
yaxis
mean_by_trial
xaxis = [int(interval.mid) for interval in yaxis.index]
yaxis.index
yaxis.index[0]
yaxis.index[0].right
del yaxis
del xaxis
trial_ranges = range(0, df.Trial.max() + bin_size, 100)
del trial_ranges
bin_size = 100
bins = range(0, df.Trial.max() + bin_size, 100)
trial_bins = pd.cut(df['Trial'], bins=bins)
RTs_bin_mean = df.groupby(['TrialRange', 'Subject'])['RespWinLen'].mean()
bin_indexes = [int(interval.right) for interval in RTs_bin_mean.index]
trial_bins = pd.cut(df['Trial'], bins=bins)
bin_size = 100
bins = range(0, df.Trial.max() + bin_size, 100)
df['TrialBins'] = pd.cut(df['Trial'], bins=bins)
RTs_bin_mean = df.groupby(['TrialBins'])['RespWinLen'].mean()
bin_indexes = [int(interval.right) for interval in RTs_bin_mean.index]
RTs_bin_sem = df.groupby(['TrialBins'])['RespWinLen'].sem().values
RTs_bin_sem
xaxis = [int(interval.mid) for interval in mean_by_trial.index]
errordata = yaxis.groupby('TrialRange').sem().values
yaxis = df.groupby(['TrialRange', 'Subject'])['RespWinLen'].mean()
mean_by_trial = yaxis.groupby('TrialRange').mean()
xaxis = [int(interval.mid) for interval in mean_by_trial.index]
errordata = yaxis.groupby('TrialRange').sem().values
errordata
RTs_bin_sem
yaxis = df.groupby(['TrialRange'])['RespWinLen'].mean()
yaxis
errordata = yaxis.groupby('TrialRange').sem().values
errordata
yaxis
plt.plot(xaxis, mean_by_trial, color='#0DA470', linewidth=3, label='Mean ± SEM', zorder=1)
plt.errorbar(xaxis, mean_by_trial, yerr=errordata, fmt='o', color="black", markersize=5, label=None, zorder=2)
plt.plot(xaxis, bin_indexes, color='#0DA470', linewidth=3, label='Mean ± SEM', zorder=1)
plt.errorbar(xaxis, bin_indexes, yerr=errordata, fmt='o', color="black", markersize=5, label=None, zorder=2)
plt.plot(bin_indexes, mean_by_trial, color='#0DA470', linewidth=3, label='Mean ± SEM', zorder=1)
plt.errorbar(bin_indexes, mean_by_trial, yerr=errordata, fmt='o', color="black", markersize=5, label=None, zorder=2)
del xaxis
del yaxis
RTs_bin_mean = df.groupby(['TrialBins'])['RespWinLen'].mean()
RTs_bin_sem = df.groupby(['TrialBins'])['RespWinLen'].sem().values
bin_indexes = [int(interval.right) for interval in RTs_bin_mean.index]
def get_RTs_subject(experiment=None, animal=None, bin_size=100):

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
    # df = df[df.Hit == 0]  # Select only error trials
    df = df[df.Hit == 1]  # Select only correct trials

    if df.Experiment.unique() == '2AFC_2':
        df = df[df.Drug.isnull()]  # Remove drug experimental sessions

    ####################################################################################################################

    n_trials = len(df)

    bin_size = 100
    bins = range(0, df.Trial.max() + bin_size, 100)
    df['TrialBins'] = pd.cut(df['Trial'], bins=bins)
    RTs_bin_mean = df.groupby(['TrialBins'])['RespWinLen'].mean()
    RTs_bin_sem = df.groupby(['TrialBins'])['RespWinLen'].sem().values
    bin_indexes = [int(interval.right) for interval in RTs_bin_mean.index]



    # session_indexes = df.SessionIndex.unique()
    # RTs = []
    #
    # for _ in range(len(session_indexes)):
    #     print(_)
    #     df_session = df[df.SessionIndex == session_indexes[_]].reset_index(drop=True)
    #     if len(df_session) < bin_size * 2:
    #         RTs.append([np.nan])
    #     else:
    #         bins = np.arange(df_session.Trial.min(), df_session.Trial.max(), bin_size)
    #         RTs_bins_means = stats.binned_statistic(df_session.Trial, df_session.RespWinLen, statistic='mean', bins=bins).statistic
    #         RTs_bins_means = list(RTs_bins_means)
    #         RTs.append(RTs_bins_means)
    #
    # RTs = pd.DataFrame(RTs)
    # RTs = RTs.mean(axis=0)



    time_end = time.time()
    runtime = time_end - time_start
    print('The script took', round(runtime, 2), 'seconds to run')

    return RTs_bin_mean, RTs_bin_sem, bin_indexes, n_trials
n_trials = len(df)
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

RTs_means = []
n_trials_sum = []
for animal in animals:
    print(animal)
    RTs_bin_mean, RTs_bin_sem, bin_indexes, n_trials = get_RTs_subject(experiment=experiment, animal=animal, bin_size=bin_size)
    # plt.plot(RTs, label=animal, marker='o', color='tab:gray', alpha=0.25)
    RTs_means.append(RTs_bin_mean)
    n_trials_sum.append(n_trials)
# plt.legend()
test = pd.DataFrame(RTs_means)
RTs_means = pd.DataFrame(RTs_means)
del test
n_trials_sum = sum(n_trials_sum)
for animal in animals:
    print(animal)
    RTs_bin_mean, RTs_bin_sem, bin_indexes, n_trials = get_RTs_subject(experiment=experiment, animal=animal, bin_size=bin_size)
    plt.plot(RTs, label=animal, marker='o', color='tab:gray', alpha=0.25)
    RTs_means.append(RTs_bin_mean)
    n_trials_sum.append(n_trials)
# plt.legend()
plt.figure(constrained_layout=True)
for animal in animals:
    print(animal)
    RTs_bin_mean, RTs_bin_sem, bin_indexes, n_trials = get_RTs_subject(experiment=experiment, animal=animal, bin_size=bin_size)
    plt.plot(RTs_bin_mean, label=animal, marker='o', color='tab:gray', alpha=0.25)
    RTs_means.append(RTs_bin_mean)
    n_trials_sum.append(n_trials)
# plt.legend()
for animal in animals:
    print(animal)
    RTs_bin_mean, RTs_bin_sem, bin_indexes, n_trials = get_RTs_subject(experiment=experiment, animal=animal, bin_size=bin_size)
    plt.plot(bin_indexes, RTs_bin_mean, label=animal, marker='o', color='tab:gray', alpha=0.25)
    RTs_means.append(RTs_bin_mean)
    n_trials_sum.append(n_trials)
# plt.legend()
def get_RTs_subject(experiment=None, animal=None, bin_size=100):

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
    # df = df[df.Hit == 0]  # Select only error trials
    df = df[df.Hit == 1]  # Select only correct trials

    if df.Experiment.unique() == '2AFC_2':
        df = df[df.Drug.isnull()]  # Remove drug experimental sessions

    ####################################################################################################################

    n_trials = len(df)

    bin_size = 100
    bins = range(0, df.Trial.max() + bin_size, 100)
    df['TrialBins'] = pd.cut(df['Trial'], bins=bins)
    RTs_bin_mean = df.groupby(['TrialBins'])['RespWinLen'].mean()
    RTs_bin_sem = df.groupby(['TrialBins'])['RespWinLen'].sem().values
    bin_indexes = [int(interval.right) for interval in RTs_bin_mean.index]

    # session_indexes = df.SessionIndex.unique()
    # RTs = []
    #
    # for _ in range(len(session_indexes)):
    #     print(_)
    #     df_session = df[df.SessionIndex == session_indexes[_]].reset_index(drop=True)
    #     if len(df_session) < bin_size * 2:
    #         RTs.append([np.nan])
    #     else:
    #         bins = np.arange(df_session.Trial.min(), df_session.Trial.max(), bin_size)
    #         RTs_bins_means = stats.binned_statistic(df_session.Trial, df_session.RespWinLen, statistic='mean', bins=bins).statistic
    #         RTs_bins_means = list(RTs_bins_means)
    #         RTs.append(RTs_bins_means)
    #
    # RTs = pd.DataFrame(RTs)
    # RTs = RTs.mean(axis=0)

    time_end = time.time()
    runtime = time_end - time_start
    print('The script took', round(runtime, 2), 'seconds to run')

    return RTs_bin_mean, RTs_bin_sem, bin_indexes, n_trials
# Curated list
if experiment == '2AFC_2':
    animals = ['325', '327', '329', '330', '332', '333', '335', '337']
elif experiment == '2AFC_3':
    animals = ['419', '420', '422', '616', '619', '623']

RTs_means = []
n_trials_sum = []
plt.figure(constrained_layout=True)
for animal in animals:
    print(animal)
    RTs_bin_mean, RTs_bin_sem, bin_indexes, n_trials = get_RTs_subject(experiment=experiment, animal=animal, bin_size=bin_size)
    plt.plot(bin_indexes, RTs_bin_mean, label=animal, marker='o', color='tab:gray', alpha=0.25)
    RTs_means.append(RTs_bin_mean)
    n_trials_sum.append(n_trials)
# plt.legend()
RTs_means.mean(axis=0).plot(marker='o', color='k')
RTs_means = pd.DataFrame(RTs_means)
RTs_means.mean(axis=0).plot(marker='o', color='k')
RTs_means = []
n_trials_sum = []
plt.figure(constrained_layout=True)
for animal in animals:
    print(animal)
    RTs_bin_mean, RTs_bin_sem, bin_indexes, n_trials = get_RTs_subject(experiment=experiment, animal=animal, bin_size=bin_size)
    plt.plot(bin_indexes, RTs_bin_mean, label=animal, marker='o', color='tab:gray', alpha=0.25)
    RTs_means.append(RTs_bin_mean)
    n_trials_sum.append(n_trials)
# plt.legend()

RTs_means = pd.DataFrame(RTs_means)
n_trials_sum = sum(n_trials_sum)
RTs_means = RTs_means.mean(axis=0)
plt.plot(RTs_means, marker='o', color='k')
RTs_means = []
n_trials_sum = []
plt.figure(constrained_layout=True)
for animal in animals:
    print(animal)
    RTs_bin_mean, RTs_bin_sem, bin_indexes, n_trials = get_RTs_subject(experiment=experiment, animal=animal, bin_size=bin_size)
    plt.plot(bin_indexes, RTs_bin_mean, label=animal, marker='o', color='tab:gray', alpha=0.25)
    RTs_means.append(RTs_bin_mean)
    n_trials_sum.append(n_trials)
# plt.legend()
RTs_means = pd.DataFrame(RTs_means)
test ) RTs_means = RTs_means.mean(axis=1)
test = RTs_means = RTs_means.mean(axis=1)
test = RTs_means = RTs_means.mean(axis=0)
RTs_means = []
n_trials_sum = []
plt.figure(constrained_layout=True)
for animal in animals:
    print(animal)
    RTs_bin_mean, RTs_bin_sem, bin_indexes, n_trials = get_RTs_subject(experiment=experiment, animal=animal, bin_size=bin_size)
    plt.plot(bin_indexes, RTs_bin_mean, label=animal, marker='o', color='tab:gray', alpha=0.25)
    RTs_means.append(RTs_bin_mean)
    n_trials_sum.append(n_trials)
RTs_means = pd.DataFrame(RTs_means)
RTs_means.mean()
test = RTs_means.mean()
plt.plot(test,'ok')
test.index
test.index[0]
test.index[0].right
test.index.right
bin_indexes = [int(interval.right) for interval in test.index]
plt.plot(bin_indexes, test, 'ko')
RTs_means = []
RTs_sems = []
bin_indexes_max = []
n_trials_sum = []
plt.figure(constrained_layout=True)
for animal in animals:
    print(animal)
    RTs_bin_mean, RTs_bin_sem, bin_indexes, n_trials = get_RTs_subject(experiment=experiment, animal=animal, bin_size=bin_size)
    plt.plot(bin_indexes, RTs_bin_mean, label=animal, marker='o', color='tab:gray', alpha=0.25)
    RTs_means.append(RTs_bin_mean)
    RTs_sems.append(RTs_bin_sem)
    bin_indexes_max.append(bin_indexes)
    n_trials_sum.append(n_trials)
# plt.legend()
n_trials_sum = sum(n_trials_sum)
RT_sem = pd.DataFrame(RT_mean)
bin_indexes_max = pd.DataFrame(bin_indexes_max)
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


def get_RTs_subject(experiment=None, animal=None, bin_size=100):

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
    # df = df[df.Hit == 0]  # Select only error trials
    df = df[df.Hit == 1]  # Select only correct trials

    if df.Experiment.unique() == '2AFC_2':
        df = df[df.Drug.isnull()]  # Remove drug experimental sessions

    ####################################################################################################################

    n_trials = len(df)

    bin_size = 100
    bins = range(0, df.Trial.max() + bin_size, 100)
    df['TrialBins'] = pd.cut(df['Trial'], bins=bins)
    RTs_bin_mean = df.groupby(['TrialBins'])['RespWinLen'].mean()
    RTs_bin_sem = df.groupby(['TrialBins'])['RespWinLen'].sem().values
    bin_indexes = [int(interval.right) for interval in RTs_bin_mean.index]

    # session_indexes = df.SessionIndex.unique()
    # RTs = []
    #
    # for _ in range(len(session_indexes)):
    #     print(_)
    #     df_session = df[df.SessionIndex == session_indexes[_]].reset_index(drop=True)
    #     if len(df_session) < bin_size * 2:
    #         RTs.append([np.nan])
    #     else:
    #         bins = np.arange(df_session.Trial.min(), df_session.Trial.max(), bin_size)
    #         RTs_bins_means = stats.binned_statistic(df_session.Trial, df_session.RespWinLen, statistic='mean', bins=bins).statistic
    #         RTs_bins_means = list(RTs_bins_means)
    #         RTs.append(RTs_bins_means)
    #
    # RTs = pd.DataFrame(RTs)
    # RTs = RTs.mean(axis=0)

    time_end = time.time()
    runtime = time_end - time_start
    print('The script took', round(runtime, 2), 'seconds to run')

    return RTs_bin_mean, RTs_bin_sem, bin_indexes, n_trials


def get_RTs_experiment(experiment=None, bin_size=100):

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

    RT_mean = []
    RT_sem = []
    bin_indexes_max = []
    n_trials_sum = []
    plt.figure(constrained_layout=True)
    for animal in animals:
        print(animal)
        RTs_bin_mean, RTs_bin_sem, bin_indexes, n_trials = get_RTs_subject(experiment=experiment, animal=animal, bin_size=bin_size)
        # plt.plot(bin_indexes, RTs_bin_mean, label=animal, marker='o', color='tab:gray', alpha=0.25)
        RT_mean.append(RTs_bin_mean)
        RT_sem.append(RTs_bin_sem)
        bin_indexes_max.append(bin_indexes)
        n_trials_sum.append(n_trials)
    # plt.legend()

    RT_mean = pd.DataFrame(RT_mean)
    RT_sem = pd.DataFrame(RT_mean)
    bin_indexes_max = pd.DataFrame(bin_indexes_max)
    n_trials_sum = sum(n_trials_sum)
    # RT_mean = RT_mean.mean()
    # plt.plot(RT_mean, marker='o', color='k')

    return RT_mean, RT_sem, bin_indexes_max, n_trials_sum


def get_RTs_mean(experiments=['2AFC_2', '2AFC_3'], bin_size=100, save=False):

    RTs_mean = pd.DataFrame()
    n_trials_sum_sum = []
    for experiment in experiments:
        print(experiment)
        RT_mean, RT_sem, bin_indexes_max, n_trials_sum = get_RTs_experiment(experiment=experiment, bin_size=bin_size)
        RTs_mean = pd.concat([RTs_mean, RT_mean])
        n_trials_sum_sum.append(n_trials_sum)

    n_trials_sum_sum = sum(n_trials_sum_sum)

    plt.figure(constrained_layout=True)
    color = 'k'
    label = ''
    # color = 'tab:red'
    # label = 'error'
    # filename = f'mean_RTs_error_experiments_{experiments}_{bin_size}_trials_bins'
    color = 'tab:green'
    label = 'correct'
    filename = f'mean_RTs_correct_experiments_{experiments}_{bin_size}_trials_bins'

    for _ in range(len(RTs_mean)):
        print()
        plt.plot(RTs_mean.iloc[_], marker='o', color='tab:gray', alpha=0.25)

    RTs_mean = RTs_mean.iloc[:, :-1]  # Remove last column as there's only one value
    RTs_mean_mean = RTs_mean.mean()
    RTs_mean_sem = RTs_mean.sem()
    plt.plot(RTs_mean_mean, marker='o', color=color, label=label)
    plt.errorbar(RTs_mean_sem.index.values, RTs_mean_mean, RTs_mean_sem, color=color, marker='o', fmt='none', mec='none', ms=0)
    plt.xlabel('Trial')
    plt.ylabel('RT (ms)')
    xticks = np.arange(0, RTs_mean.shape[1], 2)
    xticklabels = xticks * bin_size
    plt.xticks(ticks=xticks, labels=xticklabels)
    plt.title(f'N={len(RTs_mean)}, {n_trials_sum_sum} trials')
    plt.legend()
    sns.despine(trim=True)  # Despine axes triming the 0

    # filename = f'mean_RTs_experiments_{experiments}_{bin_size}_trials_bins'
    if save:
        folder_out = Path.home() / 'Documentos' / 'trial index' / 'RTs' / 'mean'
        save_fig(folder_out, filename)


# Debugging
experiment = '2AFC_2'
animal = '333'
get_RTs_mean(experiments=['2AFC_2'], bin_size=100, save=True)
# get_RTs_mean(experiments=['2AFC_2', '2AFC_3'], bin_size=100, save=True)
RTs_mean = pd.DataFrame()
n_trials_sum_sum = []
for experiment in experiments:
    print(experiment)
    RT_mean, RT_sem, bin_indexes_max, n_trials_sum = get_RTs_experiment(experiment=experiment, bin_size=bin_size)
    RTs_mean = pd.concat([RTs_mean, RT_mean])
    n_trials_sum_sum.append(n_trials_sum)

n_trials_sum_sum = sum(n_trials_sum_sum)
experiments = '2AFC_2'
RTs_mean = pd.DataFrame()
n_trials_sum_sum = []
for experiment in experiments:
    print(experiment)
    RT_mean, RT_sem, bin_indexes_max, n_trials_sum = get_RTs_experiment(experiment=experiment, bin_size=bin_size)
    RTs_mean = pd.concat([RTs_mean, RT_mean])
    n_trials_sum_sum.append(n_trials_sum)

n_trials_sum_sum = sum(n_trials_sum_sum)

plt.figure(constrained_layout=True)
color = 'k'
label = ''
# color = 'tab:red'
# label = 'error'
# filename = f'mean_RTs_error_experiments_{experiments}_{bin_size}_trials_bins'
color = 'tab:green'
label = 'correct'
filename = f'mean_RTs_correct_experiments_{experiments}_{bin_size}_trials_bins'
experiment=experiments
RTs_mean = pd.DataFrame()
n_trials_sum_sum = []
for experiment in experiments:
    print(experiment)
    RT_mean, RT_sem, bin_indexes_max, n_trials_sum = get_RTs_experiment(experiment=experiment, bin_size=bin_size)
    RTs_mean = pd.concat([RTs_mean, RT_mean])
    n_trials_sum_sum.append(n_trials_sum)
experiments=['2AFC_2', '2AFC_3']
RTs_mean = pd.DataFrame()
n_trials_sum_sum = []
for experiment in experiments:
    print(experiment)
    RT_mean, RT_sem, bin_indexes_max, n_trials_sum = get_RTs_experiment(experiment=experiment, bin_size=bin_size)
    RTs_mean = pd.concat([RTs_mean, RT_mean])
    n_trials_sum_sum.append(n_trials_sum)

n_trials_sum_sum = sum(n_trials_sum_sum)
for _ in range(len(RTs_mean)):
    print()
    plt.plot(RTs_mean.iloc[_], marker='o', color='tab:gray', alpha=0.25)
for _ in range(len(RTs_mean)):
    print()
    plt.plot(bin_indexes_max[_], RTs_mean.iloc[_], marker='o', color='tab:gray', alpha=0.25)
RTs_mean.iloc[0]
for _ in range(len(RTs_mean)):
    print()
    plt.plot(bin_indexes_max[_], RTs_mean.iloc[_], marker='o', color='tab:gray', alpha=0.25)
RTs_mean.iloc[0].index
RTs_mean.iloc[0].index.right
for _ in range(len(RTs_mean)):
    print()
    x = RTs_mean.iloc[_].index.right
    plt.plot(x, RTs_mean.iloc[_], marker='o', color='tab:gray', alpha=0.25)
for _ in range(len(RTs_mean)):
    print()
    x = RTs_mean.iloc[_].index.right
    plt.plot(x, RTs_mean.iloc[_], marker='o', color='tab:gray', alpha=0.25)
RTs_mean = RTs_mean.iloc[:, :-1]  # Remove last column as there's only one value
RTs_mean_mean = RTs_mean.mean()
RTs_mean_sem = RTs_mean.sem()
plt.plot(RTs_mean_mean, marker='o', color=color, label=label)
plt.errorbar(RTs_mean_sem.index.values, RTs_mean_mean, RTs_mean_sem, color=color, marker='o', fmt='none', mec='none', ms=0)
plt.xlabel('Trial')
plt.ylabel('RT (ms)')
xticks = np.arange(0, RTs_mean.shape[1], 2)
xticklabels = xticks * bin_size
plt.xticks(ticks=xticks, labels=xticklabels)
plt.title(f'N={len(RTs_mean)}, {n_trials_sum_sum} trials')
plt.legend()
sns.despine(trim=True)  # Despine axes triming the 0
color = 'k'
label = ''
RTs_mean = RTs_mean.iloc[:, :-1]  # Remove last column as there's only one value
RTs_mean_mean = RTs_mean.mean()
RTs_mean_sem = RTs_mean.sem()
plt.plot(RTs_mean_mean, marker='o', color=color, label=label)
plt.errorbar(RTs_mean_sem.index.values, RTs_mean_mean, RTs_mean_sem, color=color, marker='o', fmt='none', mec='none', ms=0)
plt.xlabel('Trial')
plt.ylabel('RT (ms)')
xticks = np.arange(0, RTs_mean.shape[1], 2)
xticklabels = xticks * bin_size
plt.xticks(ticks=xticks, labels=xticklabels)
plt.title(f'N={len(RTs_mean)}, {n_trials_sum_sum} trials')
plt.legend()
sns.despine(trim=True)  # Despine axes triming the 0
for _ in range(len(RTs_mean)):
    print()
    x = RTs_mean.iloc[_].index.right
    plt.plot(x, RTs_mean.iloc[_], marker='o', color='tab:gray', alpha=0.25)
RTs_mean = pd.DataFrame()
n_trials_sum_sum = []
for experiment in experiments:
    print(experiment)
    RT_mean, RT_sem, bin_indexes_max, n_trials_sum = get_RTs_experiment(experiment=experiment, bin_size=bin_size)
    RTs_mean = pd.concat([RTs_mean, RT_mean])
    n_trials_sum_sum.append(n_trials_sum)

n_trials_sum_sum = sum(n_trials_sum_sum)

plt.figure(constrained_layout=True)
color = 'k'
label = ''
# color = 'tab:red'
# label = 'error'
# filename = f'mean_RTs_error_experiments_{experiments}_{bin_size}_trials_bins'
# color = 'tab:green'
# label = 'correct'
# filename = f'mean_RTs_correct_experiments_{experiments}_{bin_size}_trials_bins'

for _ in range(len(RTs_mean)):
    print()
    x = RTs_mean.iloc[_].index.right
    plt.plot(x, RTs_mean.iloc[_], marker='o', color='tab:gray', alpha=0.25)
for _ in range(len(RTs_mean)):
    print()
    x = RTs_mean.iloc[_].index.right
    plt.plot(x, RTs_mean.iloc[_], marker='o', color='tab:gray', alpha=0.25)
for _ in range(len(RTs_mean)):
    print()
    x = RTs_mean.iloc[_].index.right
    plt.plot(x, RTs_mean.iloc[_], marker='o', color='tab:gray', alpha=0.25)
RTs_mean = RTs_mean.iloc[:, :-1]  # Remove last column as there's only one value
RTs_mean_mean = RTs_mean.mean()
RTs_mean_sem = RTs_mean.sem()
plt.plot(RTs_mean_mean, marker='o', color=color, label=label)
plt.errorbar(RTs_mean_sem.index.values, RTs_mean_mean, RTs_mean_sem, color=color, marker='o', fmt='none', mec='none', ms=0)
x = RTs_mean.index.right
x = RTs_mean_mean.index.right
x = RTs_mean_mean.index.right
RTs_mean_sem = RTs_mean.sem()
plt.plot(x, RTs_mean_mean, marker='o', color=color, label=label)
plt.errorbar(x, RTs_mean_sem.index.values, RTs_mean_mean, RTs_mean_sem, color=color, marker='o', fmt='none', mec='none', ms=0)
plt.errorbar(x, RTs_mean_mean, RTs_mean_sem, color=color, marker='o', fmt='none', mec='none', ms=0)
plt.xlabel('Trial')
plt.ylabel('RT (ms)')
plt.xlabel('Trial')
def get_RTs_subject(experiment=None, animal=None, bin_size=100):

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
    # df = df[df.Hit == 0]  # Select only error trials
    df = df[df.Hit == 1]  # Select only correct trials

    if df.Experiment.unique() == '2AFC_2':
        df = df[df.Drug.isnull()]  # Remove drug experimental sessions

    ####################################################################################################################

    n_trials = len(df)

    bin_size = 100
    bins = range(0, df.Trial.max() + bin_size, 100)
    df['TrialBins'] = pd.cut(df['Trial'], bins=bins)
    RTs_bin_mean = df.groupby(['TrialBins'])['RespWinLen'].mean()
    RTs_bin_sem = df.groupby(['TrialBins'])['RespWinLen'].sem().values
    bin_indexes = [int(interval.right) for interval in RTs_bin_mean.index]

    # session_indexes = df.SessionIndex.unique()
    # RTs = []
    #
    # for _ in range(len(session_indexes)):
    #     print(_)
    #     df_session = df[df.SessionIndex == session_indexes[_]].reset_index(drop=True)
    #     if len(df_session) < bin_size * 2:
    #         RTs.append([np.nan])
    #     else:
    #         bins = np.arange(df_session.Trial.min(), df_session.Trial.max(), bin_size)
    #         RTs_bins_means = stats.binned_statistic(df_session.Trial, df_session.RespWinLen, statistic='mean', bins=bins).statistic
    #         RTs_bins_means = list(RTs_bins_means)
    #         RTs.append(RTs_bins_means)
    #
    # RTs = pd.DataFrame(RTs)
    # RTs = RTs.mean(axis=0)

    time_end = time.time()
    runtime = time_end - time_start
    print('The script took', round(runtime, 2), 'seconds to run')

    return RTs_bin_mean, RTs_bin_sem, bin_indexes, n_trials


def get_RTs_experiment(experiment=None, bin_size=100):

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

    RT_mean = []
    RT_sem = []
    bin_indexes_max = []
    n_trials_sum = []
    # plt.figure(constrained_layout=True)
    for animal in animals:
        print(animal)
        RTs_bin_mean, RTs_bin_sem, bin_indexes, n_trials = get_RTs_subject(experiment=experiment, animal=animal, bin_size=bin_size)
        # plt.plot(bin_indexes, RTs_bin_mean, label=animal, marker='o', color='tab:gray', alpha=0.25)
        RT_mean.append(RTs_bin_mean)
        RT_sem.append(RTs_bin_sem)
        bin_indexes_max.append(bin_indexes)
        n_trials_sum.append(n_trials)
    # plt.legend()

    RT_mean = pd.DataFrame(RT_mean)
    RT_sem = pd.DataFrame(RT_mean)
    bin_indexes_max = pd.DataFrame(bin_indexes_max)
    n_trials_sum = sum(n_trials_sum)
    # RT_mean = RT_mean.mean()
    # plt.plot(RT_mean, marker='o', color='k')

    return RT_mean, RT_sem, bin_indexes_max, n_trials_sum
RTs_mean = pd.DataFrame()
n_trials_sum_sum = []
for experiment in experiments:
    print(experiment)
    RT_mean, RT_sem, bin_indexes_max, n_trials_sum = get_RTs_experiment(experiment=experiment, bin_size=bin_size)
    RTs_mean = pd.concat([RTs_mean, RT_mean])
    n_trials_sum_sum.append(n_trials_sum)

n_trials_sum_sum = sum(n_trials_sum_sum)

plt.figure(constrained_layout=True)
color = 'k'
label = ''
# color = 'tab:red'
# label = 'error'
# filename = f'mean_RTs_error_experiments_{experiments}_{bin_size}_trials_bins'
# color = 'tab:green'
# label = 'correct'
# filename = f'mean_RTs_correct_experiments_{experiments}_{bin_size}_trials_bins'

for _ in range(len(RTs_mean)):
    print()
    x = RTs_mean.iloc[_].index.right
    plt.plot(x, RTs_mean.iloc[_], marker='o', color='tab:gray', alpha=0.25)
plt.figure(constrained_layout=True)
color = 'k'
label = ''
# color = 'tab:red'
# label = 'error'
# filename = f'mean_RTs_error_experiments_{experiments}_{bin_size}_trials_bins'
# color = 'tab:green'
# label = 'correct'
# filename = f'mean_RTs_correct_experiments_{experiments}_{bin_size}_trials_bins'

for _ in range(len(RTs_mean)):
    print()
    x = RTs_mean.iloc[_].index.right
    plt.plot(x, RTs_mean.iloc[_], marker='o', color='tab:gray', alpha=0.25)
RTs_mean = RTs_mean.iloc[:, :-1]  # Remove last column as there's only one value
RTs_mean_mean = RTs_mean.mean()
x = RTs_mean_mean.index.right
RTs_mean_sem = RTs_mean.sem()
plt.plot(x, RTs_mean_mean, marker='o', color=color, label=label)
plt.errorbar(x, RTs_mean_mean, RTs_mean_sem, color=color, marker='o', fmt='none', mec='none', ms=0)
plt.xlabel('Trial')
plt.ylabel('RT (ms)')
plt.title(f'N={len(RTs_mean)}, {n_trials_sum_sum} trials')
sns.despine(trim=True)  # Despine axes triming the 0
filename = f'mean_RTs_correct_experiments_{experiments}_{bin_size}_trials_bins'
# filename = f'mean_RTs_experiments_{experiments}_{bin_size}_trials_bins'
if save:
    folder_out = Path.home() / 'Documentos' / 'trial index' / 'RTs' / 'mean'
    save_fig(folder_out, filename)
