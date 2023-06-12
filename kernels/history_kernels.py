import time
from pathlib import Path
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from matplotlib import pyplot as plt
import seaborn as sns
from collections import namedtuple
from my_fun.my_fun import get_experiment, get_animal, get_ild
from kernels.kernels_tools import *

# Plotting parameters
sns.set_theme()
sns.set_style('white')
sns.set_style('ticks')
sns.set_context('poster')


# GLM weights of previously rewarded (r+) and previously unrewarded (r-) responses. These kernels quantify the
# influence on choice of the side (left vs. right) of previous responses.
# From 'Response outcomes gate the impact of expectations on perceptual decisions', Figure 4
# (https://www-nature-com.sire.ub.edu/articles/s41467-020-14824-w)


def get_hk(experiment='2AFC_2', animal=None, drug=None, trial_lag=10, iterations=100):

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

    df = df.reset_index(drop=True)
    n_trials = len(df)
    dm_choice_history = make_choice_history_dm(df, trial_lag)
    dm_session_index = make_session_index_dm(df)
    dm_ild = make_ild_dm(df)
    exog = pd.concat([dm_choice_history, dm_session_index, dm_ild], axis=1)

    ####################################################################################################################

    endog = df.Choice
    model = sm.GLM(endog, exog, family=sm.families.Binomial(), missing='drop')  # GLM with Binomial family
    results = model.fit()
    params = results.params
    beta_std_err = results.bse
    p_values = results.pvalues
    summary = results.summary()
    print(summary)

    # Get shuffles
    shuffles = get_shuffles_GLM(endog, exog, iterations=100)

    # Remove session index and stimulus constants
    params = params.iloc[:trial_lag * 2]  # From params
    shuffles = [shuffles[i].iloc[:trial_lag * 2] for i in range(len(shuffles))]  # From shuffles
    beta_std_err = beta_std_err.iloc[:trial_lag * 2]  # From beta_std_err
    p_values = p_values.iloc[:trial_lag * 2]  # From p_values

    # Store results in a namedtuple
    HK = namedtuple('HK', ['params', 'std_err', 'p_values', 'shuffles', 'n_trials', 'trial_lag', 'experiment',
                           'animal', 'drug', 'iterations'])
    hk = HK(params=params, std_err=beta_std_err, p_values=p_values, shuffles=shuffles, n_trials=n_trials,
            trial_lag=trial_lag, experiment=experiment, animal=animal, drug=drug, iterations=iterations)

    time_end = time.time()
    runtime = time_end - time_start
    print('The script took', round(runtime, 2), 'seconds to run')

    return hk


def plot_hk(experiment='2AFC_2', animal=None, drug=None, iterations=100, save=False, format='svg', transparent=False):

    time_start = time.time()

    hk = get_hk(experiment=experiment, animal=animal, drug=drug, iterations=iterations)

    # Default plotting parameters
    color = 'k'
    color_upper_shuffle = 'tab:red'
    label = ''

    trial_lag = hk.trial_lag

    for _ in range(1, 3):

        if _ == 1:
            title = '$r^{-}$'
            params_indexes = np.arange(trial_lag * _)
        elif _ == 2:
            title = '$r^{+}$'
            params_indexes = np.arange(trial_lag, trial_lag * _)

        # Plot history kernel (responses lag beta weights)
        # + 1 to skip constant; + int(residuals) to skip ILD
        plt.figure(constrained_layout=True)
        x = np.arange(1, trial_lag + 1)
        y = hk.params.iloc[params_indexes]
        yerr = hk.std_err.iloc[params_indexes]
        plt.plot(x, y, color=color, marker='o', label=label)
        plt.errorbar(x, y, yerr=yerr, color=color, marker='o', fmt='none', mec='none', ms=0)
        plt.title(title)
        plt.xticks(x, x[::-1])
        plt.xlabel('Trial lag')
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

        shuffles_mean = np.mean(hk.shuffles, axis=0)  # Get the mean of all the shuffles
        percentiles2_5 = np.percentile(hk.shuffles, 2.5, axis=0)  # Get upper 5 percentile of the shuffled_var
        percentiles97_5 = np.percentile(hk.shuffles, 97.5, axis=0)  # Get upper 5 percentile of the shuffled_var
        plt.plot(x, shuffles_mean[params_indexes], color='tab:gray', ls='--', zorder=1.8)
        plt.plot(x, percentiles2_5[params_indexes], color=color_upper_shuffle, ls=':', zorder=1.9)
        plt.plot(x, percentiles97_5[params_indexes], color=color_upper_shuffle, ls=':', zorder=2)

        sns.despine(trim=True)  # Despine top and right axes triming them to their min/max tick

        time_end = time.time()
        runtime = time_end - time_start
        print('The script took', round(runtime, 2), 'seconds to run')


########################################################################################################################

# Debugging
experiment = '2AFC_2'
# experiments = ['2AFC_2', '2AFC_3']
animal = '333'
# animals = ['325', '327', '329', '330', '332', '333', '335', '337']  # Bach 2 (with ILDs) -326, -334
# animals = ['419', '420', '422', '616', '619', '623']  # Batch 3 (with ILDs)  -617, -620
# animals = ['332', '333', '337']  # Drug experiments
target_ilds = None
drug = None
residuals = True
iterations = 100
save = False
format = 'svg'
transparent = True
trial_lag = 10