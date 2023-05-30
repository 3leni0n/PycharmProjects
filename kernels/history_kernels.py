import time
from pathlib import Path
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from matplotlib import pyplot as plt
from scipy import stats
import seaborn as sns
from collections import namedtuple
from my_fun.my_fun import get_experiment, get_animal, get_ild

# Plotting parameters
sns.set_theme()
sns.set_style('white')
sns.set_style('ticks')
sns.set_context('poster')


def get_hk(experiment='2AFC_2', animal=None, library='sm', target_ilds=None, drug=None, zscore=True, kind=None,
           iterations=1000):

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
    n_trials_lag = 5  # Number of trials to use for lagged variables

    # GLM weights of previously rewarded (r+) and previously unrewarded (r-) responses. These kernels quantify the
    # influence on choice of the side (left vs. right) of previous responses.
    # From 'Response outcomes gate the impact of expectations on perceptual decisions', Figure 4
    # (https://www-nature-com.sire.ub.edu/articles/s41467-020-14824-w)

    # r+ (t-k)
    r_plus_5 = [1 if df.Hit[i - 5] == 1 and df.Choice[i - 5] == 1 else -1 if df.Hit[i - 5] == 1 and df.Choice[i - 5] == 0 else 0 for i in range(5, len(df))]
    r_plus_5 = [np.nan] * 5 + r_plus_5
    r_plus_4 = [1 if df.Hit[i - 4] == 1 and df.Choice[i - 4] == 1 else -1 if df.Hit[i - 4] == 1 and df.Choice[i - 4] == 0 else 0 for i in range(4, len(df))]
    r_plus_4 = [np.nan] * 4 + r_plus_4
    r_plus_3 = [1 if df.Hit[i - 3] == 1 and df.Choice[i - 3] == 1 else -1 if df.Hit[i - 3] == 1 and df.Choice[i - 3] == 0 else 0 for i in range(3, len(df))]
    r_plus_3 = [np.nan] * 3 + r_plus_3
    r_plus_2 = [1 if df.Hit[i - 2] == 1 and df.Choice[i - 2] == 1 else -1 if df.Hit[i - 2] == 1 and df.Choice[i - 2] == 0 else 0 for i in range(2, len(df))]
    r_plus_2 = [np.nan] * 2 + r_plus_2
    r_plus_1 = [1 if df.Hit[i - 1] == 1 and df.Choice[i - 1] == 1 else -1 if df.Hit[i - 1] == 1 and df.Choice[i - 1] == 0 else 0 for i in range(1, len(df))]
    r_plus_1 = [np.nan] * 1 + r_plus_1
    df['Rplus5'] = r_plus_5
    df['Rplus4'] = r_plus_4
    df['Rplus3'] = r_plus_3
    df['Rplus2'] = r_plus_2
    df['Rplus1'] = r_plus_1

    # r-(t-k)
    r_minus_5 = [1 if df.Hit[i - 5] == 0 and df.Choice[i - 5] == 1 else -1 if df.Hit[i - 5] == 0 and df.Choice[i - 5] == 0 else 0 for i in range(5, len(df))]
    r_minus_5 = [np.nan] * 5 + r_minus_5
    r_minus_4 = [1 if df.Hit[i - 4] == 0 and df.Choice[i - 4] == 1 else -1 if df.Hit[i - 4] == 0 and df.Choice[i - 4] == 0 else 0 for i in range(4, len(df))]
    r_minus_4 = [np.nan] * 4 + r_minus_4
    r_minus_3 = [1 if df.Hit[i - 3] == 0 and df.Choice[i - 3] == 1 else -1 if df.Hit[i - 3] == 0 and df.Choice[i - 3] == 0 else 0 for i in range(3, len(df))]
    r_minus_3 = [np.nan] * 3 + r_minus_3
    r_minus_2 = [1 if df.Hit[i - 2] == 0 and df.Choice[i - 2] == 1 else -1 if df.Hit[i - 2] == 0 and df.Choice[i - 2] == 0 else 0 for i in range(2, len(df))]
    r_minus_2 = [np.nan] * 2 + r_minus_2
    r_minus_1 = [1 if df.Hit[i - 1] == 0 and df.Choice[i - 1] == 1 else -1 if df.Hit[i - 1] == 0 and df.Choice[i - 1] == 0 else 0 for i in range(1, len(df))]
    r_minus_1 = [np.nan] * 1 + r_minus_1
    df['Rminus5'] = r_minus_5
    df['Rminus4'] = r_minus_4
    df['Rminus3'] = r_minus_3
    df['Rminus2'] = r_minus_2
    df['Rminus1'] = r_minus_1

    ####################################################################################################################

    choices = df.Choice
    r_plus = df[['ILD', 'Rplus5', 'Rplus4',  'Rplus3', 'Rplus2', 'Rplus1']]
    r_plus = sm.add_constant(r_plus)
    r_minus = df[['ILD', 'Rminus5', 'Rminus4', 'Rminus3', 'Rminus2', 'Rminus1']]
    r_minus = sm.add_constant(r_minus)

    if kind =='r_plus':
        exog = r_plus
    elif kind == 'r_minus':
        exog = r_minus

    model = sm.GLM(choices, exog, family=sm.families.Binomial(), missing='drop')  # GLM with Binomial family and Logit link
    results = model.fit()
    params = results.params
    beta_std_err = results.bse
    p_values = results.pvalues
    summary = results.summary()
    print(summary)

    # Permutation test (shuffled_var)
    shuffles = []
    # Shuffling the choices or the stim_strength index is the same, so it doesn't matter. Shuffling along the
    # columns of stim_strength is wrong because it breaks the temporal structure of the data. Shuffling the frames
    # within trial could be an interesting test, as it preserves the overall weight of the stimulus for each trial
    # but breaks the frame structure
    for _ in range(iterations):
        choices_shuffled = choices.sample(frac=1).reset_index(drop=True)
        # stim_strength_shuffled = stim_strength.sample(frac=1).reset_index(drop=True)
        # model = sm.Logit(choices, stim_strength)  # Discrete Logit model
        model_shuffled = sm.GLM(choices_shuffled, exog,  # Shuffled choices
                                family=sm.families.Binomial(), missing='drop')  # GLM with Binomial family and Logit link
        # model_shuffled = sm.GLM(choices, stim_strength_shuffled,  # Shuffled stim_strength
        #                         family=sm.families.Binomial())  # GLM with Binomial family and Logit link
        results_shuffled = model_shuffled.fit()
        params_shuffled = results_shuffled.params
        shuffles.append(params_shuffled)
        # plt.plot(np.arange(1, len(params_shuffled)), params_shuffled.iloc[1:11], color='tab:gray', marker=None,
        #          mfc='none', mec='none', mew=0, ms=0, label=label, alpha=0.1, zorder=1.7)  # Plot all shuffles

    # shuffles_mean = np.mean(shuffles, axis=0)  # Get the mean of all the shuffles
    # percentiles95 = np.percentile(shuffles, 95, axis=0)  # Get upper 5 percentile of the shuffled_var

    # Store results in a namedtuple
    HK = namedtuple('HK', ['params', 'std_err', 'p_values', 'shuffles', 'n_trials', 'n_trials_lag', 'experiment',
                           'animal', 'library', 'target_ilds', 'drug', 'zscore', 'kind', 'iterations'])
    hk = HK(params=params, std_err=beta_std_err, p_values=p_values, shuffles=shuffles, n_trials=n_trials,
            n_trials_lag=n_trials_lag, experiment=experiment, animal=animal, library=library, target_ilds=None,
            drug=drug, zscore=zscore, kind=kind, iterations=iterations)

    time_end = time.time()
    runtime = time_end - time_start
    print('The script took', round(runtime, 2), 'seconds to run')

    return hk


def plot_hk(experiment='2AFC_2', animal=None, library='sm', target_ilds=None, drug=None, zscore=True, kind=None,
            iterations=1000, save=False, format='svg', transparent=False):

    time_start = time.time()

    ####################################################################################################################

    hk = get_hk(experiment=experiment, animal=animal, library=library, target_ilds=target_ilds, drug=drug, zscore=zscore,
                kind=kind, iterations=iterations)

    # Default plotting parameters
    color = 'k'
    color_upper_shuffle = 'tab:red'
    label = ''

    if kind == 'r_plus':
        title = '$r^{+}$'
    elif kind == 'r_minus':
        title = '$r^{-}$'

    ylabel = 'GLM weight'

    # Plot history kernel (responses lag beta weights)
    # + 1 to skip constant; + int(residuals) to skip ILD
    plt.figure(constrained_layout=True)
    n_trials_lag = hk.n_trials_lag
    x = np.arange(1 + 1, len(hk.params))
    y = hk.params.iloc[1 + 1:len(hk.params)]
    yerr = hk.std_err.iloc[1 + 1:len(hk.params)]
    plt.plot(x, y, color=color, marker='o', label=label)
    plt.errorbar(x, y, yerr=yerr, color=color, marker='o', fmt='none', mec='none', ms=0)
    plt.title(title)
    plt.xlabel('Trial lag')
    plt.ylabel(ylabel)
    plt.legend(frameon=False)
    yticks = plt.gca().get_yticks()  # Get current axis yticks for the significance annotations

    if hk.p_values is not None:
        for i in range(n_trials_lag):
            if hk.p_values[i] <= 0.05:
                text = '*'
            else:
                # text = 'ns'
                text = ''
            plt.annotate(text, xy=(i + 1 + int(residuals), yticks[1]), xytext=(i + 1 + int(residuals), yticks[1]),
                         color=color, va='center', ha='center', fontsize='medium')

    shuffles_mean = np.mean(hk.shuffles, axis=0)  # Get the mean of all the shuffles
    percentiles95 = np.percentile(hk.shuffles, 95, axis=0)  # Get upper 5 percentile of the shuffled_var
    plt.plot(x, shuffles_mean[1 + int(residuals):len(shuffles_mean)], color='tab:gray', ls='--', zorder=1.8)
    plt.plot(x, percentiles95[1 + int(residuals):len(shuffles_mean)], color=color_upper_shuffle, ls=':', zorder=1.9)

    # Adjust xticks to number of regressors (cont + ILD + n_trials_lag)
    xticks = np.arange(2, n_trials_lag + 2, 1)
    xticklabels = np.arange(1, n_trials_lag + 1, 1)
    xticklabels = reversed(xticklabels)
    plt.xticks(xticks, xticklabels)  # Readjust xticks
    sns.despine(offset=10, trim=True)  # Despine axes triming the 0

########################################################################################################################

# Debugging
experiment = '2AFC_2'
experiments = ['2AFC_2', '2AFC_3']
animal = '333'
# animals = ['325', '327', '329', '330', '332', '333', '335', '337']  # Bach 2 (with ILDs) -326, -334
# animals = ['419', '420', '422', '616', '619', '623']  # Batch 3 (with ILDs)  -617, -620
# animals = ['332', '333', '337']  # Drug experiments
library = 'sm'
target_ilds = None
drug = None
residuals = True
zscore = False
control = None
n_mean_frames = None
iterations = 100
save = False
format = 'svg'
transparent = True
