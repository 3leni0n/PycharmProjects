import time
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.optimize import minimize
from my_fun.my_fun import *  # Or from daily_report.daily_report import daily_report

import time
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from matplotlib import pyplot as plt
from scipy import stats
import seaborn as sns

# Mel's code snippet for poster
sns.set_theme()
sns.set_style('white')
sns.set_style('ticks')
sns.set_context('poster')
# sns.despine()

########################################################################################################################

# Resources
'https://towardsdatascience.com/cross-entropy-negative-log-likelihood-and-all-that-jazz-47a95bd2e81'

########################################################################################################################

time_start = time.time()

# Load toy data
experiment = '2AFC_2'
animal = '333'
library = 'sm'
target_ilds = [-70, -8, -4, -2, 0, 2, 4, 8, 70]
# target_ilds = [-2, 0, 2]
zscore = True
control = None
n_mean_frames = None
iterations = 10
save = False
format = 'svg'
transparent = False

if experiment is None:

    folder_in = '/home/alexis/PycharmProjects/glue_sessions/'  # Where the data for all animals is
    experiments = os.listdir(folder_in)  # List experiments
    experiments.sort()  # Sort them by name
    experiments = [x for x in experiments if os.path.isdir(folder_in + x)]  # Get rid of non folders

    try:
        experiments.remove('__pycache__')  # Pycharm's archive
    except ValueError:
        pass

    print('Experiments: ' + str(experiments)[1:-1])  # Remove square brackets
    experiment = input('Enter experiment name')

folder_in = '/home/alexis/PycharmProjects/glue_sessions/' + experiment + '/'  # Where the data for all animals is

if animal is None:
    animals = os.listdir(folder_in)  # List animals
    animals.sort()  # Sort them by name
    animals = [x[:-4] for x in animals]  # Get rid of .csv extension

    print('Animals: ' + str(animals))  # Remove square brackets
    animal = input('Enter animal')  # Ask user to input animal to glue sessions from

folder_in = folder_in + animal + '.csv'

# Load sounds
# sounds_path = '/home/alexis/PycharmProjects/create_sounds/sounds.csv'
sounds_path = '/home/alexis/PycharmProjects/create_sounds/sounds_2.csv'
sounds = pd.read_csv(sounds_path)
n_frames = 10

# Left frames
left_frames_column_names = [f'EL{n:01}' for n in range(n_frames)]
frames_left = sounds[left_frames_column_names]

# Right frames
right_frames_column_names = [f'ER{n:01}' for n in range(n_frames)]
frames_right = sounds[right_frames_column_names]

# Frames ILD (elementwise)
frames_ild = pd.DataFrame(
    sounds[right_frames_column_names].values - sounds[left_frames_column_names].values)  # Directly on the dataframe

# Zscore
if zscore:
    frames_ild = pd.DataFrame(stats.zscore(frames_ild, axis=None))  # Z-score the ILDs (along axis 0 or None
    # returns same result, but not axis 1)
    ylabel = 'GLM weight (z-scored)'
else:
    ylabel = 'GLM weight'

frames_ild.insert(0, column='filename', value=sounds.filename)  # Insert behavior_filenames in first column

# Load behavioral data
df = pd.read_csv(folder_in)  # Load behavioral data
df = df[df.Choice.notna()]  # Drop misses (nan in choices), otherwise the code crashes
ilds = np.sort(df.ILD.unique())
df = df[df.ILD.isin(target_ilds)]  # Select only trials with the desired ILDs

try:
    df = df[df.Drug.isnull()]  # Remove drug experimental sessions
except AttributeError:
    pass
n_trials = len(df)

filenames = df.Filename.tolist()

# Get complete dataset compute every iteration, otherwise the 2nd time will be doing the half of the half!
choices = df.Choice.reset_index(drop=True)  # Indices must match for modeling
stim_strength = frames_ild.loc[
    [np.where(sounds.filename == np.array(filenames[i]))[0][0] for i in range(len(filenames))]].drop(
    columns=['filename'])
stim_strength.reset_index(drop=True, inplace=True)  # Indices must match for modeling

# Transform variables into arrays (the code is almost x4 times faster with nNumpy arrays than with Pandas DataFrames)
x = np.array(stim_strength)
y = np.array(choices)


########################################################################################################################


def fit_model(x, y, lapses=True, iterations=10):
    """Computes a psychometric function.
    x is a vector
    """
    # https://psychology.stackexchange.com/questions/13347/how-can-i-fit-a-psychometric-function-such-that-the-minimum-is-50-chance-level

    frame1 = x[:, 0]
    frame2 = x[:, 1]
    frame3 = x[:, 2]
    frame4 = x[:, 3]
    frame5 = x[:, 4]
    frame6 = x[:, 5]
    frame7 = x[:, 6]
    frame8 = x[:, 7]
    frame9 = x[:, 8]
    frame10 = x[:, 9]

    def sigmoid_mme(fit_params: tuple):

        if lapses:  # With lapses
            lapse1, lapse2, bias, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10 = fit_params
            # Function to fit:
            p_right = lapse1 + (1 - lapse1 - lapse2) / (
                    1 + np.exp(-(bias + k1 * frame1 + k2 * frame2 + k3 * frame3 + k4 * frame4 + k5 * frame5 +
                                 k6 * frame6 + k7 * frame7 + k8 * frame8 + k9 * frame9 + k10 * frame10)))

        else:  # Without lapses
            bias, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10 = fit_params
            # Function to fit:
            p_right = 1 / (1 + np.exp(-(bias + k1 * frame1 + k2 * frame2 + k3 * frame3 + k4 * frame4 + k5 * frame5 +
                                        k6 * frame6 + k7 * frame7 + k8 * frame8 + k9 * frame9 + k10 * frame10)))

        # # Calculate negative log likelihood:
        # neg_ll = - np.sum(stats.norm.logpdf(y, loc=p_right))  # I think this works if y is normally distributed

        # Calculate negative log likelihood# Calculate negative log likelihood (BAMB 2022)
        p_left = 1 - p_right
        ll = np.where(y == 1, np.log(p_right), np.log(p_left))
        neg_ll = - ll.sum()

        return neg_ll

    best_res = 1000000000
    best_fit = None

    for i in range(iterations):

        initial_guess_other_params = (np.random.random(11) * 2 - 1)  # So that is between -1 and 1

        if lapses:  # With lapses
            initial_guess_lapses = np.random.random(2) * 0.3
            # initial_guess_lapses = np.array([0, 0])
            initial_guess = np.hstack([initial_guess_lapses, initial_guess_other_params])
            bnds = (
                (0, 0.5), (0, 0.5), (-2, 2), (-2, 2), (-2, 2), (-2, 2), (-2, 2), (-2, 2), (-2, 2), (-2, 2), (-2, 2),
                (-2, 2),
                (-2, 2))
        else:  # Without lapses
            initial_guess = initial_guess_other_params
            bnds = (
                (-2, 2), (-2, 2), (-2, 2), (-2, 2), (-2, 2), (-2, 2), (-2, 2), (-2, 2), (-2, 2), (-2, 2), (-2, 2))

        # Run the minimizer:
        # res = minimize(sigmoid_mme, initial_guess)
        res = minimize(sigmoid_mme, initial_guess, bounds=bnds, method='Powell')

        print(f'Iteration {i}')
        print('Current result:', res.fun)

        if res.fun < best_res:
            best_fit = res
            best_res = res.fun  # What we are minimizing

        print('Best result:', best_res)
        print('')
        # plt.plot(res.x, label=i)

    print('')
    print('Solution: ', res.x)
    print('Success: ', res.success)
    print('Message: ', res.message)
    print('')
    # plt.legend()

    return best_fit


test = fit_model(x, y, iterations=10)

plt.figure(constrained_layout=True)
plt.plot(test.x, marker='o')
xtixks = np.arange(13)
plt.xticks(xtixks, ['L1', 'L2', 'B', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
plt.title(f'Mouse {df.Setup.unique()[0]}, {n_trials} trials')
plt.xlabel('Stimulus frame')
plt.ylabel('Weight')
# plt.legend(frameon=False)
sns.despine(offset=10, trim=True)  # Despine axes triming the 0
# plt.axhline(0, color='tab:gray', linestyle='--')  # Chance level

time_end = time.time()
runtime = time_end - time_start
print('The script took', round(runtime, 2), 'seconds to run')

########################################################################################################################

# Permutation test (shuffled_var)
shuffles = []
color_upper_shuffle = 'tab:red'

for _ in range(10):
    # choices_shuffled = choices.sample(frac=1).reset_index(drop=True)
    # y_shuffled = np.array(choices_shuffled)
    stim_strength_shuffled = stim_strength.sample(frac=1).reset_index(drop=True)
    x_shuffled = np.array(stim_strength_shuffled)
    # model = sm.Logit(choices, stim_strength)  # Discrete Logit model
    # choices = list(choices)  # Otherwise 'ValueError: The indices for endog and exog are not aligned'
    # model_shuffled = sm.GLM(choices, stim_strength_shuffled,
    #                         family=sm.families.Binomial())  # GLM with Binomial family and Logit link
    # results_shuffled = model_shuffled.fit()
    results_shuffled = fit_model(x_shuffled, choices, lapses=True, iterations=1)
    # params_shuffled = results_shuffled.params
    params_shuffled = results_shuffled.x
    shuffles.append(params_shuffled)
    plt.plot(np.arange(len(params_shuffled)), params_shuffled, color='tab:gray', marker=None,
             mfc='none', mec='none', mew=0, ms=0, label=_, alpha=0.2, zorder=1.7)  # Plot all shuffles

shuffles_mean = np.mean(shuffles, axis=0)  # Get the mean of all the shuffles
percentiles = np.percentile(shuffles, 95, axis=0)  # Get upper 5 percentile of the shuffled_var
# percentiles = np.percentile(shuffles, 68, axis=0)  # Get upper 32 percentile of the shuffled_var
plt.plot(np.arange(len(test.x)), shuffles_mean, color='tab:gray', ls='--', zorder=1.8)
plt.plot(np.arange(len(test.x)), percentiles, color=color_upper_shuffle, ls=':', zorder=1.9)
# plt.xticks(np.arange(1, n_frames + 1, 1))  # Put one xtick for observation for triming later
# sns.despine(offset=10, trim=True)  # Despine axes triming the 0
# plt.xticks(np.arange(2, n_frames + 1, 2))  # Readjust xticks

