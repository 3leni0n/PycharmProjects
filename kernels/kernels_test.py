"""
Notes from Genis:

The kernel stimates the weight subjects gives to each stimulus frame. It's usually computed via logistic regression
(https://en.wikipedia.org/wiki/Logistic_regression). We estimate the probability of a decision 'right' given some filters
(the betas or weights).
- p is the probability of choose right
- B0 isn't multiplied by any x and therefore is the bias. Normally is not included, but if the subject is biased, it's
best to do so. Bi are the weights of each frame, and there's one beta for each x
- x are the frames, there's one x for each B

In the wikipedia example plot, the x-axis would be the stimulus strength and the y-axis would be probability of
choose right. Then we fit the logistic regression curve. When we plot a kernel, what we're actually representing are
values of Bi. The values of beta can be computed in python with the 'logistic regression' from the 'sklearn' library
(https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).
- x will be a matrix with my stimulus strengths (1 row per stimulus, one column for each frame, so 1*10)
- y will be the subjects' choices

To do:

- Fit a line to the kernel
"""

import time
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from matplotlib import pyplot as plt
from scipy import stats


def plot_kernel(experiment='2AFC_2', animal=['325', '327'], library='sm', target_ilds=[-2, 0, 2], zscore=True, save=False):

    time_start = time.time()

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
    if zscore:
        frames_ild = pd.DataFrame(stats.zscore(frames_ild, axis=None))  # Z-score the ILDs (along axis 0 or None
    # returns same result, but not axis 1)
    frames_ild.insert(0, column='filename', value=sounds.filename)  # Insert filenames in first column

    # # Frames mean (elementwise) - not needed not used
    # sounds_concat = pd.concat((pd.DataFrame(frames_left.values), pd.DataFrame(frames_right.values)))  # DataFrame concatenating left and right frames
    # sounds_concat_indices = sounds_concat.groupby(sounds_concat.index)
    # frames_mean = sounds_concat_indices.mean()
    # frames_mean.insert(0, column='filename', value=sounds.filename)  # Insert filenames in first column
    # frames_mean_array = np.mean([frames_left_array, frames_right_array], axis=0)  # Get the column wise mean between right and left frames

    # Load behavioral data
    df = df[df.Choice.notna()]  # Drop misses (nan in choices), otherwise the code crashes
    ilds = np.sort(df.ILD.unique())
    # target_ilds = [-2, 0, 2]
    # target_ilds = [0]
    df = df[df.ILD.isin(target_ilds)]  # Select only trials with the desired ILDs
    filenames = df.Filename.tolist()
    stim_strength = frames_ild.loc[
        [np.where(sounds.filename == np.array(filenames[i]))[0][0] for i in range(len(filenames))]].drop(
        columns=['filename'])
    stim_strength.reset_index(drop=True, inplace=True)  # Indices must match for modeling
    choices = df.Choice.reset_index(drop=True)  # Indices must match for modeling

    # Random 50% vs 50% of trials without replacement
    trials_indexes = choices.index.values
    half_trials_indexes = int(np.rint(len(choices) / 2))  # Size must be in int
    random_half1_indexes = np.sort(np.random.choice(trials_indexes, half_trials_indexes, replace=False))
    random_half1_indexes_isin_trials_indexes = np.isin(trials_indexes, random_half1_indexes)
    random_half2_indexes = np.where(random_half1_indexes_isin_trials_indexes == False)[0]

    # Default plotting parameters
    color = 'k'
    label = ''

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
        # animal = input('Enter animal')
        animal = [x.strip() for x in input('Enter animal(s) split by comma').split(',')]  # Ask user to input animal

    for i in range(len(animal)):

        folder_in = folder_in + animal[i] + '.csv'
        print(len(animal))
        print(folder_in, i)
        df = pd.read_csv(folder_in)  # Read behavioral data
        folder_in = '/home/alexis/PycharmProjects/glue_sessions/' + experiment + '/'  # Clear path for next animal

        # 1st half of trials vs 2nd half of trials | Random 50% vs 50% of trials without replacement
        # for i in range(2):
        # for i in range(1):
            # if i == 0:
                # stim_strength = stim_strength.loc[:np.rint(len(stim_strength) / 2), :]  # 1st half
                # choices = choices.loc[:np.rint(len(choices) / 2)]  # 1st half

                # stim_strength = stim_strength.loc[random_half1_indexes, :]  # 1st half (random)
                # choices = choices.loc[random_half1_indexes]  # 1st half (random)

                # color = 'tab:blue'
                # label = '1st half'
                # label = '1st random half'
            # else:
                # Copy the original again otherwise is the half of the half
                # stim_strength = frames_ild.loc[
                #     [np.where(sounds.filename == np.array(filenames[i]))[0][0] for i in range(len(filenames))]].drop(
                #     columns=['filename'])
                # stim_strength.reset_index(drop=True, inplace=True)  # Indices must match for modeling
                # choices = df.Choice.reset_index(drop=True)  # Indices must match for modeling

                # stim_strength = stim_strength.loc[np.rint(len(stim_strength) / 2):, :]  # 2nd half
                # choices = choices.loc[np.rint(len(choices) / 2):]  # 2nd half

                # stim_strength = stim_strength.loc[random_half2_indexes, :]  # 2nd half (random)
                # choices = choices.loc[random_half2_indexes]  # 2nd half (random)

                # color = 'tab:orange'
                # label = '2nd half'
                # label = '2nd random half'

        # Plots
        plt.figure()

        if library == 'sklearn':  # Scikit-learn library
            clf = LogisticRegression(random_state=0).fit(stim_strength, choices)
            clf.get_params()
            plt.plot(np.arange(len(clf.coef_[0])), clf.coef_[0], marker='o', mfc='None', label='Method 1')
            # plt.title('Psychophysical kernel')
            # plt.title('Psychophysical kernel')
            # plt.xlabel('Number of frames')
            # plt.ylabel('Weight')

        elif library == 'sm':  # Statsmodels library
            # From Genis' paper analysis code (gives directly the error)
            # Paper: https://www-nature-com.sire.ub.edu/articles/s41467-021-21501-z
            # Code: https://bitbucket.org/delaRochaLab/flexible-categorization/src/master/functions/analysis_fc.py
            # GLM with Binomial family and Logit link = discrete Logit model
            stim_strength = sm.add_constant(stim_strength)  # Add constant (bias)
            # model = sm.Logit(choices, stim_strength)  # Discrete Logit model
            model = sm.GLM(choices, stim_strength,
                           family=sm.families.Binomial())  # GLM with Binomial family and Logit link
            results = model.fit()
            params = results.params
            beta_std_err = results.bse
            p_values = results.pvalues
            summary = results.summary()
            print(summary)

        # # Fit a line to the weights check primacy vs recency
        # result = stats.linregress(np.arange(len(params)-1), params.iloc[1:11])
        # intercept = result.intercept
        # pvalue = result.pvalue
        # rvalue = result.rvalue
        # slope = result.slope
        # stderr = result.stderr

        # Plot kernel
        # plt.plot(np.arange(len(params)), params, marker='o', mfc='None', label=label)  # With constant (bias)
        # plt.errorbar(np.arange(len(params)), params, yerr=beta_std_err, color='tab:blue', fmt='o',
        #              markerfacecolor='none')  # With constant (bias)

        plt.plot(np.arange(1, len(params)), params.iloc[1:11], color=color, marker='o', mfc='None', label=label)
        # Without constant (bias)
        plt.errorbar(np.arange(1, len(params)), params.iloc[1:11], yerr=beta_std_err.iloc[1:11], color=color, fmt='o',
                     markerfacecolor='none')  # Without constant (bias)

        plt.axhline(0, color='tab:gray', ls='--')
        plt.title(f'Psychophysical kernel, animal {df.Setup.unique()[0]}, {len(choices)} trials, ILDs: {target_ilds}')
        plt.xlabel('Stimulus frame')
        plt.ylabel('GLM weight (z-scored)')
        plt.legend()
        yticks = plt.gca().get_yticks()  # Get current axis yticks for the significance annotations

        # Annotate significance
        for i in range(n_frames):
            if p_values[i] <= 0.05:  # +i to skip constant
                text = '*'
            else:
                # text = 'ns'
                text = ''
            # plt.annotate(str(round(p_values[0+i], 2)),
            #              xy=(i+1, yticks[1]), xytext=(i+1, yticks[1]), color='k',
            #              va='top', ha='center', fontsize='medium')
            plt.annotate(text, xy=(i + 1, yticks[1]), xytext=(i + 1, yticks[1]), color=color, va='center', ha='center',
                         fontsize='medium')  # i+1 to skip constant

    if save:
        folder_out = '/home/alexis/Documentos/kernels/'
        if not os.path.exists(folder_out):
            os.mkdir(folder_out)
        os.chdir(folder_out)
        plt.savefig(folder_out + str(df.Setup.unique()[0]) + f'_PK_ILDs: {target_ilds}.png')
        # plt.savefig(folder_out + str(df.Setup.unique()[0]) + f'_PK_ILDs: {target_ilds}_1st_vs_2nd_half.png')
        # plt.savefig(folder_out + str(df.Setup.unique()[0]) + f'_PK_ILDs: {target_ilds}_1st_vs_2nd_half_random.png')
        plt.close()

    return params


def plot_kernels_across_animals(experiment='2AFC_2', animals=['325', '327', '329', '330', '332', '333', '335', '337']):

    time_start = time.time()

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

    params = []

    for i in range(len(animals)):
        print(folder_in + animals[i] + '.csv')
        params.append(plot_kernel(experiment=experiment, animal=animals[i], library='sm', save=True))

    params = np.array(params)
    params_mean = np.mean(params[:, 1:11], 0)
    params_sem = stats.sem(params[:, 1:11], 0)

    plt.plot(np.arange(params.shape[1] - 1), params_mean, color='k')
    plt.errorbar(np.arange(params.shape[1] - 1), params_mean, yerr=params_sem, color='k', fmt='o',
                 markerfacecolor='none')

    filename = 'mean_PK_across_animals.png'
    folder_out = '/home/alexis/Documentos/kernels/'

    plt.savefig(folder_out + filename)

    return params

########################################################################################################################

# To do:
# z-score ilds                                  Done
# ILDs = [-2, 0, 2] vs ILDs = [0]               Done
# 1st half vs 2nd half                          Done
# Left vs right kernels
# random 50% vs 50%                             Done
