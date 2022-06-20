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
- x is a matrix with my stimulus strengths (1 row per stimulus, one column for each frame, so 1*10)
- y is a vector with the subjects' choices
"""

# To do:
# Fit a line to the kernel                      To do
# z-score ilds                                  Done
# ILDs = [-2, 0, 2] vs ILDs = [0]               Done
# 1st half vs 2nd half                          Done
# Random 50% vs 50%                             Done
# Left vs right kernels                         To do


# Comments from Jaime:
# - Are you using any type of regularisation when computing the kernels?
# - Another nice control would be to generate synthetic data with an agent that e.g. only uses 1 frame (1st or n-th). Generate responses using that frame plus noise and compute kernels at different coherences.
# - Can you also try to compute kernels using the AUC method that Genis describes in his paper (Prat-Ortega et al 2020)'

# 1. For each animal and each stim evidence level, compute the mean and std. dev. of the stimuli used. Check that the means and std dev obtained numerically coincide with the nominal values.
# 2. Compute, as explained in Kiani’s paper, the mean of all the stimuli conditioned on the choice (ie mean of all stimuli of evidence X yielding a Right choice and the mean of those yielding a Left choice).
# 3. As we talked Today, compute for each stim evidence and each animal, histograms of the number of times each of the stimulus were used.

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
sns.set_style("white")
sns.set_context("poster")


def plot_kernel(experiment='2AFC_2', animal=None, library='sm', target_ilds=[-2, 0, 2], zscore=True, control=None,
                n_mean_frames=None, save=False, format='svg', transparent=False):
    """
    Compute a psychophysical kernel and plot it. The target ILDs can be added, the stimuli can be zscored and several
    options for control are available
    :param n_mean_frames: Number of mean frames (end/final frames)
    :param experiment: Batch of animals, needed to specify where the root folder with the data is
    :param animal: Mouse ID number
    :param library: library used to compute the kernel
    :param target_ilds: ILDs to use (ideally just 0)
    :param zscore: if True zscore the ILDs per frame, resulting in heavier weights nad allowing comparisons
    :param control: What control analysis to run
    :param save: If True, saves the plot
    :param format: output format of the saved figure
    :param transparent: set background transparent
    :return: GLM model parameters
    """

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

    if zscore:
        frames_ild = pd.DataFrame(stats.zscore(frames_ild, axis=None))  # Z-score the ILDs (along axis 0 or None
    # returns same result, but not axis 1)
    frames_ild.insert(0, column='filename', value=sounds.filename)  # Insert filenames in first column

    # # Frames mean (elementwise) - not needed not used
    # sounds_concat = pd.concat((pd.DataFrame(frames_left.values), pd.DataFrame(frames_right.values)))  # DataFrame concatenating left and right frames
    # sounds_concat_indices = sounds_concat.groupby(sounds_concat.index)
    # frames_mean = sounds_concat_indices.mean()
    # frames_mean.insert(0, column='filename', value=sounds.filename)  # Insert filenames in first column

    # Load behavioral data
    df = pd.read_csv(folder_in)  # Load behavioral data
    df = df[df.Choice.notna()]  # Drop misses (nan in choices), otherwise the code crashes
    ilds = np.sort(df.ILD.unique())
    df = df[df.ILD.isin(target_ilds)]  # Select only trials with the desired ILDs
    filenames = df.Filename.tolist()

    # Default plotting parameters
    color = 'k'
    label = ''
    filename = f'_PK_ILDs: {target_ilds}'

    # Control
    if control is not None:
        n_iterations = 2  # If running control, 2 plots (half vs half, 50% vs 50% random, left vs right, etc)
    else:
        n_iterations = 1  # If not running control, 1 plot

    plt.figure()

    for j in range(n_iterations):

        # Get complete dataset compute every iteration, otherwise the 2nd time will be doing the half of the half!
        choices = df.Choice.reset_index(drop=True)  # Indices must match for modeling
        stim_strength = frames_ild.loc[
            [np.where(sounds.filename == np.array(filenames[i]))[0][0] for i in range(len(filenames))]].drop(
            columns=['filename'])
        stim_strength.reset_index(drop=True, inplace=True)  # Indices must match for modeling

        # Average frames (to have more trials per regressor)
        if n_mean_frames is not None:
            # n_mean_frames = 2  # Number of mean frames (end/final frames)
            n_frames_per_mean_frame = int(
                n_frames / n_mean_frames)  # Number of frames to compute the mean of (must be an integer
            # for slicing)
            assert n_frames % n_mean_frames == 0  # Need to be exact division
            stim_strength_mean = []
            for i in range(n_mean_frames):
                stim_strength_mean.append(stim_strength.iloc[:,
                                          i * n_frames_per_mean_frame:n_frames_per_mean_frame + i * n_frames_per_mean_frame].mean(
                    axis=1))  # Get the mean per trial of every 'n_frames_per_mean_frame' frames
            stim_strength = pd.DataFrame(data=stim_strength_mean)
            stim_strength = stim_strength.T
            filename = f'_PK_ILDs: {target_ilds}, {n_mean_frames} averaged frames'

        # Random 50% vs 50% of trials without replacement
        trials_indexes = choices.index.values
        half_trials_indexes = int(np.rint(len(choices) / 2))  # Size must be in int
        random_half1_indexes = np.sort(np.random.choice(trials_indexes, half_trials_indexes, replace=False))
        random_half1_indexes_isin_trials_indexes = np.isin(trials_indexes, random_half1_indexes)
        random_half2_indexes = np.where(random_half1_indexes_isin_trials_indexes == False)[0]

        # What did the animal chose when the evidence was to choose left/right?
        stim_strength_mean = stim_strength.mean(axis=1)  # Get mean stimulus strength
        choices_mean = [0 if x < 0 else 1 for x in stim_strength_mean]  # Get the choices according to mean stimulus
        # strength (perfect agent)
        choices_mean = np.array(choices_mean)  # To np array to use np.where

        if control is not None:  # else plot regular kernel
            if j == 0:  # 1st half / 1st half random / left
                if control == 'half1_vs_half2':
                    stim_strength = stim_strength.loc[:np.rint(len(stim_strength) / 2), :]  # 1st half
                    choices = choices.loc[:np.rint(len(choices) / 2)]  # 1st half
                    label = '1st half'
                    filename = f'_PK_ILDs: {target_ilds}_1st_vs_2nd_half'
                elif control == 'half1_vs_half2_random':
                    stim_strength = stim_strength.loc[random_half1_indexes, :]  # 1st half (random)
                    choices = choices.loc[random_half1_indexes]  # 1st half (random)
                    label = '1st random half'
                    filename = f'_PK_ILDs: {target_ilds}_1st_vs_2nd_half_random'
                elif control == 'left_vs_right':
                    choices_mean_indexes = np.where(choices_mean == 0)[0]
                    # Get indexes of trials where evidence was to choose left
                    stim_strength = stim_strength.iloc[choices_mean_indexes, :]
                    # Get stimuli of trials where evidence was to choose left
                    choices = choices[choices_mean_indexes]  # Get choices of trials where evidence was to choose left
                    label = 'left'
                    filename = f'_PK_ILDs: {target_ilds}_left_vs_right'
                color = 'tab:blue'

            else:  # # 2nd half / 2nd half random / right
                if control == 'half1_vs_half2':
                    stim_strength = stim_strength.loc[np.rint(len(stim_strength) / 2):, :]  # 2nd half
                    choices = choices.loc[np.rint(len(choices) / 2):]  # 2nd half
                    label = '2nd half'
                    filename = f'_PK_ILDs: {target_ilds}_1st_vs_2nd_half'
                elif control == 'half1_vs_half2_random':
                    stim_strength = stim_strength.loc[random_half2_indexes, :]  # 2nd half (random)
                    choices = choices.loc[random_half2_indexes]  # 2nd half (random)
                    label = '2nd random half'
                    filename = f'_PK_ILDs: {target_ilds}_1st_vs_2nd_half_random'
                elif control == 'left_vs_right':
                    choices_mean_indexes = np.where(choices_mean == 1)[0]
                    # Get indexes of trials where evidence was to choose right
                    stim_strength = stim_strength.iloc[choices_mean_indexes, :]
                    # Get stimuli of trials where evidence was to choose right
                    choices = choices[choices_mean_indexes]  # Get choices of trials where evidence was to choose right
                    label = 'right'
                    filename = f'_PK_ILDs: {target_ilds}_left_vs_right'
                color = 'tab:orange'

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

        plt.plot(np.arange(1, len(params)), params.iloc[1:11], color=color, marker=None, mfc='none', mec='none', mew=0,
                 ms=0, label=label)
        # Without constant (bias)
        plt.errorbar(np.arange(1, len(params)), params.iloc[1:11], yerr=beta_std_err.iloc[1:11], color=color,
                     marker=None, mfc='none', mec='none', mew=0, ms=0)  # Without constant (bias)

        plt.axhline(0, color='tab:gray', ls='--')
        # plt.title(f'Mouse {df.Setup.unique()[0]}, {len(df)} trials, ILDs: {target_ilds}')  # With ILDs
        plt.title(f'Mouse {df.Setup.unique()[0]}, {len(df)} trials')
        plt.xlabel('Stimulus frame')
        plt.ylabel('GLM weight (z-scored)')
        # plt.legend()
        yticks = plt.gca().get_yticks()  # Get current axis yticks for the significance annotations

        # Annotate significance
        if n_mean_frames is not None:  # If averaged frames, loop over the number of averaged frames instead
            n_frames = n_mean_frames

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
        plt.savefig(folder_out + str(df.Setup.unique()[0]) + filename, foramt=format, transparent=transparent)
        plt.close()

    time_end = time.time()
    runtime = time_end - time_start
    print('The script took', round(runtime, 2), 'seconds to run')

    return params


def do_kernels(experiment='2AFC_2', animals=['325', '327', '329', '330', '332', '333', '335', '337'], library='sm',
               target_ilds=[-2, 0, 2], zscore=True, control=None, n_mean_frames=None, save=False):
    """Do the kernels for all animals of a given batch (experiment)"""

    time_start = time.time()

    if experiment is None:

        folder = '/home/alexis/PycharmProjects/glue_sessions/'  # Where the data for all animals is
        experiments = os.listdir(folder)  # List experiments
        experiments.sort()  # Sort them by name
        experiments = [x for x in experiments if os.path.isdir(folder + x)]  # Get rid of non folders

        try:
            experiments.remove('__pycache__')  # Pycharm's archive
        except ValueError:
            pass

        print('Experiments: ' + str(experiments)[1:-1])  # Remove square brackets
        experiment = input('Enter experiment name')

    folder = '/home/alexis/PycharmProjects/glue_sessions/' + experiment + '/'  # Where the data for all animals is

    for i in range(len(animals)):
        path = folder + animals[i]
        print(path)
        plot_kernel(experiment=experiment, animal=animals[i], library=library, target_ilds=target_ilds, zscore=zscore,
                    control=control, n_mean_frames=n_mean_frames, save=save)

    time_end = time.time()
    runtime = time_end - time_start
    print('The script took', round(runtime, 2), 'seconds to run')


def plot_kernels_across_animals(experiment='2AFC_2', animals=['325', '327', '329', '330', '332', '333', '335', '337'],
                                save=False, format='svg', transparent=False):

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

    n_frames = 10
    params = []
    n_trials = []

    for i in range(len(animals)):
        print(folder_in + animals[i] + '.csv')
        n_trials.append(len(pd.read_csv(folder_in + animals[i] + '.csv')))
        params.append(plot_kernel(experiment=experiment, animal=animals[i], library='sm', save=False))

    params = np.array(params)
    params_mean = np.mean(params[:, 1:11], 0)
    params_sem = stats.sem(params[:, 1:11], 0)

    plt.figure()

    plt.plot(np.arange(params.shape[1] - 1), params_mean, color='k', marker=None, mfc='none', mec='none', mew=0, ms=0)
    plt.errorbar(np.arange(params.shape[1] - 1), params_mean, yerr=params_sem, color='k', marker=None, mfc='none',
                 mec='none', mew=0, ms=0)

    plt.axhline(0, color='tab:gray', ls='--')
    # plt.title(f'Mouse {df.Setup.unique()[0]}, {len(df)} trials, ILDs: {target_ilds}')  # With ILDs
    plt.title(f'N={len(params)}, {sum(n_trials)} trials')
    plt.xlabel('Stimulus frame')
    plt.ylabel('GLM weight (z-scored)')
    # plt.legend()
    yticks = plt.gca().get_yticks()  # Get current axis yticks for the significance annotations

    if save:
        folder_out = '/home/alexis/Documentos/kernels/'
        filename = 'mean_PK_across_animals'
        if not os.path.exists(folder_out):
            os.mkdir(folder_out)
        os.chdir(folder_out)
        plt.savefig(folder_out + filename, format=format, transparent=transparent)
        plt.close()

    plt.close('all')

    time_end = time.time()
    runtime = time_end - time_start
    print('The script took', round(runtime, 2), 'seconds to run')

    return params
