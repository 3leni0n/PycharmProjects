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

# To-do/done:
# Fix z-score
# Fit a line to the kernel                      To do
# z-score ilds                                  Done
# ILDs = [-2, 0, 2] vs ILDs = [0]               Done
# 1st half vs 2nd half                          Done
# Random 50% vs 50%                             Done
# Left vs right kernels                         Done
# Rep vs Alt                                    To do
# Permutation CI                                Done
# Bootstrap errorbars                           Done (akin to bse)
# Add parameter for residuals                   Done


# Comments from Jaime:
# - Are you using any type of regularisation when computing the kernels?  No
# - Another nice control would be to generate synthetic data with an agent that e.g. only uses 1 frame (1st or n-th).
# - Generate responses using that frame plus noise and compute kernels at different coherences.
# - Can you also try to compute kernels using the AUC method that Genis describes in his paper (Prat-Ortega et. al 2020)'

# 1. For each animal and each stim evidence level, compute the mean and std. dev. of the stimuli used. Check that the
# means and std dev obtained numerically coincide with the nominal values.
# 2. Compute, as explained in Kiani’s paper, the mean of all the stimuli conditioned on the choice (ie mean of all
# stimuli of evidence X yielding a Right choice and the mean of those yielding a Left choice).
# 3. Compute for each stim evidence and each animal, histograms of the number of times each of the
# stimulus were used.


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

# Mel's code snippet for poster
sns.set_theme()
sns.set_style('white')
sns.set_style('ticks')
sns.set_context('poster')


# sns.despine()


def plot_kernel(experiment='2AFC_2', animal=None, library='sm', target_ilds=[-2, 0, 2], drug=False,
                residuals=False, zscore=True, control=None, n_mean_frames=None, iterations=1000, save=False,
                format='svg', transparent=False):
    """
    Compute a psychophysical kernel and plot it. The target ILDs can be added, the stimuli can be zscored and several
    options for control are available
    :param experiment: Batch of animals, needed to specify where the root folder with the data is
    :param animal: Mouse ID number
    :param library: Library used to compute the kernel
    :param target_ilds: ILDs to use (ideally just 0)
    :param drug: Use or drug trials/sessions or not
    :param residuals: If True substract residuals and set zscore to False
    :param zscore: If True zscore the ILDs per frame, resulting in heavier weights nad allowing comparisons
    :param control: What control analysis to run
    :param n_mean_frames: Number of mean frames (end/final frames)
    :param iterations: Number of iterations to compute the CI by permutation method
    :param save: If True, saves the plot
    :param format: Output format of the saved figure
    :param transparent: Set background transparent
    :return: GLM model parameters
    """

    time_start = time.time()

    if experiment is None:

        # folder_in = '/home/alexis/PycharmProjects/glue_sessions/'  # Where the data for all animals is
        folder_in = Path.home() / 'PycharmProjects' / 'glue_sessions'  # Where the data for all animals is
        experiments = os.listdir(folder_in)  # List experiments
        experiments.sort()  # Sort them by name
        # experiments = [x for x in experiments if os.path.isdir(folder_in + x)]  # Get rid of non folders
        experiments = [x for x in experiments if Path(folder_in / x).is_dir()]  # Get rid of non folders

        try:
            experiments.remove('__pycache__')  # Pycharm's archive
        except ValueError:
            pass

        print('Experiments: ' + str(experiments)[1:-1])  # Remove square brackets
        experiment = input('Enter experiment name')

    # folder_in = '/home/alexis/PycharmProjects/glue_sessions/' + experiment + '/'  # Where the data for all animals is
    folder_in = Path.home() / 'PycharmProjects' / 'glue_sessions' / experiment

    if animal is None:
        animals = os.listdir(folder_in)  # List animals
        animals.sort()  # Sort them by name
        animals = [x[:-4] for x in animals]  # Get rid of .csv extension

        print('Animals: ' + str(animals))  # Remove square brackets
        animal = input('Enter animal')  # Ask user to input animal to glue sessions from

    # folder_in = folder_in + animal + '.csv'
    folder_in = Path(folder_in / animal).with_suffix('.csv')

    # Load sounds
    # sounds_path = '/home/alexis/PycharmProjects/create_sounds/sounds.csv'
    # sounds_path = '/home/alexis/PycharmProjects/create_sounds/sounds_2.csv'
    sounds_path = Path.home() / 'PycharmProjects' / 'create_sounds' / 'sounds_2.csv'
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

    ####################################################################################################################
    # # After cafesito with Leonsito on 30.03.2023:
    # frames_ild = []
    # for i in range(len(sounds)):
    #     if sounds.ILD[i] < 0:  # Left trials
    #         frames_ild.append(sounds[left_frames_column_names].values[i] - sounds[right_frames_column_names].values[i])
    #     elif sounds.ILD[i] == 0:  # Impossible trials
    #         frames_ild.append(sounds[right_frames_column_names].values[i] - sounds[left_frames_column_names].values[i])
    #     elif sounds.ILD[i] > 0:  # Right trials
    #         frames_ild.append(sounds[right_frames_column_names].values[i] - sounds[left_frames_column_names].values[i])
    # frames_ild = pd.DataFrame(frames_ild)
    ####################################################################################################################

    # Residuals (https://www-nature-com.sire.ub.edu/articles/nature08275)
    if residuals:
        sounds_ild = sounds.ILD
        frames_ild = frames_ild.sub(sounds_ild, axis='rows')
        ylabel = 'GLM weight (residuals)'
    else:
        ylabel = 'GLM weight'

    frames_ild.insert(0, column='filename', value=sounds.filename)  # Insert filenames in first column

    # Frames mean (elementwise) - not needed not used
    # sounds_concat = pd.concat((pd.DataFrame(frames_left.values), pd.DataFrame(frames_right.values)))  # DataFrame concatenating left and right frames
    # sounds_concat_indices = sounds_concat.groupby(sounds_concat.index)
    # frames_mean = sounds_concat_indices.mean()
    # frames_mean.insert(0, column='filename', value=sounds.filename)  # Insert filenames in first column

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

    # Filter out some trials
    df = df[df.Choice.notna()]  # Drop misses (nan in choices), otherwise the code crashes
    ilds = np.sort(df.ILD.unique())
    df = df[df.ILD.isin(target_ilds)]  # Select only trials with the desired ILDs
    # df = df[df.Hit == 1]  # Only correct trials
    # accuracy_threshold = 0.6
    # df = df[(df.AccuracyLeft >= accuracy_threshold) & (df.AccuracyRight >= accuracy_threshold)]  # Select only trials
    # with accuracy >= threshold

    # Drug sessions/trials
    if drug:  # Select drug session trials
        df = df[df.Drug.notnull()]
    else:  # Don't select drug session trials
        try:
            df = df[df.Drug.isnull()]  # Remove drug experimental sessions
        except AttributeError:
            pass

    n_trials = len(df)

    ####################################################################################################################
    'UNDER CONSTRUCTION >>> DRUG DATA'
    ####################################################################################################################

    # # Index drug sessions
    # df_intersession = pd.read_csv('/home/alexis/PycharmProjects/intersession/' + '/' + experiment + '/' + animal +
    #                               '_intersession.csv')
    # # df_intersession = df_intersession[(df_intersession.AccuracyLeft >= 0.75) & (df_intersession.AccuracyRight >= 0.75)]
    # # Select only sessions with accuracy above threshold
    # drug_session_dates = df_intersession[df_intersession.Drug == 'MK801'].Dates
    # # df = df[df.Drug.isnull()]  # Remove drug experimental sessions
    #
    # df.drop(index=df[(df.Date == '2022-05-25') & (df.Setup == 337)].index, inplace=True)
    # df.drop(index=df[(df.Date == '2022-05-24') & (df.Setup == 337)].index, inplace=True)
    # df.drop(index=df[(df.Date == '2022-05-26') & (df.Setup == 332)].index, inplace=True)
    # df.drop(index=df[(df.Date == '2022-05-27') & (df.Setup == 333)].index, inplace=True)
    # df.drop(index=df[(df.Date == '2022-05-31') & (df.Setup == 333)].index, inplace=True)
    #
    # # df = df[df.Drug == 'saline']
    # df = df[df.Drug == 'MK801']
    # # df = df[df.Drug == 'rest']
    # n_trials = len(df)

    ####################################################################################################################

    filenames = df.Filename.tolist()

    # Default plotting parameters
    color = 'k'
    color_upper_shuffle = 'tab:red'
    label = ''
    filename = f'_PK_ILDs: {target_ilds}'

    # Control
    if control is not None:
        n_plots = 2  # If running control, 2 plots (half vs half, 50% vs 50% random, left vs right, etc)
    else:
        n_plots = 1  # If not running control, 1 plot

    plt.figure(constrained_layout=True)

    for j in range(n_plots):

        # Get complete dataset compute every iteration, otherwise the 2nd time will be doing the half of the half!
        choices = df.Choice.reset_index(drop=True)  # Indices must match for modeling
        stim_strength = frames_ild.loc[
            [np.where(sounds.filename == np.array(filenames[i]))[0][0] for i in range(len(filenames))]].drop(
            columns=['filename'])

        # Zscore
        if not residuals:  # To not do both (otherwise I'd be subtracting the mean twice)
            if zscore:
                stim_strength = pd.DataFrame(stats.zscore(stim_strength, axis=0))  # Z-score the ILDs (along axis 0 or None
                # returns same result, but not axis 1). 0 along trials that's what I wanna do :)
                ylabel = 'GLM weight (z-scored)'
            else:
                ylabel = 'GLM weight'

        stim_strength.reset_index(drop=True, inplace=True)  # Indices must match for modeling

        # Average frames (to have more trials per regressor)
        if n_mean_frames is not None:
            n_frames_per_mean_frame = int(n_frames / n_mean_frames)  # Number of frames to compute the mean of (must be
            # an integer for slicing)
            assert n_frames % n_mean_frames == 0  # Need to be exact division
            stim_strength_mean = []
            for i in range(n_mean_frames):
                stim_strength_mean.append(stim_strength.iloc[:,
                                          i * n_frames_per_mean_frame:n_frames_per_mean_frame + i * n_frames_per_mean_frame].mean(
                    axis=1))  # Get the mean per trial of every 'n_frames_per_mean_frame' frames
            stim_strength = pd.DataFrame(data=stim_strength_mean).T
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
                color_upper_shuffle = 'tab:blue'

            else:  # 2nd half / 2nd half random / right
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
                color_upper_shuffle = 'tab:orange'

        # Add nominal ILDs to design matrix as regressor
        if residuals:
            trials_ild = df.ILD.reset_index(drop=True)  # Nominal ILDs per trial
            stim_strength.insert(0, 'ILD', trials_ild)  # Add nominal ILSs to stim_strength

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

        # Plot kernel (stimulus frames beta weights)
        # + i to skip constant; + int(residuals) to skip ILD
        x = np.arange(1 + int(residuals), len(params))
        y = params.iloc[1 + int(residuals):len(params)]
        yerr = beta_std_err.iloc[1 + int(residuals):len(params)]
        plt.plot(x, y, color=color, marker='o', label=label)
        plt.errorbar(x, y, yerr=yerr, color=color, marker='o', fmt='none', mec='none', ms=0)
        plt.title(f'Mouse {df.Setup.unique()[0]}, {n_trials} trials')
        plt.xlabel('Stimulus frame')
        plt.ylabel(ylabel)
        plt.legend(frameon=False)
        yticks = plt.gca().get_yticks()  # Get current axis yticks for the significance annotations

        # Annotate significance
        if n_mean_frames is not None:  # If averaged frames, loop over the number of averaged frames instead
            n_frames = n_mean_frames

        for i in range(n_frames):
            if p_values[i] <= 0.05:
                text = '*'
            else:
                # text = 'ns'
                text = ''
            plt.annotate(text, xy=(i + 1 + int(residuals), yticks[1]), xytext=(i + 1 + int(residuals), yticks[1]),
                         color=color, va='center', ha='center', fontsize='medium')

        # Permutation test (shuffled_var)
        shuffles = []
        # Shuffling the choices or the stim_strength index is the same, so it doesn't matter. Shuffling along the
        # columns of stim_strength is wrong because it breaks the temporal structure of the data. Shuffling the frames
        # within trial could be an interesting test, as it preserves the overall weight of the stimulus for each trial
        # but breaks the frame structure
        for _ in range(iterations):
            choices_shuffled = choices.sample(frac=1).reset_index(drop=True)
            stim_strength_shuffled = stim_strength.sample(frac=1).reset_index(drop=True)
            # model = sm.Logit(choices, stim_strength)  # Discrete Logit model
            # model_shuffled = sm.GLM(choices_shuffled, stim_strength,  # Shuffled choices
            #                         family=sm.families.Binomial())  # GLM with Binomial family and Logit link
            model_shuffled = sm.GLM(choices, stim_strength_shuffled,  # Shuffled stim_strength
                                    family=sm.families.Binomial())  # GLM with Binomial family and Logit link
            results_shuffled = model_shuffled.fit()
            params_shuffled = results_shuffled.params
            shuffles.append(params_shuffled)
            # plt.plot(np.arange(1, len(params_shuffled)), params_shuffled.iloc[1:11], color='tab:gray', marker=None,
            #          mfc='none', mec='none', mew=0, ms=0, label=label, alpha=0.1, zorder=1.7)  # Plot all shuffles

        shuffles_mean = np.mean(shuffles, axis=0)  # Get the mean of all the shuffles
        percentiles95 = np.percentile(shuffles, 95, axis=0)  # Get upper 5 percentile of the shuffled_var
        plt.plot(x, shuffles_mean[1 + int(residuals):len(shuffles_mean)], color='tab:gray', ls='--', zorder=1.8)
        plt.plot(x, percentiles95[1 + int(residuals):len(shuffles_mean)], color=color_upper_shuffle, ls=':', zorder=1.9)

        # Adjust xticks to number of regressors (cont/ILD + frames)
        if residuals:
            xticks = np.arange(2, n_frames + 2, 1)
            plt.xticks(xticks)  # Put one xtick for observation for triming later
            xticks = np.arange(2 + int(residuals), n_frames + 1 + int(residuals), 2)
            xticklabels = xticks - 1
            sns.despine(offset=10, trim=True)  # Despine axes triming the 0
            plt.xticks(xticks, xticklabels)  # Readjust xticks
        else:
            xticks = np.arange(1, n_frames + 1, 1)
            plt.xticks(xticks)  # Put one xtick for observation for triming later
            sns.despine(offset=10, trim=True)  # Despine axes triming the 0
            plt.xticks(np.arange(2, n_frames + 1, 2))  # Readjust xticks


    if n_mean_frames == 2:
        plt.xticks([1, 2])  # Readjust xticks

    if save:
        # folder_out = '/home/alexis/Documentos/kernels/' + experiment + '/'
        folder_out = Path.home() / 'Documentos' / 'kernels' / experiment
        # if not os.path.exists(folder_out):
        #     os.mkdir(folder_out)
        if not folder_out.exists():
            folder_out.mkdir(parents=True, exist_ok=True)
        os.chdir(folder_out)
        # plt.savefig(folder_out + str(df.Setup.unique()[0]) + filename + '.' + format, format=format, transparent=transparent)
        plt.savefig(Path(folder_out, str(df.Setup.unique()[0]) + filename + '.' + format), format=format,
                    transparent=transparent)
        plt.close()

    # plt.close()

    time_end = time.time()
    runtime = time_end - time_start
    print('The script took', round(runtime, 2), 'seconds to run')

    return params, shuffles, shuffles_mean, percentiles95, n_trials


def do_kernels(experiment='2AFC_2', animals=['325', '327', '329', '330', '332', '333', '335', '337'], library='sm',
               target_ilds=[-8, -4, -2, 0, 2, 4, 8], drug=False, residuals=False, zscore=True, control=None,
               n_mean_frames=None, iterations=1000, save=False, format='svg', transparent=False):
    """
    Do the kernels for all animals of a given batch (experiment)
    :param experiment: Batch of animals, needed to specify where the root folder with the data is
    :param animals: Mouse ID number
    :param library: Library used to compute the kernel
    :param target_ilds: ILDs to use (ideally just 0)
    :param drug: Use or drug trials/sessions or not
    :param residuals: If True substract residuals and set zscore to False
    :param zscore: If True zscore the ILDs per frame, resulting in heavier weights nad allowing comparisons
    :param control: What control analysis to run
    :param n_mean_frames: Number of mean frames (end/final frames)
    :param iterations: Number of iterations to compute the CI by permutation method
    :param save: If True, saves the plot
    :param format: Output format of the saved figure
    :param transparent: Set background transparent
    :return: Nothing
    """

    time_start = time.time()

    if experiment is None:

        # folder = '/home/alexis/PycharmProjects/glue_sessions/'  # Where the data for all animals is
        folder = Path.home() / 'PycharmProjects' / 'glue_sessions'  # Where the data for all animals is
        experiments = os.listdir(folder)  # List experiments
        experiments.sort()  # Sort them by name
        # experiments = [x for x in experiments if os.path.isdir(folder + x)]  # Get rid of non folders
        experiments = [x for x in experiments if Path(folder / x).is_dir()]  # Get rid of non folders

        try:
            experiments.remove('__pycache__')  # Pycharm's archive
        except ValueError:
            pass

        print('Experiments: ' + str(experiments)[1:-1])  # Remove square brackets
        experiment = input('Enter experiment name')

    # folder = '/home/alexis/PycharmProjects/glue_sessions/' + experiment + '/'  # Where the data for all animals is
    folder = Path.home() / 'PycharmProjects' / 'glue_sessions' / experiment

    for i in range(len(animals)):
        # path = folder + animals[i]
        path = Path(folder, animals[i])
        print(path)
        plot_kernel(experiment=experiment, animal=animals[i], library=library, target_ilds=target_ilds, drug=drug,
                    residuals=residuals, zscore=zscore, control=control, n_mean_frames=n_mean_frames,
                    iterations=iterations, save=save, format=format, transparent=transparent)

    time_end = time.time()
    runtime = time_end - time_start
    print('The script took', round(runtime, 2), 'seconds to run')


def plot_kernels_across_animals(experiment='2AFC_2', animals=['325', '327', '329', '330', '332', '333', '335', '337'],
                                library='sm', target_ilds=[-8, -4, -2, 0, 2, 4, 8], drug=False, residuals=False,
                                zscore=True, control=None, n_mean_frames=None, iterations=1000, save=False,
                                format='svg', transparent=False):
    """
    Do the kernels for all animals of a given batch (experiment)
    :param experiment: Batch of animals, needed to specify where the root folder with the data is
    :param animals: Mouse ID number
    :param library: Library used to compute the kernel
    :param target_ilds: ILDs to use (ideally just 0)
    :param drug: Use or drug trials/sessions or not
    :param residuals: If True substract residuals and set zscore to False
    :param zscore: If True zscore the ILDs per frame, resulting in heavier weights nad allowing comparisons
    :param control: What control analysis to run
    :param n_mean_frames: Number of mean frames (end/final frames)
    :param iterations: Number of iterations to compute the CI by permutation method
    :param save: If True, saves the plot
    :param format: Output format of the saved figure
    :param transparent: Set background transparent
    :return: GLM model parameters
    """

    time_start = time.time()

    if experiment is None:

        # folder_in = '/home/alexis/PycharmProjects/glue_sessions/'  # Where the data for all animals is
        folder_in = Path.home() / 'PycharmProjects' / 'glue_sessions'  # Where the data for all animals is
        experiments = os.listdir(folder_in)  # List experiments
        experiments.sort()  # Sort them by name
        # experiments = [x for x in experiments if os.path.isdir(folder_in + x)]  # Get rid of non folders
        experiments = [x for x in experiments if Path(folder_in / x).is_dir()]  # Get rid of non folders

        try:
            experiments.remove('__pycache__')  # Pycharm's archive
        except ValueError:
            pass

        print('Experiments: ' + str(experiments)[1:-1])  # Remove square brackets
        experiment = input('Enter experiment name')

    # folder_in = '/home/alexis/PycharmProjects/glue_sessions/' + experiment + '/'  # Where the data for all animals is
    folder_in = Path.home() / 'PycharmProjects' / 'glue_sessions' / experiment

    params_across_animals = []
    shuffles_across_animals = []
    shuffles_means_across_animals = []
    percentiles95_across_animals = []
    n_trials_across_animals = []

    for i in range(len(animals)):
        print(f'Doing kernel of animal {animals[i]} ({i+1}/{len(animals)})')
        params, shuffles, shuffles_mean, percentiles95, n_trials = plot_kernel(experiment=experiment, animal=animals[i],
                                                                               library=library,
                                                                               target_ilds=target_ilds, drug=drug,
                                                                               control=control,
                                                                               n_mean_frames=n_mean_frames,
                                                                               iterations=iterations,
                                                                               residuals=residuals, zscore=zscore,
                                                                               save=save)
        params_across_animals.append(params)
        shuffles_across_animals.append(shuffles)
        shuffles_means_across_animals.append(shuffles_mean)
        percentiles95_across_animals.append(percentiles95)
        n_trials_across_animals.append(n_trials)
        # n_trials.append(len(pd.read_csv(folder_in + animals[i] + '.csv')))

    # plt.close('all')
    n_frames = len(params) - 1

    params_across_animals = np.array(params_across_animals)
    params_mean_across_animals = np.mean(params_across_animals, 0)
    params_sem_across_animals = stats.sem(params_across_animals, 0)

    shuffles_across_animals = np.array(shuffles_across_animals)  # Convert list os lists to 3 dim array (animal x
    # iterations x params)
    shuffles_means_across_animals = np.mean(shuffles_across_animals, 0)
    shuffles_means_mean_across_animals = np.mean(shuffles_means_across_animals, 0)
    percentiles95_across_animals = np.percentile(shuffles_means_across_animals, 95,
                                                 axis=0)  # Get upper 5 percentile of the shuffled_var

    # Wrong old method
    # shuffles_means_across_animals = np.array(shuffles_means_across_animals)
    # shuffles_means_mean_across_animals = np.mean(shuffles_means_across_animals, 0)
    # percentiles95_across_animals = np.array(percentiles95_across_animals)
    # percentiles95_mean_across_animals = np.mean(percentiles95_across_animals, 0)

    plt.figure(constrained_layout=True)

    color = 'k'
    # color = 'tab:pink'
    # color = 'tab:gray'

    # Plot kernel
    plt.plot(np.arange(1, len(params_mean_across_animals)), params_mean_across_animals[1:11], color=color, marker='o')
    plt.errorbar(np.arange(1, len(params_mean_across_animals)), params_mean_across_animals[1:11],
                 yerr=params_sem_across_animals[1:11], color=color, marker='o', fmt='none', mec='none')
    plt.plot(np.arange(1, len(shuffles_means_mean_across_animals)), shuffles_means_mean_across_animals[1:11],
             color='tab:gray', ls='--', zorder=1.8)
    plt.plot(np.arange(1, len(percentiles95_across_animals)), percentiles95_across_animals[1:11], color='tab:red',
             ls=':', zorder=1.9)

    if n_mean_frames is not None:
        n_frames = n_mean_frames

    plt.xticks(np.arange(1, n_frames + 1, 1))  # Put one xtick for observation for triming later
    sns.despine(offset=10, trim=True)  # Despine axes triming the 0
    plt.xticks(np.arange(2, n_frames + 1, 2))  # Readjust xticks

    if n_mean_frames == 2:
        plt.xticks([1, 2])  # Readjust xticks

    # plt.axhline(0, color='tab:gray', ls='--')
    # plt.title(f'Mouse {df.Setup.unique()[0]}, {len(df)} trials, ILDs: {target_ilds}')  # With ILDs
    plt.title(f'N={len(params_across_animals)}, {sum(n_trials_across_animals)} trials')
    plt.xlabel('Stimulus frame')

    if zscore:
        plt.ylabel('GLM weight (z-scored)')
    else:
        plt.ylabel('GLM weight')

    # plt.legend()
    yticks = plt.gca().get_yticks()  # Get current axis yticks for the significance annotations

    if save:
        # folder_out = '/home/alexis/Documentos/kernels/'
        folder_out = Path.home() / 'Documentos' / 'kernels'
        filename = f'mean_PK_across_animals: ILDs: {target_ilds}, {n_mean_frames} averaged frames' + '.' + format
        # if not os.path.exists(folder_out):
        #     os.mkdir(folder_out)
        if not folder_out.exists():
            folder_out.mkdir(parents=True, exist_ok=True)
        os.chdir(folder_out)
        # plt.savefig(folder_out + filename, format=format, transparent=transparent)
        plt.savefig(Path(folder_out, filename, format=format), transparent=transparent)
        # plt.close()

    time_end = time.time()
    runtime = time_end - time_start
    print('The script took', round(runtime, 2), 'seconds to run')

    return params_across_animals, shuffles_means_across_animals, percentiles95_across_animals


# Debug
plot_kernel(experiment='2AFC_2', animal='333', library='sm', target_ilds=[-70, -8, -4, -2, 0, 2, 4, 8, 70], drug=False,
            residuals=True, zscore=False, control=None, n_mean_frames=None, iterations=10, save=False,
            format='svg', transparent=False)

# plot_kernels_across_animals(experiment='2AFC_2', animals=['325', '327', '329', '330', '332', '333', '335', '337'],
#                                 library='sm', target_ilds=[-2, 0, 2], drug=False, residuals=False,
#                                 zscore=False, control=None, n_mean_frames=None, iterations=10, save=False,
#                                 format='svg', transparent=False)

# Good animals batch 2:['325', '327', '329', '330', '332', '333', '335', '337']
# Good animals batch 3: ['419', '420', '422', '616', '617', '619', '623']

experiment = '2AFC_2'
animal = '325'
library = 'sm'
target_ilds = [-2, 0, 2]
drug = False
residuals = True
zscore = False
control = None
n_mean_frames = None
iterations = 1000
save = False
format = 'svg'
transparent = False