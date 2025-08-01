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
# Rep vs Alt                                    To do

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
import statsmodels.api as sm
from matplotlib import pyplot as plt
from scipy import stats
import seaborn as sns
from collections import namedtuple
from my_fun.my_fun import get_experiment, get_animal, save_fig, timer, filter_drug_sessions
from kernels.kernels_tools import *

# Plotting parameters
sns.set_theme()
sns.set_style('white')
sns.set_style('ticks')
sns.set_context('poster')


@timer
def get_pk(experiment='2AFC_6', animal=None, target_ilds=None, drug=None, residuals=True, zscore=False,
           control=None, n_mean_frames=None, iterations=1000):
    """
    Compute a psychophysical kernel and plot it. The target ILDs can be added, the stimuli can be zscored and several
    options for control are available
    :param experiment: Batch of animals, needed to specify where the root folder with the data is
    :param animal: Mouse ID number
    :param target_ilds: ILDs to use (ideally just 0)
    :param drug: Use drug trials/sessions or not. If so, specify which drug; if not, specify None
    :param residuals: If True substract residuals and set zscore to False
    :param zscore: If True zscore the ILDs per frame, resulting in heavier weights nad allowing comparisons
    :param control: What control analysis to run
    :param n_mean_frames: Number of mean frames (end/final frames)
    :param iterations: Number of iterations to compute the CI by permutation method
    :return: GLM model parameters
    """

    # Get the path to the data
    experiment, folder_in = get_experiment(experiment)
    animal = get_animal(experiment=experiment, path_session='glue_sessions', animal=animal)
    folder_in = Path(folder_in / animal).with_suffix('.csv')

    # Load behavioral data
    df = pd.read_csv(folder_in)

    ####################################################################################################################

    # Load intersession data
    path_intersession = Path.home() / 'PycharmProjects' / 'intersession' / experiment / (str(int(animal)) + '_intersession.csv')
    # str(int(animal)) to remove the 0 padding in ID
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
    if target_ilds is not None:
        df = df[df.ILD.isin(target_ilds)]  # Select only trials with the desired ILDs
    accuracy_threshold = 0.5
    df = df[(df.AccuracyLeft >= accuracy_threshold) & (df.AccuracyRight >= accuracy_threshold)]  # Select only trials
    # with accuracy >= threshold

    ####################################################################################################################

    # Drug sessions/trials
    if drug is not None:  # Select drug session trials

        if experiment == '2AFC_2':
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

        # Drop sessions in which the animal didn't do the task. This can be achieved by using an accuracy threshold of 0.5
        # df.drop(index=df[(df.Date == '2022-05-24') & (df.Setup == 337)].index, inplace=True)  # Left accuracy 4%
        # df.drop(index=df[(df.Date == '2022-05-25') & (df.Setup == 337)].index, inplace=True)  # Left accuracy 3%
        # df.drop(index=df[(df.Date == '2022-06-01') & (df.Setup == 337)].index, inplace=True)  # Left accuracy 43%
        # df.drop(index=df[(df.Date == '2022-05-26') & (df.Setup == 332)].index, inplace=True)  # Miss rate 50%
        # df.drop(index=df[(df.Date == '2022-05-27') & (df.Setup == 333)].index, inplace=True)  # Miss rate 83%

        elif experiment == '2AFC_6':
            if drug in [0, 1]:
                df = filter_drug_sessions(df)
                df = df[df.Drug == drug]

    else:  # Don't select drug session trials
        try:
            df = df[df.Drug.isnull()]  # Remove drug experimental sessions
        except AttributeError:
            pass

    ####################################################################################################################

    n_trials = len(df)
    # behavior_filenames = df.Filename.tolist()

    # Control
    if control is not None:
        n_plots = 2  # If running control, 2 plots (half vs half, 50% vs 50% random, left vs right, etc)
    else:
        n_plots = 1  # If not running control, 1 plot

    for j in range(n_plots):

        # Get complete dataset compute every iteration, otherwise the 2nd time will be doing the half of the half!
        choices = df.Choice.reset_index(drop=True)  # Indices must match for modeling
        # stim_strength = frames_ild.loc[
        #     [np.where(sounds.filename == np.array(behavior_filenames[i]))[0][0] for i in range(len(behavior_filenames))]].drop(
        #     columns=['filename'])
        # stim_strength.reset_index(drop=True, inplace=True)  # Indices must match for modeling
        #
        # # Zscore
        # if not residuals:  # To not do both (otherwise I'd be subtracting the mean twice)
        #     if zscore:
        #         stim_strength = pd.DataFrame(
        #             stats.zscore(stim_strength, axis=0))  # Z-score the ILDs (along axis 0 or None
        #         # returns same result, but not axis 1). 0 along trials that's what I wanna do :)

        stim_strength, n_frames = make_frames_dm(df, stim_set=6, residuals=residuals, zscore=zscore)

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

        endog = choices
        dm_session_index = make_session_index_dm(df)  # Add bias (constant) per session
        if residuals:
            dm_net_ild = make_net_ild_dm(df)
            exog = pd.concat([stim_strength, dm_session_index, dm_net_ild], axis=1)
        else:
            exog = pd.concat([stim_strength, dm_session_index], axis=1)

        # From Genis' paper analysis code (gives directly the error)
        # Paper: https://www-nature-com.sire.ub.edu/articles/s41467-021-21501-z
        # Code: https://bitbucket.org/delaRochaLab/flexible-categorization/src/master/functions/analysis_fc.py
        # GLM with Binomial family and Logit link = discrete Logit model
        # stim_strength = sm.add_constant(stim_strength)  # Add constant (bias)
        model = sm.GLM(endog, exog, family=sm.families.Binomial(), missing='drop')  # GLM with Binomial family
        results = model.fit()
        params = results.params
        beta_std_err = results.bse
        p_values = results.pvalues
        summary = results.summary()
        print(summary)

        # Get shuffles
        # shuffles = get_shuffles_GLM(choices, stim_strength, iterations=iterations)  # Only frames

        ################################################################################################################

        # WORK IN PROGRESS
        net_ilds = list(df.ILD.abs().unique())
        net_ilds.sort()
        net_ilds = net_ilds[1:]  # Remove 0

        shuffles = get_shuffles_GLM(endog, exog, iterations=iterations, kind='pk_session_index')
        session_index_params = params[n_frames:-len(net_ilds)]
        session_index_std_error = beta_std_err[n_frames:-len(net_ilds)]
        session_index_p_values = p_values[n_frames:-len(net_ilds)]
        shuffles_session_index = [shuffles[i].iloc[n_frames:-len(net_ilds)] for i in range(len(shuffles))]  # From shuffles

        shuffles = get_shuffles_GLM(endog, exog, iterations=iterations, kind='pk_net_stim')
        net_stim_params = params[-len(net_ilds):]
        net_stim_std_err = beta_std_err[-len(net_ilds):]
        net_stim_p_values = p_values[-len(net_ilds):]
        net_stim_shuffles = [shuffles[i].iloc[-len(net_ilds):] for i in range(len(shuffles))]  # From shuffles

        ################################################################################################################

        shuffles = get_shuffles_GLM(endog, exog, iterations=iterations, kind='pk_frames')

        # Remove session index and stimulus constants
        params = params.iloc[:n_frames]  # From params
        shuffles = [shuffles[i].iloc[:n_frames] for i in range(len(shuffles))]  # From shuffles
        beta_std_err = beta_std_err.iloc[:n_frames]  # From beta_std_err
        p_values = p_values.iloc[:n_frames]  # From p_values

        # Store results in a namedtuple
        PK = namedtuple('PK', ['params', 'session_index_params', 'net_stim_params', 'std_err', 'session_index_std_error',
                               'net_stim_std_err', 'p_values', 'session_index_p_values', 'net_stim_p_values', 'shuffles',
                               'shuffles_session_index', 'net_stim_shuffles', 'n_trials', 'n_frames', 'experiment',
                               'animal', 'target_ilds', 'drug', 'residuals', 'zscore', 'control', 'n_mean_frames',
                               'iterations'])
        pk = PK(params=params, session_index_params=session_index_params, net_stim_params=net_stim_params,
                std_err=beta_std_err, session_index_std_error=session_index_std_error, net_stim_std_err=net_stim_std_err,
                p_values=p_values, session_index_p_values=session_index_p_values, net_stim_p_values=net_stim_p_values,
                shuffles=shuffles, shuffles_session_index=shuffles_session_index, net_stim_shuffles=net_stim_shuffles,
                n_trials=n_trials, n_frames=n_frames, experiment=experiment, animal=animal, target_ilds=target_ilds,
                drug=drug, residuals=residuals, zscore=zscore, control=control, n_mean_frames=n_mean_frames,
                iterations=iterations)

    return pk


def plot_pk(experiment='2AFC_6', animal=None, target_ilds=None, drug=None, residuals=True, zscore=False,
            control=None, n_mean_frames=None, iterations=1000, save=False):

    if type(experiment) == list:
        pk = get_mean_pk(experiments=experiment, animals=None, target_ilds=target_ilds, drug=drug,
                         residuals=residuals, zscore=zscore, control=control, n_mean_frames=n_mean_frames, iterations=iterations)
        title = f'N={len(pk.animal)}, {pk.n_trials} trials'
        filename_prefix = 'mean_'
    else:
        pk = get_pk(experiment=experiment, animal=animal, target_ilds=target_ilds, drug=drug,
                    residuals=residuals, zscore=zscore, control=control, n_mean_frames=n_mean_frames,
                    iterations=iterations)
        title = f'Mouse {pk.animal}, {pk.n_trials} trials'
        filename_prefix = ''

    # Default plotting parameters
    color = 'k'
    color_upper_shuffle = 'tab:red'
    label = ''

    ####################################################################################################################

    # PLOT FRAMES KERNEL

    # Residuals
    if residuals:
        ylabel = 'GLM weight (residuals)'
    else:
        if zscore:   # To not do both (otherwise I'd be subtracting the mean twice)
            ylabel = 'GLM weight (z-scored)'
        else:
            ylabel = 'GLM weight'

    plt.figure(constrained_layout=True)
    n_frames = pk.n_frames
    x = np.arange(n_frames)
    y = pk.params
    yerr = pk.std_err
    plt.plot(x, y, color=color, marker='o', label=label)
    plt.errorbar(x, y, yerr=yerr, color=color, marker='o', fmt='none', mec='none', ms=0)
    plt.title(title)
    plt.xlabel('Stimulus frame')
    plt.ylabel(ylabel)
    plt.legend(frameon=False)

    # Annotate significance
    if n_mean_frames is not None:  # If averaged frames, loop over the number of averaged frames instead
        n_frames = n_mean_frames

    # if pk.p_values is not None:
    #     for i in range(n_frames):
    #         if pk.p_values[i] <= 0.05:
    #             text = '*'
    #         else:
    #             # text = 'ns'
    #             text = ''
    #         plt.annotate(text, xy=(i + 1 + int(residuals), yticks[1]), xytext=(i + 1 + int(residuals), yticks[1]),
    #                      color=color, va='center', ha='center', fontsize='medium')

    shuffles_mean = np.mean(pk.shuffles, axis=0)  # Get the mean of all the shuffles
    percentiles95 = np.percentile(pk.shuffles, 95, axis=0)  # Get upper 5 percentile of the shuffled_var
    plt.plot(x, shuffles_mean, color='tab:gray', ls='--', zorder=1.8)
    plt.plot(x, percentiles95, color=color_upper_shuffle, ls=':', zorder=1.9)
    xticks = np.arange(1, n_frames + 1, 2)
    xticklabels = xticks + 1
    plt.xticks(xticks, xticklabels)
    sns.despine()  # Despine axes triming the 0

    if n_mean_frames == 2:
        plt.xticks([1, 2])  # Readjust xticks

    if save:
        filename = f'{pk.animal}_PK_ILDs_{target_ilds}, {n_mean_frames} averaged frames'
        filename = filename_prefix + filename
        folder_out = Path.home() / 'OneDrive' / 'Imágenes' / 'Figures' / 'kernels' / 'PK' / experiment
        if not folder_out.exists():
            folder_out.mkdir(parents=True, exist_ok=True)
        save_fig(folder_out, filename)
        plt.close()

    ####################################################################################################################

    # PLOT NET STIMULUS KERNEL

    plt.figure(constrained_layout=True)
    x = pk.net_stim_params.index.values
    x[-1] = 16  # Trick to zoom in
    y = pk.net_stim_params
    yerr = pk.net_stim_std_err
    plt.plot(x, y, color=color, marker='o')
    plt.errorbar(x, y, yerr=yerr, color=color, marker='o', fmt='none', mec='none', ms=0)
    plt.title(title)
    plt.xlabel('Net stimuli')
    plt.ylabel('GLM weight')
    # plt.legend(frameon=False)
    plt.xticks(x, ['2', '4', '8', '70'])

    shuffles_mean = np.mean(pk.net_stim_shuffles, axis=0)  # Get the mean of all the shuffles
    percentiles95 = np.percentile(pk.net_stim_shuffles, 95, axis=0)  # Get upper 5 percentile of the shuffled_var
    plt.plot(x, shuffles_mean, color='tab:gray', ls='--', zorder=1.8)
    plt.plot(x, percentiles95, color='tab:red', ls=':', zorder=1.9)
    sns.despine()

    if save:
        filename = f'{pk.animal}_PK_net_stim_{target_ilds}, {n_mean_frames} averaged frames'
        filename = filename_prefix + filename
        folder_out = Path.home() / 'Documentos' / 'kernels' / 'PK' / experiment
        save_fig(folder_out, filename)
        plt.close()

    ####################################################################################################################

    # PLOT SESSION INDEX KERNEL

    if type(experiment) == str:  # Don't do it for the mean kernel

        plt.figure(constrained_layout=True)
        x = pk.session_index_params.index.values
        y = pk.session_index_params
        yerr = pk.session_index_std_error
        plt.plot(x, y, color=color, marker='o')
        plt.errorbar(x, y, yerr=yerr, color=color, marker='o', fmt='none', mec='none', ms=0)
        plt.title(title)
        plt.xlabel('Session index')
        plt.ylabel('GLM weight')
        # plt.legend(frameon=False)
        # plt.xticks(x, ['2', '4', '8', '70'])
        shuffles_mean = np.mean(pk.shuffles_session_index, axis=0)  # Get the mean of all the shuffles
        percentiles2dot5 = np.percentile(pk.shuffles_session_index, 2.5, axis=0)  # Get upper 5 percentile of the shuffled_var
        percentiles97dot5 = np.percentile(pk.shuffles_session_index, 97.5, axis=0)  # Get upper 5 percentile of the shuffled_var
        plt.plot(x, shuffles_mean, color='tab:gray', ls='--', zorder=1.8)
        plt.plot(x, percentiles2dot5, color='tab:red', ls=':', zorder=1.85)
        plt.plot(x, percentiles97dot5, color='tab:red', ls=':', zorder=1.9)
        sns.despine()

        if save:
            filename = f'{pk.animal}_PK_session_index_{target_ilds}, {n_mean_frames} averaged frames'
            folder_out = Path.home() / 'Documentos' / 'kernels' / 'PK' / experiment
            save_fig(folder_out, filename)
            plt.close()


def plot_pks(experiment='2AFC_6', animals=['014', '016', '017', '020', '021', '022', '023', '024', '025'],
             target_ilds=None, drug=None, residuals=True, zscore=False, control=None,
             n_mean_frames=None, iterations=1000, save=False):
    """
    Do the kernels for all animals of a given batch (experiment)
    :param experiment: Batch of animals, needed to specify where the root folder with the data is
    :param animals: Mouse ID number
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
    :return:
    """

    time_start = time.time()

    experiment = get_experiment(experiment)

    # folder = '/home/alexis/PycharmProjects/glue_sessions/' + experiment + '/'  # Where the data for all animals is
    folder = Path.home() / 'PycharmProjects' / 'glue_sessions' / experiment

    for i in range(len(animals)):
        # path = folder + animals[i]
        path = Path(folder, animals[i])
        print(path)
        print(f'Plotting kernel of animal {animals[i]} ({i + 1}/{len(animals)})')
        plot_pk(experiment=experiment, animal=animals[i], target_ilds=target_ilds, drug=drug, residuals=residuals,
                zscore=zscore, control=control, n_mean_frames=n_mean_frames, iterations=iterations, save=save)

    time_end = time.time()
    runtime = time_end - time_start
    print('The script took', round(runtime, 2), 'seconds to run')


@timer
def get_mean_pk(experiments=['2AFC_2', '2AFC_3'], animals=None, target_ilds=None, drug=None, residuals=True,
                zscore=False, control=None, n_mean_frames=None, iterations=1000):
    """
    Get the kernels for all animals of a given batch (single string experiment) or across batches (list of experiments)
    :param experiment: Batch of animals, needed to specify where the root folder with the data is. If a list, get mean
    kernel across batches
    :param animals: Mouse ID number
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

    experiments_across_batches = []
    animals_across_batches = []
    params_across_animals = []
    session_index_params_across_animals = []
    net_stim_params_across_animals = []
    shuffles_across_animals = []
    shuffles_session_index_across_animals = []
    net_stim_shuffles_across_animals = []
    n_trials_across_animals = []
    pks = []

    for j in range(len(experiments)):

        experiment, _ = get_experiment(experiments[j])
        print(experiment)

        if experiments[j] == '2AFC_2':
            # animals = ['325', '327', '329', '330', '332', '333', '335', '337']
            animals = ['332', '333', '337']  # Drug experiments
        elif experiments[j] == '2AFC_3':
            animals = ['419', '420', '422', '616', '619', '623']
        elif experiments[j] == '2AFC_6':
            animals = ['014', '016', '017', '020', '021', '022', '023', '024', '025']

        folder_in = Path.home() / 'PycharmProjects' / 'glue_sessions' / experiment

        print(folder_in)

        for i in range(len(animals)):
            print(f'Getting kernel of animal {animals[i]} ({i + 1}/{len(animals)})')
            pk = get_pk(experiment=experiment, animal=animals[i], target_ilds=target_ilds, drug=drug,
                        residuals=residuals, zscore=zscore, control=control, n_mean_frames=n_mean_frames,
                        iterations=iterations)
            animals_across_batches.append(pk.animal)
            params_across_animals.append(pk.params)
            session_index_params_across_animals.append(pk.session_index_params)
            net_stim_params_across_animals.append(pk.net_stim_params)
            shuffles_across_animals.append(pk.shuffles)
            shuffles_session_index_across_animals.append(pk.shuffles_session_index)
            net_stim_shuffles_across_animals.append(pk.net_stim_shuffles)
            n_trials_across_animals.append(pk.n_trials)
            pks.append(pk)
        experiments_across_batches.append(experiment)

    n_trials = sum(n_trials_across_animals)
    n_frames = pk.n_frames

    # Get mean and sem of the parameters across animals
    params_across_animals = np.array(params_across_animals)
    params_mean_across_animals = np.mean(params_across_animals, 0)
    params_sem_across_animals = stats.sem(params_across_animals, 0)

    # Get mean and sem of the net stimuli parameters across animals
    net_stim_params_across_animals = np.array(net_stim_params_across_animals)
    net_stim_params_mean_across_animals = np.mean(net_stim_params_across_animals, 0)
    net_stim_params_sem_across_animals = stats.sem(net_stim_params_across_animals, 0)

    # Get the mean of the shuffles across animals
    shuffles_across_animals = np.array(shuffles_across_animals)  # Convert list of lists to 3 dim array (animal x
    # iterations x params)
    shuffles_means_across_animals = np.mean(shuffles_across_animals, 0)

    # Get the mean of the net stimuli shuffles across animals
    net_stim_shuffles_across_animals = np.array(net_stim_shuffles_across_animals)  # Convert list of lists to 3 dim array (animal x
    # iterations x params)
    net_stim_shuffles_across_animals = np.mean(net_stim_shuffles_across_animals, 0)

    # Store results in a namedtuple
    params_mean_across_animals = pd.Series(params_mean_across_animals)
    params_sem_across_animals = pd.Series(params_sem_across_animals)
    net_stim_params_mean_across_animals = pd.Series(net_stim_params_mean_across_animals)
    net_stim_params_sem_across_animals = pd.Series(net_stim_params_sem_across_animals)

    # Transform suffles_means_across_animals into a list of pd.Series
    shuffles_means_across_animals = [pd.Series(shuffles_means_across_animals[i, :]) for i in
                                     range(len(shuffles_means_across_animals))]
    # Transform suffles_means_across_animals into a list of pd.Series
    net_stim_shuffles_across_animals = [pd.Series(net_stim_shuffles_across_animals[i, :]) for i in
                                     range(len(net_stim_shuffles_across_animals))]
    # Rename the first element of each pd.Series as 'const' instead of '0'
    # shuffles_means_across_animals = [shuffles_means_across_animals[i].rename({0: 'const'}) for i in
    #                                  range(len(shuffles_means_across_animals))]

    # Store results in a namedtuple
    MeanPK = namedtuple('MeanPK', ['params', 'net_stim_params', 'std_err', 'net_stim_std_err',
                                   'p_values', 'shuffles', 'net_stim_shuffles', 'n_trials', 'n_frames', 'experiment', 'animal', 'target_ilds',
                                   'drug', 'residuals', 'zscore', 'control', 'n_mean_frames', 'iterations'])

    mean_pk = MeanPK(params=params_mean_across_animals, net_stim_params=net_stim_params_mean_across_animals,
                     std_err=params_sem_across_animals, net_stim_std_err=net_stim_params_sem_across_animals,
                     p_values=None, shuffles=shuffles_means_across_animals, net_stim_shuffles=net_stim_shuffles_across_animals,
                     n_trials=n_trials, n_frames=n_frames, experiment=experiments_across_batches, animal=animals_across_batches,
                     target_ilds=target_ilds, drug=drug, residuals=residuals, zscore=zscore, control=control, n_mean_frames=n_mean_frames,
                     iterations=iterations)

    return mean_pk
    # return pks


########################################################################################################################

# From 'Flexible categorization in perceptual decision making'
# Paper: (https://www-nature-com.sire.ub.edu/articles/s41467-021-21501-z#Sec11)
# Code: https://bitbucket.org/delaRochaLab/flexible-categorization/src/master/functions/analysis_fc.py
# betas = pd.Series([0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0])  # Primacy (for testing)
# betas = pd.Series([0, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5])  # Recency (for testing)
def normalized_pi_pk_area(pk):
    """
    Compute the PK area normalized by the area of a PI
    :param pk: Psychophysical kernel as namedtuple
    :return:
    """
    if pk.n_mean_frames is not None:
        n_frames = pk.n_mean_frames
    else:
        n_frames = pk.n_frames
    betas = pk.params.iloc[-n_frames:]
    area_pi = n_frames * (0.5 + 2 / np.pi * np.arctan(1 / np.sqrt(2 * n_frames - 1))) - 0.5 * n_frames
    npk_pi_area = np.sum(betas - 0.5) / area_pi
    return npk_pi_area


def normalized_pk_slope(pk):
    """
    The normalized PK slope is the slope of a linear regression of the PK, normalized between −1 (decaying PK, primacy)
    to +1 (increasing PK, recency). Because we wanted the PK slope to quantify the shape of the PK rather than its
    magnitude (which is captured by the PK area), we first normalized the PK to have unit area, where T is the stimulus
    duration. We then fit the NPK with a linear function of time, where β1 is the PK slope and k=12⋅var(t) is a factor
    that normalizes the PK slope to the interval (−1, +1).
    :param pk: Psychophysical kernel as namedtuple
    :return: Normalizaed PK slope
    """
    if pk.n_mean_frames is not None:
        n_frames = pk.n_mean_frames
    else:
        n_frames = pk.n_frames
    # aux = np.linspace(1, -1, n_frames)  # Like this primacy is positive and recency is negative
    aux = np.linspace(-1, 1, n_frames)
    betas = pk.params.iloc[-n_frames:]
    npk = betas - 0.5
    npk = npk / sum(npk)  # Normalized pk, must sum 1
    # npk_slope = -sum(aux * npk)  # Remove the minus if using aux = np.linspace(1, -1, n_frames)
    results = stats.linregress(aux, npk.values)
    npk_slope = results.slope
    return npk_slope


def primacy_recency_index(pk):
    """
    Where β1 and β2 are the coefficients of a logistic regression with the coherence of the first and second part of
    the stimuli as predictors. Similar to the Normalized PK slope, the primacy-recency index ranges from −1 (primacy) to
    1 (recency).
    :param pk: Psychophysical kernel as namedtuple
    :return: Primacy-recency index
    """
    if pk.n_mean_frames is not None:
        n_frames = pk.n_mean_frames
    else:
        n_frames = pk.n_frames
    betas = pk.params.iloc[-n_frames:]
    beta1 = betas.iloc[0:int(len(betas) / 2)].mean()
    beta2 = betas.iloc[int(len(betas) / 2):].mean()
    index = (beta2 - beta1) / (beta1 + beta2)
    return index


########################################################################################################################

# # Debugging
# experiment = '2AFC_2'
# # experiment = ['2AFC_2', '2AFC_3']
# experiments = ['2AFC_2']
# # animal = '333'
# # animals = ['325', '327', '329', '330', '332', '333', '335', '337']  # Bach 2 (with ILDs) -326, -334
# # animals = ['419', '420', '422', '616', '619', '623']  # Batch 3 (with ILDs)  -617, -620
# animals = ['332', '333', '337']  # Drug experiments (batch 2)
# target_ilds = None
# drug = 'MK801'
# residuals = True
# zscore = False
# control = None
# n_mean_frames = None
# iterations = 10
# save = False


# Get PK
# pk = get_pk(experiment=experiment, animal=animal, target_ilds=target_ilds, drug=drug,
#                 residuals=residuals, zscore=zscore, control=control, n_mean_frames=n_mean_frames, iterations=iterations)

# Get mean PK
# mean_pk = get_mean_pk(experiments=experiments, animals=None, target_ilds=target_ilds, drug=drug,
#                       residuals=residuals, zscore=zscore, control=control, n_mean_frames=n_mean_frames,
#                       iterations=iterations)

# Plot individual PKs
# plot_pks(experiment=experiment, animals=animals, target_ilds=target_ilds, drug=drug,
#             residuals=residuals, zscore=zscore, control=control, n_mean_frames=n_mean_frames, iterations=iterations,
#             save=save)

# Plot  mean PK
# plot_pk(experiment=experiments, animal=animals, target_ilds=target_ilds, drug=drug,
#             residuals=residuals, zscore=zscore, control=control, n_mean_frames=n_mean_frames, iterations=iterations,
#             save=save)

# plot_pks(experiment=experiment, animals=animals, target_ilds=target_ilds, drug=drug,
#             residuals=residuals, zscore=zscore, control=control, n_mean_frames=n_mean_frames, iterations=iterations,
#             save=save)
# plot_pks(experiment='2AFC_3', animals=['419', '420', '422', '616', '619', '623'], target_ilds=target_ilds, drug=drug,
#             residuals=residuals, zscore=zscore, control=control, n_mean_frames=n_mean_frames, iterations=iterations,
#             save=save)

def plot_drug_pk():
    mean_pk_saline = get_mean_pk(experiments=['2AFC_2'], animals=None, target_ilds=None, drug='saline', residuals=True,
                                 zscore=False, control=None, n_mean_frames=2, iterations=10)
    mean_pk_mk801 = get_mean_pk(experiments=['2AFC_2'], animals=None, target_ilds=None, drug='MK801', residuals=True,
                                 zscore=False, control=None, n_mean_frames=2, iterations=10)

    title = f'N={3}, {mean_pk_saline.n_trials + mean_pk_mk801.n_trials} trials'
    filename_prefix = 'drug_'
    color_saline = 'tab:gray'
    color_drug = 'tab:pink'

    ####################################################################################################################

    # PLOT FRAMES KERNEL

    # Residuals
    if residuals:
        ylabel = 'GLM weight (residuals)'
    else:
        if zscore:  # To not do both (otherwise I'd be subtracting the mean twice)
            ylabel = 'GLM weight (z-scored)'
        else:
            ylabel = 'GLM weight'


    plt.figure(constrained_layout=True)
    n_frames = mean_pk_saline.n_frames
    x = np.arange(n_frames)
    y_saline = mean_pk_saline.params
    y_drug = mean_pk_mk801.params
    yerr = mean_pk_saline.std_err
    plt.plot(x, y_saline, color=color_saline, marker='o', label='saline')
    plt.errorbar(x, y_saline, yerr=yerr, color=color_saline, marker='o', fmt='none', mec='none', ms=0)

    plt.plot(x, y_drug, color=color_drug, marker='o', label='MK801')
    plt.errorbar(x, y_drug, yerr=yerr, color=color_drug, marker='o', fmt='none', mec='none', ms=0)


    plt.title(title)
    plt.xlabel('Stimulus frame')
    plt.ylabel(ylabel)
    plt.legend(frameon=False)

    # Annotate significance
    if n_mean_frames is not None:  # If averaged frames, loop over the number of averaged frames instead
        n_frames = n_mean_frames

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
    xticks = np.arange(1, n_frames + 1, 2)
    xticklabels = xticks + 1
    plt.xticks(xticks, xticklabels)
    sns.despine(trim=True)  # Despine axes triming the 0

    if n_mean_frames == 2:
        plt.xticks([1, 2])  # Readjust xticks

    if save:
        filename = f'{mean_pk_saline.animal}_PK_ILDs_{target_ilds}, {n_mean_frames} averaged frames'
        filename = filename_prefix + filename
        folder_out = Path.home() / 'Documentos' / 'kernels' / 'PK' / 'drug'
        save_fig(folder_out, filename)
        plt.close()

    ####################################################################################################################

    # PLOT NET STIMULUS KERNEL

    plt.figure(constrained_layout=True)
    x = mean_pk_saline.net_stim_params.index.values
    x[-1] = 16  # Trick to zoom in
    x = [2, 4, 8, 16]
    y_saline = mean_pk_saline.net_stim_params
    yerr_saline = mean_pk_saline.net_stim_std_err
    plt.plot(x, y_saline, color=color_saline, marker='o', label='saline')
    plt.errorbar(x, y_saline, yerr=yerr_saline, color=color_saline, marker='o', fmt='none', mec='none', ms=0)

    y_drug = mean_pk_mk801.net_stim_params
    yerr_drug = mean_pk_mk801.net_stim_std_err
    plt.plot(x, y_drug, color=color_drug, marker='o', label='MK801')
    plt.errorbar(x, y_drug, yerr=yerr_drug, color=color_drug, marker='o', fmt='none', mec='none', ms=0)
    plt.title(title)
    plt.xlabel('Net stimuli')
    plt.ylabel('GLM weight')
    plt.legend(frameon=False)
    plt.xticks(x, ['2', '4', '8', '70'])

    # shuffles_mean = np.mean(pk.net_stim_shuffles, axis=0)  # Get the mean of all the shuffles
    # percentiles95 = np.percentile(pk.net_stim_shuffles, 95, axis=0)  # Get upper 5 percentile of the shuffled_var
    # plt.plot(x, shuffles_mean, color='tab:gray', ls='--', zorder=1.8)
    # plt.plot(x, percentiles95, color='tab:red', ls=':', zorder=1.9)
    sns.despine(trim=True)

    if save:
        filename = f'{mean_pk_saline.animal}_PK_net_stim_{target_ilds}, {n_mean_frames} averaged frames'
        filename = filename_prefix + filename
        folder_out = Path.home() / 'Documentos' / 'kernels' / 'PK' / 'drug'
        save_fig(folder_out, filename)
        plt.close()
