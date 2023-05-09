# Import libraries
import time
import os
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from matplotlib import pyplot as plt
import seaborn as sns

from my_fun.my_fun import select_ilds, compute_psych_curve
from create_sounds.create_sounds_v2 import create_sounds_v2

# Mel's code snippet for poster
sns.set_theme()
sns.set_style('white')
sns.set_style('ticks')
sns.set_context('poster')


def perfect_agent(n_trials=1000, sim_stim=False, sigma=1, plot=False):
    """
    Simulates a perfect agent that perform an 2AFC ILD discrimination task
    :param n_trials: Number of trials
    :param sim_stim: Simulates stimuli. If False uses real stimuli used for training mice
    :param sigma: std of the envelope fluctuations. Only used when sim_stim=True
    :param plot: If plot the results or not
    :return: DataFrame with the number of unfair trials, errors, hits and accuracy
    """

    time_start = time.time()

    # Select the folder where to save the plot or create it if it doesn't exist
    folder = '/home/alexis/Documentos/perfect agent/'
    if not os.path.exists(folder):
        os.mkdir(folder)
    os.chdir(folder)

    # If True create a simulated stimuli dataset, otherwise use the existing one to train mice
    if sim_stim:
        df = create_sounds_v2(max_vol=70, fs=44100, cutoff=[2000, 20000], amp=1, dur=1, fn=10000, normalize=True,
                              n_frames=10, sigma=sigma, save=False)
    else:
        df = pd.read_csv('/home/alexis/PycharmProjects/create_sounds/sounds_2.csv')

    trial_types = [0, 1]  # 0=left, 1=right
    trial_list = np.random.choice(trial_types, n_trials).tolist()  # Generate random trial vector of length n_trials
    ilds = np.sort(df.ILD.unique().astype('int'))

    # Set cycling colors
    color_cycle = ['tab:blue',
                   'tab:orange',
                   'tab:green',
                   'tab:red',
                   'tab:purple',
                   'tab:brown',
                   'tab:pink',
                   'tab:gray',
                   'tab:olive',
                   'tab:cyan']

    # Initialize empty lists to evaluate performance
    unfair_trials = []
    errors = []
    hits = []
    accuracy = []

    for j in range(df.n_frames.unique()[0]):

        # Initialize empty lists for simulated data
        sim_sound = []
        sim_ild = []
        sim_mean_ild = []
        sim_choice = []

        for i in range(n_trials):

            ild = select_ilds(ilds, 1, trial_list[i])  # Select ild
            sample_index = df[df.ILD == ild].index  # Get indexes of sounds with selected evidence
            sound_index = np.random.choice(sample_index)  # Choose a random sound from sample

            # Append values to list
            sim_sound.append(df.filename.iloc[sound_index])
            sim_ild.append(df.ILD.iloc[sound_index])
            # sim_mean_ild.append(np.mean(df.loc[sound_index, 'ER0':'ER9'].values - df.loc[sound_index, 'EL0':'EL9'].values))
            # All frames
            sim_mean_ild.append(np.mean(df.iloc[sound_index, 12:13 + j].values - df.iloc[sound_index, 2:3 + j].values))
            # Accumulating frames each iteration

            if sim_mean_ild[i] < 0:
                sim_choice.append(0)
            else:
                sim_choice.append(1)

        # Evaluate performance
        unfair_trials.append(np.where(np.array(trial_list) != sim_choice)[0])  # Return indices where the choice of the
        # perfect agent doesn't match with the known outcome of the trial
        errors.append(len(unfair_trials[j]))
        hits.append(n_trials - errors[j])
        accuracy.append(hits[j] / n_trials)

        if plot:
            # Compute psychometric curves
            psych_curve_perfect_agent = compute_psych_curve(sim_ild, sim_choice)  # Perfect agent
            # psych_curve_cheater_agent = compute_psych_curve(sim_ild, trial_list)  # Cheater agent

            # Plot horizontal and vertical lines
            plt.axhline(0.5, color='tab:gray', ls='--')
            plt.axvline(0., color='tab:gray', ls='--')

            # Plot Perfect Agent's psychometric curves and errorbars
            plt.plot(np.linspace(np.min(df.ILD), np.max(df.ILD), len(psych_curve_perfect_agent.fit)),
                     psych_curve_perfect_agent.fit,
                     color=color_cycle[j], label=f'{j + 1} frames, acc.={accuracy[j]}')
            plt.errorbar(psych_curve_perfect_agent.xdata, psych_curve_perfect_agent.ydata,
                         yerr=psych_curve_perfect_agent.fit_error,
                         color=color_cycle[j], fmt='o', markerfacecolor='none')

            # # Plot Cheater Agent's psychometric curves and errorbars
            # plt.plot(np.linspace(np.min(df.ILD), np.max(df.ILD), len(psych_curve_cheater_agent.fit)), psych_curve_cheater_agent.fit,
            #          color=color_cycle[j], label=f'{j+1} frames, acc.={accuracy[j]}')
            # plt.errorbar(psych_curve_cheater_agent.xdata, psych_curve_cheater_agent.ydata, yerr=psych_curve_cheater_agent.fit_error,
            #              color=color_cycle[j], fmt='o', markerfacecolor='none')

            # plt.title('Psychometric curves \n(' + str(n_trials) + ' trials, ' + str(errors) + ' unfair)')
            plt.title(f'Perfect agents, {n_trials} trials')
            plt.xscale('symlog', linthreshx=20)  # Set symmetric logarithmic spacing to zoom in the middle
            plt.xticks(ilds, list(ilds))
            plt.minorticks_off()  # Remove minor ticks
            plt.xlabel('Interaural Level Difference (dB)')
            plt.ylabel('Probability choose right')
            plt.legend(loc="lower right", frameon=False)
            # plt.spines['top'].set_visible(False)
            # plt.spines['right'].set_visible(False)
            plt.savefig(folder + 'perfect_agent.png')

    # Construct DataFrame
    columns = ['UnfairTrials', 'Errors', 'Hits', 'Accuracy']
    data = list(zip(unfair_trials, errors, hits, accuracy))
    df = pd.DataFrame(data=data, columns=columns)

    time_end = time.time()
    runtime = time_end - time_start
    print('The script took', round(runtime, 2), 'seconds to run')

    return df


def test_stim_set(n_iterations=3, n_trials=1000, plot=True):
    time_start = time.time()

    # Select the folder where to save the plot or create it if it doesn't exist
    folder = '/home/alexis/Documentos/perfect agent/'
    if not os.path.exists(folder):
        os.mkdir(folder)
    os.chdir(folder)

    df_list = []

    for i in range(n_iterations):
        df = perfect_agent(n_trials=n_trials, sim_stim=True, sigma=i + 1)  # + 1 to avoid sigma=0
        df_list.append(df)

        if plot:
            plt.plot(df_list[i].Accuracy, marker='o', label=f'std={i + 1}')  # + 1 to avoid sigma=0

        plt.title(f"Perfect Agents' accuracy ({n_trials} trials)")
        plt.xlabel('Number of integrated frames')
        plt.xticks(np.arange(0, 10), labels=np.arange(1, 11))
        plt.ylabel('Accuracy')
        # plt.ylim([0.5, 1])
        plt.legend(loc='lower right')
        plt.savefig(folder + 'perfect_agent_test.png')

    time_end = time.time()
    runtime = time_end - time_start
    print('The script took', round(runtime, 2), 'seconds to run')

    return df_list


def test_kernels(experiment='2AFC_2', animal='333', frame_index=None, target_ilds=[-2, 0, 2], zscore=True,
                 iterations=100, save=False, format='svg', transparent=False):
    """
    Simulates a perfect agent that perform an 2AFC ILD discrimination task
    :param n_trials: Number of trials
    :param sim_stim: Simulates stimuli. If False uses real stimuli used for training mice
    :param sigma: std of the envelope fluctuations. Only used when sim_stim=True
    :param plot: If plot the results or not
    :return: DataFrame with the number of unfair trials, errors, hits and accuracy
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
    frames_ild.insert(0, column='filename', value=sounds.filename)  # Insert filenames in first column

    # Load behavioral data
    df = pd.read_csv(folder_in)
    df = df[df.Choice.notna()]  # Drop misses (nan in choices), otherwise the code crashes
    df = df[df.ILD.isin(target_ilds)]  # Select only trials with the desired ILDs
    n_trials = len(df)
    filenames = df.Filename.tolist()
    stim_strength = frames_ild.loc[
        [np.where(sounds.filename == np.array(filenames[i]))[0][0] for i in range(len(filenames))]].drop(
        columns=['filename'])
    n_frames = stim_strength.shape[1]

    # Evidence accumulation regime
    if frame_index == 'all':  # Perfect integrator
        accum_evi = stim_strength.mean(axis=1).tolist()
    elif frame_index == 'random_snapshot':  # Random integrator of 1 frame
        random_frames_indexes = [np.random.choice(np.arange(10)) for i in range(len(stim_strength))]
        accum_evi = [stim_strength.iloc[i, random_frames_indexes[i]] for i in range(len(stim_strength))]
    else:  # Discrete integrator
        accum_evi = stim_strength.iloc[:, frame_index].tolist()

    # Create choices vector according the evidence accumulation regime
    choices = []
    for i in range(len(accum_evi)):
        # noise = np.random.choice([-1, 1]) * np.random.random()
        noise = np.random.normal()  # loc=0.0, scale=1.0  # Same but more correct
        if accum_evi[i] + noise < 0:
            choices.append(0)
        else:
            choices.append(1)

    # Zscore
    # if not residuals:  # To not do both (otherwise I'd be subtracting the mean twice)
    if zscore:
        stim_strength = pd.DataFrame(stats.zscore(stim_strength, axis=0))  # Z-score the ILDs (along axis 0 or None
        # returns same result, but not axis 1). 0 along trials that's what I wamnna do :)
        ylabel = 'GLM weight (z-scored)'
    else:
        ylabel = 'GLM weight'

    stim_strength.reset_index(drop=True, inplace=True)  # Indices must match for modeling
    stim_strength = sm.add_constant(stim_strength)  # Add constant (bias)
    model = sm.GLM(choices, stim_strength, family=sm.families.Binomial())  # GLM with Binomial family and Logit link
    results = model.fit()
    params = results.params
    beta_std_err = results.bse
    p_values = results.pvalues
    summary = results.summary()
    print(summary)

    # Default plotting parameters
    color = 'k'
    color_upper_shuffle = 'tab:red'
    label = ''
    filename = f'_PK_PA_frame_{frame_index}_ILDs: {target_ilds}'

    plt.figure(constrained_layout=True)

    # Plot kernel
    plt.plot(np.arange(1, len(params)), params.iloc[1:11], color=color, marker='o', label=label)
    plt.errorbar(np.arange(1, len(params)), params.iloc[1:11], yerr=beta_std_err.iloc[1:11], color=color,
                 marker='o', fmt='none', mec='none', ms=0)  # Without constant (bias)
    plt.title(f'Mouse {df.Setup.unique()[0]}, {n_trials} trials')
    plt.xlabel('Stimulus frame')
    plt.ylabel(ylabel)
    plt.legend(frameon=False)
    yticks = plt.gca().get_yticks()  # Get current axis yticks for the significance annotations

    # # Annotate significance
    # if n_mean_frames is not None:  # If averaged frames, loop over the number of averaged frames instead
    #     n_frames = n_mean_frames

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

    # Permutation test (shuffled_var)
    shuffles = []
    for _ in range(iterations):
        # choices_shuffled = choices.sample(frac=1).reset_index(drop=True)
        stim_strength_shuffled = stim_strength.sample(frac=1).reset_index(drop=True)
        # model = sm.Logit(choices, stim_strength)  # Discrete Logit model
        choices = list(choices)  # Otherwise 'ValueError: The indices for endog and exog are not aligned'
        model_shuffled = sm.GLM(choices, stim_strength_shuffled,
                                family=sm.families.Binomial())  # GLM with Binomial family and Logit link
        results_shuffled = model_shuffled.fit()
        params_shuffled = results_shuffled.params
        shuffles.append(params_shuffled)
        # plt.plot(np.arange(1, len(params_shuffled)), params_shuffled.iloc[1:11], color='tab:gray', marker=None,
        #          mfc='none', mec='none', mew=0, ms=0, label=label, alpha=0.1, zorder=1.7)  # Plot all shuffles

    shuffles_mean = np.mean(shuffles, axis=0)  # Get the mean of all the shuffles
    percentiles = np.percentile(shuffles, 95, axis=0)  # Get upper 5 percentile of the shuffled_var
    # percentiles = np.percentile(shuffles, 68, axis=0)  # Get upper 32 percentile of the shuffled_var
    plt.plot(np.arange(1, len(params)), shuffles_mean[1:11], color='tab:gray', ls='--', zorder=1.8)
    plt.plot(np.arange(1, len(params)), percentiles[1:11], color=color_upper_shuffle, ls=':', zorder=1.9)
    plt.xticks(np.arange(1, n_frames + 1, 1))  # Put one xtick for observation for triming later
    sns.despine(offset=10, trim=True)  # Despine axes triming the 0
    plt.xticks(np.arange(2, n_frames + 1, 2))  # Readjust xticks

    if save:
        folder_out = '/home/alexis/Documentos/perfect agent/' + experiment + '/'
        if not os.path.exists(folder_out):
            os.mkdir(folder_out)
        os.chdir(folder_out)
        plt.savefig(folder_out + str(df.Setup.unique()[0]) + filename + '.' + format, format=format,
                    transparent=transparent)
        plt.close()

    time_end = time.time()
    runtime = time_end - time_start
    print('The script took', round(runtime, 2), 'seconds to run')

    return print(results.summary())


# test_kernels(experiment='2AFC_2', animal='333', frame_index='random_snapshot', target_ilds=[-70, -8, -4, -2, 0, 2,
# 4, 8, 70], zscore=True, iterations=100, save=True, format='png', transparent=False)
test_kernels(experiment='2AFC_2', animal='333', frame_index='random_snapshot', target_ilds=[0],
             zscore=True, iterations=100, save=True, format='png', transparent=False)
