# Import libraries
import time
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from my_fun.my_fun import select_ilds, compute_psych_curve
from create_sounds.create_sounds_v2 import create_sounds_v2


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

    # Select the folder where to save the PDF or create it if it doesn't exist
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

    # Select the folder where to save the PDF or create it if it doesn't exist
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
