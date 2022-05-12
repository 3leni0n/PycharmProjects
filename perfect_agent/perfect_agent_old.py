# Import libraries
import numpy as np
from matplotlib import pyplot as plt
from my_fun.my_fun import find_power_dB_par, ild, my_select_evidence_old, compute_psych_curve
from glue_sessions.glue_sessions import glue_sessions
from create_sounds.create_sounds import create_sounds

########################################################################################################################

df_ild = ild()
df_ild_summary = df_ild.groupby('Evidence', as_index=False).mean()  # SQL-style index
n_trials = 1000
trial_types = [0, 1]  # 0=left, 1=right
trial_list = np.random.choice(trial_types, n_trials).tolist()  # Generate random trial vector of length n_trials
evidences = df_ild.Evidence.unique()
target_evidences = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
target_ilds = list(df_ild_summary.Mean[df_ild_summary.Evidence.isin(target_evidences)].round().astype('int'))
ilds = df_ild_summary.Mean[df_ild_summary.Evidence.isin(evidences)].reset_index(drop=True, inplace=False).round()
df_sample = df_ild[df_ild.Evidence.isin(evidences)].reset_index(drop=True, inplace=False)

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

for j in range(10):

    # Initialize empty lists for simulated data
    sim_sound = []
    sim_evidence = []
    sim_mean_ild = []
    sim_choice = []

    for i in range(n_trials):

        evidence = my_select_evidence_old(trial_list[i], evidences)  # Select evidence
        sample_index = df_sample[df_sample.Evidence == evidence].index  # Get indexes of sounds with selected evidence
        sound_index = np.random.choice(sample_index)  # Choose a random sound from sample

        # Append values to list
        sim_sound.append(df_sample.Filename.iloc[sound_index])
        sim_evidence.append(df_sample.Evidence.iloc[sound_index])
        # sim_mean_ild.append(df_sample.Mean.iloc[sound_index])

        sim_mean_ild.append(df_sample.iloc[sound_index, 2:3+j].values.mean())  # ILD0

        if sim_mean_ild[i] < 0:
            sim_choice.append(0)
        else:
            sim_choice.append(1)

    # Evaluate performance
    unfair_trials = np.where(np.array(trial_list) != sim_choice)[0]  # Return indices where the choice of the perfect agent
    # doesn't match with the known outcome of the trial
    errors = len(unfair_trials)
    hits = n_trials - errors
    accuracy = hits / n_trials

    # Compute psychometric curves
    psych_curve_perfect_agent = compute_psych_curve(sim_evidence, sim_choice)  # Perfect agent
    # psych_curve_cheater_agent = compute_psych_curve(sim_evidence, trial_list)  # Cheater agent

    # Plot horizontal and vertical lines
    plt.axhline(0.5, color='tab:gray', ls='--')
    plt.axvline(0., color='tab:gray', ls='--')

    # Plot Perfect Agent's psychometric curves and errorbars
    plt.plot(np.linspace(-1, 1, 30), psych_curve_perfect_agent.fit, color=color_cycle[j], label=f'{j+1} frames')
    plt.errorbar(psych_curve_perfect_agent.xdata, psych_curve_perfect_agent.ydata, yerr=psych_curve_perfect_agent.fit_error,
                 color=color_cycle[j], fmt='o', markerfacecolor='none')

    # Plot Cheater Agent's psychometric curves and errorbars
    # plt.plot(np.linspace(-1, 1, 30), psych_curve_cheater_agent.fit, color='tab:orange', label='Cheater agent')
    # plt.errorbar(psych_curve_cheater_agent.xdata, psych_curve_cheater_agent.ydata, yerr=psych_curve_cheater_agent.fit_error,
    #              color='tab:orange', fmt='o', markerfacecolor='none')

    # plt.title('Psychometric curves \n(' + str(n_trials) + ' trials, ' + str(errors) + ' unfair)')
    plt.title(f'Perfect agents, {n_trials} trials')
    plt.xlabel('Interaural Level Difference (dB)')
    # plt.xticks(target_evidences, target_ilds)
    plt.xticks(evidences,
               ['-40', '', '', '-17', '', '-8', '', '', '-4', '', '0', '', '-4', '', '', '-8', '', '-17', '', '', '-40'])
    plt.ylabel('Probability choose right')
    plt.legend(loc="lower right", frameon=False)
    # plt.spines['top'].set_visible(False)
    # plt.spines['right'].set_visible(False)
