# Import libraries
import numpy as np
from matplotlib import pyplot as plt
from my_fun.my_fun import find_power_dB_par, ild, my_select_evidence, compute_psych_curve
from glue_sessions.glue_sessions import glue_sessions
from create_sounds.create_sounds_v1 import create_sounds

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

# Initialize empty lists for simulated data
sim_sound = []
sim_evidence = []
sim_mean_ild = []
sim_choice = []

for i in range(n_trials):

    evidence = my_select_evidence(trial_list[i], evidences)  # Select evidence
    sample_index = df_sample[df_sample.Evidence == evidence].index  # Get indexes of sounds with selected evidence
    sound_index = np.random.choice(sample_index)  # Choose a random sound from sample

    # Append values to list
    sim_sound.append(df_sample.Filename.iloc[sound_index])
    sim_evidence.append(df_sample.Evidence.iloc[sound_index])
    sim_mean_ild.append(df_sample.Mean.iloc[sound_index])

    if sim_mean_ild[i] < 0:
        sim_choice.append(0)
    else:
        sim_choice.append(1)

unfair_trials = np.where(np.array(trial_list) != sim_choice)[0]  # Return indices where the choice of the perfect agent
# doesn't match with the known outcome of the trial
errors = len(unfair_trials)
hits = n_trials - errors
accuracy = hits / n_trials

# Compute psychometric curves
psych_curve_perfect_agent = compute_psych_curve(sim_evidence, sim_choice)  # Perfect agent
psych_curve_cheater_agent = compute_psych_curve(sim_evidence, trial_list)  # Cheater agent

# Plot horizontal and vertical lines
plt.axhline(0.5, color='tab:gray', ls='--')
plt.axvline(0., color='tab:gray', ls='--')

# Plot psychometric curves and errorbars
plt.plot(np.linspace(-1, 1, 30), psych_curve_perfect_agent.fit, color='tab:blue', label='Perfect agent')
plt.errorbar(psych_curve_perfect_agent.xdata, psych_curve_perfect_agent.ydata, yerr=psych_curve_perfect_agent.fit_error,
             color='tab:blue', fmt='o', markerfacecolor='none')

plt.plot(np.linspace(-1, 1, 30), psych_curve_cheater_agent.fit, color='tab:orange', label='Cheater agent')
plt.errorbar(psych_curve_cheater_agent.xdata, psych_curve_cheater_agent.ydata, yerr=psych_curve_cheater_agent.fit_error,
             color='tab:orange', fmt='o', markerfacecolor='none')

plt.title('Psychometric curves \n(' + str(n_trials) + ' trials, ' + str(errors) + ' unfair)')
# plt.xlabel('Evidence')
plt.xlabel('Interaural Level Difference (dB)')
# plt.xticks(target_evidences, target_ilds)
plt.xticks(evidences,
           ['-40', '', '', '-17', '', '-8', '', '', '-4', '', '0', '', '-4', '', '', '-8', '', '-17', '', '', '-40'])
plt.ylabel('Probability choose right')
plt.legend(loc="lower right", frameon=False)
# plt.spines['top'].set_visible(False)
# plt.spines['right'].set_visible(False)

########################################################################################################################

# Same but running the perfect and cheating agents on the same trials I did (instead of simulating them)

# Glue sessions I (Alexis) did
df_Alexis = glue_sessions()
df_Alexis = df_Alexis[df_Alexis.Response == 1].reset_index(drop=True, inplace=False)  # Discard missed trials
# df_Alexis = df_Alexis[df_Alexis.Response == 1][:1000].reset_index(drop=True, inplace=False)  # Discard missed trials
n_trials = len(df_Alexis)
evidences = df_ild.Evidence.unique()

# Perfect agent
df_Alexis_filenames = df_Alexis.Filename  # Get filenames of the dataset
alexis_ild = []  # Create empty list
perfect_agent_choice = []

for i in range(n_trials):
    alexis_ild.append(
        df_ild.Mean[df_ild.Filename == df_Alexis_filenames[i]].values[0])  # Append the mean ILD for that sound

    # alexis_ild.append(df_ild[df_ild.Filename == df_Alexis_filenames[i]].values[0][2:7])  # First 5 frames
    # alexis_ild.append(df_ild[df_ild.Filename == df_Alexis_filenames[i]].values[0][7:12])  # First 5 frames

    if alexis_ild[i] < 0:
        perfect_agent_choice.append(0)
    else:
        perfect_agent_choice.append(1)

unfair_trials = np.where(np.array(df_Alexis.Side) != perfect_agent_choice)[
    0]  # Return indices where the choice of the perfect agent
# doesn't match with the known outcome of the trial

# Compute psychometric curves
psych_curve_Alexis_agent = compute_psych_curve(df_Alexis.Evidence, df_Alexis.Choice)  # Alexis agent
psych_curve_perfect_agent = compute_psych_curve(df_Alexis.Evidence, perfect_agent_choice)  # Perfect agent
psych_curve_cheater_agent = compute_psych_curve(df_Alexis.Evidence, df_Alexis.Side)  # Cheater agent

# Plot horizontal and vertical lines
plt.axhline(0.5, color='tab:gray', ls='--')
plt.axvline(0., color='tab:gray', ls='--')

# Plot psychometric curves and errorbars
plt.plot(np.linspace(-1, 1, 30), psych_curve_Alexis_agent.fit, color='tab:blue', label='Alexis agent')
plt.errorbar(psych_curve_Alexis_agent.xdata, psych_curve_Alexis_agent.ydata, yerr=psych_curve_Alexis_agent.fit_error,
             color='tab:blue', fmt='o', markerfacecolor='none')

plt.plot(np.linspace(-1, 1, 30), psych_curve_perfect_agent.fit, color='tab:orange', label='Perfect agent')
plt.errorbar(psych_curve_perfect_agent.xdata, psych_curve_perfect_agent.ydata, yerr=psych_curve_perfect_agent.fit_error,
             color='tab:orange', fmt='o', markerfacecolor='none')

plt.plot(np.linspace(-1, 1, 30), psych_curve_cheater_agent.fit, color='tab:green', label='Cheater agent')
plt.errorbar(psych_curve_cheater_agent.xdata, psych_curve_cheater_agent.ydata, yerr=psych_curve_cheater_agent.fit_error,
             color='tab:green', fmt='o', markerfacecolor='none')

plt.title('Psychometric curves \n(' + str(len(df_Alexis)) + ' trials, ' + str(len(unfair_trials)) + ' unfair)')
# plt.xlabel('Evidence')
plt.xlabel('Interaural Level Difference (dB)')
plt.xticks(target_evidences, target_ilds)
plt.ylabel('Probability choose right')
plt.legend(loc="lower right", frameon=False)

########################################################################################################################

df_sounds = create_sounds_test(save=False)
df_ild = ild(df_sounds)
