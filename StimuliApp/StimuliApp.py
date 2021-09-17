import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from my_fun.my_fun import compute_psych_curve

# Path to sessions
path1 = '/home/alexis/StimuliApp Data/testsoundalexis20210913183327/testSoundAlexis 2021-09-13 18_33_27 trials.csv'
path2 = '/home/alexis/StimuliApp Data/testsoundalexis20210913184648/testSoundAlexis 2021-09-13 18_46_48 trials.csv'
path3 = '/home/alexis/StimuliApp Data/testsoundalexis20210913190101/testSoundAlexis 2021-09-13 19_01_01 trials.csv'
path4 = '/home/alexis/StimuliApp Data/testsoundalexis20210913191033/testSoundAlexis 2021-09-13 19_10_33 trials.csv'
path5 = '/home/alexis/StimuliApp Data/testsoundalexis20210913192000/testSoundAlexis 2021-09-13 19_20_00 trials.csv'

# Import csv
df1 = pd.read_csv(path1)
df2 = pd.read_csv(path2)
df3 = pd.read_csv(path3)
df4 = pd.read_csv(path4)
df5 = pd.read_csv(path5)

# Merge all
df = pd.concat([df1, df2, df3, df4, df5])

# Get variables
trials = df.trial.to_list()
correct = df.correct.to_list()
sound_number = df.soundPlay_object1_audioNumber.to_list()
evidence = []
x_pos = df.soundPlay_touchPositionX.to_list()
choice = []

for i in range(len(trials)):

    if x_pos[i] < 0:
        choice.append(0)
    else:
        choice.append(1)

    if 1 <= sound_number[i] <= 10:
        evidence.append(-1)

    if 11 <= sound_number[i] <= 20:
        evidence.append(-0.5)

    if 21 <= sound_number[i] <= 30:
        evidence.append(-0.25)

    if 31 <= sound_number[i] <= 40:
        evidence.append(0)

    if 41 <= sound_number[i] <= 50:
        evidence.append(0.25)

    if 51 <= sound_number[i] <= 60:
        evidence.append(0.5)

    if 61 <= sound_number[i] <= 70:
        evidence.append(1)

# Compute psychometric curve
psych_curve = compute_psych_curve(evidence, choice)

# Plot horizontal and vertical lines
plt.axhline(0.5, color='tab:gray', ls='--')
plt.axvline(0., color='tab:gray', ls='--')

# Plot psychometric curves and errorbars
plt.plot(np.linspace(-1, 1, 30), psych_curve.fit, color='tab:orange', label='Alexis agent StimuliApp')
plt.errorbar(psych_curve.xdata, psych_curve.ydata, yerr=psych_curve.fit_error,
             color='tab:orange', fmt='o', markerfacecolor='none')

plt.title('Psychometric curves \n(' + str(len(trials)) + ' trials)')
# plt.xlabel('Evidence')
plt.xlabel('Interaural Level Difference (dB)')
# plt.xticks(target_evidences, target_ilds)
plt.xticks(np.unique(evidence), ['-40', '-8', '-4', '0', '-4', '-8', '-40'])
plt.ylabel('Probability choose right')
plt.legend(loc="lower right", frameon=False)
# plt.spines['top'].set_visible(False)
# plt.spines['right'].set_visible(False)