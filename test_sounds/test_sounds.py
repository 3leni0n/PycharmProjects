import time
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

"""
To do:
Can you plot the mean and std dev of the stim as a function of frame position for each stim evidence (diff plots)?
"""

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
# if zscore:
#     frames_ild = pd.DataFrame(stats.zscore(frames_ild, axis=None))  # Z-score the ILDs (along axis 0 or None
# returns same result, but not axis 1)
frames_ild.insert(0, column='filename', value=sounds.filename)  # Insert filenames in first column

# Split sounds set by ilds
ilds = sounds.ILD.unique()
sounds_split = np.split(sounds.iloc[1:-1, 1:], len(ilds) - 2)  # [1:-1] to skip the first and last sound

EL_mean = []
EL_std = []
ER_mean = []
ER_std = []
ILD_mean = []

for i in range(len(sounds_split)):
    EL_mean.append(sounds_split[i].loc[:, 'EL0':'EL9'].mean().mean())
    EL_std.append(sounds_split[i].loc[:, 'EL0':'EL9'].std().mean())
    ER_mean.append(sounds_split[i].loc[:, 'ER0':'ER9'].mean().mean())
    ER_std.append(sounds_split[i].loc[:, 'ER0':'ER9'].std().mean())
    ILD_mean.append(ER_mean[i] - EL_mean[i])

columns = ['EL_mean', 'EL_std', 'ER_mean', 'ER_std', 'ILD_mean']
data = list(zip(EL_mean, EL_std, ER_mean, ER_std, ILD_mean))
df = pd.DataFrame(data=data, columns=columns)
df.to_csv('/home/alexis/Escritorio/sounds_stats_test.csv')


########################################################################################################################


def sounds_per_ild(experiment='2AFC_2', animal=None, save=True):
    """
    Plot a histogram of the sounds per ILD
    :param experiment: Batch of animals, needed to specify where the root folder with the data is
    :param animal: Mouse ID number
    :param save: If True, saves the plot
    :return:
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

    # Load behavioral data
    df = pd.read_csv(folder_in)  # Load behavioral data
    df = df[df.Choice.notna()]  # Drop misses (nan in choices), otherwise the code crashes
    ilds = np.sort(df.ILD.unique()).astype('int')
    ilds = np.flip(np.unique(abs(ilds))).astype('int')
    trials_max_evi = len(df[(df.ILD == -70) | (df.ILD == 70)])
    trials_non_max_evi = len(df) - trials_max_evi
    assert trials_max_evi + trials_non_max_evi == len(df)

    # Load sounds
    # sounds_path = '/home/alexis/PycharmProjects/create_sounds/sounds.csv'
    sounds_path = '/home/alexis/PycharmProjects/create_sounds/sounds_2.csv'
    sounds = pd.read_csv(sounds_path)
    sounds_per_ild = (len(sounds) - 2) / (len(sounds.ILD.unique()) - 2)

    # Create figure with subplots
    fig, axs = plt.subplots(len(ilds) - 1, 1, figsize=(21.69, 8.27))  # A4 size in inches landscape. -1 to skip +-70 dB

    for j in range(len(ilds) - 1):  # -1 to skip max evidence (+-70 dB ILD)

        ild = ilds[j + 1]  # + 1 to start in the first non max evidence (+-8 dB ILD)
        trials_ild_total = len(df[(df.ILD == -abs(ild)) | (df.ILD == abs(ild))])  # Total trials per ILD (left + right)

        for i in range(2):

            # If first iteration left, (else) second right
            if i == 0:
                ild = -ild
                color = 'tab:blue'
                label = 'Left'
                print(f'iteration {j}: left')

            else:
                ild = abs(ild)
                color = 'tab:orange'
                label = 'Right'
                print(f'iteration {j}: right')

            # If last iteration (evidence 0) gray color
            if j == 3:
                color = 'tab:gray'

            print(ild)

            # Index filenames by ILD
            filenames = df.Filename[df.ILD == ild].sort_values()  # So xticklabels (filename) are in order

            if i == 0:
                sounds_played_left = len(np.unique(filenames))
                sounds_not_played_left = sounds_per_ild - sounds_played_left
                print('sounds never left', len(np.unique(filenames)))
                n_sounds_left = len(df[df.ILD == ild])
                if j == 3:
                    n_sounds_left = len(df[(df.ILD == ild) & (df.Side == i)])
            else:
                sounds_played_right = len(np.unique(filenames))
                sounds_not_played_right = sounds_per_ild - sounds_played_right
                print('sounds never right', len(np.unique(filenames)))
                n_sounds_right = len(df[df.ILD == ild])
                if j == 3:
                    n_sounds_right = len(df[(df.ILD == ild) & (df.Side == i)])

            # With plt.hist
            # n, bins, patches = plt.hist(filenames, bins=len(np.unique(filenames)), color=color, label=label)  # n = counts
            axs[j].hist(filenames, bins=len(np.unique(filenames)), color=color, label=label)  # n = counts
            axs[j].set_xticks([])
            axs[j].set_ylabel('Counts')

            axs[j].spines['top'].set_visible(False)
            axs[j].spines['bottom'].set_visible(False)

            # Legend only in first plot
            if j == 0:
                axs[j].legend(loc='upper right')

            # xlabel only in last plot
            elif j == 3:
                axs[j].set_xlabel('Filenames')
                axs[j].spines['bottom'].set_visible(True)

            # # With np.unique + plt.bar
            # values, counts = np.unique(filenames, return_counts=True)  # counts = n
            # plt.bar(values, counts, align='center', width=1, color=color, label=label)
            # plt.gca().set_xticks(values)
            # plt.xticks(ticks=plt.gca().get_xticks(), labels=[])

        # Need to be out of side (left/right) loop
        sounds_played = sounds_played_left + sounds_played_right
        # axs[j].set_title(f'ILD = ±{int(abs(ild))}, {trials_ild_total} trials ({n_sounds_left} L, {n_sounds_right} R), '
        #                  f'never played L / R = {int(sounds_not_played_left / sounds_per_ild * 100)}% / '
        #                  f'{int(sounds_not_played_right / sounds_per_ild * 100)}%')

        axs[j].set_title(f'ILD = ±{int(abs(ild))}, {trials_ild_total} trials ({n_sounds_left} L, {n_sounds_right} R), '
                         f'{int((sounds_played / (sounds_per_ild * 2)) * 100)}% sounds played '
                         f'({int(sounds_played_left / sounds_per_ild * 100)}% L, {int(sounds_played_right / sounds_per_ild * 100)}% R)')

        # Instantiate a second axes that shares the same x-axis
        axs_twin = axs[j].twinx()
        axs_twin.set_ylim(axs[j].get_ylim())
        axs_twin.spines['top'].set_visible(False)
        axs_twin.spines['bottom'].set_visible(False)

        assert n_sounds_left + n_sounds_right == trials_ild_total

    fig.suptitle(f'Animal {animal}: ILDs = ±{ilds[1:]}, {trials_non_max_evi} of {len(df)} trials')
    filename = '_test_sounds.png'

    if save:
        folder_out = '/home/alexis/Documentos/test_sounds/'
        if not os.path.exists(folder_out):
            os.mkdir(folder_out)
        os.chdir(folder_out)
        fig.savefig(folder_out + animal + filename)
        plt.close()

    time_end = time.time()
    runtime = time_end - time_start
    print('The script took', round(runtime, 2), 'seconds to run')


sounds_per_ild(experiment='2AFC_2', animal='325', save=True)
sounds_per_ild(experiment='2AFC_2', animal='327', save=True)
sounds_per_ild(experiment='2AFC_2', animal='329', save=True)
sounds_per_ild(experiment='2AFC_2', animal='330', save=True)
sounds_per_ild(experiment='2AFC_2', animal='332', save=True)
sounds_per_ild(experiment='2AFC_2', animal='333', save=True)
sounds_per_ild(experiment='2AFC_2', animal='335', save=True)
sounds_per_ild(experiment='2AFC_2', animal='337', save=True)