# Import libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from my_fun.my_fun import find_power_dB_par, ild, my_select_evidence, compute_psych_curve
from create_sounds.create_sounds_v2 import create_sounds_v2

########################################################################################################################

df = create_sounds_v2()  # Simulate sounds dataset
n_trials = 10000
trial_types = [0, 1]  # 0=left, 1=right
trial_list = np.random.choice(trial_types, n_trials).tolist()  # Generate random trial vector of length n_trials
ilds = df.ILD.unique()

# Initialize empty lists for simulated data
sim_sound = []
sim_evidence = []
sim_mean_ild = []
sim_choice = []

for i in range(n_trials):

    evidence = my_select_evidence(trial_list[i], ilds)  # Select evidence
    sample_index = df[df.ILD == evidence].index  # Get indexes of sounds with selected evidence
    sound_index = np.random.choice(sample_index)  # Choose a random sound from sample

    # Append values to list
    sim_sound.append(df.filename.iloc[sound_index])
    sim_evidence.append(df.ILD.iloc[sound_index])
    sim_mean_ild.append(df.iloc[sound_index][2:22].mean())

    if sim_mean_ild[i] < 0:
        sim_choice.append(0)
    else:
        sim_choice.append(1)

plt.hist(sim_evidence, bins=50)


def my_select_evidence(trial_type, evidences, p=None):  # Adapted from UtilsR
# def my_select_evidence(trial_type, evidences, shape='uniform'):  # Adapted from UtilsR
    """
    Reduce the prob of 0 evidence to 1/2 as it is part of both left and right trials. This function would be equivalent
    to repeat each evidence in the array except for 0
    trial_type: int, 0=left, 1=right
    evidences: np.array with all possible evidences
    returns: a randomly selected evidence from the available ones according to trial_type and withdrawn with equal prob
    NOTE, uniform works only without evidence 0
    """

    evidences = np.array(evidences)

    if trial_type == 0:
        available = evidences[evidences <= 0]  # Evidences corresponding to the left
        if p is not None:
            p = p
        # if shape == 'u-shape':
            # p = [((1 / 3) + (1 / 3 / 3)), (1 / 3), ((1 / 3) - (1 / 3 / 3))]
        # elif shape == 'uniform':
            # p = list(np.repeat(1 / len(available), len(available)))
    elif trial_type == 1:
        available = evidences[evidences >= 0]  # Evidences corresponding to the right
        if p is not None:
            # p.reverse()
            p = p[::-1]
         # if shape == 'u-shape':
            # p = [((1 / 3) - (1 / 3 / 3)), (1 / 3), ((1 / 3) + (1 / 3 / 3))]
        # elif shape == 'uniform':
            # p = list(np.repeat(1 / len(available), len(available)))

    if 0 not in evidences:  # just pick one randomly
        # selected_evidence = np.random.choice(available)
        selected_evidence = np.random.choice(available, p=p)
    else:  # find it and set its prob of being taken by np.random.choice by 1/2 of the rest
        zero_loc = np.where(available == 0)[0][0]  # index of 0 in our vector available
        prob = 1 / len(evidences)  # prob of any particular evidence
        prob_vec = np.repeat(prob * 2, len(available))  # Make them all double
        prob_vec[zero_loc] = prob  # Set prob for evidence 0 to 1/2 of the rest so it appears with the same prob to
        # other evidences if added both reward sides

        if p is not None:
            zero_loc = np.where(available == 0)[0][0]
            p_corr = p.copy()  # Very important to copy() the original list, otherwise (p_corr = p) both would be linked
            p_corr[zero_loc] = p_corr[zero_loc] / 2  # Make p0 the half of it
            rest = 1 - sum(p_corr)  # Get the remainder half of p0
            non_zero_loc = np.where(available != 0)[0]  # Find the indexes of non zero evidences in available vector
            for i in range(len(non_zero_loc)):
                p_corr[non_zero_loc[i]] = p_corr[non_zero_loc[i]] + rest / len(non_zero_loc)  # Sum the other half of p0 to the p of the non zero
                # evidences (so the whole sums 1)
            selected_evidence = np.random.choice(available, p=p_corr)
        else:
            selected_evidence = np.random.choice(available, p=prob_vec)

    return selected_evidence  # UtilsR one returns coherence