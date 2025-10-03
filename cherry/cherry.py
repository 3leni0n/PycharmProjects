import numpy as np
from matplotlib import pyplot as plt

from glue_sessions import glue_groups
from psychometric_curves import plot_pc


def check_valid_trials(df):
    """
    Check number of good trials per subject (responded & P > 0)
    :param df: DataFrame with behavior data
    :return: Dictionary with number of good trials per subject
    """
    valid_trials_subject = {}
    subjects = df.Subject.unique()
    print('Number of responded trials in data collection (with evidences):')
    for s in subjects:
        subdf = df[(df.Subject == int(s)) & (df.P > 0) & (df.Miss == 0)]
        n_trials = len(subdf)
        valid_trials_subject[s] = n_trials
        print(f'{s}: {n_trials} trials')
    print('\n')
    return valid_trials_subject


def find_left_behind(valid_trials_subject, threshold=1000):
    """Return subjects with less than threshold good trials
    :param good_trials_per_subject: Dictionary with number of good trials per subject
    :param threshold: Minimum number of good trials"""

    print('Subjects left behind (never learnt):')
    left_behind = []
    for s, n in valid_trials_subject.items():
        if n < threshold:
            print(f'{s}: {n} trials')
            left_behind.append(s)
    print('\n')
    return left_behind


def find_bad_subjects(psych_curves, max_lapse=2/3):
    """
    Find bad subjects based on psychometric performance (lapse rates)
    :param psych_curves: Psychometric curve objects
    :param max_lapse: Maximum allowed lapse rate (sum of lower and upper)
    :return: Indices of bad subjects and their lapses
    """

    # Unpack spych curves parmaters
    sensitivity = []
    bias = []
    lr_lower = []
    lr_upper = []

    for psych_curve in psych_curves:
        sensitivity_subject, bias_subject, lr_lower_subject, lr_upper_subject = psych_curve.params
        sensitivity.append(sensitivity_subject)
        bias.append(bias_subject)
        lr_lower.append(lr_lower_subject)
        lr_upper.append(lr_upper_subject)

    lr_lower = np.array(lr_lower)
    lr_upper = np.array(lr_upper)

    # Concatenate lr_lower and upper
    lapses = np.vstack((lr_lower, lr_upper)).T
    total_lapses = np.sum(lapses, axis=1)

    # Find indices where either lapse was higher than max_lapse (bas subjects)
    # indices = np.where(np.any(lapses > max_lapse, axis=1))[0]  # For lapses = 1/3
    indices = np.where(total_lapses > max_lapse)[0]  # For lapses = 2/3

    # return indices, lapses
    return indices, total_lapses


def cherry_pick(df_behavior, experiment, plot=False):
    """
    Cherrypick the best subjects for a given experiment (actually drop the bad ones)
    :param experiment: Experiment name ('2AFC_2-6')
    :return: Psychometric curve plots for the good subjects
    """

    # Find bad subjects
    df = df_behavior[df_behavior.Experiment == experiment]
    subjects = df.Subject.unique().astype(list)
    subjects = [str(int(s)) for s in subjects]  # Convert to list of strings of integers
    good_trials_per_subject = check_valid_trials(df)  # Check valid trials per subject
    left_behind = find_left_behind(good_trials_per_subject)  # Find subjects with less than threshold good trials
    left_behind = [str(s) for s in left_behind]  # Convert bad_subjects to str
    left_behind = [float(s) for s in left_behind]  # Transform bad subjects back to floats

    # Remove subjects left behind from df
    df = df[~df.Subject.isin(left_behind)]
    animals = df.Subject.unique().astype(list)
    animals = [str(int(s)) for s in animals]  # Convert to list of strings of integers
    animals = [s.zfill(3) for s in animals]  # Pad with zeros to have 3 digits (needed for group #6)
    # print(f'Remaining subjects: {animals}')

    # Plot psychometric curves
    psych_curves = plot_pc(experiment=experiment, animal=animals, kind='prob_right', drug=None, save=False, format='png',
                           transparent=False)
    plt.close('all')

    # Find bad subjects (returns indices of bad curves)
    # indices, lapses = find_bad_subjects(psych_curves)
    indices, total_lapses = find_bad_subjects(psych_curves)
    # lapses = np.delete(lapses, indices, axis=0)  # Remove bad subjects from lapses
    total_lapses = np.delete(total_lapses, indices, axis=0)  # Remove bad subjects from lapses
    # total_lapses = np.sum(lapses, axis=1)

    # Map indices back to animal IDs
    bad_subjects = [animals[i] for i in indices]
    print(f'Bad subjects (based on lapses): {bad_subjects}\n')

    good_subjects = [animals[i] for i in range(len(animals)) if i not in indices]
    print('Good subjects:')
    for subj, lapse in zip(good_subjects, total_lapses):
        print(f'{subj}: {round(lapse, 2)} lapses')
    print('\n')

    if plot:
        # Plot psychometric curves
        psych_curves = plot_pc(experiment=experiment, animal=good_subjects, kind='prob_right', drug=None, save=False,
                               format='png', transparent=False)

    return good_subjects


def main(experiments=None):
    """
    Main function to cherry-pick subjects from experiments.
    :param experiments: List of experiment names or single experiment name as string.
    If None, defaults to all experiments.
    :return: Dictionary with good subjects per experiment
    """

    if experiments is None:
        experiments = [
            '2AFC_2',
            '2AFC_3',
            '2AFC_4',    # Ephys pilot group (loads of infections). FSM changes (0.15s motor in). 0.5s delay introduced
            # '2AFC_5',    # Ephys group (no evidences)
            '2AFC_6',    # Pharma group. 11th frame with 0 evidence
        ]
    elif isinstance(experiments, str):
        experiments = [experiments]  # Wrap single string into list

    df_behavior = glue_groups(experiments)

    cherries = {}
    for experiment in experiments:
        print(f'Cherry picking for experiment: {experiment}')
        good_subjects = cherry_pick(df_behavior, experiment)
        cherries[experiment] = (good_subjects)

    return cherries