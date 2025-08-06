import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from networkx.classes import edges

from ephys.analysis import get_peri_stim_licks

# df_behavior = pd.read_csv(r'C:\Users\Usuario\PycharmProjects\glue_sessions\2AFC_2\333.csv', low_memory=False)
df_behavior = pd.read_csv(r'C:\Users\Usuario\PycharmProjects\glue_sessions\2AFC_5\007.csv', low_memory=False)

# Filters for groups 1-3
# df_behavior = df_behavior[df_behavior.P > 0].reset_index(drop=True)

# Filters for groups 4-5 (otherwise bump in lick rate before stimulus onset)
df_behavior = df_behavior[df_behavior.Task == 'FD'].reset_index(drop=True)
df_behavior = df_behavior[df_behavior.Delay == 0.5].reset_index(drop=True)


# Define lick functions
def curate_licks(licks: pd.Series, df_behavior: pd.DataFrame) -> pd.Series:
    """
    Curate licks for a given behavioral session (remove licks before Response Window opens or after ITI ends).
    Note that timestamps (in seconds) of licks and events are relative to the trial onset (0 s).
    :return: Series with curated licks
    """

    premature_lick_trials = []  # List of indices of corrupted trials where licks happened out of Response Window
    tolerance = 0.0  # In seconds, tolerance for licks outside the response window (delay of motor Arduino code)
    time_window = 1  # Time window to consider for licks (in seconds). This will result N licks = lick rate (Hz)

    for trial in range(len(licks)):
        resp_win_start = df_behavior.RespWinStart.iloc[trial]
        resp_win_end = df_behavior.RespWinEnd.iloc[trial]

        # Check if any licks before response window start to detect trials in which the motor was stuck
        # (do not include in the condition licks after response window end, as these depend on the time the lickport was
        # available (which varies depending on trial outcome)
        if any((lick < resp_win_start - tolerance) for lick in licks.iloc[trial]):
            premature_lick_trials.append(trial)

        licks[trial] = [lick for lick in licks.iloc[trial]
                        if resp_win_start - tolerance <= lick <= resp_win_end + time_window + tolerance]

    return licks, premature_lick_trials


def get_peri_stim_licks(df_behavior, event='StimStart'):
    """
    Get peri-stimulus licks for a given behavioral session
    :param df_behavior: DataFrame with behavior data
    :return: pd.Series with a list of peri-stimulus licks per trial
    """

    # Convert string representations of lists back to actual lists
    df_behavior["Port1In"] = df_behavior["Port1In"].apply(ast.literal_eval)
    df_behavior["Port2In"] = df_behavior["Port2In"].apply(ast.literal_eval)

    licks_left = df_behavior.Port1In.copy()
    licks_right = df_behavior.Port2In.copy()

    # Curate licks
    licks_left, premature_lick_trials_left = curate_licks(licks_left, df_behavior)
    licks_right, premature_lick_trials_right = curate_licks(licks_right, df_behavior)

    for trial in range(len(df_behavior)):

        # Align licks to the event time
        event_time = df_behavior[event].iloc[trial]
        licks_left[trial] = [x - event_time for x in licks_left.iloc[trial]]  # Left
        licks_right[trial] = [x - event_time for x in licks_right.iloc[trial]]  # Right

    licks = [licks_left] + [licks_right]
    premature_lick_trials = [premature_lick_trials_left] + [premature_lick_trials_right]

    return licks, premature_lick_trials


def compute_psth(peri_stim_licks, time_win=[-1, 3], bin_size=0.1):
    """
    Compute a PSTH of a given cluster aligned to a specific event.
    :param peri_stim_spikes: Lick times of a given subject (output of get_peri_stim_licks)
    :param time_win: List with the time window around the event (default: [-1, 3])
    :param bin_size: Size of the bins for the PSTH (default: 0.1 s)
    """

    n_bins = int((time_win[1] - time_win[0]) / bin_size) + 1
    bins = np.linspace(time_win[0], time_win[1], n_bins)  # linspace is preferred over arange for PSTHs

    psth = []
    # Loop over trials (timestamps of stimulus onset)
    for trial in range(len(peri_stim_licks)):
        hist, _ = np.histogram(peri_stim_licks.iloc[trial], bins)  # Ignore the bin_edges output
        psth.append(hist)
    psth = np.array(psth)  # Convert to numpy array

    return bins, psth


def inter_lick_interval(licks: pd.Series) -> list:
    """
    Compute the inter-lick interval (ILI) of the licks of a behavioral session.
    :return: Inter-lick interval (ILI) of the licks per trial
    """

    ili = []

    for trial in range(len(licks)):
        licks_trial = (licks.iloc[trial])
        licks_trial.sort()  # Sort licks in ascending order
        n_licks = len(licks_trial)
        if n_licks < 2:  # Minimum of 2 licks required to compute ILI
            ili.append(np.nan)
        else:
            # intervals = np.diff(licks_trial)
            # ili.append(np.mean(intervals))
            ili.append(np.diff(licks_trial))

    return ili


def get_rt(df_behavior):
    """
    Compute the reaction time (RT) of a behavioral session from event data.
    :param df_behavior: DataFrame with the behavioral data
    :return: Reaction time (RT) of the licks per trial
    """

    rt = []

    for trial in range(len(df_behavior)):

        if df_behavior.Miss.iloc[trial] == 1:
            rt.append(np.nan)
        else:
            rt.append(df_behavior.RespWinLen.iloc[trial])

    return rt


def get_rt2(licks, df_behavior):
    """
    Compute the reaction time (RT) of a behavioral session from lick data.
    :param df_behavior: DataFrame with the behavioral data
    :return: Reaction time (RT) of the licks per trial
    """

    rt2 = []

    # Combine left and right licks into a single Series
    licks = pd.Series([left + right for left, right in zip(licks[0], licks[1])])

    for trial in range(len(licks)):

        trial_licks = licks.iloc[trial]
        trial_licks.sort()  # Sort licks in ascending order

        if not trial_licks or df_behavior.Miss.iloc[trial] == 1:  # If no licks in the trial
            rt2.append(np.nan)
        else:
            # Align response window start to stimulus onset
            stim_start = df_behavior.StimStart.iloc[trial]
            resp_win_start = df_behavior.RespWinStart.iloc[trial]
            resp_win_start_aligned = resp_win_start - stim_start

            # Find the first lick within the response window
            first_lick = min(trial_licks)
            first_lick_aligned = first_lick - resp_win_start_aligned

            if first_lick_aligned < 0:  # If the first lick is before the response window
                rt2.append(np.nan)
            else:
                rt2.append(first_lick_aligned)  # Use the first lick as RT

    return rt2


def add_lick_data(df_behavior: pd.DataFrame) -> pd.DataFrame:
    """
    Main function to compute licks and reaction times for a given behavioral session.
    :param df_behavior: DataFrame with the behavioral data of a session
    :return: DataFrame with additional columns for licks and reaction times
    """

    # Apply lick functions
    licks, premature_lick_trials = get_peri_stim_licks(df_behavior, event='StimStart')

    # Combine left and right corrupted trials
    premature_lick_trials = sorted(set(premature_lick_trials[0] + premature_lick_trials[1]))

    # Print % corrupted trials
    try:
        print(f'{len(premature_lick_trials) / len(df_behavior) * 100:.2f}% of trials corrupted due to licks outside response window')
    except ZeroDivisionError as e:
        print('0% of trials corrupted due to licks outside response window')

    # # Drop corrupted trials from DataFrame
    # df_behavior = df_behavior.drop(index=premature_lick_trials).reset_index(drop=True)

    licks_left = licks[0]
    licks_right = licks[1]
    n_licks_left = [len(lick) for lick in licks_left]
    n_licks_right = [len(lick) for lick in licks_right]
    ili_left = inter_lick_interval(licks_left)
    ili_right = inter_lick_interval(licks_right)
    rt = get_rt(df_behavior)

    # Add lick vars to DataFrame
    df_behavior['LicksLeft'] = licks[0]
    df_behavior['LicksRight'] = licks[1]
    df_behavior['nLicksLeft'] = n_licks_left
    df_behavior['nLicksRight'] = n_licks_right
    df_behavior['ILI_Left'] = ili_left
    df_behavior['ILI_Right'] = ili_right
    df_behavior['RT'] = rt

    return df_behavior


########################################################################################################################

def plot_licks_psth(bins, left_psth, right_psth):
    # Plot PSTH
    plt.figure(constrained_layout=True)
    plt.plot(bins, left_psth.mean(axis=0), label='Left licks', color='tab:blue')
    plt.plot(bins, right_psth.mean(axis=0), label='Right licks', color='tab:orange')
    plt.axvline(0, color='black', linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel('Lick Rate')
    plt.title('Peri-Stimulus Lick Histogram')
    plt.legend(frameon=False)
    sns.despine()


def plot_rts(rt):
    plt.figure(constrained_layout=True)
    # plt.hist(rt2, bins=1000, density=False, label='RT2', color='tab:orange', edgecolor='none', alpha=0.5)
    plt.hist(rt, bins=1000, density=False, label='RT', color='tab:blue', edgecolor='none', alpha=0.5)
    plt.title(f'Reaction Time Histogram ({df_behavior.Subject.unique()[0]}, N={len(df_behavior)})')
    plt.xlabel('Reaction Time (s)')
    # plt.ylabel('Frequency')
    sns.despine()
