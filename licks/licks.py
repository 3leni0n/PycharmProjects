import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from ephys.analysis import get_peri_stim_licks

# df_behavior = pd.read_csv(r'C:\Users\Usuario\PycharmProjects\glue_sessions\2AFC_2\333.csv', low_memory=False)
df_behavior = pd.read_csv(r'C:\Users\Usuario\PycharmProjects\glue_sessions\2AFC_5\007.csv', low_memory=False)

# Filters for groups 1-3
# df_behavior = df_behavior[df_behavior.P > 0].reset_index(drop=True)

# Filters for groups 4-5 (otherwise bump in lick rate before stimulus onset)
df_behavior = df_behavior[df_behavior.Task == 'FD'].reset_index(drop=True)
df_behavior = df_behavior[df_behavior.Delay == 0.5].reset_index(drop=True)

# Convert string representations of lists back to actual lists
df_behavior["Port1In"] = df_behavior["Port1In"].apply(ast.literal_eval)
df_behavior["Port2In"] = df_behavior["Port2In"].apply(ast.literal_eval)


# Define lick functions
def curate_licks(licks: pd.Series, df_behavior: pd.DataFrame) -> pd.Series:
    """
    Curate licks for a given behavioral session (remove licks before Response Window opens or after ITI ends).
    :return: Series with curated licks
    """

    corrupted_trials = []  # List of indices of corrupted trials where licks happened out of Response Window
    tolerance = 0.1  # In seconds, tolerance for licks outside the response window (delay of motor Arduino code)

    for trial in range(len(licks)):
        resp_win_start = df_behavior.RespWinStart.iloc[trial]
        resp_win_end = df_behavior.RespWinEnd.iloc[trial]
        iti = df_behavior.ITI.iloc[trial]

        # Check if any licks are outside valid window
        if any((lick < resp_win_start - tolerance or lick > resp_win_end + iti + tolerance) for lick in licks.iloc[trial]):
            corrupted_trials.append(trial)

        # n_licks = len(licks.iloc[trial])  # Before curation
        licks[trial] = [lick for lick in licks.iloc[trial] if resp_win_start - tolerance <= lick <= resp_win_end + iti + tolerance]

        # # After curation, compare the number of licks before and after
        # if len(licks.iloc[trial]) != n_licks:
        #     print(f'Trial {trial} licks curated: {n_licks} -> {len(licks.iloc[trial])}')
        #     corrupted_trials.append(trial)

    return licks, corrupted_trials


def get_peri_stim_licks(df_behavior, event='StimStart'):
    """
    Get peri-stimulus licks for a given behavioral session
    :param df_behavior: DataFrame with behavior data
    :return: pd.Series with a list of peri-stimulus licks per trial
    """

    licks_left = df_behavior.Port1In.copy()
    licks_right = df_behavior.Port2In.copy()

    # Curate licks
    licks_left, corrupted_trials_left = curate_licks(licks_left, df_behavior)
    licks_right, corrupted_trials_right = curate_licks(licks_right, df_behavior)

    for trial in range(len(df_behavior)):

        # Align licks to the event time
        event_time = df_behavior[event].iloc[trial]
        licks_left[trial] = [x - event_time for x in licks_left.iloc[trial]]  # Left
        licks_right[trial] = [x - event_time for x in licks_right.iloc[trial]]  # Right

    licks = [licks_left] + [licks_right]
    corrupted_trials = [corrupted_trials_left] + [corrupted_trials_right]

    return licks, corrupted_trials


def compute_psth(peri_stim_licks, time_win=[-1, 3], bin_size=0.1):
    """
    Compute a PSTH of a given cluster aligned to a specific event.
    :param peri_stim_spikes: Spike times of a given cluster (output of get_peri_stim_spikes)
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
            intervals = np.diff(licks_trial)
            ili.append(np.mean(intervals))

    return ili


def get_rt(df_behavior):
    """
    Compute the reaction time (RT) of the licks of a behavioral session.
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


def get_rt2(licks):
    """
    Compute the reaction time (RT) of the licks of a behavioral session.
    :param df_behavior: DataFrame with the behavioral data
    :return: Reaction time (RT) of the licks per trial
    """

    rt2 = []

    # Combine left and right licks into a single Series
    licks = pd.Series([left + right for left, right in zip(licks[0], licks[1])])

    for trial in range(len(licks)):

        trial_licks = licks.iloc[trial]
        trial_licks.sort()  # Sort licks in ascending order

        if not trial_licks:  # If no licks in the trial
            rt2.append(np.nan)
        else:
            rt2.append(min(trial_licks))  # Use the first lick as RT

    return rt2


# Apply lick functions
licks, corrupted_trials = get_peri_stim_licks(df_behavior, event='StimStart')

# Combine left and right corrupted trials
corrupted_trials = sorted(set(corrupted_trials[0] + corrupted_trials[1]))

# Print % corrupted trials
try:
    print(f'{len(corrupted_trials) / len(df_behavior) * 100:.2f}% of trials corrupted due to licks outside response window')
except ZeroDivisionError as e:
    print('0% of trials corrupted due to licks outside response window')

# # Drop corrupted trials from DataFrame
# df_behavior = df_behavior.drop(index=corrupted_trials).reset_index(drop=True)

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

bins, left_psth = compute_psth(licks_left, time_win=[-1, 3], bin_size=0.1)
bins, right_psth = compute_psth(licks_right, time_win=[-1, 3], bin_size=0.1)
bins = bins[:-1]  # Remove the last bin edge to match the histogram length

def plot_licks_psth():
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

plot_licks_psth()



rt2 = get_rt2(licks)
# Check minimum and maximum RT
print(f'Minimum RT: {np.nanmin(rt2):.3f} s | Maximum RT: {np.nanmax(rt2):.3f} s')

# Count number of trials with RT < 1 s
n_trials_fast_rt = sum(np.array(rt2) < 1)
print(f'Number of trials with RT < 1 s: {n_trials_fast_rt}')


# Plot hist of rt and rt 2 to compare
plt.figure(constrained_layout=True)
plt.hist(rt)
plt.hist(rt2)