# Standard libraries
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import zscore, sem
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt

# Plotting parameters
import seaborn as sns
sns.set_theme()
sns.set_style('ticks')
# sns.set_context('poster')
sns.despine()
import warnings

warnings.filterwarnings('ignore')

# Ephys specific libraries
from quantities import ms, s, Hz
from neo.core import SpikeTrain
from elephant.statistics import time_histogram, instantaneous_rate, fanofactor, mean_firing_rate
from elephant.kernels import GaussianKernel
from elephant.conversion import BinnedSpikeTrain
from elephant.spike_train_correlation import cross_correlation_histogram

# My own libraries
from parse.parse_v2 import parse_v2
from ephys.preprocessing import *

########################################################################################################################

# Run preprocessing

# Define the session ID and directory
id = '007_2024-06-23_12-46-55'
# id = '007_2024-06-24_17-47-22'
# directory = Path.home() / 'Documents' / 'Open Ephys' / id  # Ephys PC
directory = Path() / 'D:' / 'Data' / id  # Personal laptop

# Load raw Open Ephys data
continuous, events = load_oe_data(directory, sync=True, stream='AP')
sample_rate = continuous.metadata['sample_rate']

# Get TTLs from continuous or/and event data
df_ttl = get_ttls(continuous, events)

# Get the sound filenames and sound orders from TTLs
df_keys = decode_ttls(df_ttl)

# Load behavior data
path_behavior = r"D:\Data\007_2024-06-23_12-46-55\007_stage_training_v5_20240623-130152\007_stage_training_v5_20240623-130152.csv"
# path_behavior = r"D:\Data\007_2024-06-24_17-47-22\007_stage_training_v5_20240624-180217\007_stage_training_v5_20240624-180217.csv"
df_behavior = parse_v2(path_behavior)

# Check if the behavior and ephys data match and get the number of trials common to both
n_trials, sounds_mismatch_index = check_data(df_behavior, df_keys)

# Load spike sorted data (KS4)
path_ks4 = Path() / directory / 'Record Node 101' / 'experiment1' / 'recording1' / 'continuous' / \
           'Neuropix-PXI-109.ProbeA-AP' / 'kilosort4'
path_phy2 = Path() / directory / 'Record Node 101' / 'experiment1' / 'recording1' / 'continuous' / \
            'Neuropix-PXI-109.ProbeA-AP' / 'Phy2'
df_spikes = load_spike_sorted_data(path_ks4, path_phy2, sample_rate)

# Sort clusters by the number of spikes
df_clusters = sort_clusters(df_spikes)
clusters = df_clusters.cluster.unique()
n_clusters = len(df_clusters.cluster.unique())

# Print session info
print_session_info(continuous, events, df_behavior, df_spikes)

# Temporal alignment of ephys and behavior data (skip for now)
# df_aligned, df_spikes = temp_align(df_ttl, df_behavior, df_spikes)

# Temporal alignment of ephys and behavior data (from continuous data)
df_ttl = df_ttl[df_ttl['key'] == 'play']  # Keep only rows with key == play (1 TTL per trial)
df_ttl['Trial'] = np.arange(len(df_ttl))  # Prepare a column with trial indexes for merging
df_ttl = df_ttl.iloc[:len(df_behavior)]  # Keep only the first n TTLs (n = number of trials in behavior data)
df_ttl.reset_index(drop=True, inplace=True)  # Reset index
assert len(df_ttl) == len(df_behavior), 'Number of stimulus onset TTLs and trials in behavior data do not match'


########################################################################################################################

# Define helper functions

def get_trial_indexes(df_behavior, condition='outcome'):
    """
    Get trial indexes given a condition. Skip misses by default (hit and choice are nan in misses).
    :param df_behavior: DataFrame with behavior data
    :param condition: Condition to get the trial indexes (default: 'outcome')
    :return: Indexes of trials for the given condition
    """

    # Get the trial indexes of condition
    if condition == 'outcome':  # Already excludes misses
        indexes0 = df_behavior[df_behavior.Hit == 0].Trial.values  # Error
        indexes1 = df_behavior[df_behavior.Hit == 1].Trial.values  # Correct

    elif condition == 'choice':  # Already excludes misses
        indexes0 = df_behavior[df_behavior.Choice == 0].Trial.values  # Choice left
        indexes1 = df_behavior[df_behavior.Choice == 1].Trial.values  # Choice right

    elif condition == 'stimulus':  # Need exclude misses manually
        indexes0 = df_behavior[(df_behavior.Side == 0) & (df_behavior.Miss == 0)].Trial.values  # Stimulus left
        indexes1 = df_behavior[(df_behavior.Side == 1) & (df_behavior.Miss == 0)].Trial.values  # Stimulus right

    # Store indexes in a list
    indexes0 = indexes0.tolist()
    indexes1 = indexes1.tolist()
    indexes = [indexes0, indexes1]

    return indexes


def get_peri_stim_spikes(df_cluster,  df_ttl, time_win=2, scale=0):
    """
    Get peri-stimulus spikes for a cluster.
    :param df_cluster: DataFrame with spike times of a given cluster
    :param time_win: Time window around the event (in seconds) before and after
    :param df_ttl: DataFrame with TTL events
    :param scale: Scale of the jitter (default: 0; no jitter)
    """

    peri_stim_spikes = []
    # Loop over trials (timestamps of stimulus onset)
    for trial in range(len(df_ttl)):
        # print(f'Trial {trial}')
        jitter = np.random.normal(0, scale)  # Jitter the stimulus onset timestamps
        stim_onset = df_ttl.ON[trial] + jitter  # Get the stimulus onset timestamp
        # Select only spikes within the time window of interest around the event
        spikes_trial = df_cluster[(df_cluster.spike_times > stim_onset - time_win) &
                                  (df_cluster.spike_times < stim_onset + time_win)].spike_times
        spikes_trial = spikes_trial - stim_onset  # Align spikes to the event
        peri_stim_spikes.append(spikes_trial)

    return peri_stim_spikes


# Smoothing
def moving_average(data, window):
    """
    Compute the moving average of a 1D array.
    :param data: 1D array
    :param window: Window size
    :return: Moving average
    """
    return np.convolve(data, np.ones(window), 'same') / window


def convolve_psth(psth, sigma=1):
    """
    Convolve a PSTH with a Gaussian kernel.
    :param psth: PSTH
    :param sigma: Standard deviation of the Gaussian kernel in terms of bin size (default: 1). The Gaussian kernel will
    spread across approximately 3 standard deviations, averaging the values in that range.
    :return: Convolved PSTH
    """
    return gaussian_filter1d(psth, sigma)


########################################################################################################################
# SINGLE UNIT ANALYSES
########################################################################################################################

# Raster plots
def plot_raster(peri_stim_spikes, colors=None, ax=None):
    """
    Plot a raster plot of a given cluster aligned to a specific event. Uses an approach completely independent of
    the behavior data. Loops over the TTL events related to stimuli registered in the ephys
    :param peri_stim_spikes: Spike times of a given cluster (output of get_peri_stim_spikes)
    :param colors: Colors of the raster plot (default: None)
    :param ax: Axes to plot the raster (default: None)
    """

    # If no Axes is provided, create a new one
    if ax is None:
        fig, ax = plt.subplots()
        responded_trials = df_behavior[df_behavior.Response == 1].Trial.values
        peri_stim_spikes = [peri_stim_spikes[_] for _ in responded_trials]  # Only trials with a response

    if colors is None:
        colors = ['k'] * len(peri_stim_spikes)

    ax.eventplot(peri_stim_spikes, lineoffsets=range(len(peri_stim_spikes)), colors=colors, linelengths=3)

    # # Loop over trials (timestamps of stimulus onset)
    # for trial in range(len(peri_stim_spikes)):
    #     # print(f'Trial {trial}')
    #     # Skip misses
    #     # Skip trials with no spikes
    #     spikes_trial = peri_stim_spikes[trial]
    #     if spikes_trial.empty:
    #         # continue  # This solution requires to manually set the ylims to show the total number of trials
    #         ax.plot(0, trial, marker='|', color='none')  # Invisible placeholder for empty trials
    #     else:
    #         # ax.eventplot(spikes_trial, lineoffsets=trial, color='k')
    #         for spike in range(len(spikes_trial)):
    #             ax.plot(spikes_trial.iloc[spike], trial, marker='|', linestyle=None, color=color[trial])

    ax.axvline(0, color='tab:red', label='Stimulus')
    ax.axvline(delay, color='tab:gray', label='Delay')
    ax.axvline(go_cue, color='tab:blue', label='Go cue')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Trial')
    ax.set_title(f'Cluster {cluster} ({group})')
    ax.legend(loc='upper left', frameon=False)


def plot_raster_split(condition='outcome', ax=None):
    """
    Plot a raster plot of a given cluster aligned to a specific event split by condition.
    """

    # If no Axes is provided, create a new one
    if ax is None:
        fig, ax = plt.subplots()

    if condition == 'outcome':
        colors = ['tab:red', 'tab:green']
    elif condition == 'choice':
        colors = ['tab:blue', 'tab:orange']
    elif condition == 'stimulus':
        colors = ['tab:blue', 'tab:orange']

    indexes = get_trial_indexes(df_behavior, condition=condition)
    peri_stim_spikes = []

    for _ in range(len(indexes)):
        peri_stim_spikes.append(
            get_peri_stim_spikes(df_cluster, df_ttl.iloc[indexes[_]].reset_index(drop=True)))

    peri_stim_spikes = peri_stim_spikes[0] + peri_stim_spikes[1]  # Concatenate lists
    colors = [colors[0]] * len(indexes[0]) + [colors[1]] * len(indexes[1])  # Concatenate colors
    plot_raster(peri_stim_spikes, colors=colors, ax=ax)  # Plot raster plot split by condition
    ax.legend().remove()


########################################################################################################################

# Peri-Stimulus Time Histograms (PSTHs)
def plot_psth_elephant(cluster, align='StimStart'):
    """
    Use ELEPHANT library
    Plot a PSTH of a given cluster aligned to a specific event.
    :param cluster: Cluster ID
    :param align: Event to align the PSTH to (default: StimStart)
    """

    df_cluster = df_spikes[df_spikes.cluster == cluster]  # Slice DataFrame of given cluster
    df_cluster['spike_times_aligned'] = df_cluster.spike_times - df_cluster[align]  # Align spike times to the event

    # Get the minimum and maximum spike times of the cluster (instead of the trial)
    # Like that all trials will have the same time window, regardless of the trial length
    t_start = df_cluster.spike_times_aligned.min() * 1000  # In ms
    t_stop = df_cluster.spike_times_aligned.max() * 1000  # In ms

    trials = df_behavior.Trial

    mfr = []
    hist_times = []
    hist_firing = []
    gauss_times = []
    gauss_firing = []

    for trial in trials:
        # Slice DataFrame of given cluster of given trial
        df_cluster_trial = df_cluster[df_cluster.Trial == trial]

        # In ms (much faster to work in ms since the binning and convolution will be in ms)
        times = (df_cluster_trial.spike_times_aligned) * 1000
        # times = (df_cluster_trial.spike_times - df_cluster_trial[align]) * 1000
        # t_start = df_cluster_trial.spike_times.min() * 1000
        # t_stop = df_cluster_trial.spike_times.max() * 1000
        units = ms

        # Compute spike train with Neo (https://neo.readthedocs.io/en/latest/api_reference.html#neo.core.SpikeTrain)
        spiketrain = SpikeTrain(times, t_start=t_start, t_stop=t_stop, units=units)

        # Elephant tutorial: https://elephant.readthedocs.io/en/latest/tutorials/statistics.html
        # Compute mean firing rate with Elephant
        # https://elephant.readthedocs.io/en/latest/reference/_toctree/statistics/elephant.statistics.mean_firing_rate.html#elephant.statistics.mean_firing_rate
        mfr.append(np.round(mean_firing_rate(spiketrain).magnitude * 1000, n_decimals))  # In spikes/s
        # print(f"The mean firing rate of cluster {cluster} spiketrain is", mfr)

        # Compute time histogram with Elephant
        # https://elephant.readthedocs.io/en/latest/reference/_toctree/statistics/elephant.statistics.time_histogram.html#elephant.statistics.time_histogram
        bin_size = 1  # ms
        hist_rate = time_histogram(spiketrain, bin_size * ms, output='rate')
        hist_times.append(
            hist_rate.times.rescale(s).magnitude)  # Convert to seconds and store as a numpy array (not a Quantity)
        hist_firing.append(hist_rate.magnitude.flatten())

        # Compute instantaneous rate with Elephant
        # https://elephant.readthedocs.io/en/latest/reference/_toctree/statistics/elephant.statistics.instantaneous_rate.html#elephant.statistics.instantaneous_rate
        sigma = 30  # In ms (from Suzuki & Gottlieb)
        sampling_period = bin_size * ms
        kernel = GaussianKernel(sigma * ms)
        gauss_rate = instantaneous_rate(spiketrain, sampling_period=sampling_period, kernel=kernel)
        gauss_times.append(
            gauss_rate.times.rescale(s).magnitude)  # Convert to seconds and store as a numpy array (not a Quantity)
        gauss_firing.append(gauss_rate.magnitude.flatten())
        # gauss_firing = gauss_rate.rescale(hist_rate.dimensionality).magnitude.flatten()
        # gauss_firing = conv_firing * 1000  # Convert to spikes/s

    # Create a DataFrame with the firing rates
    df_fr = pd.DataFrame({'mfr': mfr, 'hist_times': hist_times, 'hist_firing': hist_firing, 'gauss_times': gauss_times,
                          'gauss_firing': gauss_firing})

    test_times = pd.DataFrame(gauss_times)
    test_firing = pd.DataFrame(gauss_firing)

    # test_hist_times = pd.DataFrame(hist_times)
    # test_hist_firing = pd.DataFrame(hist_firing)

    # Plot standard error of the mean (sem) of firing rate across trials
    plt.figure()
    plt.plot(test_times.mean(axis=0), test_firing.mean(axis=0), color='k')
    plt.fill_between(test_times.mean(axis=0), test_firing.mean(axis=0) - test_firing.sem(axis=0),
                     test_firing.mean(axis=0) + test_firing.sem(axis=0), color='k', alpha=0.1)
    plt.axvline(0, color='tab:red', label='Stimulus')
    plt.axvline(df_cluster.Delay.unique()[0], color='tab:blue', label='Delay')
    plt.axvline(df_cluster.StimDur.unique()[0] + df_cluster.Delay.unique()[0], color='tab:green', label='Response')
    plt.xlabel('Time (s)')
    plt.ylabel('Trial')
    plt.title(f'PSTH aligned to {align} (cluster {cluster})')
    plt.legend(loc='upper right', frameon=False)

    # Save the figure in Desktop folder called 'psths'
    plt.savefig(Path.home() / 'OneDrive' / 'Escritorio' / 'psths' / f'cluster_{cluster}_aligned_{align}.png')
    plt.close()


def compute_psth(peri_stim_spikes, time_win=2, bin_size=0.1):
    """
    Compute a PSTH of a given cluster aligned to a specific event.
    :param peri_stim_spikes: Spike times of a given cluster (output of get_peri_stim_spikes)
    :param time_win: Time window of interest before and after the event (in seconds)
    :param bin_size: Size of the bins for the PSTH (default: 0.1 s)
    """

    n_bins = int((2 * time_win) / bin_size) + 1
    bins = np.linspace(-time_win, time_win, n_bins)  # linspace is preferred over arange for PSTHs

    psth = []
    # Loop over trials (timestamps of stimulus onset)
    for trial in range(len(peri_stim_spikes)):
        hist, _ = np.histogram(peri_stim_spikes[trial], bins)  # Ignore the bin_edges output
        psth.append(hist)
    psth = np.array(psth)  # Convert to numpy array

    return bins, psth


def compute_psth_shuffles(n_shuffles=1000, scale=2):
    """
    Compute PSTHs for shuffled spike times.
    :param n_shuffles: Number of shuffles (default: 1000)
    :param scale: Scale of the jitter (default: 2)
    """

    psth_shuffles = []
    for _ in range(n_shuffles):
        peri_stim_spikes = get_peri_stim_spikes(df_cluster, df_ttl, time_win, scale=scale)  # Get jittered spikes
        bins, psth = compute_psth(peri_stim_spikes)  # Compute the PSTH of the jittered spikes
        psth = np.mean(psth, axis=0)  # Average across trials
        psth = psth / bin_size  # Convert to spikes/s
        psth_shuffles.append(psth)  # Store the PSTH of the shuffled spikes
    psth_shuffles = np.array(psth_shuffles)  # Convert to numpy array

    return bins, psth_shuffles


def plot_psth(bins, psth, psth_shuffles, bin_size, color=None, label=None, ax=None):
    """
    Plot a PSTH of a given cluster aligned to a specific event.
    :param bins: Bins of the PSTH
    :param psth: Histograms of the PSTH
    :param df_behavior: DataFrame with behavior data
    :param bin_size: Size of the bins when coputing the PSTH (default: 0.1 s)
    :param color: Color of the PSTH (default: None)
    :param label: Label of the PSTH (default: None)
    :param ax: Axes to plot the PSTH (default: None)
    """

    # Stats across trials
    psth_mean = psth.mean(axis=0)
    psth_sem = sem(psth, axis=0)

    # Convert to spikes/s
    psth_mean = psth_mean / bin_size
    psth_sem = psth_sem / bin_size

    # If no Axes is provided, create a new one
    if ax is None:
        fig, ax = plt.subplots()

    if color is None:
        color = 'k'

    # Compute the 95% confidence interval
    if color == 'k':
        # If only one PSTH, compare to null hypothesis (shuffled spikes). Use % as sem scales with N shuffles
        psth_shuffle_mean = psth_shuffles.mean(axis=0)
        ax.plot(bins[:-1], psth_shuffle_mean, color='tab:gray', ls='--')
        bound = np.percentile(psth_shuffles, [2.5, 97.5], axis=0)  # The 95% confidence interval of the shuffles
    else:
        # If multiple PSTHs, compare sem (sem * 1.96 is 95% CI)
        bound = [psth_mean - psth_sem, psth_mean + psth_sem]

    # Plot PSTH
    alpha = 0.1
    ax.plot(bins[:-1], psth_mean, color=color, label=label)
    ax.fill_between(bins[:-1], bound[0], bound[1], color=color, alpha=alpha)
    ax.axvline(0, color='tab:red', label='Stimulus' if label is None else '')
    ax.axvline(delay, color='tab:gray', label='Delay' if label is None else '')
    ax.axvline(go_cue, color='tab:blue', label='Go cue' if label is None else '')
    ax.set_xlabel('Time (s)')
    ax.set_ylim(bottom=0)
    ax.set_ylabel('Firing Rate (spikes/s)')
    ax.set_title(f'PSTH for cluster {cluster} ({group})')
    ax.legend(loc='upper left', frameon=False)

    return ax


def plot_psth_split(condition='outcome', ax=None):
    """
    Plot a PSTH of a given cluster aligned to a specific event split by condition.
    """

    if ax is None:
        fig, ax = plt.subplots()

    indexes = get_trial_indexes(df_behavior, condition=condition)

    if condition == 'outcome':
        color = ['tab:red', 'tab:green']
        labels = ['Error', 'Correct']
    elif condition == 'choice':
        color = ['tab:blue', 'tab:orange']
        labels = ['Choice left', 'Choice right']
    elif condition == 'stimulus':
        color = ['tab:blue', 'tab:orange']
        labels = ['Stimulus left', 'Stimulus right']

    for _ in range(len(indexes)):
        peri_stim_spikes = get_peri_stim_spikes(df_cluster, df_ttl.iloc[indexes[_]].reset_index(drop=True), time_win)
        bins, psth = compute_psth(peri_stim_spikes)
        plot_psth(bins, psth, psth_shuffles, bin_size, color=color[_], label=labels[_], ax=ax)

    ax.set_title(f'Cluster {cluster} ({group})')


########################################################################################################################

# Plot both raster and PSTH
def plot_raster_psth(ax=[None, None]):
    """
    Plot a raster plot and PSTH of a given cluster aligned to a specific event.
    """

    if ax[0] is None and ax[1] is None:
        fig, ax = plt.subplots(2, 1, sharex=True)

    responded_trials = df_behavior[df_behavior.Response == 1].Trial.values
    peri_stim_spikes = get_peri_stim_spikes(df_cluster, df_ttl, time_win)
    peri_stim_spikes = [peri_stim_spikes[_] for _ in responded_trials]  # Only trials with a response

    plot_raster(peri_stim_spikes, colors=['k'] * len(peri_stim_spikes), ax=ax[0])
    ax[0].set_title('')
    ax[0].set_xlabel('')
    ax[0].legend().remove()
    plot_psth(bins, psth, psth_shuffles, bin_size, ax=ax[1])
    ax[1].set_title('')
    plt.suptitle(f'Cluster {cluster} ({group})')
    plt.tight_layout()


def plot_raster_psth_split(condition='outcome', ax=[None, None]):
    """
    Plot a raster and a PSTH split by a condition.
    """

    if ax[0] is None and ax[1] is None:
        fig, ax = plt.subplots(2, 1, sharex=True)

    plot_raster_split(condition=condition, ax=ax[0])
    plot_psth_split(condition=condition, ax=ax[1])
    ax[0].set_title('')
    ax[1].set_title('')
    ax[0].set_xlabel('')
    plt.suptitle(f'Cluster {cluster} ({group})')
    plt.tight_layout()


def cluster_report():
    """
    Plot a raster and PSTH of a given cluster aligned to a specific event.
    """

    default_figsize = plt.rcParams["figure.figsize"]
    default_width = default_figsize[0]
    default_heigth = default_figsize[1]
    fig, ax = plt.subplots(2, 4, figsize=(default_width * 4, default_heigth * 2), height_ratios=[2, 1], sharex=True)
    # fig, ax = plt.subplots(2, 3, figsize=(11.69, 8.27), sharex=True)  # A4 size in inches landscape

    plot_raster_psth(ax=[ax[0, 0], ax[1, 0]])
    plot_raster_psth_split(condition='outcome', ax=[ax[0, 1], ax[1, 1]])
    plot_raster_psth_split(condition='choice', ax=[ax[0, 2], ax[1, 2]])
    plot_raster_psth_split(condition='stimulus', ax=[ax[0, 3], ax[1, 3]])

    # Remove legends
    ax[1, 0].legend().remove()

    # Remove y-labels
    ax[0, 1].set_ylabel('')
    ax[1, 1].set_ylabel('')
    ax[0, 2].set_ylabel('')
    ax[1, 2].set_ylabel('')
    ax[0, 3].set_ylabel('')
    ax[1, 3].set_ylabel('')

    # Set same y-limits for PSTHs
    y_max = np.max([ax[1, 0].get_ylim()[1], ax[1, 1].get_ylim()[1], ax[1, 2].get_ylim()[1], ax[1, 3].get_ylim()[1]])
    ax[1, 0].set_ylim(0, y_max)
    ax[1, 1].set_ylim(0, y_max)
    ax[1, 2].set_ylim(0, y_max)
    ax[1, 3].set_ylim(0, y_max)

    # Set titles
    ax[0, 0].set_title('All')
    ax[0, 1].set_title('Outcome')
    ax[0, 2].set_title('Choice')
    ax[0, 3].set_title('Stimulus')

    plt.tight_layout()

    # Save figure using pathlib in Desktop (Escritorio) inside a folder called
    plt.savefig(Path.home() / 'OneDrive' / 'Escritorio' / 'cluster report' / f'cluster {cluster} .png')
    plt.close()


########################################################################################################################

# Get behavioral events
stim_dur = df_behavior.StimDur.unique()[0]
delay = df_behavior.Delay.unique()[0]
go_cue = stim_dur + delay

# Set parameters
time_win = 2  # Time window of interest before and after the event (in seconds)
bin_size = 0.1  # In seconds


# for cluster in df_clusters[df_clusters.group == 'good'].cluster:
for cluster in df_clusters[df_clusters.group == 'good'].cluster:

    print(f'Cluster {cluster}')

    # Plot a raster and PSTH for a given cluster
    cluster = 881
    df_cluster = df_spikes[df_spikes.cluster == cluster]  # Slice DataFrame of given cluster
    group = df_cluster.group.unique()[0]

    # DEBUGGINGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG
    # df_ttl = get_ttls(continuous, events)
    # df_ttl.ON = df_ttl.ON - first_timestamp
    # df_ttl.OFF = df_ttl.OFF - first_timestamp

    peri_stim_spikes = get_peri_stim_spikes(df_cluster, df_ttl, time_win)
    bins, psth = compute_psth(peri_stim_spikes, time_win, bin_size)
    bins, psth_shuffles = compute_psth_shuffles(n_shuffles=10, scale=2)

    cluster_report()


########################################################################################################################
# POPULATION ACTIVITY ANALYSIS
########################################################################################################################

# PLOT RAW DATA (ft. Umberto Olcese)

# # Plot all clusters PSTH for a limited time window (with Elephant)
#
# win_len = 1 * 60  # In seconds
#
# # Slice DataFrame of given time window
# df_test = df[(df.spike_times > first_event) & (df.spike_times < first_event + win_len)]
#
# # Get the minimum and maximum spike times of the clusters
# times = df_test.spike_times * 1000
# t_start = df_test.spike_times.min() * 1000  # In ms
# t_stop = df_test.spike_times.max() * 1000  # In ms
# units = ms
#
# # Compute spike train with Neo (https://neo.readthedocs.io/en/latest/api_reference.html#neo.core.SpikeTrain)
# spiketrain = SpikeTrain(times, t_start=t_start, t_stop=t_stop, units=units)
#
# # Compute time histogram with Elephant
# # https://elephant.readthedocs.io/en/latest/reference/_toctree/statistics/elephant.statistics.time_histogram.html#elephant.statistics.time_histogram
# bin_size = 500 # ms
# hist_rate = time_histogram(spiketrain, bin_size * ms, output='rate')
# hist_times = hist_rate.times.rescale(s).magnitude  # Convert to seconds and store as a numpy array (not a Quantity)
# hist_firing = hist_rate.magnitude.flatten()
# hist_firing = hist_firing * 1000  # Convert to spikes/s
#
# plt.plot(hist_times, hist_firing)
#
# events_win = df_ttl[(df_ttl.ON > first_event) & (df_ttl.ON < first_event + win_len)].ON
#
# for _ in range(len(events_win)):
#     plt.axvline(events_win[_], color='r')

########################################################################################################################


def plot_pop_raw(df_spikes=df_spikes, slice='trials', bin_size=0.1):
    """
    Plot population activity of all clusters in 2 subplots: raster (above) and PSTH (below).
    Short time window (a few trials/seconds).
    :param df_spikes: DataFrame with spike times and clusters
    :param slice: Slice the data in trials or time (default: trials)
    :param bin_size: Size of the bins for the PSTH (default: 0.1 s)
    """
    stim_dur = df_behavior.StimDur.unique()[0]
    delay = df_behavior.Delay.unique()[0]
    go_cue = stim_dur + delay
    # bin_size = 0.1  # In seconds

    # # Slice time vs slice trials
    # slice = 'trials'  # 'trials' or 'time

    if slice == 'trials':
        # Slice DataFrame of given time window after first event (behavior started)
        win_trials = 977, 980  # Edges of trials to plot
        print(f' Plotting trials: {np.arange(win_trials[0], win_trials[1] - 1)}')
        win_events = df_ttl.ON.iloc[win_trials[0]:win_trials[1]]
        df_slice = df_spikes[
            (df_spikes.spike_times > win_events.iloc[0]) & (df_spikes.spike_times < win_events.iloc[-1] + go_cue)]
        df_clusters = sort_clusters(df_slice)  # Sort clusters by number of spikes
        title = (f"Population activity of {len(df_clusters)} clusters "
                 f"({round(len(df_clusters[df_clusters.group == 'good']) / len(df_clusters) * 100)}% 'good')")
        bins = np.arange(win_events.iloc[0], win_events.iloc[-1] + go_cue, bin_size)
    elif slice == 'time':
        # Slice DataFrame of given time window after first event (behavior started)
        win_time = 1922, 1927  # Edges of time window to plot
        print(f' Plotting time: {win_time}')
        df_slice = df_spikes[(df_spikes.spike_times > win_time[0]) & (df_spikes.spike_times < win_time[1])]
        df_clusters = sort_clusters(df_slice)  # Sort clusters by number of spikes
        title = (f"Population activity of {len(df_clusters)} clusters "
                 f"({round(len(df_clusters[df_clusters.group == 'good']) / len(df_clusters) * 100)}% 'good')")
        bins = np.arange(win_time[0], win_time[1], bin_size)

    plt.figure()

    # plt.figure()
    plt.subplot(211)

    # Plot raster
    # All clusters, few trials horizontally stacked (vs. the usual one cluster, all trials vertically stacked)
    # The y-axis is the cluster ID sorted by number of spikes and the x-axis is the time of the spike
    for _ in range(len(df_clusters)):
        # print(f'Cluster {df_clusters.cluster[_]} ({df_clusters.group[_]}): {df_clusters.n_spikes[_]} spikes')
        spikes = df_slice[df_slice.cluster == df_clusters.cluster[_]].spike_times
        plt.eventplot(spikes, lineoffsets=_, color='k')

    if slice == 'trials':
        # Plot events
        for _ in win_events.index.values:
            plt.axvline(win_events[_], color='tab:red', label='Stimulus')
            plt.axvline(win_events[_] + 1, color='tab:blue', label='Go cue')

    # plt.xlabel('Time (s)')
    # plt.xticks(win_events, [])
    plt.gca().set_xticklabels([])
    plt.ylabel('Cluster')
    # plt.title(f'Raster plot ({win_trials} trials)')

    # Plot PSTH
    # All clusters, few trials horizontally stacked (vs. the usual one cluster, all trials vertically stacked)
    # The y-axis is the firing rate and the x-axis is the time of the spike

    # Make a histogram of the number of spikes per bin
    HIST = []
    BIN_EDGES = []
    for _ in range(len(df_clusters)):
        spikes = df_slice[df_slice.cluster == df_clusters.cluster[_]].spike_times
        hist, bin_edges = np.histogram(spikes, bins)
        HIST.append(hist)
        BIN_EDGES.append(bin_edges)

    # Convert to numpy arrays
    HIST = np.array(HIST)
    BIN_EDGES = np.array(BIN_EDGES)

    HIST = HIST / bin_size  # Convert to spikes/s

    # Average across clusters
    HIST = HIST.mean(axis=0)
    BIN_EDGES = BIN_EDGES.mean(axis=0)

    # plt.figure()
    plt.subplot(212)

    # plt.bar(BIN_EDGES[:-1], HIST, width=bin_size, color='k')
    plt.plot(BIN_EDGES[:-1], HIST, color='k')

    if slice == 'trials':
        # Plot events
        for _ in win_events.index.values:
            plt.axvline(win_events[_], color='tab:red', label='Stimulus')
            plt.axvline(win_events[_] + 1, color='tab:blue', label='Go cue')

    plt.xlabel('Time (s)')
    # plt.xticks(win_events, np.arange(win_trials))
    plt.ylabel('FR (spikes/s')
    # plt.title(f'Raster plot ({win_trials} trials)')
    plt.suptitle(title)


time_window = 2  # In seconds
bin_size = 0.1  # In seconds

# Plot PSTH for all clusters concatenaninting the trials +- 1 s around the stimulus onset
bins = np.arange(-time_window, time_window, bin_size)
# run through good units
# df_good = df_spikes[df_spikes.group == 'good']


# POP_HIST = []
HIST = []
BIN_EDGES = []
# for neuron in df_good.cluster.unique()[:30]:
#     print(neuron)
#     df_neuron = df_good[df_good.cluster == neuron]
#     HIST = []
for _ in range(len(df_ttl)):
    # if df_behavior.Miss[_] == 1:  # Skip missed trials
    #     continue
    # else:
    # print(_)
    spikes = df_spikes[(df_spikes.spike_times > df_ttl.ON[_] - time_window) & (
            df_spikes.spike_times < df_ttl.ON[_] + time_window)].spike_times
    spikes = spikes - df_ttl.ON[_]  # Align to stimulus onset
    hist, bin_edges = np.histogram(spikes, bins)
    HIST.append(hist)
    BIN_EDGES.append(bin_edges)
# POP_HIST.append(np.mean(HIST, axis=0)) # Average across trials (neuron x time)

# POP_HIST = np.array(POP_HIST)/ bin_size
# plt.plot(np.mean(zscore(POP_HIST,1), axis=0))
HIST = np.array(HIST)
BIN_EDGES = np.array(BIN_EDGES)

HIST = np.array(HIST) / bin_size  # Convert to spikes/s

HIST = HIST / len(df_clusters)  # Average across clusters

# HIST = HIST.mean(axis=0)  # Average across clusters
BIN_EDGES = BIN_EDGES.mean(axis=0)  # Average across clusters

# Plot histogram
plt.figure()
# plt.subplot(212)

# plt.bar(BIN_EDGES[:-1], HIST, width=bin_size, color='k')
# plt.plot(BIN_EDGES[:-1], HIST, color='k')
# Plot sem of HIST
plt.plot(BIN_EDGES[:-1], HIST.mean(axis=0), color='k')
plt.fill_between(BIN_EDGES[:-1], HIST.mean(axis=0) - sem(HIST, axis=0), HIST.mean(axis=0) + sem(HIST, axis=0),
                 color='k', alpha=0.1)
# Set yaxis limits to start at 0
plt.ylim(ymin=0)

plt.axvline(0, color='tab:red', label='Stimulus')
plt.axvline(1, color='tab:blue', label='Go cue')
plt.axvline(-0.30, color='tab:gray')
plt.axvline(-1.30, color='tab:gray')
plt.legend(frameon=False)
plt.xlabel('Time (s)')
plt.ylabel('FR (spikes/s')
plt.title(f'Population PSTH around stimulus onset (all clusters, all trials)')

"""
Questions Jaime:

1. Are we certain that the mouse responded to these 10 trials in this particular session? The modulation of licks will 
be more visible than that of the stimulus response.                                                                     DONE

2. Can you look for the last 10 valid trials of the session and see if you see population synchrony? DONE
3. Can you show the same plot for a second session?                                                                     TO DO

4. I think that the stimulus should evoke little population response. I would look more for (1) preparatory activity 
between stim onset and Go cue (port approach). (2) rate modulation associated with licking. For this, you could sort the 
units in the raster not according to their overall firing rate, but to the firing rate computed only between stimulus 
onset and Go cue. This is the way Tiffany did it and you can then see some modulation of the population during this 
delay period:                                                                                                           TO DO

5. In general however, the fluctuations in population activity are huge and comparable with the peaks you obtain for the
stim response or the licking. This is particularly true, when the brain state is synchronized (towards the end of the 
session) when the up-down-like transitions make the population rate fluctuate largely:                                  TO OBSERVE

6. What you suggest about computing the stimulus-triggered average of the pop. instantaneous rate across trials is a 
very good idea. I would only include there, valid trials though (with licking).                                         TO DO


Comments:
1. You may also want to show only 1-3 trials to have better temporal resolution.                                        DONE

2. Very little is observed in the population firing rate (MUA FR) or in the raster. If anything, there is a bit more 
activity during the licking but not in response to Stim onset (and this recording was in Audit Ctx!!).                  TO OBSERVE

3. Ideally, high firing neurons on top of the raster.                                                                   DONE

4. Another question: I thought you had a few hundred clusters per session. HEre I only see around 100 neurons. To see 
the up-down activity, the more clusters (even MUA), the better.                                                         DONE

5. Would be good to plot the licks in a separate plot below, like in the Reato et al paper I shared above.  It is not 
the same correct (6-8 licks) than error choices (1-2 licks). When averaging the pop inst rate across trials, I would do 
it separatly for correct (many licks) and error trials (few licks).                                                     TO DO
"""

# Plot population autocorrelogram
t_start = df_spikes.spike_times.min() * 1000 * ms  # In ms
t_stop = df_spikes.spike_times.max() * 1000 * ms  # In ms
units = ms
spike_train = SpikeTrain(df_spikes.spike_times, t_start=t_start, t_stop=t_stop, units=units)

bin_size = 5
window = 10

binned_spike_train = BinnedSpikeTrain(spike_train, bin_size=bin_size * units)

cc_hist, bins = cross_correlation_histogram(binned_spike_train, binned_spike_train, window=[-window, window])
plt.figure()
plt.plot(bins, cc_hist)
