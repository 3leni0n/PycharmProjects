# Standard libraries
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import zscore, sem, poisson
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt

# Plotting parameters
import seaborn as sns
sns.set_theme()
sns.set_style('ticks')
# sns.set_context('poster')
import warnings

warnings.filterwarnings('ignore')

# Ephys specific libraries
from quantities import ms, s
from neo.core import SpikeTrain
from elephant.statistics import time_histogram, instantaneous_rate, fanofactor, mean_firing_rate, isi, cv
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
df_spikes, cluster_info = load_spike_sorted_data(path_ks4, path_phy2)
clusters = cluster_info.cluster_id.unique()
n_clusters = len(cluster_info)

# Print session info
show_session_info(continuous, events, df_behavior, df_spikes, plot=False)

# Temporal alignment of ephys and behavior data (skip for now)
# df_aligned, df_spikes = temp_align(df_ttl, df_behavior, df_spikes)

# Temporal alignment of ephys and behavior data (from continuous data)
df_ttl = df_ttl[df_ttl['key'] == 'play']  # Keep only rows with key == play (stimulus onset, 1 TTL per trial)
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


def get_peri_stim_spikes(df_cluster, df_ttl, time_win=2, scale=0):
    """
    Get peri-stimulus spikes for a cluster.
    :param df_cluster: DataFrame with spike times of a given cluster
    :param time_win: Time window around the event (in seconds) before and after
    :param df_ttl: DataFrame with TTL events
    :param scale: Scale of the jitter (default: 0; no jitter)
    """

    peri_stim_spikes = []
    # Loop over trials (timestamps of stimulus onset)

    # Check if df_ttl is a Series and convert to DataFrame to iterate over single trial
    if isinstance(df_ttl, pd.Series):
        df_ttl = pd.DataFrame([df_ttl])

    for trial in range(len(df_ttl)):
        # print(f'Trial {trial}')
        jitter = np.random.normal(0, scale)  # Jitter the stimulus onset timestamps
        stim_onset = df_ttl.OFF[trial] + jitter  # Get the stimulus onset timestamp
        # Select only spikes within the time window of interest around the event
        spikes_trial = df_cluster[(df_cluster.times > stim_onset - time_win) &
                                  (df_cluster.times < stim_onset + time_win)].times
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

    ax.eventplot(peri_stim_spikes, lineoffsets=range(len(peri_stim_spikes)), colors=colors)

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
    df_cluster['times_aligned'] = df_cluster.times - df_cluster[align]  # Align spike times to the event

    # Get the minimum and maximum spike times of the cluster (instead of the trial)
    # Like that all trials will have the same time window, regardless of the trial length
    t_start = df_cluster.times_aligned.min() * 1000  # In ms
    t_stop = df_cluster.times_aligned.max() * 1000  # In ms

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
        times = (df_cluster_trial.times_aligned) * 1000
        # times = (df_cluster_trial.times - df_cluster_trial[align]) * 1000
        # t_start = df_cluster_trial.times.min() * 1000
        # t_stop = df_cluster_trial.times.max() * 1000
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


def compute_psth_shuffles(df_cluster, n_shuffles=1000, scale=2):
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
    ax.set_ylabel('FR (spikes/s)')
    ax.set_title(f'PSTH for cluster {cluster} ({group})')
    ax.legend(loc='upper left', frameon=False)

    return ax


def plot_psth_split(condition='outcome', ax=None):
    """
    Plot a PSTH of a given cluster aligned to a specific event split by condition.
    """

    if ax is None:
        fig, ax = plt.subplots()
        title = f'Cluster {cluster} ({group})'
    else:
        title = ''

    indexes = get_trial_indexes(df_behavior, condition=condition)

    if condition == 'outcome':
        color = ['tab:red', 'tab:green']
        labels = ['Error', 'Correct']
    elif condition == 'choice':
        color = ['tab:blue', 'tab:orange']
        labels = ['Choice left', 'Choice right'] if ax is None else ['Left', 'Right']
    elif condition == 'stimulus':
        color = ['tab:blue', 'tab:orange']
        labels = ['Stimulus left', 'Stimulus right'] if ax is None else ['Left', 'Right']

    for _ in range(len(indexes)):
        peri_stim_spikes = get_peri_stim_spikes(df_cluster, df_ttl.iloc[indexes[_]].reset_index(drop=True), time_win)
        bins, psth = compute_psth(peri_stim_spikes)
        plot_psth(bins, psth, psth_shuffles, bin_size, color=color[_], label=labels[_], ax=ax)

    ax.set_title(title)


########################################################################################################################

# Plot both raster and PSTH
def plot_raster_psth(ax=[None, None]):
    """
    Plot a raster plot and PSTH of a given cluster aligned to a specific event.
    """

    if ax[0] is None and ax[1] is None:
        fig, ax = plt.subplots(2, 1, sharex=True)
        title = f'Cluster {cluster} ({group})'
    else:
        title = ''

    responded_trials = df_behavior[df_behavior.Response == 1].Trial.values
    peri_stim_spikes = get_peri_stim_spikes(df_cluster, df_ttl, time_win)
    peri_stim_spikes = [peri_stim_spikes[_] for _ in responded_trials]  # Only trials with a response

    plot_raster(peri_stim_spikes, colors=['k'] * len(peri_stim_spikes), ax=ax[0])
    ax[0].set_title('')
    ax[0].set_xlabel('')
    ax[0].legend().remove()
    plot_psth(bins, psth, psth_shuffles, bin_size, ax=ax[1])
    ax[1].set_title('')
    plt.suptitle(title)
    plt.tight_layout()


def plot_raster_psth_split(condition='outcome', ax=[None, None]):
    """
    Plot a raster and a PSTH split by a condition.
    """

    if ax[0] is None and ax[1] is None:
        fig, ax = plt.subplots(2, 1, sharex=True)
        title = f'Cluster {cluster} ({group})'
    else:
        title = ''

    plot_raster_split(condition=condition, ax=ax[0])
    plot_psth_split(condition=condition, ax=ax[1])
    ax[0].set_title('')
    ax[1].set_title('')
    ax[0].set_xlabel('')
    plt.suptitle(title)
    plt.tight_layout()


def cluster_report(save=False):
    """
    Plot a raster and PSTH of a given cluster aligned to a specific event.
    """

    default_figsize = plt.rcParams["figure.figsize"]
    default_width = default_figsize[0]
    default_heigth = default_figsize[1]
    # figsize = (default_width * 4, default_heigth * 2)
    figsize = (11.69, 8.27)
    width_ratios = [1, 1, 1, 1]
    height_ratios = [1, 2, 1]

    # Set subplots layout with mosaic
    mosaic = [['Auto', 'MFR', 'MFR', 'MFR'],  # Autocorrelogram and mean firing rate
              ['RasterAll', 'RasterOutcome', 'RasterChoice', 'RasterStimulus'],  # Rasters
              ['RasterAll', 'RasterOutcome', 'RasterChoice', 'RasterStimulus'],  # Rasters
              ['PSTHAll', 'PSTHOutcome', 'PSTHChoice', 'PSTHStimulus']]  # PSTHs
    fig, ax_dict = plt.subplot_mosaic(mosaic, figsize=figsize)

    # Plot panels
    plot_autocorrelogram(df_cluster, bin_size=0.001, window=[-50, 50], ax=ax_dict['Auto'])
    plot_mfr(psth, ax=ax_dict['MFR'])
    plot_raster_psth(ax=[ax_dict['RasterAll'], ax_dict['PSTHAll']])
    plot_raster_psth_split(condition='outcome', ax=[ax_dict['RasterOutcome'], ax_dict['PSTHOutcome']])
    plot_raster_psth_split(condition='choice', ax=[ax_dict['RasterChoice'], ax_dict['PSTHChoice']])
    plot_raster_psth_split(condition='stimulus', ax=[ax_dict['RasterStimulus'], ax_dict['PSTHStimulus']])
    # isis = plot_isi(peri_stim_spikes, ax=ax[0, 2])
    # plot_cv(isis, ax=ax[0, 3])

    # Remove xticklabels
    ax_dict['RasterAll'].set_xticklabels([])
    ax_dict['RasterOutcome'].set_xticklabels([])
    ax_dict['RasterChoice'].set_xticklabels([])
    ax_dict['RasterStimulus'].set_xticklabels([])

    # Remove yticklabels
    ax_dict['RasterOutcome'].set_yticklabels([])
    ax_dict['RasterChoice'].set_yticklabels([])
    ax_dict['RasterStimulus'].set_yticklabels([])
    ax_dict['PSTHOutcome'].set_yticklabels([])
    ax_dict['PSTHChoice'].set_yticklabels([])
    ax_dict['PSTHStimulus'].set_yticklabels([])

    # Remove legends
    ax_dict['PSTHAll'].legend().remove()

    # Remove ylabels
    ax_dict['RasterOutcome'].set_ylabel('')
    ax_dict['RasterChoice'].set_ylabel('')
    ax_dict['RasterStimulus'].set_ylabel('')
    ax_dict['PSTHOutcome'].set_ylabel('')
    ax_dict['PSTHChoice'].set_ylabel('')
    ax_dict['PSTHStimulus'].set_ylabel('')

    responses = df_behavior[df_behavior.Response == 1].Response.sum()
    ax_dict['RasterAll'].set_ylim(0, responses)
    ax_dict['RasterOutcome'].set_ylim(0, responses)
    ax_dict['RasterChoice'].set_ylim(0, responses)
    ax_dict['RasterStimulus'].set_ylim(0, responses)

    # Set same ylims for PSTHs
    y_max = np.max([ax_dict['PSTHAll'].get_ylim()[1],
                    ax_dict['PSTHOutcome'].get_ylim()[1],
                    ax_dict['PSTHChoice'].get_ylim()[1],
                    ax_dict['PSTHStimulus'].get_ylim()[1]])
    ax_dict['PSTHAll'].set_ylim(0, y_max)
    ax_dict['PSTHOutcome'].set_ylim(0, y_max)
    ax_dict['PSTHChoice'].set_ylim(0, y_max)
    ax_dict['PSTHStimulus'].set_ylim(0, y_max)

    # Add gridlines to the PSTHs
    ax_dict['PSTHAll'].grid(axis='y')
    ax_dict['PSTHOutcome'].grid(axis='y')
    ax_dict['PSTHChoice'].grid(axis='y')
    ax_dict['PSTHStimulus'].grid(axis='y')

    # Remove axes margins
    ax_dict['MFR'].margins(x=0)
    ax_dict['MFR'].margins(y=0)
    ax_dict['RasterAll'].margins(x=0)
    ax_dict['RasterOutcome'].margins(x=0)
    ax_dict['RasterChoice'].margins(x=0)
    ax_dict['RasterStimulus'].margins(x=0)
    ax_dict['PSTHAll'].margins(x=0)
    ax_dict['PSTHOutcome'].margins(x=0)
    ax_dict['PSTHChoice'].margins(x=0)
    ax_dict['PSTHStimulus'].margins(x=0)

    # Set titles
    ax_dict['RasterAll'].set_title('All')
    ax_dict['RasterOutcome'].set_title('Outcome')
    ax_dict['RasterChoice'].set_title('Choice')
    ax_dict['RasterStimulus'].set_title('Stimulus')

    # Despine axes
    sns.despine(ax=ax_dict['MFR'])
    sns.despine(ax=ax_dict['Auto'])
    sns.despine(ax=ax_dict['RasterAll'], bottom=True)
    sns.despine(ax=ax_dict['RasterOutcome'], left=True, bottom=True)
    sns.despine(ax=ax_dict['RasterChoice'], left=True, bottom=True)
    sns.despine(ax=ax_dict['RasterStimulus'], left=True, bottom=True)
    sns.despine(ax=ax_dict['PSTHAll'])
    sns.despine(ax=ax_dict['PSTHOutcome'], left=True)
    sns.despine(ax=ax_dict['PSTHChoice'], left=True)
    sns.despine(ax=ax_dict['PSTHStimulus'], left=True)

    # Set figure title with cluster info
    isis = plot_isi(spikes=df_cluster, ax=None)
    coeff_var = plot_cv(isis, ax=None)
    fano = fano_factor(peri_stim_spikes)
    fig.suptitle(f'Cluster {cluster} ({group}): '
                 f'\n'
                 f'depth={round(depth/1000, 2)} mm, '
                 f'mean ISI={round(np.mean(isis), 2)}, '
                 f'CV={round(coeff_var, 2)}, '
                 f'Fano factor={round(fano, 2)}')

    fig.tight_layout()

    if save:
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

# for cluster in cluster_info[cluster_info.group == 'good'].cluster_id:

# print(f'Cluster {cluster}')

# Plot a raster and PSTH for a given cluster
cluster = 881
df_cluster = df_spikes[df_spikes.cluster == cluster]  # Slice DataFrame of given cluster
group = cluster_info[cluster_info.cluster_id==cluster].group.iloc[0]
depth = cluster_info[cluster_info.cluster_id==cluster].depth.iloc[0]

peri_stim_spikes = get_peri_stim_spikes(df_cluster, df_ttl, time_win)
bins, psth = compute_psth(peri_stim_spikes, time_win, bin_size)
bins, psth_shuffles = compute_psth_shuffles(df_cluster, n_shuffles=1, scale=2)

cluster_report()


########################################################################################################################
# POPULATION ACTIVITY ANALYSIS
########################################################################################################################


def plot_pop_raw(df_spikes, cluster_info, slice='trials', sort_by='n_spikes', bin_size=0.01):
    """
    Plot population activity of all clusters in 2 subplots: raster (above) and PSTH (below).
    Short time window (a few trials/seconds).
    :param df_spikes: DataFrame with spike times and clusters
    :param slice: Slice the data in trials or time (default: trials)
    :param sort: Sort clusters by attribute of cluster_info (default: n_spikes)
    :param bin_size: Size of the bins for the PSTH (default: 0.1 s)
    """

    # Sort clusters by number of spikes
    cluster_info.sort_values(sort_by, ascending=True).reset_index(drop=True, inplace=True)

    if slice == 'trials':
        # Slice DataFrame of given time window after first event (behavior started)
        win_trials = 647, 650  # Edges of trials to plot
        print(f' Plotting trials: {np.arange(win_trials[0], win_trials[1] - 1)}')
        win_events = df_ttl.OFF.iloc[win_trials[0]:win_trials[1]]
        df_slice = df_spikes[
            (df_spikes.times > win_events.iloc[0]) & (df_spikes.times < win_events.iloc[-1] + go_cue)]
        # cluster_info = df_slice.sort('n_spikes')  # Sort clusters by number of spikes
        title = (f"Population activity of {len(cluster_info)} clusters "
                 f"({round(len(cluster_info[cluster_info.group == 'good']) / len(cluster_info) * 100)}% 'good')")
        bins = np.arange(win_events.iloc[0], win_events.iloc[-1] + go_cue, bin_size)
    elif slice == 'time':
        # Slice DataFrame of given time window after first event (behavior started)
        win_time = 1922, 1927  # Edges of time window to plot
        print(f' Plotting time: {win_time}')
        df_slice = df_spikes[(df_spikes.times > win_time[0]) & (df_spikes.times < win_time[1])]
        # cluster_info = df_slice.sort('n_spikes')  # Sort clusters by number of spikes
        title = (f"Population activity of {len(cluster_info)} clusters "
                 f"({round(len(cluster_info[cluster_info.group == 'good']) / len(cluster_info) * 100)}% 'good')")
        bins = np.arange(win_time[0], win_time[1], bin_size)

    fig, ax = plt.subplots(2, 1, sharex=True)

    # Plot population raster (cluster in y-axis)
    HIST = []
    BIN_EDGES = []
    for _ in range(len(cluster_info)):
        # print(f'Cluster {cluster_info.cluster_id[_]} ({cluster_info.group[_]}): {cluster_info.n_spikes[_]} spikes')
        spikes = df_slice[df_slice.cluster == cluster_info.cluster_id[_]].times
        ax[0].eventplot(spikes, lineoffsets=_, color='k')

        # Make a histogram of the number of spikes per bin
        hist, bin_edges = np.histogram(spikes, bins)
        HIST.append(hist)
        BIN_EDGES.append(bin_edges)

    # Plot population PSTH

    # Convert to numpy arrays
    HIST = np.array(HIST)
    BIN_EDGES = np.array(BIN_EDGES)

    HIST = HIST / bin_size  # Convert to spikes/s

    # Average across clusters
    HIST = HIST.mean(axis=0)
    BIN_EDGES = BIN_EDGES.mean(axis=0)

    # plt.bar(BIN_EDGES[:-1], HIST, width=bin_size, color='k')
    ax[1].plot(BIN_EDGES[:-1], HIST, color='k')

    # Plot events
    if slice == 'trials':
        for i in range(len(ax)):
            for _ in win_events.index.values:
                ax[i].axvline(win_events[_], color='tab:red', label='Stimulus')
                ax[i].axvline(win_events[_] + 0.5, color='tab:gray', label='Stimulus')
                ax[i].axvline(win_events[_] + 1, color='tab:blue', label='Go cue')

    ax[0].set_ylabel('Cluster')
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylim(bottom=0)
    ax[1].set_ylabel('FR (spikes/s)')
    plt.suptitle(title)
    plt.tight_layout()


def plot_pop_psth():
    """
    Plot population PSTH treating all the spikes as coming from a single superneuron.
    """
    peri_stim_spikes = get_peri_stim_spikes(df_spikes, df_ttl, time_win)
    bins, psth = compute_psth(peri_stim_spikes, time_win, bin_size)
    _, psth_shuffles = compute_psth_shuffles(df_spikes, n_shuffles=10, scale=2)
    plot_psth(bins, psth/len(cluster_info), psth_shuffles/len(cluster_info), bin_size)
    plt.title('Population PSTH')


def plot_autocorrelogram(df_cluster, bin_size=0.001, window=[-50, 50], ax=None):
    """
    Plot the autocorrelogram of a given cluster.
    :param df_cluster: DataFrame with spike times of a given cluster
    :param bin_size: Size of the bins in seconds for the autocorrelogram (default: 0.001 s)
    :param window: List of integers representing the left and right extremes (expressed as number of bins) where the
    autocorrelogram is computed. Same units as bin_size.
    """

    if ax is None:
        fig, ax = plt.subplots()
        title = 'Autocorrelogram'
    else:
        title = ''

    times = df_cluster.times
    t_start = times.min()
    t_stop = times.max()
    units = s

    spike_train = SpikeTrain(times, t_start=t_start, t_stop=t_stop, units=units)

    bin_size = bin_size * units
    # window = [-50, 50]

    binned_spike_train = BinnedSpikeTrain(spike_train, bin_size=bin_size)

    cch, lags = cross_correlation_histogram(binned_spike_train, binned_spike_train, window=window,
                                            cross_correlation_coefficient=True)
    cch, lags = np.delete(cch.magnitude.flatten(), lags == 0), np.delete(lags, lags == 0)

    ax.plot(lags, cch, color='k')
    refractory_period = 2  # In number of bins
    ax.axvline(-refractory_period, color='tab:gray')
    ax.axvline(refractory_period, color='tab:gray')
    ax.set_title(title)
    ax.set_xlabel('Time lag (ms)')
    ax.set_ylabel('Correlation')


def plot_mfr(psth, ax=None):
    """
    Compute the mean firing rate per trial of a PSTH.
    :param psth: PSTH
    :param ax: Axes to plot the mean firing rate (default: None)
    return: Mean and standard error of the mean firing rate
    """

    if ax is None:
        fig, ax = plt.subplots()
        title = 'Mean Firing Rate'
    else:
        title = ''

    mfr = np.mean(psth, axis=1)
    sfr = sem(psth, axis=1)

    # Convert to spikes/s
    mfr = mfr / bin_size
    sfr = sfr / bin_size

    mfr_mean = np.mean(mfr)
    percentile = 20
    mfr_percentile = np.percentile(mfr, percentile)

    ax.plot(mfr, color='k')
    ax.fill_between(np.arange(len(mfr)), mfr - sfr, mfr + sfr, color='k', alpha=0.1)
    ax.axhline(mfr_mean, color='tab:red', label='mean')
    ax.axhline(mfr_percentile, color='tab:gray', linestyle='--', label=f'{percentile}th percentile')
    ax.set_xlabel('Trial')
    ax.set_ylabel('FR (spikes/s)')
    ax.legend(loc='upper right', frameon=False)
    ax.set_title(title)

    return mfr, sfr


def plot_mfr_dist(mfr, ax=None):

    if ax is None:
        fig, ax = plt.subplots()
        title = 'Mean Firing Rate Distribution'
    else:
        title = ''

    # Plot the distribution of the mean firing rate
    sns.histplot(y=mfr, kde=True, color='k', bins='auto', ax=ax)  # Horizontal
    ax.axhline(np.mean(mfr), color='tab:red')
    ax.axhline(np.percentile(mfr, 20), color='tab:gray', linestyle='--')
    ax.set_xlabel('Count')
    ax.set_title(title)


def get_baseline(psth):
    """
    Compute the baseline firing rate of a PSTH. Assumes a symmetric PSTH in number of bins around the event.
    :param psth: PSTH
    """

    baseline = psth[:, 0:int(psth.shape[1] / 2)]  # Take the bins before the event (first half of the PSTH)
    baseline = baseline.mean()  # Compute the mean firing rate across bins (per trial)

    return baseline


def plot_isi(spikes=df_cluster, ax=None):
    """
    Plot the Inter Spike Intervals (ISI) of a given cluster.
    :param spikes: DataFrame with spike times of a given cluster (df_cluster) or list of spike times around an event
    (peri_stim_spikes)
    :param ax: Axes to plot the ISI distribution (default: None)
    return: ISI per trial
    """

    # Plot distribution
    if ax is None:
        fig, ax = plt.subplots()

    # With df_cluster (isis will be a single np.array)
    if isinstance(spikes, pd.DataFrame):
        times = df_cluster.times
        isis = np.diff(times)
        # isis = isi(times)  # With elephant
        ax.hist(isis * 1000, bins=100, range=(0, 1000), color='k')

    # With peri_stim_spikes (isis will be a list of np.arrays, one per trial)
    elif isinstance(spikes, list):
        # ISIs per trial (requires peri_stim_spikes)
        isis = []
        for trial in range(len(peri_stim_spikes)):
            spikes = peri_stim_spikes[trial]
            isis.append(np.diff(spikes))
            # isis.append(isi(spikes))  # With elephant
        isis_flatten = np.concatenate(isis)
        ax.hist(isis_flatten * 1000, bins=100, range=(0, 1000), color='k')

    # ax.hist(isis * 1000, bins=100, range=(0, 1000), color='k')
    ax.set_xlabel('Inter Spike Interval (ms)')
    ax.set_ylabel('Count')
    ax.set_title('ISI distribution')

    return isis


def plot_cv(isis, ax=None):
    """
    Compute the coefficient of variation (CV) of a cluster.
    :param isis: Inter Spike Intervals (ISI) of a given cluster. Can be a list (one np.array per trial) or a np.array
    return: Coefficient of variation (CV) of the Inter Spike intervals (ISI) per trial
    """

    # Check if isis is a list of arrays (per trial). coeff_var will be a list of CVs
    if isinstance(isis[0], np.ndarray):
        coeff_var = []
        for trial in range(len(isis)):
            isis_mean = np.mean(isis[trial])
            isis_std = np.std(isis[trial])
            coeff_var.append(isis_std / isis_mean)
            # coeff_var.append(cv(isis[trial]))  # With elephant

        # Plot distribution
        if ax is None:
            fig, ax = plt.subplots()

        ax.hist(coeff_var, bins=100, color='k')
        ax.set_xlabel('Coefficient of Variation')
        ax.set_ylabel('Count')
        ax.set_title('CV distribution')

    # If isis is a single array (per cluster). coeff_var will be a single CV
    elif isinstance(isis[0], np.float64):
        isis_mean = np.mean(isis)
        isis_std = np.std(isis)
        coeff_var = isis_std / isis_mean
        # coeff_var = cv(isis)  # With elephant

    return coeff_var


def fano_factor(peri_stim_spikes):
    """
    Compute the Fano factor of a PSTH.
    :param peri_stim_spikes: Spike times of a given cluster (output of get_peri_stim_spikes)
    return: Fano factor
    """

    spike_counts = [len(series) for series in peri_stim_spikes]
    # spike_counts = np.sum(psth, axis=1)  # Should give the same result
    fano = np.var(spike_counts) / np.mean(spike_counts)
    # fano = fanofactor(peri_stim_spikes)  # With elephant

    return fano





########################################################################################################################

"""
Jaime's Qs/comments:
in this plot it is very clear that you need at leas 1-2 seconds more at the end of the x-axis to show the entire rate 
response. I think  there are many interesting questions about the licks that we may want to address. Remember that the 
people doing Calcium imaging cannot resolve well the timing of the spiking of the neurons locked to the licks. Plus in 
many lick detectors that work with capacitor, the licks can cause an artifact that obscures the spiking activity just at 
the time of the licks. You are in a position to dig into the neural correlates of these licks and answer questions like:
What causes the variability in the lick RT?
What causes the variability in the lick rate (ie inter-lick-interval)?
What makes some correct responses  have 2-3 licks and some 8-10?
Are all response selective neurons locked to the licks?

smooth mean FR across trials by taking longer bins to compute the means rate (e.g. 2-5 trials)



TO DO:

- Resize the window size in PSTHs and rasters to include 1-2 s more post-response. window_size = [-1, 3] s

Make a population report per session with these parts:
1. Population raster and PSTH of the first and last responded trial of the session. Check synchrony                     TO DO
2. Population raster and PSTH of the first second of the last minute of the waiting period before running the task      
and the first second of the last minute of the recording after the waiting period after the task. Check for up-down 
states                                                                                                                  TO DO
3. I think that the stimulus should evoke little population response. I would look more for (1) preparatory activity
between stim onset and Go cue (port approach). (2) rate modulation associated with licking. For this, you could sort the 
units in the raster not according to their overall firing rate, but to the firing rate computed only between stimulus 
onset and Go cue. This is the way Tiffany did it and you can then see some modulation of the population during this     
delay period:                                                                                                           TO DO
4. Would be good to plot the licks in a separate plot below, like in the Reato et al paper I shared above.  It is not 
the same correct (6-8 licks) than error choices (1-2 licks). When averaging the pop inst rate across trials, I would do 
it separatly for correct (many licks) and error trials (few licks).                                                     TO DO
"""