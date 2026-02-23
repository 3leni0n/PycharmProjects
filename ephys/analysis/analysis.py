# Standard libraries
import ast
import matplotlib as mpl
import seaborn as sns
sns.set_theme()
sns.set_style('ticks')
# sns.set_context('talk')
from scipy.ndimage import gaussian_filter1d
from scipy.stats import sem
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime

# Ephys specific libraries
from quantities import ms, s
from neo.core import SpikeTrain
from elephant.statistics import time_histogram, instantaneous_rate, mean_firing_rate
from elephant.kernels import GaussianKernel
from elephant.conversion import BinnedSpikeTrain
from elephant.spike_train_correlation import cross_correlation_histogram

# My libraries
from my_fun.my_fun import compute_window, fig_size
from ephys.preprocessing import *

########################################################################################################################

# Define helper functions

def find_disengaged(df_behavior, threshold=0.5, min_trial=200, win_len=20, plot=False):
    """
    Find the first trial where the animal disengages from the task based on side accuracy.
    :param df_behavior: DataFrame with behavioral data
    :param threshold: threshold accuracy to consider the animal disengaged
    :param min_trial: minimum trial to start looking for disengagement
    :param win_len: window length to compute rolling average
    :return: first_trial (int)
    """

    x_total, y_total, x_0, y_0, x_1, y_1 = get_roll_avg(df_behavior, kind='side')

    # Convert indices to lists to ensure compatibility
    x_0, x_1, x_total = list(x_0), list(x_1), list(x_total)

    # Adjust minimum trial to account for the running window
    min_valid_trial = min_trial + win_len

    # Filter trials starting from min_valid_trial
    filtered_x_total = [(x, y) for x, y in zip(x_total, y_total) if x >= min_valid_trial]
    filtered_x_0 = [(x, y) for x, y in zip(x_0, y_0) if x >= min_valid_trial]
    filtered_x_1 = [(x, y) for x, y in zip(x_1, y_1) if x >= min_valid_trial]

    # Find first trial where y_total reaches threshold
    idx_total = next(((x, y) for x, y in filtered_x_total if y <= threshold), None)

    # Find first trial where y_0 or y_1 reaches threshold, mapped back to absolute trials
    idx_0 = next(((x_total[x_total.index(x)], y) for x, y in filtered_x_0 if y <= threshold), None) if filtered_x_0 else None
    idx_1 = next(((x_total[x_total.index(x)], y) for x, y in filtered_x_1 if y <= threshold), None) if filtered_x_1 else None

    # Get the earliest occurrence and corresponding y-value
    disengaged_trial, disengaged_y = min(filter(None, [idx_total, idx_0, idx_1]), default=(None, None))

    print(f'Disengagement in trial {disengaged_trial}')

    # Side accuracy plot
    if plot:
        plt.figure(constrained_layout=True)
        plt.plot(x_total, y_total, color='k', label='Total')
        plt.plot(x_0, y_0, color='tab:blue', label='Left')
        plt.plot(x_1, y_1, color='tab:orange', label='Right')
        plt.axhline(0.25, color='tab:gray', ls=':')
        plt.axhline(0.5, color='tab:gray', ls='--')
        plt.axhline(0.75, color='tab:gray', ls=':')

        # Plot the red dot at the correct x (absolute trial number) and y=0.5
        if disengaged_trial is not None:
            plt.plot(disengaged_trial, threshold, 'ro')

        plt.xlabel('Trial')
        plt.ylabel('Accuracy')
        plt.title(df_behavior.Session.unique()[0])
        plt.legend(frameon=False)
        sns.despine()

    return disengaged_trial


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

    elif condition == 'repeat':
        indexes0 = df_behavior[df_behavior.RepTrial == 0].Trial.values
        indexes1 = df_behavior[df_behavior.RepTrial == 1].Trial.values

    elif condition == 'prev_out':
        indexes0 = df_behavior[df_behavior.AfterHit == 0].Trial.values
        indexes1 = df_behavior[df_behavior.AfterHit == 1].Trial.values

    elif condition == 'miss':
        indexes0 = df_behavior[df_behavior.Miss == 0].Trial.values
        indexes1 = df_behavior[df_behavior.Miss == 1].Trial.values


    # Store indexes in a list
    indexes0 = indexes0.tolist()
    indexes1 = indexes1.tolist()
    indexes = [indexes0, indexes1]

    return indexes


def select_cluster(cluster_info, group='all'):
    """
    Select cluster indices based on their group (good, mua, all)
    :param cluster_info: cluster_info dataframe
    :param group: 'good', 'mua', 'all'
    :return: cluster indices
    """

    if group == 'all':  # All clusters
        cond = (cluster_info.group == 'good') | (cluster_info.group == 'mua')
    elif group == 'good':
        cond = cluster_info.group == 'good'
    elif group == 'mua':
        cond = cluster_info.group == 'mua'

    return cluster_info[cond].cluster_id


def get_peri_event_spikes(df_cluster, df_ttl, align='stim', time_win=[-1, 3], scale=0):
    """
    Get peri-stimulus spikes for a cluster.
    :param df_cluster: DataFrame with spike times of a given cluster
    :param time_win: List with the time window around the event (default: [-1, 3])
    :param df_ttl: DataFrame with TTL events
    :param scale: Scale of the jitter (default: 0; no jitter)
    """

    peri_event_spikes = []
    # Loop over trials (timestamps of stimulus onset)

    # Check if df_ttl is a Series and convert to DataFrame to iterate over single trial
    if isinstance(df_ttl, pd.Series):
        df_ttl = pd.DataFrame([df_ttl])

    for trial in range(len(df_ttl)):
        # print(f'Trial {trial}')
        jitter = np.random.normal(0, scale)  # Jitter the stimulus onset timestamps

        if align == 'stim':
            alignment = df_ttl.OFF[trial] + jitter  # Stimulus onset timestamp
        elif align == 'go_cue':
            alignment = df_ttl.GoCue[trial] + jitter  # Go cue timestamp
        elif align == 'resp':
            alignment = df_ttl.RT[trial] + jitter  # First lick timestamp

        # Select only spikes within the time window of interest around the event
        spikes_trial = df_cluster[(df_cluster.times >= alignment - abs(time_win[0])) &
                                  (df_cluster.times <= alignment + abs(time_win[1]))].times
        spikes_trial = spikes_trial - alignment  # Align spikes to the event
        peri_event_spikes.append(spikes_trial)

    return peri_event_spikes


def get_peri_event_licks(df_behavior, event='StimStart'):
    """
    Get peri-event licks for a given behavioral session.
    :param df_behavior: DataFrame with behavior data
    :param event: Event to align the licks to (default: 'StimStart', 'RespWinEnd' for first lick)
    :return: pd.Series with a list of peri-stimulus licks per trial
    """

    # Convert string of lists back to lists
    try:
        df_behavior["Port1In"] = df_behavior["Port1In"].apply(ast.literal_eval)
        df_behavior["Port2In"] = df_behavior["Port2In"].apply(ast.literal_eval)
    except ValueError:
        # print('Port1In or Port2In are already lists')
        pass

    licks_left = df_behavior.Port1In.copy()
    licks_right = df_behavior.Port2In.copy()
    for trial in range(len(df_behavior)):
        licks_left[trial] = [x - df_behavior[event].iloc[trial] for x in licks_left[trial]]  # Left
        licks_right[trial] = [x - df_behavior[event].iloc[trial] for x in licks_right[trial]]  # Right
    licks = licks_left + licks_right

    return licks


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
def plot_raster(df_behavior, peri_event_spikes, cluster, group, colors=None, ax=None):
    """
    Plot a raster plot of a given cluster aligned to a specific event. Uses an approach completely independent of
    the behavior data. Loops over the TTL events related to stimuli registered in the ephys
    :param peri_event_spikes: Spike times of a given cluster (output of get_peri_event_spikes)
    :param colors: Colors of the raster plot (default: None)
    :param ax: Axes to plot the raster (default: None)
    """

    stim_dur = df_behavior.StimDur.unique()[0]
    delay = df_behavior.Delay.unique()[0]
    go_cue = stim_dur + delay

    # If no Axes is provided, create a new one
    if ax is None:
        fig, ax = plt.subplots()
        responded_trials = df_behavior[df_behavior.Response == 1].Trial.values
        peri_event_spikes = [peri_event_spikes[_] for _ in responded_trials]  # Only trials with a response
        title = f'Cluster {cluster} ({group})'
    else:
        title = ''

    if colors is None:
        colors = ['k'] * len(peri_event_spikes)

    # for trial, spikes in enumerate(peri_event_spikes):
    #     ax.scatter(spikes, np.full_like(spikes, trial), marker='.', color=colors[trial])

    ax.eventplot(peri_event_spikes, orientation='horizontal', lineoffsets=range(len(peri_event_spikes)), colors=colors)
    ax.axvline(0, color='tab:gray', label='Stimulus')
    ax.axvline(delay, color='tab:gray', linestyle='--', label='Delay')
    ax.axvline(go_cue, color='tab:gray', label='Go cue')
    ax.set_xlabel('Time from stim. onset (s)')
    ax.set_ylabel('Trial')
    ax.set_title(title)
    ax.legend(loc='upper left', frameon=False)


def plot_raster_split(df_behavior, df_cluster, df_ttl, condition='outcome', ax=None):
    """
    Plot a raster plot of a given cluster aligned to a specific event split by condition.
    """

    # If no Axes is provided, create a new one
    if ax is None:
        fig, ax = plt.subplots()

    cluster = df_cluster.cluster.unique()[0]
    group = df_cluster.group.unique()[0]

    if condition == 'outcome':
        colors = ['tab:red', 'tab:green']
    elif condition == 'choice':
        colors = ['tab:blue', 'tab:orange']
    elif condition == 'stimulus':
        colors = ['tab:blue', 'tab:orange']
    elif condition == 'repeat':
        colors = ['tab:purple', 'tab:brown']

    indexes = get_trial_indexes(df_behavior, condition=condition)
    peri_event_spikes = []

    for _ in range(len(indexes)):
        peri_event_spikes.append(
            get_peri_event_spikes(df_cluster, df_ttl.iloc[indexes[_]].reset_index(drop=True)))

    peri_event_spikes = peri_event_spikes[0] + peri_event_spikes[1]  # Concatenate lists
    colors = [colors[0]] * len(indexes[0]) + [colors[1]] * len(indexes[1])  # Concatenate colors
    plot_raster(df_behavior, peri_event_spikes, cluster, group, colors=colors, ax=ax)  # Plot raster plot split by condition
    ax.legend().remove()


########################################################################################################################

# Peri-Stimulus Time Histograms (PSTHs)
def plot_psth_elephant(df_spikes, cluster, df_behavior, align='StimStart'):
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
        mfr.append(np.round(mean_firing_rate(spiketrain).magnitude * 1000, 2))  # In spikes/s
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
    plt.axvline(0, color='tab:red', label='Stimulus')
    plt.axvline(df_cluster.Delay.unique()[0], color='tab:blue', label='Delay')
    plt.axvline(df_cluster.StimDur.unique()[0] + df_cluster.Delay.unique()[0], color='tab:green', label='Response')
    plt.plot(test_times.mean(axis=0), test_firing.mean(axis=0), color='k')
    plt.fill_between(test_times.mean(axis=0), test_firing.mean(axis=0) - test_firing.sem(axis=0),
                     test_firing.mean(axis=0) + test_firing.sem(axis=0), color='k', alpha=0.25, edgecolor='none')

    plt.xlabel('Time (s)')
    plt.ylabel('Trial')
    plt.title(f'PSTH aligned to {align} (cluster {cluster})')
    plt.legend(loc='upper right', frameon=False)

    # Save the figure in Desktop folder called 'psths'
    plt.savefig(Path.home() / 'OneDrive' / 'Escritorio' / 'psths' / f'cluster_{cluster}_aligned_{align}.png')
    plt.close()


def compute_psth(peri_event_spikes, time_win=[-1, 3], bin_size=0.1):
    """
    Compute a PSTH of a given cluster aligned to a specific event.
    :param peri_event_spikes: Spike times of a given cluster (output of get_peri_event_spikes)
    :param time_win: List with the time window around the event (default: [-1, 3])
    :param bin_size: Size of the bins for the PSTH (default: 0.1 s)
    """

    n_bins = int((time_win[1] - time_win[0]) / bin_size) + 1
    bins = np.linspace(time_win[0], time_win[1], n_bins)  # linspace is preferred over arange for PSTHs

    psth = []
    # Loop over trials (timestamps of stimulus onset)
    for trial in range(len(peri_event_spikes)):
        hist, _ = np.histogram(peri_event_spikes[trial], bins)  # Ignore the bin_edges output
        psth.append(hist)
    psth = np.array(psth)  # Convert to numpy array

    return bins, psth


@timer
def get_all_psth(cluster_info, df_spikes, df_ttl, n_trials, align='stim', time_win=[-1, 3], bin_size=0.1):
    """
    Create 3-dimensional array (trial x bin x cluster) with all PSTHs for all clusters of one session
    :param df_spikes: dataframe with spike times
    :param cluster_info: dataframe with cluster information
    :param time_win: time window around event
    :param bin_size: bin size
    :return: bins, all_psth
    """

    n_bins = int((time_win[1] - time_win[0]) / bin_size)
    n_clusters = len(cluster_info)
    all_psth = np.zeros((n_trials, n_bins, n_clusters))  # Initialize array to store all PSTHs

    for i, cluster in enumerate(cluster_info.cluster_id):
        # print(f'{i}: cluster {cluster}')
        df_cluster = df_spikes[df_spikes.cluster == cluster]
        peri_event_spikes = get_peri_event_spikes(df_cluster, df_ttl, align=align, time_win=time_win, scale=0)
        bins, psth = compute_psth(peri_event_spikes, time_win=time_win, bin_size=bin_size)
        all_psth[:, :, i] = psth

    return bins, all_psth


def compute_psth_shuffles(df_cluster, df_ttl, n_shuffles=1000, time_win=[-1, 3], scale=2, bin_size=0.1):
    """
    Compute PSTHs for shuffled spike times.
    :param n_shuffles: Number of shuffles (default: 1000)
    :param scale: Scale of the jitter (default: 2)
    """

    psth_shuffles = []
    for _ in range(n_shuffles):
        peri_event_spikes = get_peri_event_spikes(df_cluster, df_ttl, time_win, scale=scale)  # Get jittered spikes
        bins, psth = compute_psth(peri_event_spikes, time_win=time_win, bin_size=bin_size)  # Compute the PSTH of the jittered spikes
        psth = np.mean(psth, axis=0)  # Average across trials
        psth = psth / bin_size  # Convert to spikes/s
        psth_shuffles.append(psth)  # Store the PSTH of the shuffled spikes
    psth_shuffles = np.array(psth_shuffles)  # Convert to numpy array

    return bins, psth_shuffles


def plot_psth(bins, psth, bin_size=0.1, color=None, label=None, ax=None):
    """
    Plot a PSTH of a given cluster aligned to a specific event.
    :param bins: Bins of the PSTH
    :param psth: Histograms of the PSTH
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

    # # Compute the 95% confidence interval
    # if color == 'k':
    #     # If only one PSTH, compare to null hypothesis (shuffled spikes). Use % as sem scales with N shuffles
    #     psth_shuffle_mean = psth_shuffles.mean(axis=0)
    #     ax.plot(bins[:-1], psth_shuffle_mean, color='tab:gray', ls='--')
    #     bound = np.percentile(psth_shuffles, [2.5, 97.5], axis=0)  # The 95% confidence interval of the shuffles
    # else:
    #     # If multiple PSTHs, compare sem (sem * 1.96 is 95% CI)
    #     bound = [psth_mean - psth_sem, psth_mean + psth_sem]

    bound = [psth_mean - psth_sem, psth_mean + psth_sem]

    # Plot PSTH
    ax.axvline(0, color='tab:gray')
    ax.axvline(0.5, color='tab:gray', ls='--')
    ax.axvline(1, color='tab:gray')
    ax.plot(bins[:-1], psth_mean, color=color, label=label)
    ax.fill_between(bins[:-1], bound[0], bound[1], color=color, alpha=0.25, edgecolor='none')
    ax.set_xlabel('Time (s)')
    ax.set_ylim(bottom=0)
    ax.set_ylabel('Firing rate')
    # ax.set_title(f'PSTH for cluster {cluster} ({group})')
    # ax.legend(loc='upper left', frameon=False)


def plot_psth_split(df_behavior, df_cluster, df_ttl, condition='outcome', over='spikes', align='stim', time_win=[-1, 3],
                    bin_size=0.1, ax=None):
    """
    Plot a PSTH of a given cluster aligned to a specific event split by condition.
    """

    if ax is None:
        fig, ax = plt.subplots()
        cluster = df_cluster.cluster.unique()[0]
        group = df_cluster.group.unique()[0]
        title = f'Cluster {cluster} ({group})'
    else:
        title = ''

    indexes = get_trial_indexes(df_behavior, condition=condition)

    if condition == 'outcome':
        color = ['tab:red', 'tab:green']
        # labels = ['Error', 'Correct']
        labels = ['E', 'C']
    elif condition == 'choice':
        color = ['tab:blue', 'tab:orange']
        # labels = ['Left', 'Right']
        labels = ['L', 'R']
    elif condition == 'stimulus':
        color = ['tab:blue', 'tab:orange']
        # labels = ['Left', 'Right']
        labels = ['L', 'R']
    elif condition == 'repeat':
        color = ['tab:purple', 'tab:brown']
        # labels = ['Alternate', 'Repeat']
        labels = ['A', 'R']

    for _ in range(len(indexes)):

        if over == 'spikes':
            peri_stim = get_peri_event_spikes(df_cluster, df_ttl.iloc[indexes[_]].reset_index(drop=True), align, time_win)
            ylabel = 'Firing rate'
        elif over == 'licks':
            peri_stim = get_peri_event_licks(df_behavior.iloc[indexes[_]].reset_index(drop=True))
            ylabel = 'Lick rate'

        bins, psth = compute_psth(peri_stim)
        plot_psth(bins, psth, bin_size, color=color[_], label=labels[_], ax=ax)
        # plot_psth(bins, psth, psth_shuffles, bin_size, color=color[_], label=labels[_], ax=ax)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(loc='upper right', frameon=False, handlelength=1, fontsize='small', bbox_to_anchor=(1, 1), borderaxespad=0)


########################################################################################################################

# Plot both raster and PSTH
def plot_raster_psth(df_cluster, df_behavior, df_ttl, time_win=[-1, 3], bin_size=0.1, ax=[None, None]):
    """
    Plot a raster plot and PSTH of a given cluster aligned to a specific event.
    """

    cluster = df_cluster.cluster.unique()[0]
    group = df_cluster.group.unique()[0]

    if ax[0] is None and ax[1] is None:
        fig, ax = plt.subplots(2, 1, sharex=True)
        title = f'Cluster {cluster} ({group})'
    else:
        title = ''

    responded_trials = df_behavior[df_behavior.Response == 1].Trial.values
    peri_event_spikes = get_peri_event_spikes(df_cluster, df_ttl, time_win=time_win)
    peri_event_spikes = [peri_event_spikes[_] for _ in responded_trials]  # Only trials with a response

    plot_raster(df_behavior, peri_event_spikes, cluster, group, colors=['k'] * len(peri_event_spikes), ax=ax[0])
    ax[0].set_title('')
    ax[0].set_xlabel('')
    ax[0].legend().remove()
    bins, psth = compute_psth(peri_event_spikes)
    plot_psth(bins, psth, bin_size, ax=ax[1])
    # plot_psth(bins, psth, psth_shuffles, bin_size, ax=ax[1])
    ax[1].set_title('')
    plt.suptitle(title)
    # plt.tight_layout()


def plot_raster_psth_split(df_behavior, df_cluster, df_ttl, condition='outcome', ax=[None, None]):
    """
    Plot a raster and a PSTH split by a condition.
    """

    if ax[0] is None and ax[1] is None:
        fig, ax = plt.subplots(2, 1, sharex=True)
        cluster = df_cluster.cluster.unique()[0]
        group = df_cluster.group.unique()[0]
        title = f'Cluster {cluster} ({group})'
    else:
        title = ''

    plot_raster_split(df_behavior, df_cluster, df_ttl, condition=condition, ax=ax[0])
    plot_psth_split(df_behavior, df_cluster, df_ttl, condition=condition, ax=ax[1])
    ax[0].set_title('')
    ax[1].set_title('')
    ax[0].set_xlabel('')
    plt.suptitle(title)
    # plt.tight_layout()


########################################################################################################################
# POPULATION ACTIVITY ANALYSIS
########################################################################################################################


def plot_pop_raw(df_spikes, df_ttl, df_behavior, cluster_info, slice='trials', win_edges=(549, 551), sort_by='depth',
                 bin_size=0.02):
    """
    Plot population activity of all clusters in 2 subplots: raster (above) and PSTH (below).
    Short time window (a few trials/seconds).
    :param df_spikes: DataFrame with spike times and clusters
    :param slice: Slice the data in trials or time (default: trials)
    :param win_edges: Edges of the slice (trials or time (s))
    :param sort: Sort clusters by attribute of cluster_info (default: n_spikes)
    :param bin_size: Size of the bins for the PSTH (default: 0.1 s)
    """

    # Sort clusters
    if sort_by == 'n_spikes':
        ascending = True
    elif sort_by == 'depth':
        ascending = False

    cluster_info.sort_values(sort_by, ascending=ascending, inplace=True)
    cluster_info.reset_index(drop=True, inplace=True)

    go_cue = df_behavior.StimDur.unique()[0] + df_behavior.Delay.unique()[0]
    baseline = 1  # s

    if slice == 'trials':
        # Slice DataFrame of given time window after first event (behavior started)
        # win_edges = 549, 551  # Edges of trials to plot
        print(f' Plotting trials: {np.arange(win_edges[0], win_edges[1] - 1)}')
        win_events = df_ttl.OFF.iloc[win_edges[0]:win_edges[1]]
        df_slice = df_spikes[
            (df_spikes.times > win_events.iloc[0] - baseline) & (df_spikes.times < win_events.iloc[-1] + go_cue)]
        title = (f"Population activity of {len(cluster_info)} clusters "
                 f"({round(len(cluster_info[cluster_info.group == 'good']) / len(cluster_info) * 100)}% 'good')")
        bins = np.arange(win_events.iloc[0] - baseline, win_events.iloc[-1] + go_cue, bin_size)
    elif slice == 'time':
        # Slice DataFrame of given time window after first event (behavior started)
        # win_edges = 1922, 1927  # Edges of time window to plot
        print(f' Plotting time: {win_edges}')
        df_slice = df_spikes[(df_spikes.times > win_edges[0]) & (df_spikes.times < win_edges[1])]
        title = (f"Population activity of {len(cluster_info)} clusters "
                 f"({round(len(cluster_info[cluster_info.group == 'good']) / len(cluster_info) * 100)}% 'good')")
        bins = np.arange(win_edges[0], win_edges[1], bin_size)

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

    # Plot events and licks ( aligned to stimulus onset)
    if slice == 'trials':
        licks = get_peri_event_licks(df_behavior)
        for i in range(len(ax)):
            for _ in win_events.index.values:
                ax[i].axvline(win_events[_], color='tab:red', label='Stimulus')
                ax[i].axvline(win_events[_] + 0.5, color='tab:gray', label='Stimulus')
                ax[i].axvline(win_events[_] + 1, color='tab:blue', label='Go cue')
            # for _ in licks[win_edges[0]]:
            #     ax[i].axvline(_ + df_ttl.OFF.iloc[win_edges[0]], color='cyan')

    ax[0].set_ylabel('Cluster')
    ax[1].set_xlabel(f'Time (s)')
    ax[0].set_title(f'{df_behavior.Session.unique()[0]}: Trial {np.arange(win_edges[0], win_edges[1] - 1)}')
    ax[1].set_ylim(bottom=0)
    ax[1].set_ylabel('FR (spikes/s)')
    plt.suptitle(title)
    plt.tight_layout()


def plot_pop_psth(df_spikes, df_ttl, cluster_info, time_win=[-1, 3], bin_size=0.1, ax=None):
    """
    Plot population PSTH treating all the spikes as coming from a single superneuron.
    :param time_win: Time window around the event (default: [-1, 3])
    :param ax: Axes to plot the population PSTH (default: None)
    """

    if ax is None:
        fig, ax = plt.subplots()

    peri_event_spikes = get_peri_event_spikes(df_spikes, df_ttl, time_win)
    bins, psth = compute_psth(peri_event_spikes, time_win, bin_size)
    # _, psth_shuffles = compute_psth_shuffles(df_spikes, df_ttl, n_shuffles=10, scale=2)
    psth = psth/len(cluster_info)
    plot_psth(bins, psth, bin_size, ax=ax)
    # plot_psth(bins, psth / len(cluster_info), psth_shuffles / len(cluster_info), bin_size)
    ax.set_title('Population PSTH')
    sns.despine(ax=ax)

    return bins, psth


########################################################################################################################
# STATS
########################################################################################################################
def plot_autocorrelogram(df_cluster, bin_size=0.001, window=[-50, 50], cross_corr_coeff=True,
                         ax=None):
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
    binned_spike_train = BinnedSpikeTrain(spike_train, bin_size=bin_size)
    cch, lags = cross_correlation_histogram(binned_spike_train, binned_spike_train, window=window,
                                            cross_correlation_coefficient=cross_corr_coeff)

    # # Normalize to correalation
    # binned_spike_train = binned_spike_train.to_array()
    # mean = np.mean(binned_spike_train)
    # variance = np.var(binned_spike_train)
    # if variance > 0:
    #     cch = (cch - len(binned_spike_train) * mean ** 2) / (len(binned_spike_train) * variance)
    # else:
    #     cch = np.zeros_like(cch)

    cch, lags = np.delete(cch.magnitude.flatten(), lags == 0), np.delete(lags, lags == 0)

    refractory_period = 2  # In number of bins
    ax.axvline(-refractory_period, color='tab:gray')
    ax.axvline(refractory_period, color='tab:gray')
    ax.plot(lags, cch, color='k')
    ax.set_title(title)
    ax.set_xlabel('Time lag (ms)')
    ax.set_ylabel('Correlation')

    return cch, lags


def plot_mfr(psth, bin_size=0.1, ax=None):
    """
    Compute the mean firing rate per trial of a PSTH.
    :param psth: PSTH
    :param bin_size: Size of the bins when coputing the PSTH (default: 0.1 s)
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
    # mfr_percentile = np.percentile(mfr, percentile)

    ax.plot(mfr, color='k')
    ax.fill_between(np.arange(len(mfr)), mfr - sfr, mfr + sfr, color='k', alpha=0.25, edgecolor='none')
    # ax.axhline(mfr_mean, color='tab:red', label='mean')
    # ax.axhline(mfr_percentile, color='tab:gray', linestyle='--', label=f'{percentile}th percentile')
    ax.set_xlabel('Trial')
    ax.set_ylim(bottom=0)
    ax.set_ylabel('Firing rate')
    # ax.legend(loc='upper right', frameon=False)
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


def plot_isi(df_cluster, spikes, ax=None):
    """
    Plot the Inter Spike Intervals (ISI) of a given cluster.
    :param spikes: DataFrame with spike times of a given cluster (df_cluster) or list of spike times around an event
    (peri_event_spikes)
    :param ax: Axes to plot the ISI distribution (default: None)
    return: ISI per trial
    """

    # Plot distribution
    if ax is None:
        fig, ax = plt.subplots()
        plt.close()  # Close the figure to avoid showing it. Only plotting it inside cluster_report

    # With df_cluster (isis will be a single np.array)
    if isinstance(spikes, pd.DataFrame):
        times = df_cluster.times
        isis = np.diff(times)
        # isis = isi(times)  # With elephant
        ax.hist(isis * 1000, bins=100, range=(0, 1000), color='k')

    # With peri_event_spikes (isis will be a list of np.arrays, one per trial)
    elif isinstance(spikes, list):
        # ISIs per trial (requires peri_event_spikes)
        isis = []
        for trial in range(len(spikes)):
            spikes = spikes[trial]
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


def fano_factor(peri_event_spikes):
    """
    Compute the Fano factor of a PSTH.
    :param peri_event_spikes: Spike times of a given cluster (output of get_peri_event_spikes)
    return: Fano factor
    """

    spike_counts = [len(series) for series in peri_event_spikes]
    # spike_counts = np.sum(peri_event_spikes, axis=1)  # Should give the same result
    fano = np.var(spike_counts) / np.mean(spike_counts)
    # fano = fanofactor(peri_event_spikes)  # With elephant

    return fano


def get_session_fano(all_psth):
    """
    Compute Fano factor for each cluster in a session from a precomputed PSTH array.
    :param all_psth: np.array of shape (trials, bins, clusters)
    :return: list of Fano factors for each cluster in the session
    """

    fano_session = []
    n_clusters = all_psth.shape[2]

    for cluster in range(n_clusters):
        # sum spikes across bins for each trial to get spike counts
        spike_counts = all_psth[:, :, cluster].sum(axis=1)  # shape: (n_trials,)
        ff = np.var(spike_counts) / np.mean(spike_counts) if spike_counts.mean() > 0 else np.nan
        fano_session.append(ff)

    # fano_session = np.array(fano_session)

    return fano_session


def filter_units(bins, all_psth, cluster_info, min_fr=0.1, max_fano=10, group=None, depth=None):
    """
    Filter units based on firing rate, Fano factor, group and depth. Should be run by session.
    :param bins: bins used to compute the PSTH (output of get_all_psth)
    :param all_psth: np.array of shape (trials, bins, clusters) with the PSTH of all clusters in a session (output of
    get_all_psth)
    :param cluster_info: DataFrame with cluster information (output of get_cluster_info)
    :param min_fr: Minimum firing rate to include a unit (default: 0.1 Hz)
    :param max_fano: Maximum Fano factor to include a unit (default: 10)
    :param group: Group to include (default: None, includes all groups)
    :param depth: tuple depth range in mm to include (default: None, includes all depths)
    """

    n_units = all_psth.shape[2]
    mask = np.ones(n_units, dtype=bool) # Start with all units included

    # By firing rate
    if min_fr:
        bin_size = bins[1] - bins[0]
        fr = all_psth.mean(axis=(0, 1)) / bin_size
        count = mask.sum()
        mask &= fr >= min_fr
        print(f'Removed {count - mask.sum()} units by FR')

    # By Fano factor
    if max_fano:
        fano_session = get_session_fano(all_psth)
        fano_session = np.array(fano_session)
        cluster_info['fano'] = fano_session
        count = mask.sum()
        mask &= fano_session <= max_fano
        print(f'Removed {count - mask.sum()} units by Fano factor')

    if group:
        count = mask.sum()
        mask &= (cluster_info.group == group).to_numpy()
        print(f'Removed {count - mask.sum()} units by group')

    if depth:
        cluster_info['depth'] /= 1000
        min_depth, max_depth = depth
        surface = cluster_info.depth.max()
        # Compute depth from surface (originally distance from probe tip)
        depth_from_surface = surface - cluster_info['depth']
        count = mask.sum()
        mask &= (depth_from_surface >= min_depth) & (depth_from_surface <= max_depth)
        print(f'Removed {count - mask.sum()} units by depth')

    print(f'{sum(mask)} / {n_units} units kept')
    all_psth = all_psth[:, :, mask]
    cluster_info = cluster_info[mask].reset_index(drop=True)

    return all_psth, cluster_info


########################################################################################################################


# cluster = 0
# df_cluster = df_spikes[df_spikes.cluster == cluster]

def cluster_report(df_cluster, cluster_info, df_ttl, df_behavior, align='stim', time_win=[-1, 3], bin_size=0.1, save=False):
    """
    Plot a raster and PSTH of a given cluster aligned to a specific event.
    """

    figsize = fig_size(n_cols=0, ratio=None)  # A4 size in inches without margins
    figsize = (figsize[1], figsize[0])  # Landscape
    figsize = (figsize[0], figsize[1] - 1)  # Reduce height by 1 inch to make space for figure caption
    # Set subplots layout with mosaic
    mosaic = [['Auto', 'MFR', 'MFR', 'MFR'],#, 'MFR'],  # Autocorrelogram and mean firing rate
              ['RasterAll', 'RasterStimulus', 'RasterChoice', 'RasterOutcome'],#, 'RasterRepeat'],  # Rasters
              ['RasterAll', 'RasterStimulus', 'RasterChoice', 'RasterOutcome'],#, 'RasterRepeat'],  # Rasters
              ['PSTHAll', 'PSTHStimulus', 'PSTHChoice', 'PSTHOutcome'],#, 'PSTHRepeat'],  # PSTHs
              ['LicksAll', 'LicksStimulus', 'LicksChoice', 'LicksOutcome']]# 'LicksRepeat']]  # Licks
    fig, ax_dict = plt.subplot_mosaic(mosaic, figsize=figsize, constrained_layout=True, dpi=300)

    cluster = df_cluster.cluster.unique()[0]
    group = df_cluster.group.unique()[0]

    print(f'Doing report of cluster {cluster}...')

    # df_cluster = df_spikes[df_spikes.cluster == cluster]  # Slice DataFrame of given cluster
    # group = cluster_info[cluster_info.cluster_id == cluster].group.iloc[0]
    depth = cluster_info[cluster_info.cluster_id == cluster].depth.iloc[0]
    fr = cluster_info[cluster_info.cluster_id == cluster].fr.iloc[0]

    peri_event_spikes = get_peri_event_spikes(df_cluster, df_ttl, align, time_win)
    bins, psth = compute_psth(peri_event_spikes, time_win, bin_size)
    peri_stim_licks = get_peri_event_licks(df_behavior)
    bins, licks_psth = compute_psth(peri_stim_licks, bin_size=bin_size, time_win=time_win)
    # bins, psth_shuffles = compute_psth_shuffles(df_cluster, df_ttl, n_shuffles=1, scale=2)

    # Plot panels
    plot_autocorrelogram(df_cluster, bin_size=0.001, window=[-50, 50], cross_corr_coeff=True, ax=ax_dict['Auto'])
    plot_mfr(psth, bin_size, ax=ax_dict['MFR'])
    plot_raster_psth(df_cluster, df_behavior, df_ttl, time_win=time_win, bin_size=bin_size, ax=[ax_dict['RasterAll'], ax_dict['PSTHAll']])
    plot_raster_psth_split(df_behavior, df_cluster, df_ttl, condition='stimulus', ax=[ax_dict['RasterStimulus'], ax_dict['PSTHStimulus']])
    plot_raster_psth_split(df_behavior, df_cluster, df_ttl, condition='choice', ax=[ax_dict['RasterChoice'], ax_dict['PSTHChoice']])
    plot_raster_psth_split(df_behavior, df_cluster, df_ttl, condition='outcome', ax=[ax_dict['RasterOutcome'], ax_dict['PSTHOutcome']])
    # plot_raster_psth_split(df_behavior, df_cluster, df_ttl, condition='repeat', ax=[ax_dict['RasterRepeat'], ax_dict['PSTHRepeat']])
    plot_psth(bins, licks_psth, bin_size, ax=ax_dict['LicksAll'])
    plot_psth_split(df_behavior, df_cluster, df_ttl, condition='stimulus', over='licks', ax=ax_dict['LicksStimulus'])
    plot_psth_split(df_behavior, df_cluster, df_ttl, condition='choice', over='licks', ax=ax_dict['LicksChoice'])
    plot_psth_split(df_behavior, df_cluster, df_ttl, condition='outcome', over='licks', ax=ax_dict['LicksOutcome'])
    # plot_psth_split(df_behavior, df_cluster, df_ttl, condition='repeat', over='licks', ax=ax_dict['LicksRepeat'])

    # Remove xlabels
    ax_dict['PSTHAll'].set_xlabel('')
    ax_dict['PSTHStimulus'].set_xlabel('')
    ax_dict['PSTHChoice'].set_xlabel('')
    ax_dict['PSTHOutcome'].set_xlabel('')
    # ax_dict['PSTHRepeat'].set_xlabel('')
    ax_dict['LicksAll'].set_xlabel('')
    ax_dict['LicksStimulus'].set_xlabel('')
    ax_dict['LicksChoice'].set_xlabel('')
    ax_dict['LicksOutcome'].set_xlabel('')
    # ax_dict['LicksRepeat'].set_xlabel('')

    # Remove xticklabels
    ax_dict['RasterAll'].set_xticklabels([])
    ax_dict['RasterStimulus'].set_xticklabels([])
    ax_dict['RasterChoice'].set_xticklabels([])
    ax_dict['RasterOutcome'].set_xticklabels([])
    # ax_dict['RasterRepeat'].set_xticklabels([])
    ax_dict['PSTHAll'].set_xticklabels([])
    ax_dict['PSTHStimulus'].set_xticklabels([])
    ax_dict['PSTHChoice'].set_xticklabels([])
    ax_dict['PSTHOutcome'].set_xticklabels([])
    # ax_dict['PSTHRepeat'].set_xticklabels([])

    # Remove ylabels
    ax_dict['RasterStimulus'].set_ylabel('')
    ax_dict['RasterChoice'].set_ylabel('')
    ax_dict['RasterOutcome'].set_ylabel('')
    # ax_dict['RasterRepeat'].set_ylabel('')
    ax_dict['PSTHStimulus'].set_ylabel('')
    ax_dict['PSTHChoice'].set_ylabel('')
    ax_dict['PSTHOutcome'].set_ylabel('')
    # ax_dict['PSTHRepeat'].set_ylabel('')
    ax_dict['LicksStimulus'].set_ylabel('')
    ax_dict['LicksChoice'].set_ylabel('')
    ax_dict['LicksOutcome'].set_ylabel('')
    # ax_dict['LicksRepeat'].set_ylabel('')

    # Set ylabels
    ax_dict['LicksAll'].set_ylabel('Lick rate')

    # Remove yticklabels
    ax_dict['RasterStimulus'].set_yticklabels([])
    ax_dict['RasterChoice'].set_yticklabels([])
    ax_dict['RasterOutcome'].set_yticklabels([])
    # ax_dict['RasterRepeat'].set_yticklabels([])
    ax_dict['PSTHStimulus'].set_yticklabels([])
    ax_dict['PSTHChoice'].set_yticklabels([])
    ax_dict['PSTHOutcome'].set_yticklabels([])
    # ax_dict['PSTHRepeat'].set_yticklabels([])
    ax_dict['LicksStimulus'].set_yticklabels([])
    ax_dict['LicksChoice'].set_yticklabels([])
    ax_dict['LicksOutcome'].set_yticklabels([])
    # ax_dict['LicksRepeat'].set_yticklabels([])

    # Remove white space in ylims for Rasters
    responses = df_behavior[df_behavior.Response == 1].Response.sum()
    ax_dict['RasterAll'].set_ylim(0, responses)
    ax_dict['RasterStimulus'].set_ylim(0, responses)
    ax_dict['RasterChoice'].set_ylim(0, responses)
    ax_dict['RasterOutcome'].set_ylim(0, responses)
    # ax_dict['RasterRepeat'].set_ylim(0, responses)

    # Set same ylims for PSTHs
    y_max = np.max([ax_dict['PSTHAll'].get_ylim()[1],
                    ax_dict['PSTHStimulus'].get_ylim()[1],
                    ax_dict['PSTHChoice'].get_ylim()[1],
                    ax_dict['PSTHOutcome'].get_ylim()[1]])
                    # ax_dict['PSTHRepeat'].get_ylim()[1]])
    ax_dict['PSTHAll'].set_ylim(0, y_max * 1.05)
    ax_dict['PSTHStimulus'].set_ylim(0, y_max * 1.05)
    ax_dict['PSTHChoice'].set_ylim(0, y_max * 1.05)
    ax_dict['PSTHOutcome'].set_ylim(0, y_max * 1.05)
    # ax_dict['PSTHRepeat'].set_ylim(0, y_max * 1.05)

    # Set same ylims for Licks
    y_max = np.max([ax_dict['LicksAll'].get_ylim()[1],
                    ax_dict['LicksStimulus'].get_ylim()[1],
                    ax_dict['LicksChoice'].get_ylim()[1],
                    ax_dict['LicksOutcome'].get_ylim()[1]])
                    # ax_dict['LicksRepeat'].get_ylim()[1]])
    ax_dict['LicksAll'].set_ylim(0, y_max * 1.05)
    ax_dict['LicksStimulus'].set_ylim(0, y_max * 1.05)
    ax_dict['LicksChoice'].set_ylim(0, y_max * 1.05)
    ax_dict['LicksOutcome'].set_ylim(0, y_max * 1.05)
    # ax_dict['LicksRepeat'].set_ylim(0, y_max * 1.05)

    # Remove axes margins
    ax_dict['MFR'].margins(x=0)
    ax_dict['MFR'].margins(y=0)
    ax_dict['RasterAll'].margins(x=0)
    ax_dict['RasterStimulus'].margins(x=0)
    ax_dict['RasterChoice'].margins(x=0)
    ax_dict['RasterOutcome'].margins(x=0)
    # ax_dict['RasterRepeat'].margins(x=0)
    ax_dict['PSTHAll'].margins(x=0)
    ax_dict['PSTHStimulus'].margins(x=0)
    ax_dict['PSTHChoice'].margins(x=0)
    ax_dict['PSTHOutcome'].margins(x=0)
    # ax_dict['PSTHRepeat'].margins(x=0)
    ax_dict['LicksAll'].margins(x=0)
    ax_dict['LicksStimulus'].margins(x=0)
    ax_dict['LicksChoice'].margins(x=0)
    ax_dict['LicksOutcome'].margins(x=0)
    # ax_dict['LicksRepeat'].margins(x=0)

    # Set titles
    ax_dict['RasterAll'].set_title('All')
    ax_dict['RasterStimulus'].set_title('Stimulus')
    ax_dict['RasterChoice'].set_title('Choice')
    ax_dict['RasterOutcome'].set_title('Outcome')
    # ax_dict['RasterRepeat'].set_title('Repeat')
    ax_dict['LicksAll'].set_title('')

    # Despine axes
    sns.despine(ax=ax_dict['MFR'])
    sns.despine(ax=ax_dict['Auto'])
    sns.despine(ax=ax_dict['RasterAll'], bottom=True)
    sns.despine(ax=ax_dict['RasterStimulus'], left=True, bottom=True)
    sns.despine(ax=ax_dict['RasterChoice'], left=True, bottom=True)
    sns.despine(ax=ax_dict['RasterOutcome'], left=True, bottom=True)
    # sns.despine(ax=ax_dict['RasterRepeat'], left=True, bottom=True)
    sns.despine(ax=ax_dict['PSTHAll'], bottom=True)
    sns.despine(ax=ax_dict['PSTHStimulus'], left=True, bottom=True)
    sns.despine(ax=ax_dict['PSTHChoice'], left=True, bottom=True)
    sns.despine(ax=ax_dict['PSTHOutcome'], left=True, bottom=True)
    # sns.despine(ax=ax_dict['PSTHRepeat'], left=True, bottom=True)
    sns.despine(ax=ax_dict['LicksAll'])
    sns.despine(ax=ax_dict['LicksStimulus'], left=True)
    sns.despine(ax=ax_dict['LicksChoice'], left=True)
    sns.despine(ax=ax_dict['LicksOutcome'], left=True)
    # sns.despine(ax=ax_dict['LicksRepeat'], left=True)

    # Remove legends from lick PSTHs
    ax_dict['LicksStimulus'].legend().remove()
    ax_dict['LicksChoice'].legend().remove()
    ax_dict['LicksOutcome'].legend().remove()
    # ax_dict['LicksRepeat'].legend().remove()

    # Set figure title with cluster info
    isis = plot_isi(df_cluster, spikes=df_cluster, ax=None)
    coeff_var = plot_cv(isis, ax=None)
    fano = fano_factor(peri_event_spikes)
    fig.suptitle(f'Cluster {cluster}')# ({group}, {round(depth/1000, 2)} mm): '
                 # f'\n'
                 # f'mean FR={round(np.mean(fr), 2)}, '
                 # f'CV={round(coeff_var, 2)}, '
                 # f'Fano factor={round(fano, 2)}')
    fig.supxlabel('Time (s) from stim. onset', fontsize=plt.rcParams['axes.labelsize'])

    # fig.tight_layout()

    if save:
        development = dev()
        if development == 'remote':
            subject = df_behavior.Subject.unique()[0]
            session = df_behavior.Session.unique()[0]
            date = session[-15:-7]

            dt = datetime.strptime(date, '%Y%m%d')
            ephys_id = f'{subject}_{dt.strftime("%Y-%m-%d")}'
            path = Path.home() / 'data' / subject
            folders = [p.name for p in path.iterdir() if p.is_dir()]
            ephys_id = [f for f in folders if ephys_id in f]
            ephys_id = ephys_id[0]
            path = Path.home() / 'data' / subject / ephys_id / 'cluster reports' / group

        path.mkdir(exist_ok=True)
        plt.savefig(path / f'cluster {cluster}.png')
        plt.close()


def do_cluster_reports(ephys_id, align='stim', time_win=[-1, 3], bin_size=0.1, group='good'):
    """
    Loop over all clusters and plot a report for each one.
    :param ephys_id: Ephys session ID
    :param align: Event to align the spikes, 'stim' (default) or 'resp'
    :param time_win: Time window around the event (default: [-1, 3])
    :param bin_size: Size of the bins for the PSTH (default: 0.1 s)
    :param group: Cluster group to plot, 'good' (default), 'mua' or None (all)
    """

    preprocessed = preprocess(ephys_id)
    cluster_info = preprocessed.cluster_info
    df_spikes = preprocessed.df_spikes
    df_ttl = preprocessed.df_ttl
    df_behavior = preprocessed.df_behavior
    disengagement = find_disengaged(df_behavior, threshold=0.5, min_trial=200, win_len=20, plot=False)

    # Slice out disengaged trials (based on accuracy)
    df_behavior = df_behavior[df_behavior.Trial < disengagement]
    df_ttl = df_ttl.iloc[:len(df_behavior)]

    if group in ['good', 'mua']:
        clusters = cluster_info[cluster_info.group == group].cluster_id
    elif group is None:
        clusters = cluster_info.cluster_id  # All

    for cluster in clusters:
        df_cluster = df_spikes[df_spikes.cluster == cluster]
        cluster_report(df_cluster, cluster_info, df_ttl, df_behavior, align=align, time_win=time_win, bin_size=bin_size,
                       save=True)


########################################################################################################################
# SESSION REPORT
########################################################################################################################

def get_roll_avg(df_behavior, kind='side'):

    # if ax is None:
    #     fig, ax = plt.subplots()

    win_len = 20

    if kind == 'side':

        # color = ['black', 'tab:blue', 'tab:orange']
        # alpha = [1, 1, 1]
        # label = ['Total', 'Left', 'Right']
        # ylabel = 'Accuracy'

        # Compute accuracy rolling average
        x_total = df_behavior.Hit.index
        y_total = compute_window(df_behavior.Hit, win_len)  # All responded trials
        x_0 = df_behavior.Hit[df_behavior.Side == 0].index
        y_0 = compute_window(df_behavior.Hit[df_behavior.Side == 0], win_len)  # Left responded trials
        x_1 = df_behavior.Hit[df_behavior.Side == 1].index
        y_1 = compute_window(df_behavior.Hit[df_behavior.Side == 1], win_len)  # Right responded trials

    elif kind == 'repeat':

        # color = ['black', 'tab:purple', 'tab:brown']
        # alpha = [1, 1, 1]
        # label = ['Total', 'Alternate', 'Repeat']
        # ylabel = 'Accuracy'

        # Compute accuracy for repeating vs alternating rolling average
        x_total = df_behavior.Hit.index
        y_total = compute_window(df_behavior.Hit, win_len)  # All responded trials
        x_0 = df_behavior.Hit[df_behavior.RepTrial == 0].index
        y_0 = compute_window(df_behavior.Hit[df_behavior.RepTrial == 0], win_len)  # Alternate
        x_1 = df_behavior.Hit[df_behavior.RepTrial == 1].index
        y_1 = compute_window(df_behavior.Hit[df_behavior.RepTrial == 1], win_len)  # Repeat

    elif kind == 'rep_bias':

        # color = ['black', 'tab:blue', 'tab:orange']
        # alpha = [1, 0, 0]
        # label = ['Total', 'Left', 'Right']
        # ylabel = 'Repeat bias'

        x_total = df_behavior.RepChoice.index
        y_total = compute_window(df_behavior.RepChoice, win_len)
        x_0 = df_behavior[df_behavior.Choice == 0].index.tolist()
        y_0 = compute_window(df_behavior.RepChoice[df_behavior.Choice == 0], win_len)  # Repeat rate left
        x_1 = df_behavior[df_behavior.Choice == 1].index.tolist()
        y_1 = compute_window(df_behavior.RepChoice[df_behavior.Choice == 1], win_len)  # Repeat rate right

    elif kind == 'miss':

        # color = ['black', 'tab:blue', 'tab:orange']
        # alpha = [1, 1, 1]
        # label = ['Total', 'Left', 'Right']
        # ylabel = 'Miss rate'

        # Compute miss rolling average
        x_total = df_behavior.index
        y_total = compute_window(df_behavior.Miss, win_len)  # All responded trials
        x_0 = df_behavior[df_behavior.Side == 0].index
        y_0 = compute_window(df_behavior.Miss[df_behavior.Side == 0], win_len)  # Left responded trials
        x_1 = df_behavior[df_behavior.Side == 1].index
        y_1 = compute_window(df_behavior.Miss[df_behavior.Side == 1], win_len)  # Right responded trials

    # marker = 'o'
    # ms = mpl.rcParams['lines.markersize'] / 2  # Half of default
    #
    # # Plot rolling averages
    # ax.plot(x_total, y_total, marker=marker, ms=ms, color=color[0], alpha=alpha[0], label=label[0])
    # ax.plot(x_0, y_0, marker=marker, ms=ms, color=color[1], alpha=alpha[1], label=label[1])
    # ax.plot(x_1, y_1, marker=marker, ms=ms, color=color[2], alpha=alpha[2], label=label[2])
    #
    # # Plot horizontal lines
    # ax.axhline(0.25, color='tab:gray', linestyle=':')  # Accuracy 0.25
    # ax.axhline(0.5, color='tab:gray', linestyle='--')  # Chance level
    # ax.axhline(0.75, color='tab:gray', linestyle=':')  # Accuracy 0.75
    #
    # ax.set_xlim([1, len(df_behavior)])  # 1 to not plot trial 0
    # ax.set_ylabel(ylabel)
    # ax.set_yticks(list(np.arange(0, 1.25, 0.25)))
    # ax.set_yticklabels(['0', '', '0.5', '', '1'])
    # sns.despine(ax=ax, bottom=True)

    return x_total, y_total, x_0, y_0, x_1, y_1


def plot_roll_avg(x_total, y_total, x_0, y_0, x_1, y_1, kind='side', ax=None):

    if ax is None:
        fig, ax = plt.subplots()

    if kind == 'side':

        color = ['black', 'tab:blue', 'tab:orange']
        alpha = [1, 1, 1]
        label = ['Total', 'Left', 'Right']
        ylabel = 'Accuracy'

    elif kind == 'repeat':
        color = ['black', 'tab:purple', 'tab:brown']
        alpha = [1, 1, 1]
        label = ['Total', 'Alternate', 'Repeat']
        ylabel = 'Accuracy'

    elif kind == 'rep_bias':
        color = ['black', 'tab:blue', 'tab:orange']
        alpha = [1, 0, 0]
        label = ['Total', 'Left', 'Right']
        ylabel = 'Repeat bias'

    elif kind == 'miss':
        color = ['black', 'tab:blue', 'tab:orange']
        alpha = [1, 1, 1]
        label = ['Total', 'Left', 'Right']
        ylabel = 'Miss rate'

    marker = 'o'
    ms = mpl.rcParams['lines.markersize'] / 2  # Half of default

    # Plot rolling averages
    ax.plot(x_total, y_total, marker=marker, ms=ms, color=color[0], alpha=alpha[0], label=label[0])
    # ax.plot(x_0, y_0, marker=marker, ms=ms, color=color[1], alpha=alpha[1], label=label[1])
    # ax.plot(x_1, y_1, marker=marker, ms=ms, color=color[2], alpha=alpha[2], label=label[2])

    # Plot horizontal lines
    ax.axhline(0.25, color='tab:gray', linestyle=':')  # Accuracy 0.25
    ax.axhline(0.5, color='tab:gray', linestyle='--')  # Chance level
    ax.axhline(0.75, color='tab:gray', linestyle=':')  # Accuracy 0.75

    ax.set_xlim([1, len(x_total)])  # 1 to not plot trial 0
    ax.set_ylabel(ylabel)
    ax.set_yticks(list(np.arange(0, 1.25, 0.25)))
    ax.set_yticklabels(['0', '', '0.5', '', '1'])
    sns.despine(ax=ax, bottom=True)


def get_sync(df_spikes, df_ttl, time_win=[-2, 0], bin_size=0.02, method='anal', smooth=False):
    """
    Compute the synchrony of a PSTH. This is a measure computed per trial.
    :param df_spikes: DataFrame with spike times of a given cluster
    :param time_win: Time window of interest before and after the event (in seconds)
    :param bin_size: Size of the bins in seconds for the PSTH (default: 0.02 s)
    :param method: Method to compute synchrony. Options: 'anal' (analytical formula) or 'shuffles' (shuffles method)
    """

    # # Sort in place cluster_info by depth
    # cluster_info.sort_values('depth', inplace=True)
    # # Find clusters with depth minor or equal to 1500
    # cortex_clusters = cluster_info[cluster_info.depth <= 1500].cluster_id
    # # Slice df_spikes with cortex clusters
    # df_spikes = df_spikes[df_spikes.cluster.isin(cortex_clusters)]
    # # Sort in place cluster_info by Amplitude
    # cluster_info.sort_values('Amplitude', ascending=False, inplace=True)
    # amplitude_clusters = cluster_info[0:len(cortex_clusters)].cluster_id

    peri_event_spikes = get_peri_event_spikes(df_spikes, df_ttl, time_win=time_win, scale=0)
    bins, psth = compute_psth(peri_event_spikes, time_win=time_win, bin_size=bin_size)
    # psth = psth / df_spikes.cluster.nunique() / bin_size  # Normalize by the number of clusters and bin size
    psth_mean = np.mean(psth, axis=1)
    psth_std = np.std(psth, axis=1)

    # Analytical formula for synchrony
    # (denominator is the std of the mean as a proxy for lambda in a Poisson process). Fast, dependent on normalization
    if method == 'anal':
        sync = psth_std / np.sqrt(psth_mean)

    # # Shuffles method for synchrony (slow and computationally expensive)
    # elif method == 'shuffles':
    #     psth_std_shuffles = []
    #     for i in range(10):
    #         peri_event_spikes = get_peri_event_spikes(df_spikes, df_ttl, time_win=time_win, scale=2)
    #         bins, psth = compute_psth(peri_event_spikes, time_win=time_win, bin_size=bin_size)
    #         psth = psth / df_spikes.cluster.nunique() / bin_size  # Normalize by the number of clusters and bin size
    #         psth_std_shuffles.append(np.std(psth, axis=1))
    #     psth_std_shuffles = np.array(psth_std_shuffles)
    #     psth_std_shuffles = np.mean(psth_std_shuffles, axis=0)
    #     sync = psth_std / psth_std_shuffles

    # Invariant to scaling
    elif method == "shuffles":
        psth_std_shuffles = []
        for spike_times in peri_event_spikes:
            spike_times = spike_times.values
            shuffled_spike_times = []
            for s in range(10):
                isis = np.diff(spike_times)  # Calculate ISIs
                np.random.shuffle(isis)  # Shuffle ISIs
                new_spike_times = np.cumsum(np.insert(isis, 0, spike_times[0]))
                shuffled_spike_times.append(new_spike_times)
            _, psth_shuffle = compute_psth(shuffled_spike_times, time_win=time_win, bin_size=bin_size)
            # psth_shuffle = psth_shuffle/ (df_spikes.cluster.nunique() * bin_size)  # Normalize by the number of clusters and bin size
            psth_std_shuffles.append(np.mean(np.std(psth_shuffle, axis=1)))
        sync = psth_std / psth_std_shuffles

    # Normalize
    # sync = (sync - 1) / df_spikes.cluster.nunique()

    if smooth:
        # Compute rolling average
        sync = compute_window(sync, 20)
        sync = np.array(sync)

    return sync


def plot_sync(df_spikes, df_ttl, df_behavior, time_win=[-2, 0], method='anal', smooth=False, ax=None):
    """
    Plot the synchrony of a PSTH.
    :param sync: Synchrony
    :param ax: Axes to plot the synchrony (default: None)
    """

    # Get default figure window size
    figsize_default = mpl.rcParams['figure.figsize']
    figsize = (figsize_default[0] * 3, figsize_default[1])

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    sync = get_sync(df_spikes, df_ttl, time_win=time_win, bin_size=0.02, method=method, smooth=smooth)

    # Plot synchrony
    ax.plot(sync, color='k')
    ax.axhline(np.mean(sync), color='tab:red')
    ax.plot(sync.argmin(), sync.min(), marker='o', color='tab:red')
    ax.plot(sync.argmax(), sync.max(), marker='o', color='tab:red')
    ax.set_xlim([1, len(df_behavior)])  # 1 to not plot trial 0
    ax.set_xlabel('Trial')
    ax.set_ylabel('Sync')
    ax.set_title(f'{df_behavior.Session.unique()[0]} ({method} method)')

    print('The minimum sync is', sync.min(), 'at trial', sync.argmin())
    print('The maximum sync is', sync.max(), 'at trial', sync.argmax())


def plot_sync_split(df_spikes, df_ttl, df_behavior):
    # Plot autocorrelogram for high vs low sync trials
    sync = get_sync(df_spikes, df_ttl, time_win=[-2, 0], bin_size=0.02, method='anal', smooth=False)
    mean_sync = np.mean(sync)

    # Find trials of high and low sync trials as above and below the mean
    high_sync = df_behavior[sync > mean_sync].Trial.tolist()
    low_sync = df_behavior[sync < mean_sync].Trial.tolist()

    high_ccg, high_lags = plot_autocorrelogram(df_spikes[df_spikes.Trial.isin(high_sync)], bin_size=0.001, window=[-1000, 1000], ax=None)
    low_ccg, low_lags = plot_autocorrelogram(df_spikes[df_spikes.Trial.isin(low_sync)], bin_size=0.001, window=[-1000, 1000], ax=None)

    plt.figure()
    plt.plot(high_lags, high_ccg, color='tab:purple', label='High sync')
    plt.plot(low_lags, low_ccg, color='tab:green', label='Low sync')
    plt.xlabel('Time lag (ms)')
    plt.ylabel('Correlation')
    plt.legend(frameon=False)
    sns.despine()


def plot_corr_session(df_spikes, df_ttl, cluster_info, df_behavior, time_win=[-1, 3], bin_size=0.1, ax=None):
    """
    Plot the correlation matrix of the session.
    :param ax: Axes to plot the correlation matrix (default: None)
    """

    if ax is None:
        fig, ax = plt.subplots()
        annot = True
    else:
        annot = False

    subject = df_behavior.Subject.unique()[0]
    date = df_behavior.Date.unique()[0]

    peri_event_spikes = get_peri_event_spikes(df_spikes, df_ttl, time_win)
    _, psth = compute_psth(peri_event_spikes, time_win, bin_size)
    psth = psth / len(cluster_info)
    mfr, sfr = plot_mfr(psth, bin_size, ax=None)
    plt.close()  # Close mfr plot
    sync = get_sync(df_spikes, df_ttl, time_win=[-2, 0], bin_size=0.02, method='anal', smooth=False)
    _, rep_bias, _, _, _, _ = get_roll_avg(df_behavior, kind='rep_bias')
    _, accuracy, _, _, _, _ = get_roll_avg(df_behavior, kind='side')
    _, misses, _, _, _, _ = get_roll_avg(df_behavior, kind='miss')

    # Create DataFrame
    data = {
        'mfr': mfr,
        'sync': sync,
        'rep_bias': rep_bias,
        'accuracy': accuracy,
        'misses': misses
    }
    df = pd.DataFrame(data)
    corr = df.corr()

    # Plot correlation matrix
    sns.heatmap(corr, annot=annot, cmap='coolwarm', vmin=-1, vmax=1, square=True, ax=ax)
    ax.set_title(f'Mouse {subject}: {date}')


def session_report(ephys_id):

    preprocessed = preprocess(ephys_id)
    df_ttl, df_behavior, n_trials, df_spikes, cluster_info, timeline = (
        tuple(preprocessed))
    subject = df_behavior.Subject.unique()[0]
    date = df_behavior.Date.unique()[0]

    # figsize = (11.69, 8.27)  # A4 size in inches landscape
    figsize = (8.27, 11.69)  # A4 size in inches portrait

    # Set subplots layout with mosaic
    mosaic = [['Timeline', 'PopGroupDist', 'PopPSTH', 'CorrMatrix'],
              ['MFR', 'MFR', 'MFR', 'MFR'],
              ['Sync', 'Sync', 'Sync', 'Sync'],
              ['RepBias', 'RepBias', 'RepBias', 'RepBias'],
              ['AccuracySide', 'AccuracySide', 'AccuracySide', 'AccuracySide'],
              ['Misses', 'Misses', 'Misses', 'Misses']]
    fig, ax_dict = plt.subplot_mosaic(mosaic, figsize=figsize)

    # Plot panels
    # timeline = print_timeline(continuous, events, df_behavior, df_spikes)
    plot_timeline(timeline, ax=ax_dict['Timeline'])
    plot_group_clusters_dist(cluster_info, ax=ax_dict['PopGroupDist'])
    bins, psth = plot_pop_psth(df_spikes, df_ttl, cluster_info, ax=ax_dict['PopPSTH'])
    plot_mfr(psth, ax=ax_dict['MFR'])
    plot_sync(df_spikes, df_ttl, df_behavior, time_win=[-2, 0], method='anal', smooth=True, ax=ax_dict['Sync'])

    x_total, y_total, x_0, y_0, x_1, y_1 = get_roll_avg(df_behavior, kind='side')
    plot_roll_avg(x_total, y_total, x_0, y_0, x_1, y_1, kind='side', ax=ax_dict['AccuracySide'])
    x_total, y_total, x_0, y_0, x_1, y_1 = get_roll_avg(df_behavior, kind='rep_bias')
    plot_roll_avg(x_total, y_total, x_0, y_0, x_1, y_1, kind='rep_bias', ax=ax_dict['RepBias'])
    x_total, y_total, x_0, y_0, x_1, y_1 = get_roll_avg(df_behavior, kind='miss')
    plot_roll_avg(x_total, y_total, x_0, y_0, x_1, y_1, kind='miss', ax=ax_dict['Misses'])

    plot_corr_session(df_spikes, df_ttl, cluster_info, df_behavior, time_win=[-1, 3], bin_size=0.1, ax=ax_dict['CorrMatrix'])

    # Aesthetics
    ax_dict['Timeline'].set_title('')
    ax_dict['PopGroupDist'].set_xlabel('')
    ax_dict['PopPSTH'].set_title('')
    ax_dict['CorrMatrix'].set_title('')
    ax_dict['MFR'].set_xlabel('')
    ax_dict['MFR'].set_xticklabels([])
    ax_dict['Sync'].set_title('')
    ax_dict['Sync'].set_xlabel('')
    ax_dict['Sync'].set_xticklabels([])
    ax_dict['AccuracySide'].set_xticklabels([])
    ax_dict['Misses'].spines['bottom'].set_visible(True)
    ax_dict['Misses'].set_xlabel('Trial')
    ax_dict['MFR'].set_xlim([1, len(df_behavior)])  # 1 to not plot trial 0
    sns.despine(ax=ax_dict['MFR'], bottom=True)
    sns.despine(ax=ax_dict['Sync'], bottom=True)

    plt.suptitle(f'Mouse {subject}: {date}')
    plt.tight_layout()


def get_rt(df_behavior):
    """
    Compute the reaction time (RT) of the licks of a behavioral session.
    :param df_behavior: DataFrame with the behavioral data
    :return: Reaction time (RT) of the licks per trial
    """

    licks_left = df_behavior.Port1In.copy()
    licks_right = df_behavior.Port2In.copy()
    licks = licks_left + licks_right

    rt = []
    n_licks = []

    for trial in range(len(df_behavior)):

        # Curate licks (remove those that happened before the response window open or after the ITI ends)
        licks[trial] = [lick for lick in licks[trial] if df_behavior.RespWinStart[trial] <= lick <=
                        df_behavior.RespWinEnd[trial] + df_behavior.ITI[trial]]

        n_licks.append(len(licks[trial]))

        if df_behavior.Miss[trial] == 1:
            rt.append(np.nan)
        else:
            # rt.append(df_behavior.RespWinEnd[trial] - df_behavior.RespWinStart[trial])
            rt.append(df_behavior.RespWinLen[trial])

        # Align licks to StimStart
        licks[trial] = [lick - df_behavior.StimStart[trial] for lick in licks[trial]]

    return licks, n_licks, rt


########################################################################################################################

"""
TO DO:
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