# Python standard libraries
from pathlib import Path
from open_ephys.analysis import Session
import numpy as np
import pandas as pd
from scipy.stats import zscore, sem
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Ephys specific libraries
from neo.core import SpikeTrain
from quantities import ms, s, Hz
from elephant.statistics import time_histogram, instantaneous_rate, fanofactor, mean_firing_rate
from elephant.kernels import GaussianKernel

# My own libraries
from my_fun.my_fun import do_sounds_dict
from parse.parse_v2 import parse_v2

########################################################################################################################

# Load raw ephys data
# From https://github.com/open-ephys/open-ephys-python-tools/blob/main/src/open_ephys/analysis/README.md

# Create Session object
id = '007_2024-06-24_17-47-22'  # Define the session ID
# directory = Path.home() / 'Documents' / 'Open Ephys' / id  # Ephys PC
directory = Path() / 'D:' / 'Data' / id  # Personal laptop
session = Session(directory)
recordnode = session.recordnodes[0]  # Get the first recordnode
recording = recordnode.recordings[0]  # Get the first recording

########################################################################################################################

# Synchronizing timestamps
# This must be done prior to accessing the continuous or event data
recording.add_sync_line(1,  # TTL line number
                        109,  # processor ID
                        'ProbeA-AP',  # stream name
                        main=True)  # use as the main stream

recording.add_sync_line(1,  # TTL line number
                        109,  # processor ID
                        'ProbeA-LFP',  # stream name
                        main=False)  # align to the main stream

# recording.compute_global_timestamps()  # Generate new global timestamps column
recording.compute_global_timestamps(overwrite=True)  # Overwrite existing timestamps
# recording.events.sort_values('global_timestamp', inplace=True)  # Sort events by global timestamp
recording.events.sort_values('timestamp', inplace=True)  # Sort events by global timestamp

########################################################################################################################

# Loading continuous data
continuous = recording.continuous  # Get the continuous data
continuous_AP = continuous[0]  # Get the continuous AP data
continuous_LFP = continuous[1]  # Get the continuous LFP data
# data = recording.continuous[0].get_samples(start_sample_index=0, end_sample_index=10000)
# data = continuous_AP.get_samples(start_sample_index=0, end_sample_index=continuous_AP.sample_numbers[-1])  # Too large
# data = pd.DataFrame(data)  # Convert data to a pandas DataFrame

########################################################################################################################

# Loading event data
events = recording.events
events_AP = events[events.stream_index == 0]  # Action Potential (AP) stream
events_AP.reset_index(drop=True, inplace=True)
events_LFP = events[events.stream_index == 1]  # Local Field Potential (LFP) stream
events_LFP.reset_index(drop=True, inplace=True)

stream_index = 0  # 0 for AP stream, 1 for LFP stream
# Only works with AP stream so far. AP stream always has mora data (maybe due to higher sampling rate?)
if stream_index == 0:
    events_stream = events_AP
elif stream_index == 1:  # LFP stream
    events_stream = events_LFP

########################################################################################################################
########################################################################################################################

# Recover TTLs from ephys data

# TTLs from event data (much faster). Default
ttl_precision = 4  # Precision of the TTLs (in decimal places of seconds)
events_stream.loc[events_stream['state'] == 1, 'ON'] = events_stream['timestamp']  # Onset of TTL
events_stream.loc[events_stream['state'] == 0, 'OFF'] = events_stream['timestamp']  # Offset of TTL
events_stream.OFF = events_stream.OFF.shift(-1)  # Shift the OFF column one row up
events_stream['Length'] = events_stream['OFF'] - events_stream['ON']  # Calculate length of TTLs
# events_stream['Length'] = events_stream.timestamp.diff()  # Even indexes are LOW and odd indexes are HIGH TTL lengths (matching state)
events_stream.Length = events_stream.Length.round(ttl_precision)  # Round TTLs to the desired precision
# events_stream = events_stream.length[1::2]  # Get only HIGH TTL lengths (the odd indexes)
# events_stream = events_stream.length[events_stream.state == 0]  # Get only HIGH TTL lengths  (alternative method)
events_stream = events_stream.dropna()  # Drop rows in which samples == 0  (ON and OFF are not NaN)
events_stream.reset_index(drop=True, inplace=True)
TTLs_events = events_stream.Length

# TTLs from continuous data (much slower)
TTLs_from_continuous = False  # Flag to use TTLs from continuous data

if TTLs_from_continuous:
    # Only possible if the sync signal was included as a continuous channel ("+" button of Neuropix pluggin was active
    # https://open-ephys.github.io/gui-docs/User-Manual/Plugins/Neuropixels-PXI.html
    if continuous_AP.metadata['num_channels'] == 385:  # If there is a 385th channel with TTLs
        samples = continuous_AP.samples[:, 384]  # Get the samples from channel 385 (TTLs)
        timestamps = continuous_AP.timestamps
        s1 = pd.Series(samples, name='samples')
        s2 = pd.Series(timestamps, name='timestamps')
        df_ttl = pd.concat([s1, s2], axis=1)  # Put the data in a single dataframe
        df_ttl['diff'] = df_ttl.samples.diff()  # Look for the places where there is a change in the TTL state
        df_ttl = df_ttl.loc[(df_ttl['diff'] == 1) | (df_ttl['diff'] == -1)]  # Remove values without a change (diff = 0)
        df_ttl.loc[df_ttl['samples'] == 1, 'ON'] = df_ttl['timestamps']  # Onset of TTL
        df_ttl.loc[df_ttl['samples'] == 0, 'OFF'] = df_ttl['timestamps']  # Offset of TTL
        df_ttl.OFF = df_ttl.OFF.shift(-1)  # Shift the OFF column one row up
        df_ttl['Length'] = df_ttl['OFF'] - df_ttl['ON']  # Calculate length of TTLs
        # df_ttl['Length'] = df_ttl['timestamps'].diff()  # Even indexes are LOW and odd indexes are HIGH TTL lengths (matching state)
        df_ttl.Length = df_ttl.Length.round(ttl_precision)  # Round TTLs to the desired precision
        # df_ttl.length = df_ttl.Length.shift(-1)  # Shift the TTL column one row up
        df_ttl = df_ttl.dropna()  # Drop rows in which samples == 0  (ON and OFF are not NaN)
        df_ttl.reset_index(drop=True, inplace=True)
        # TTLs_continuous = df_ttl.Length[1::2]  # Get only HIGH TTL lengths (the odd indexes)
        # TTLs_continuous = df_ttl.Length[df_ttl.samples == 0]  # Get only HIGH TTL lengths  (alternative method)
        TTLs_continuous = df_ttl.Length
        TTLs = TTLs_continuous  # Use TTLs from continuous data

        # Check if TTLs from continuous and event data match
        assert TTLs_continuous.equals(TTLs_events), 'TTLs from continuous and event data do not match'
    else:
        print('There is no 385th channel with TTLs in the continuous data. Using TTLs from event data only.')
        TTLs = TTLs_events  # Use TTLs from event data
else:
    TTLs = TTLs_events  # Use TTLs from event data

# Create variable called 'test' that contains the columns: 'ON', 'OFF' and 'length' from events_stream
df_ttl = events_stream[['ON', 'OFF', 'Length']]  # Create a new DataFrame with only the columns 'ON', 'OFF' and 'Length'

########################################################################################################################

# Recover sounds filenames and sounds orders from TTLs

sounds_dict = do_sounds_dict(0.0003, 0.0078, 26, ttl_precision)  # Create TTL-letter mapping dictionary
sounds_dict.update({'load': 0.0085, 'play': 0.0090, 'stop': 0.0095, 'wait': 0.1})  # Update dictionary with sound orders
sounds_dict_keys = list(sounds_dict.keys())  # Get sounds_dict keys
sounds_dict_values = list(sounds_dict.values())  # Get sounds_dict values
tolerance = 0.0003 / 2  # Half of the step size between TTL lengths


def do_sounds_dict_inv(TTLs):  # Move function to my_fun
    """
    Get the key of a dictionary given a value. The inverse of the do_sounds_dict function.
    """
    for i in range(len(sounds_dict)):
        if sounds_dict_values[i] - tolerance < TTLs < sounds_dict_values[i] + tolerance:
            return sounds_dict_keys[i]
        else:  # If the TTL length is not in the dictionary
            pass  # Do nothing


keys = TTLs.apply(do_sounds_dict_inv).dropna().to_list()  # Get sounds_dict keys given a TTL length as a value
df_ttl['key'] = keys  # Add keys column to df_ttl

# Get the sound filenames per trial
letters = [key for key in keys if len(key) == 1]  # Get only the keys with 1 element (letters)
ephys_filenames = [letters[i:i + 3] for i in range(0, len(letters), 3)]  # Create a list of lists with 3 keys each
ephys_filenames = [ephys_filenames[i][0] + ephys_filenames[i][1] + ephys_filenames[i][2] for i in
                   range(len(ephys_filenames))]  # Concatenate the 3 keys into a single string
ephys_filenames = pd.Series(ephys_filenames)  # Convert list to pandas Series

# Get the sound orders per trial
orders = [key for key in keys if len(key) == 4]  # Get only the keys  with 4 elements (orders)
orders = [orders[i:i + 3] for i in range(0, len(orders), 3)]  # Make a list of lists with 3 orders each
orders = pd.Series(orders)

# Check if the number of filenames and orders match (should be 1 per trial)
assert len(ephys_filenames) == len(orders), 'Number of filenames and orders do not match'

########################################################################################################################
########################################################################################################################

# Loading behavior data
path_behavior = r"C:\Users\alexi\Downloads\007_stage_training_v5_20240624-180217\007_stage_training_v5_20240624-180217.csv"
df_behavior = parse_v2(path_behavior)

########################################################################################################################
########################################################################################################################

# Compare sounds filenames from behavior and ephys data. Way to recover the trials structure from ephys data

# Found sounds sent from Bpod (Filename) that does not match with those received by Arduino (Filename2) and get indexes
index = df_behavior[df_behavior['FilesMatch'] == 0].index

if len(index) == 0:
    print('All sounds sent by Bpod match those received by Arduino')
    behavior_filenames = df_behavior['Filename']
else:
    print('Trials which sounds sent by Bpod do not match those received by Arduino:')
    print(df_behavior.iloc[index])
    behavior_filenames = df_behavior['Filename2']

# Get the number of trials from behavior and ephys data
n_trials_behavior = len(behavior_filenames)
n_trials_ephys = len(ephys_filenames)
diff_n_trials = n_trials_behavior - n_trials_ephys

# Check if the number of trials match with a tolerance of 1
assert abs(diff_n_trials) <= 1, 'Number of trials from behavior and ephys data do not match'

# Check if the number of filenames match
if len(behavior_filenames) != len(ephys_filenames):
    if np.sign(diff_n_trials) == -1:
        print(f'There is(are) {abs(diff_n_trials)} more ephys trial(s) than behavior trial(s)')
    elif np.sign(diff_n_trials) == 1:
        print(f'There is(are) {abs(diff_n_trials)} more behavior trial(s) than ephys trial(s)')

# Find shorter (minimum) length of both number of trials (from ephys and behavior) to avoid error when comparing
# Usually, the behavior data has 1 less filename than the ephys data because the last trial did not finish
n_trials = min(len(ephys_filenames), len(behavior_filenames))

# Compare ephys_filenames and behavior_filenames
sounds_match = [ephys_filenames[i] == behavior_filenames[i] for i in range(n_trials)]

# Check if all sounds from behavior and ephys match
assert all(sounds_match), 'Sounds from behavior and ephys do not match'

# Check if all values of sounds_match are True
if all(sounds_match):
    print('All sounds from behavior and ephys match')
else:
    print('Sounds from behavior and ephys do not match')
    # Get the indexes of the sounds that do not match
    sounds_mismatch_index = [i for i, x in enumerate(sounds_match) if not x]

########################################################################################################################
########################################################################################################################

# Loading spike sorted data (KS4)
path_ks4 = Path() / directory / 'Record Node 101' / 'experiment1' / 'recording1' / 'continuous' / \
          'Neuropix-PXI-109.ProbeA-AP' / 'kilosort4'
path_phy2 = Path.home() / 'Downloads' / 'Phy2'

# https://phy.readthedocs.io/en/latest/visualization/
spikes_times = np.load(path_ks4 / 'spike_times.npy')  # From KS4
spike_clusters = np.load(path_phy2 / 'spike_clusters.npy')  # From Phy2
cluster_group = pd.read_csv(path_phy2 / 'cluster_group.tsv', sep='\t')  # From Phy2

# Create pandas DataFrame with spike times and clusters
df_spikes = pd.DataFrame({'spike_times': spikes_times, 'spike_clusters': spike_clusters})

# Merge df_spikes with cluster_group
df_spikes = pd.merge(df_spikes, cluster_group, left_on='spike_clusters', right_on='cluster_id')

# Drop clusters_id column (redundant)
df_spikes.drop('cluster_id', axis=1, inplace=True)

# Rename spike_clusters column to cluster
df_spikes.rename(columns={'spike_clusters': 'cluster'}, inplace=True)

# Get the number of good and mua clusters (KSLabel)
n_clusters = len(cluster_group)
n_good_clusters = len(cluster_group[cluster_group['group'] == 'good'])
n_mua_clusters = len(cluster_group[cluster_group['group'] == 'mua'])
n_noise_clusters = len(cluster_group[cluster_group['group'] == 'noise'])

# Plot a bar graph with the number of good, mua and noise clusters
plt.figure()
x = ['good', 'mua', 'noise']
height = [n_good_clusters, n_mua_clusters, n_noise_clusters]
labels = ['good', 'mua', 'noise']
colors = ['tab:green', 'tab:orange', 'tab:gray']
plt.bar(x, height, label=labels, color=colors)
plt.legend()

print(f'Total number of clusters: {n_clusters}\n'
      f'Number of good clusters: {n_good_clusters} ({round(n_good_clusters/n_clusters*100)}%)\n'
      f'Number of mua clusters: {n_mua_clusters} ({round(n_mua_clusters/n_clusters*100)}%)\n'
      f'Number of noise clusters: {n_noise_clusters} ({round(n_noise_clusters/n_clusters*100)}%)')

# Select only clusters labelled as good or mua  (drop noise clusters)
df_spikes = df_spikes.loc[(df_spikes.group == 'good') | (df_spikes.group == 'mua')]

# Update the number of clusters
n_clusters = len(df_spikes.cluster.unique())


def sort_clusters(df_spikes):
    """
    Sort clusters from df by the number of spikes
    :param df_spikes: DataFrame with spike times and clusters
    """

    # Get the number of spikes per cluster
    n_spikes = df_spikes.groupby('cluster')['spike_times'].count()
    n_spikes = n_spikes.sort_values(ascending=True)
    clusters = n_spikes.index
    df_clusters = pd.DataFrame({'cluster': clusters, 'n_spikes': n_spikes})
    df_clusters.reset_index(drop=True, inplace=True)
    # Add cluster_group to df_clusters_sum
    df_clusters = pd.merge(df_clusters, cluster_group, left_on='cluster', right_on='cluster_id')
    df_clusters.drop('cluster_id', axis=1, inplace=True)
    return df_clusters


# Sort clusters by the number of spikes
df_clusters = sort_clusters(df_spikes)
clusters = df_clusters.cluster.unique()  # Get the unique clusters

# Transform spike times to seconds
sample_rate = continuous_AP.metadata['sample_rate']
# df.insert(1, 'spike_times_s', df.spike_times / sample_rate)  # Insert spike_times_s column
df_spikes['spike_times'] = df_spikes['spike_times'] / sample_rate  # Overwrites original spike_times column
print(f'Min spike time: {round(min(df_spikes.spike_times / 60))} min\n'
      f'Max spike time: {round(max(df_spikes.spike_times / 60))} min')

# Plot the first minute
plt.figure()
plt.scatter(df_spikes[(df_spikes.spike_times < 60)].spike_times,
            df_spikes[(df_spikes.spike_times < 60)].cluster,
            marker='|', linestyle='None', color='k')
plt.xlabel('Time (s)')
plt.ylabel('Cluster ID')
plt.title('First min. of recording')

########################################################################################################################

# Print information about the ephys and behavior data

# Ephys recording session info
first_timestamp = continuous_AP.timestamps[0]
last_timestamp = continuous_AP.timestamps[-1]
len_recording = last_timestamp - first_timestamp
first_event = events_stream.timestamp.iloc[0]
last_event = events_stream.timestamp.iloc[-1]
len_ephys_n_behavior = (last_event - first_event)
first_spike = df_spikes.spike_times.iloc[0]
last_spike = df_spikes.spike_times.iloc[-1]
len_spikes = last_spike - first_spike

print(f'The recording started after {round(first_timestamp / 60)} min. of acquisition')
print(f'The behavior first event happened after {round(first_event / 60)} min. of acquisition '
      f'({round((first_event - first_timestamp) / 60)} min. after recording started)')
print(f'The recording lasted {round(len_recording / 60)} min.')
print(f'The behavior (from ephys data) lasted {round(len_ephys_n_behavior / 60)} min.')
print(f'Spikes were recorded from {round(first_spike / 60)} min. to {round(last_spike / 60)} min. ')


# Behavior session info
behavior_start = df_behavior['TrialStart'].iloc[0]  # Get the first trial start
behavior_end = df_behavior['TrialEnd'].iloc[-1]  # Get the last trial end
behavior_len = behavior_end - behavior_start  # Get the total behavior length

print(f'The behavior (from behavior data) lasted {round(behavior_len / 60)} min.')

# Check if the length of the behavioral session from behavioral and ephys data match
assert round(len_ephys_n_behavior / 60) == round(behavior_len / 60), \
    'Length of behavioral session from behavioral and ephys data do not match'

########################################################################################################################

# Temporal alignment of ephys and behavior data (from continuous data)
df_ttl = df_ttl[df_ttl['key'] == 'play']  # Keep only rows with key == play (1 TTL per trial)
df_ttl['Trial'] = np.arange(len(df_ttl))  # Prepare a column with trial indexes for merging
df_ttl = df_ttl.iloc[:len(df_behavior)]  # Keep only the first n TTLs (n = number of trials in behavior data)
df_ttl.reset_index(drop=True, inplace=True)  # Reset index
assert len(df_ttl) == len(df_behavior), 'Number of stimulus onset TTLs and trials in behavior data do not match'

# Align timestamps to the first timestamp (start at 0). Not in use
# df_ttl.timestamps = df_ttl.timestamps - first_timestamp

df_aligned = pd.merge(df_behavior, df_ttl, on=['Trial'])  # Merge behavior and TTLs dataframes

# Transform behavioral FSM states from relative (0 = start of trial) to absolute (cumulative) timestamps
# Only 'TrialStart' and 'TrialEnd' are in absolute timestamps, the rest are in relative timestamps
transform_states = ['StimStart', 'StimEnd', 'RespWinStart', 'RespWinEnd']

for state in transform_states:
    df_aligned[state] = df_aligned[state] + df_aligned.TrialStart

# When did the behavioral session start in the ephys clock
df_aligned['BehaviorStart'] = df_aligned.ON - df_aligned.StimStart

# Plot the drift of the behavioral session start timestamps
plt.figure()
plt.plot(df_aligned.BehaviorStart)
plt.xlabel('Trial')
plt.ylabel('Time (s)')
plt.title(f'Drift behavioral session start timestamps '
          f'({round((df_aligned.BehaviorStart.max() - df_aligned.BehaviorStart.min())*1000)} ms)')

assert all(round(df_behavior.TrialStart + df_aligned.BehaviorStart + df_behavior.StimStart, ttl_precision) ==
           round(df_ttl.ON, ttl_precision))

# Align FSM states to the start of the behavioral session in the ephys clock
aligned_states = ['TrialStart', 'TrialEnd', 'StimStart', 'StimEnd', 'RespWinStart', 'RespWinEnd']

for state in aligned_states:
    df_aligned[state] = df_aligned[state] + df_aligned.BehaviorStart

# Check if the lengths of the states before the alignment match after the alignment
assert all(round(df_aligned.TrialEnd - df_aligned.TrialStart, ttl_precision) == round(df_aligned.TrialLen, ttl_precision))
assert all(round(df_aligned.StimEnd - df_aligned.StimStart, ttl_precision) == round(df_aligned.StimLen, ttl_precision))
assert all(round(df_aligned.RespWinEnd - df_aligned.RespWinStart, ttl_precision) == round(df_aligned.RespWinLen, ttl_precision))

assert all(df_aligned.StimStart == df_aligned.ON)  # Check if StimStart and df_ttl.ON match

# Align timestamps of PortXIn/Out to the start of the behavioral session in the ephys clock
port_states = ['Port1In', 'Port1Out', 'Port2In', 'Port2Out']
for j in range(len(port_states)):
    for i in range(len(df_aligned)):
            df_aligned[port_states[j]][i] = [x + df_aligned['TrialStart'][i] for x in df_aligned[port_states[j]][i]]

# Assign spikes to the corresponding trial
df_spikes['Trial'] = np.nan  # Create a new column with NaN values (default trial number)

# With np.select (overkill as there is only one condition)
# for i, row in df_aligned.iterrows():
#     # create a list of our conditions
#     condlist = [(df_spikes.spike_times > df_aligned['TrialStart'].iloc[i]) & (df_spikes.spike_times < df_aligned['TrialEnd'].iloc[i]),
#                 # Spike times within the trial
#                  (df_spikes.spike_times < df_aligned['TrialStart'].iloc[i]) | (df_spikes.spike_times > df_aligned['TrialEnd'].iloc[i])]
#                 # Spike times outside the trial
#
#     # create a list of the values we want to assign for each condition
#     choices = [df_aligned['Trial'].iloc[i].astype(int), df_spikes['Trial']]
#
#     # create a new column and use np.select to assign values to it using our lists as arguments
#     df_spikes['Trial'] = np.select(condlist, choices)

# With np.where (more appropriate as there is only one condition)
for i, row in df_aligned.iterrows():
    condition = (df_spikes.spike_times > df_aligned['TrialStart'].iloc[i]) & (df_spikes.spike_times < df_aligned['TrialEnd'].iloc[i])
    df_spikes['Trial'] = np.where(condition, df_aligned['Trial'].iloc[i].astype(int), df_spikes['Trial'])

# df_spikes.dropna(inplace=True)  # Drop rows with NaN values in the Trial column  (spikes outside trial times)

# Check if the number of trials in behavior and spikes dataframes match
# assert len(df_spikes.Trial.unique()) == len(df_aligned), \
#     'Number of trials in behavior and spikes dataframes do not match'

# df = pd.merge(df_aligned, df_spikes, on=['Trial'])  # Merge behavior and spikes dataframes. Very heavy and redundant
# Actually only need the trial column in df_spikes. No need to copy the +80 columns from df_aligned for every spike

# df = pd.merge(df_aligned.Trial, df_spikes, on=['Trial'])  # Merge trials and spikes dataframes.

########################################################################################################################
########################################################################################################################

# For debugging purposes
cluster = 8
align = 'StimStart'

########################################################################################################################

# Plot raw data per unit

# Plot rasters

def plot_raster(cluster, align='StimStart'):
    """
    Plot a raster plot of a given cluster aligned to a specific event.
    :param cluster: Cluster ID
    :param align: Event to align the raster plot to (default: StimStart)
    """

    df_cluster = df_spikes[df_spikes.cluster == cluster]  # Slice DataFrame of given cluster
    # df_cluster = df_cluster[df_cluster.Hit == 1]  # Select only correct trials

    plt.figure()
    # This line is if df_spikes also contains behavior data like trial number, stimulus start, etc.
    # plt.plot(df_cluster.spike_times - df_cluster[align], df_cluster.Trial, marker='|', linestyle='None', color='k')

    plt.axvline(0, color='r', label='Stimulus')
    plt.axvline(df_cluster.Delay.unique()[0], color='b', label='Delay')
    plt.axvline(df_cluster.StimDur.unique()[0] + df_cluster.Delay.unique()[0], color='g', label='Response')
    plt.xlabel('Time (s)')
    plt.ylabel('Trial')
    plt.title(f'Raster aligned to {align} (cluster {cluster}: {df_cluster.group.unique()[0]})')
    plt.legend(loc='upper right', frameon=False)

    # Save the figure in Desktop folder called 'rasters'
    plt.savefig(Path.home() / 'OneDrive' / 'Escritorio' / 'rasters' / f'cluster_{cluster}_aligned_{align}.png')
    plt.close()


def plot_raster2(cluster):
    """
    THIS USES AN APPROACH THAT IS COMPLETELY INDEPENDENT OF THE BEHAVIOR DATA!!!
    LOOP OVER STIM ONSET TTLs INSTEAD OF TRIALS, THE EVENTS REGISTERED IN THE EPHYS DATA

    Plot a raster plot of a given cluster aligned to a specific event. 2 rasters (correct and error trials).
    :param cluster: Cluster ID
    :param align: Event to align the raster plot to (default: StimStart)
    """

    df_cluster = df_spikes[df_spikes.cluster == cluster]  # Slice DataFrame of given cluster
    # df_cluster = df_cluster[df_cluster.Hit == 1]  # Select only correct trials
    group = df_cluster.group.unique()[0]

    stim_dur = df_behavior.StimDur.unique()[0]
    delay = df_behavior.Delay.unique()[0]
    go_cue = stim_dur + delay

    # plt.figure()
    fig, ax = plt.subplots(2, 1)

    time_win = 2  # Time window of interest before and after the event (in seconds). Needs to be positive!

    # Loop over trials (timestamps of stimulus onset)
    for trial in range(n_trials):

        # Skip missed trials
        if df_behavior.Miss[trial] == 1:
            continue
        else:
            # print(f'Trial {trial}')
            stim_onset = df_ttl.ON[trial]  # Get the stimulus onset timestamp
            # Select only spikes within the time window of interest around the event
            spikes = df_cluster[(df_cluster.spike_times > stim_onset - time_win) &
                                (df_cluster.spike_times < stim_onset + time_win)].spike_times
            spikes = spikes - stim_onset  # Align spikes to the event
            # plt.eventplot(spikes, lineoffsets=trial, color='k')

            # Plot correct and error trials in different subplots
            if df_behavior.Hit[trial] == 0:  # Error trial
                ax[0].eventplot(spikes, lineoffsets=trial, color='tab:red')
            elif df_behavior.Hit[trial] == 1:  # Correct trial
                ax[1].eventplot(spikes, lineoffsets=trial, color='tab:green')

    ax[0].axvline(0, color='k', label='Stimulus')
    ax[1].axvline(0, color='k', label='Stimulus')
    ax[0].axvline(delay, ls='--', color='tab:gray', label='Delay')
    ax[1].axvline(delay, ls='--', color='tab:gray', label='Delay')
    ax[0].axvline(go_cue, ls='--', color='tab:blue', label='Go cue')
    ax[1].axvline(go_cue, ls='--', color='tab:blue', label='Go cue')

    ax[0].set_xticklabels([])
    ax[0].set_ylabel('Trial')
    ax[1].set_ylabel('Trial')
    ax[1].set_xlabel('Time (s)')

    ax[0].set_title('Error')
    ax[1].set_title('Correct')

    ax[0].legend(loc='upper left', frameon=False)
    fig.suptitle(f'Cluster {cluster} ({group})')

    # Save the figure in Desktop folder called 'rasters'
    plt.savefig(Path.home() / 'OneDrive' / 'Escritorio' / 'rasters2' / f'cluster_{cluster}.png')
    plt.close()



# # Plot rasters for all clusters
# for cluster in clusters:
#     print(f'Cluster {cluster} ({df_clusters[df_clusters.cluster == cluster].group.unique()[0]})')
#     # plot_raster(cluster, align='StimStart')
#     plot_raster2(cluster)

########################################################################################################################

# Plot PSTHs

def plot_psth(cluster, align='StimStart'):
    """
    Plot a PSTH of a given cluster aligned to a specific event.
    :param cluster: Cluster ID
    :param align: Event to align the PSTH to (default: StimStart)
    """

    df_cluster = df[df.cluster == cluster]  # Slice DataFrame of given cluster
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
        mfr.append(np.round(mean_firing_rate(spiketrain).magnitude * 1000, ttl_precision))  # In spikes/s
        # print(f"The mean firing rate of cluster {cluster} spiketrain is", mfr)

        # Compute time histogram with Elephant
        # https://elephant.readthedocs.io/en/latest/reference/_toctree/statistics/elephant.statistics.time_histogram.html#elephant.statistics.time_histogram
        bin_size = 1  # ms
        hist_rate = time_histogram(spiketrain, bin_size * ms, output='rate')
        hist_times.append(hist_rate.times.rescale(s).magnitude)  # Convert to seconds and store as a numpy array (not a Quantity)
        hist_firing.append(hist_rate.magnitude.flatten())

        # Compute instantaneous rate with Elephant
        # https://elephant.readthedocs.io/en/latest/reference/_toctree/statistics/elephant.statistics.instantaneous_rate.html#elephant.statistics.instantaneous_rate
        sigma = 30  # In ms (from Suzuki & Gottlieb)
        sampling_period = bin_size * ms
        kernel = GaussianKernel(sigma * ms)
        gauss_rate = instantaneous_rate(spiketrain, sampling_period=sampling_period, kernel=kernel)
        gauss_times.append(gauss_rate.times.rescale(s).magnitude)  # Convert to seconds and store as a numpy array (not a Quantity)
        gauss_firing.append(gauss_rate.magnitude.flatten())
        # gauss_firing = gauss_rate.rescale(hist_rate.dimensionality).magnitude.flatten()
        # gauss_firing = conv_firing * 1000  # Convert to spikes/s

    # Create a DataFrame with the firing rates
    df_fr = pd.DataFrame({'mfr': mfr, 'hist_times': hist_times, 'hist_firing': hist_firing, 'gauss_times': gauss_times, 'gauss_firing': gauss_firing})

    test_times = pd.DataFrame(gauss_times)
    test_firing = pd.DataFrame(gauss_firing)

    # test_hist_times = pd.DataFrame(hist_times)
    # test_hist_firing = pd.DataFrame(hist_firing)

    # Plot standard error of the mean (sem) of firing rate across trials
    plt.figure()
    plt.plot(test_times.mean(axis=0), test_firing.mean(axis=0), color='k')
    plt.fill_between(test_times.mean(axis=0), test_firing.mean(axis=0) - test_firing.sem(axis=0),
                     test_firing.mean(axis=0) + test_firing.sem(axis=0), color='k', alpha=0.25)
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



def plot_psth2(cluster, align='StimStart'):
    """
    Plot a PSTH of a given cluster aligned to a specific event.
    :param cluster: Cluster ID
    :param align: Event to align the PSTH to (default: StimStart)
    """

    df_cluster = df_spikes[df_spikes.cluster == cluster]  # Slice DataFrame of given cluster
    # df_cluster['spike_times_aligned'] = df_cluster.spike_times - df_cluster[align]  # Align spike times to the event

    time_win = 2  # Time window of interest before and after the event (in seconds). Needs to be positive!
    bins = np.arange(-time_win, time_win, bin_size)

    HIST = []
    BIN_EDGES = []
    # Loop over trials (timestamps of stimulus onset)
    for trial in range(n_trials):

        print(f'Trial {trial}')
        stim_onset = df_ttl.ON[trial]  # Get the stimulus onset timestamp
        # Select only spikes within the time window of interest around the event
        spikes = df_cluster[(df_cluster.spike_times > stim_onset - time_win) &
                            (df_cluster.spike_times < stim_onset + time_win)].spike_times
        spikes = spikes - stim_onset  # Align spikes to the event
        hist, bin_edges = np.histogram(spikes, bins)
        HIST.append(hist)
        BIN_EDGES.append(bin_edges)

    # Convert to numpy arrays
    HIST = np.array(HIST)
    BIN_EDGES = np.array(BIN_EDGES)

    # Average across trials
    HIST_mean = HIST.mean(axis=0)
    HIST_sem = sem(HIST, axis=0)
    BIN_EDGES = BIN_EDGES.mean(axis=0)

    HIST = np.array(HIST) / bin_size  # Convert to spikes/s

    # Plot histogram
    plt.figure()

    # Plot sem of HIST
    plt.plot(BIN_EDGES[:-1], HIST, color='k')
    plt.fill_between(BIN_EDGES[:-1], HIST_mean - HIST_sem, HIST_mean + HIST_sem, color='k', alpha=0.2)












# # Plot rasters for all clusters
# for cluster in clusters:
#     plot_psth(cluster, align='StimStart')

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
        df_slice = df_spikes[(df_spikes.spike_times > win_events.iloc[0]) & (df_spikes.spike_times < win_events.iloc[-1] + go_cue)]
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
for _ in range(980):
    # if df_behavior.Miss[_] == 1:  # Skip missed trials
    #     continue
    # else:
    # print(_)
    spikes = df_spikes[(df_spikes.spike_times > df_ttl.ON[_] - time_window) & (df_spikes.spike_times < df_ttl.ON[_] + time_window)].spike_times
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
plt.fill_between(BIN_EDGES[:-1], HIST.mean(axis=0) - sem(HIST, axis=0), HIST.mean(axis=0) + sem(HIST, axis=0), color='k', alpha=0.2)
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
