from pathlib import Path
from open_ephys.analysis import Session
import numpy as np
from my_fun.my_fun import do_sounds_dict
import pandas as pd
import matplotlib

matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
from parse.parse_v2 import parse_v2


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
events_stream['TTL'] = events_stream.timestamp.diff()  # Even indexes are LOW and odd indexes are HIGH TTL lengths (matching state)
events_stream.TTL = events_stream.TTL.round(ttl_precision)  # Round TTLs to the desired precision
TTLs_events = events_stream.TTL[1::2]  # Get only HIGH TTL lengths (the odd indexes)

# TTLs from continuous data (much slower)
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
    df_ttl['TTL'] = df_ttl['timestamps'].diff()  # Even indexes are LOW and odd indexes are HIGH TTL lengths (matching state)
    df_ttl.TTL = df_ttl.TTL.round(ttl_precision)  # Round TTLs to the desired precision
    df_ttl.reset_index(drop=True, inplace=True)
    TTLs_continuous = df_ttl.TTL[1::2]  # Get only HIGH TTL lengths (the odd indexes)

    assert TTLs_continuous.equals(TTLs_events)  # Check if TTLs from continuous and event data match
    if TTLs_continuous.equals(TTLs_events):
        print('TTLs from continuous and event data match')
    else:
        print('TTLs from continuous and event data do not match')

TTLs = TTLs_events  # Use TTLs from event data

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

# Get the sound filenames per trial
letters = [key for key in keys if len(key) == 1]  # Get only the keys with 1 element (letters)
ephys_filenames = [letters[i:i + 3] for i in range(0, len(letters), 3)]  # Create a list of lists with 3 keys each
ephys_filenames = [ephys_filenames[i][0] + ephys_filenames[i][1] + ephys_filenames[i][2] for i in
                   range(len(ephys_filenames))]  # Concatenate the 3 keys into a single string
ephys_filenames = pd.Series(ephys_filenames)  # Convert list to pandas Series

# Get the sound orders per trial
orders = [key for key in keys if len(key) == 4]  # Get only the keys  with 4 elements (orders)
orders = [orders[i:i + 3] for i in range(0, len(orders), 3)]  # Make a list of lists with 3 orders each

assert len(ephys_filenames) == len(orders)  # Check if the number of filenames and orders match

########################################################################################################################
########################################################################################################################

# Loading behavior data
path_behavior = r"C:\Users\alexi\Downloads\007_stage_training_v5_20240624-180217\007_stage_training_v5_20240624-180217.csv"
df_behavior = parse_v2(path_behavior)

########################################################################################################################
########################################################################################################################

# Compare sounds filenames from behavior and ephys data

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
assert abs(diff_n_trials) <= 1

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

assert all(sounds_match)  # Check if all sounds from behavior and ephys match

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

# Drop clusters_id column
df_spikes.drop('cluster_id', axis=1, inplace=True)

# Rename spike_clusters column to cluster
df_spikes.rename(columns={'spike_clusters': 'cluster'}, inplace=True)

# Get the number of good and mua clusters (KSLabel)
n_clusters = len(cluster_group)
n_good_clusters = len(cluster_group[cluster_group['group'] == 'good'])
n_mua_clusters = len(cluster_group[cluster_group['group'] == 'mua'])
n_noise_clusters = len(cluster_group[cluster_group['group'] == 'noise'])

# Plot a bar graph with the number of good, mua and noise clusters
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

# Select only clusters labelled as good or mua
df_spikes = df_spikes.loc[(df_spikes.group == 'good') | (df_spikes.group == 'mua')]

# Transform spike times to seconds
sample_rate = continuous_AP.metadata['sample_rate']
# df.insert(1, 'spike_times_s', df.spike_times / sample_rate)  # Insert spike_times_s column
df_spikes['spike_times'] = df_spikes['spike_times'] / sample_rate  # Overwrites original spike_times column
print(f'Min spike time (min): {round(min(df_spikes.spike_times / 60))}\n'
      f'Max spike time (min): {round(max(df_spikes.spike_times / 60))}')

# Plot the first minute
clusters = df_spikes.cluster.unique()
df_spikes_60s = df_spikes[(df_spikes.spike_times < 60) & (df_spikes.group == 'good')]
clusters_60s = df_spikes_60s.cluster.unique()
plt.scatter(df_spikes_60s.spike_times, df_spikes_60s.cluster, color='k')
plt.xlabel('Time (s)')
plt.ylabel('Cluster ID')
plt.title('First min. of recording')

########################################################################################################################

events_stream = events_stream[events_stream['state'] == 0]  # Keep only length of HIGH TTLs
events_stream['keys'] = keys  # Add keys column to events_stream

# Keep only rows of DataFrame events_stream with keys == play
events_stream_test = events_stream_test[events_stream_test['keys'] == 'play']
events_stream['Trial'] = np.arange(1, len(events_stream) + 1)  # Add Trial column to events_stream







# Start timestamps at 0
df_ttl['timestamps'] = df_ttl['timestamps'] - df_ttl['timestamps'].iloc[0]

df_ttl.loc[df_ttl['samples'] == 1, 'Delay_ON'] = df_ttl['timestamps']  # Mark onset of delays
df_ttl.loc[df_ttl['samples'] == 0, 'Delay_OFF_next'] = df_ttl['timestamps']  # Mark offset of delay

# Create new colum with delay offset to measure the delay duration and then remove it
df_ttl['Delay_OFF'] = df_ttl['Delay_OFF_next'].shift(-1)
df_ttl['Delay_length'] = df_ttl['Delay_OFF']  - df_ttl['Delay_ON']
df_ttl.drop('Delay_OFF_next',axis='columns', inplace=True)

# Round Delay length to the desired precision
df_ttl['Delay_length'] = df_ttl['Delay_length'].round(ttl_precision)

# Keep only rows of DataFrame df_ttl with Delay-length == 0.009
df_ttl = df_ttl[df_ttl['Delay_length'] == 0.009]

# Prepare a column with trial index. start in 1 because trial 0 doesn't have a delay and is not there.
df_ttl['Trial'] = np.arange(len(df_ttl))
# df_behavior['trials'] = np.arange(len(df_behavior))+1

# Merge with cluster labels, use trial to associate each one
# df_behavior.rename(columns= {'trials': 'trial'},inplace=True)
df2_behavior = pd.merge(df_behavior, df_ttl, on=['Trial'])




didff = (df2_behavior['StimStart'].iloc[0] +df2_behavior['TrialStart'].iloc[0])
df2_behavior['START'] = didff + df2_behavior.Delay_ON
didff = (df2_behavior['StimStart'].iloc[0] +df2_behavior['TrialStart'].iloc[0])



