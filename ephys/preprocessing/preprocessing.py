# Standard libraries
import socket  # Check remote vs local
import platform  # Check remote vs local
import getpass  # Check remote vs local
import os
from pathlib import Path
import numpy as np
import pandas as pd
from open_ephys.analysis import Session
import runpy
from matplotlib import pyplot as plt
import seaborn as sns
from collections import namedtuple

# My libraries
from my_fun.my_fun import do_sounds_dict_inv, timer
from parse.parse_v2 import parse_v2


def dev():
    """
    Check if the code is running in a remote server or in a local machine.
    """

    # Check remote vs local
    hostname = socket.gethostname()  # Machine name (remote if SSH)
    username = getpass.getuser()  # Logged in user
    os_type = platform.system()  # 'Linux', 'Darwin' (macOS), 'Windows'
    if (hostname == 'headnode' or hostname == 'minibaps' or hostname == 'minibaps2') \
        and os_type == 'Linux' and username == 'alexis':
        development = 'remote'
    else:
        development = 'local'  # Local machine (Ephys PC)
    print(f'Development mode: {development} ({hostname}, {username}, {os_type})\n')

    return development


def get_behavior_id(ephys_id):
    """
    Takes an ephys session ID and finds the corresponding path to the matching .csv file in the behavior folder
    :return: Path to the behavior file
    """

    subject = ephys_id[:3]  # Get the subject ID from the first 3 characters of the ephys session ID
    date_ephys = ephys_id[4:14]  # Get the date from the ephys session ID
    date_ephys = date_ephys[:4] + date_ephys[5:7] + date_ephys[8:]  # Remove - characters in ephys date to match Bpod dates

    # Check if the code is running in a remote server or in a local machine
    development = dev()
    if development == 'local':
        folder_parent = Path.home() / 'pv_nmdar_eranet/experiments/Ephys/setups' / subject / 'sessions'
    elif development == 'remote':
        folder_parent = Path('/archive/alexis/pv_nmdar_eranet/experiments/Ephys/setups') / subject / 'sessions'

    # List all the child folders in the parent folder
    sessions = os.listdir(folder_parent)
    dates_sessions = [session[-15:-7] for session in sessions]

    # Find the indices of the dates that match the ephys date (it should be only one)
    index = [i for i, date in enumerate(dates_sessions) if date == date_ephys]
    assert len(index) == 1, 'There is more than one behavior session with the same date as the ephys session'
    behavior_id = sessions[index[0]]  # Get the behavior session ID
    path = Path(folder_parent / behavior_id / behavior_id).with_suffix('.csv')  # Get path behavior id

    return path


def get_ephys_sessions(subject):
    """
    Get the ephys sessions for a given subject by searching in both C: and D: drives.
    :param subject: str, subject number (format: '000')
    :return: list of ephys session folder names that match the subject number
    """

    folder_name_len = 23  # Length of the ephys sessions folder name

    development = dev()

    if development == 'local':

        # Define the paths for ephys sessions on C: and D: drives on Ephys PC
        C_drive = Path('C:/Users/Usuario/Documents/Open Ephys')
        D_drive = Path(f'D:/{subject}')

        # Get all folders in C: and D: (non-recursive), excluding folders that do not match the expected length
        folders_C = [p.name for p in C_drive.iterdir() if p.is_dir() and len(p.name) == folder_name_len]
        folders_D = [p.name for p in D_drive.iterdir() if p.is_dir() and len(p.name) == folder_name_len]

        folders = folders_C + folders_D  # Combine the lists of folders from both drives

    elif development == 'remote':
        # remote_drive = Path('/archive/mouse/Alexis ephys/spike_sorting') / subject
        remote_drive = Path('/archive/alexis/ephys/spike_sorting') / subject
        folders = [p.name for p in remote_drive.iterdir() if p.is_dir() and len(p.name) == folder_name_len]

    subject_folders = [f for f in folders if f.startswith(subject)]  # Filter folders that start with subject number
    subject_folders.sort()  # Sort the subject folders
    print(subject_folders, '\n')  # Print the sorted subject folders

    return subject_folders


def load_oe_data(directory, sync=True, stream='AP'):
    """
    Load raw Open Ephys data
    From https://github.com/open-ephys/open-ephys-python-tools/blob/main/src/open_ephys/analysis/README.md
    :param directory: Directory where the data is stored
    :param sync: Synchronize timestamps (default: True)
    :param stream: Stream to load (default: 'AP')
    :return: continuous, events
    """

    session = Session(directory)  # Create Session object
    recordnode = session.recordnodes[0]  # Get the first recordnode
    recording = recordnode.recordings[0]  # Get the first recording

    # Synchronizing timestamps
    if sync:
        # This must be done prior to accessing the continuous or event data
        recording.add_sync_line(1,  # TTL line number
                                109,  # processor ID
                                'ProbeA-AP',  # stream name
                                main=True)  # use as the main stream

        recording.add_sync_line(1,  # TTL line number
                                109,  # processor ID
                                'ProbeA-LFP',  # stream name
                                main=False)  # align to the main stream

        # # Convert to dict for backward compatibility
        # for c in recording.continuous:
        #     c.metadata = vars(c.metadata)

        # Convert to dict for backward compatibility
        for c in recording.continuous:
            c.metadata = vars(c.metadata)

        # recording.compute_global_timestamps()  # Generate new global timestamps column
        recording.compute_global_timestamps(overwrite=True)  # Overwrite existing timestamps
        # recording.events.sort_values('global_timestamp', inplace=True)  # Sort events by global timestamp
        recording.events.sort_values('timestamp', inplace=True)  # Sort events by global timestamp
        print('\n')

    # Loading continuous and event data
    continuous = recording.continuous  # Get the continuous data
    events = recording.events  # Get the event data
    if stream == 'AP':
        continuous = continuous[0]  # Get the continuous AP data
        # events = events[events.stream_index == 0]  # Do not use, stream_index and stream_name match is OS dependent
        events = events[events.stream_name == 'ProbeA-AP']  # Action Potential (AP) stream
    elif stream == 'LFP':
        continuous = continuous[1]  # Get the continuous LFP data
        # events = events[events.stream_index == 1]  # Do not use, stream_index and stream_name match is OS dependent
        events = events[events.stream_name == 'ProbeA-LFP']  # Local Field Potential (LFP) stream
    # data = recording.continuous[0].get_samples(start_sample_index=0, end_sample_index=10000)
    events.reset_index(drop=True, inplace=True)

    return continuous, events


def get_ttls(continuous, events, n_decimals=4, double_check=False):
    """
    Get TTLs from Open Ephys data.
    :param continuous: Continuous data
    :param events: Event data
    :param n_decimals: Precision of the TTLs (in decimal places of seconds)
    :param double_check: Double check TTLs from event and continuous data (default: False)
    :return: pd.DataFrame with TTLs
    """

    # TTLs from event data (much faster). Default
    events.loc[events['state'] == 1, 'ON'] = events['timestamp']  # Onset of TTL
    events.loc[events['state'] == 0, 'OFF'] = events['timestamp']  # Offset of TTL
    events.OFF = events.OFF.shift(-1)  # Shift the OFF column one row up
    events['Length'] = events['OFF'] - events['ON']  # Calculate length of TTLs
    events.Length = events.Length.round(n_decimals)  # Round TTLs to the desired n_decimals
    events = events.dropna()  # Drop rows in which samples == 0  (ON and OFF are not NaN)
    events.reset_index(drop=True, inplace=True)
    TTLs_events = events.Length

    # TTLs from continuous data (much slower)
    if double_check:  # Flag to use TTLs from continuous data
        # Only possible if the sync signal was included as a continuous channel ("+" button of Neuropix pluggin was active
        # https://open-ephys.github.io/gui-docs/User-Manual/Plugins/Neuropixels-PXI.html
        if continuous.metadata['num_channels'] == 385:  # If there is a 385th channel with TTLs
            samples = continuous.samples[:, 384]  # Get the samples from channel 385 (TTLs)
            timestamps = continuous.timestamps
            s1 = pd.Series(samples, name='samples')
            s2 = pd.Series(timestamps, name='timestamps')
            df_ttl = pd.concat([s1, s2], axis=1)  # Put the data in a single dataframe
            df_ttl['diff'] = df_ttl.samples.diff()  # Look for the places where there is a change in the TTL state
            df_ttl = df_ttl.loc[(df_ttl['diff'] != 0)]  # Remove values without a change (diff = 0)
            df_ttl.loc[df_ttl['samples'] == 1, 'ON'] = df_ttl['timestamps']  # Onset of TTL
            df_ttl.loc[df_ttl['samples'] == 0, 'OFF'] = df_ttl['timestamps']  # Offset of TTL
            df_ttl.OFF = df_ttl.OFF.shift(-1)  # Shift the OFF column one row up
            df_ttl['Length'] = df_ttl['OFF'] - df_ttl['ON']  # Calculate length of TTLs
            df_ttl.Length = df_ttl.Length.round(n_decimals)  # Round TTLs to the desired n_decimals
            df_ttl = df_ttl.dropna()  # Drop rows in which samples == 0  (ON and OFF are not NaN)
            df_ttl.reset_index(drop=True, inplace=True)
            df_ttl = events[['ON', 'OFF', 'Length']]  # Keep only the columns 'ON', 'OFF' and 'Length'
            TTLs_continuous = df_ttl.Length
            TTLs = TTLs_continuous  # Use TTLs from continuous data

            # Check if TTLs from continuous and event data match
            assert TTLs_continuous.equals(TTLs_events), 'TTLs from continuous and event data do not match'
        else:
            print('There is no 385th channel with TTLs in the continuous data. Using TTLs from event data only.')
            TTLs = TTLs_events  # Use TTLs from event data
    else:
        TTLs = TTLs_events  # Use TTLs from event data

    # Create a new DataFrame with only the columns 'ON', 'OFF' and 'Length'
    df_ttl = events[['ON', 'OFF', 'Length']].copy()  # Copy the DataFrame to avoid modifying the original one

    # Align TTLs (do not start at 0) to the first timestamp (to start at 0)
    # This is equivalent (but much slower) to adding the first timestamp to the spikes timestamps (that do start at 0)
    first_timestamp = continuous.timestamps[0]
    df_ttl.ON = df_ttl.ON - first_timestamp
    df_ttl.OFF = df_ttl.OFF - first_timestamp

    # Recover sounds filenames and sounds orders from TTLs
    keys = TTLs.apply(do_sounds_dict_inv).dropna().to_list()  # Get sounds_dict keys given a TTL length as a value
    df_ttl['key'] = keys  # Add keys column to df_ttl

    return df_ttl


def decode_ttls(df_ttl):
    """
    Get the TTL code from the TTLs DataFrame. 1 TTL per trial.
    :param df_ttl: DataFrame with TTLs
    :return: pd.Series with the TTL code
    """

    # Get the sound filenames per trial
    keys = df_ttl['key']
    letters = [key for key in keys if len(key) == 1]  # Get only the keys with 1 element (letters)
    ephys_filenames = [letters[i:i + 3] for i in range(0, len(letters), 3)]  # Create a list of lists with 3 keys each
    ephys_filenames = [ephys_filenames[i][0] + ephys_filenames[i][1] + ephys_filenames[i][2] for i in
                       range(len(ephys_filenames))]  # Concatenate the 3 keys into a single string
    ephys_filenames = pd.Series(ephys_filenames)  # Convert list to pandas Series

    # Get the sound orders per trial
    orders = [key for key in keys if len(key) == 4]  # Get only the keys  with 4 elements (load/play/stop)
    orders = [orders[i:i + 3] for i in range(0, len(orders), 3)]  # Make a list of lists with 3 orders each
    orders = pd.Series(orders)

    # Check if the number of filenames and orders match (should be 1 per trial)
    assert len(ephys_filenames) == len(orders), 'Number of filenames and orders do not match'

    df_keys = pd.DataFrame({'ephys_filenames': ephys_filenames, 'orders': orders})

    return df_keys


def check_data(df_behavior, df_keys):
    """
    Compare the behavior and ephys data and check if they match. Do so by comparing the sounds filenames, which also
    retrieves the number of trials.
    """

    ephys_filenames = df_keys.ephys_filenames
    orders = df_keys.orders

    # Find sounds sent from Bpod (Filename) that does not match with those received by Arduino (Filename2) and get
    # their indexes
    index = df_behavior[df_behavior['FilesMatch'] == 0].index

    if len(index) == 0:
        print('All sounds sent by Bpod match those received by Arduino')
    else:
        print('Trials which sounds sent by Bpod do not match those received by Arduino:')
        print(df_behavior.iloc[index].Trial)

    # Get the number of trials from behavior and ephys data
    behavior_filenames = df_behavior['Filename']
    n_trials_behavior = len(behavior_filenames)
    n_trials_ephys = len(ephys_filenames)
    diff_n_trials = n_trials_behavior - n_trials_ephys

    # Check if the number of trials match with a tolerance of 1
    assert abs(diff_n_trials) <= 1, 'Number of trials from behavior and ephys data do not match with a tolerance of 1'

    # Check if the number of filenames match
    if len(behavior_filenames) != len(ephys_filenames):
        if np.sign(diff_n_trials) == -1:
            print(f'There is(are) {abs(diff_n_trials)} more ephys trial(s) than behavior trial(s)')
        elif np.sign(diff_n_trials) == 1:
            print(f'There is(are) {abs(diff_n_trials)} more behavior trial(s) than ephys trial(s)')

    # Find shorter (minimum) length of both number of trials (from ephys and behavior) to avoid error when comparing
    # Usually, the behavior data has 1 less filename than the ephys data because the last trial did not finish
    n_trials = min(len(ephys_filenames), len(behavior_filenames))
    print(f'{n_trials} trials common to both data sources. Using it for later slicing')

    # Compare ephys_filenames and behavior_filenames
    sounds_match = [ephys_filenames[i] == behavior_filenames[i] for i in range(n_trials)]

    # Check if all sounds from behavior and ephys match
    assert all(sounds_match), 'Sounds from behavior and ephys do not match'

    # Check if all values of sounds_match are True
    if all(sounds_match):
        print('All sounds from behavior and ephys match')
        sounds_mismatch_index = []  # Get the indexes of the sounds that do not match
    else:
        print('Sounds from behavior and ephys do not match')
        # Get the indexes of the sounds that do not match
        sounds_mismatch_index = [i for i, x in enumerate(sounds_match) if not x]

    print('\n')

    return n_trials, sounds_mismatch_index


def load_spike_sorted_data(path_ks4, path_phy2):
    """
    Load spike data sorted with Kilosort 4 (KS4) and manually curated with Phy2.
    """

    # Note: only spike_clusters.npy and TSV files are ever modified by phy.
    # The rest of the data files are open in read-only mode.
    # From https://phy.readthedocs.io/en/latest/visualization/
    spikes_times = np.load(path_ks4 / 'spike_times.npy')  # From KS4

    # From Phy2
    spike_clusters = np.load(path_phy2 / 'spike_clusters.npy')
    cluster_group = pd.read_csv(path_phy2 / 'cluster_group.tsv', sep='\t')
    cluster_info = pd.read_csv(path_phy2 / 'cluster_info.tsv', sep='\t')
    # cluster_Amplitude = pd.read_csv(path_phy2 / 'cluster_Amplitude.tsv', sep='\t')  # Already in cluster_info
    # cluster_Contam_pct = pd.read_csv(path_phy2 / 'cluster_ContamPct.tsv', sep='\t')  # Already in cluster_info
    # cluster_KSLabel = pd.read_csv(path_phy2 / 'cluster_KSLabel.tsv', sep='\t')  # Already in cluster_info
    params = runpy.run_path(path_ks4 / 'params.py')
    sample_rate = params['sample_rate']

    # Create pandas DataFrame with spike times and clusters
    df_spikes = pd.DataFrame({'spike_times': spikes_times, 'spike_clusters': spike_clusters})

    # Merge df_spikes with cluster_group
    df_spikes = pd.merge(df_spikes, cluster_group, left_on='spike_clusters', right_on='cluster_id')

    # Drop clusters_id column (redundant)
    df_spikes.drop('cluster_id', axis=1, inplace=True)

    # Rename columns
    df_spikes.rename(columns={'spike_times': 'times'}, inplace=True)
    df_spikes.rename(columns={'spike_clusters': 'cluster'}, inplace=True)

    # Transform spike times to seconds
    df_spikes['times'] = df_spikes['times'] / sample_rate  # Overwrites original times column

    print(f'Min spike time: {round(min(df_spikes.times / 60))} min\n'
          f'Max spike time: {round(max(df_spikes.times / 60))} min')
    print('\n')

    # Drop noise clusters
    df_spikes = df_spikes.loc[(df_spikes.group != 'noise')]
    cluster_info = cluster_info.loc[(cluster_info.group != 'noise')].reset_index(drop=True)

    # Keep only good or single units (drop unsorted or other categories)
    df_spikes = df_spikes.loc[df_spikes.group.isin(['good', 'mua'])]
    cluster_info = cluster_info.loc[cluster_info.group.isin(['good', 'mua'])].reset_index(drop=True)

    # Drop sparse units
    # fr_threshold = 0.1  # Firing rate in Hz
    # sparse_clusters = cluster_info[cluster_info.fr < fr_threshold].cluster_id.to_list()
    # cluster_info = cluster_info[~cluster_info.cluster_id.isin(sparse_clusters)].reset_index(drop=True)
    # df_spikes = df_spikes[~df_spikes.cluster.isin(sparse_clusters)].reset_index(drop=True)

    # # Plot the first minute
    # plt.figure()
    # plt.scatter(df_spikes[(df_spikes.times < 60)].times,
    #             df_spikes[(df_spikes.times < 60)].cluster,
    #             marker='|', linestyle='None', color='k')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Cluster ID')
    # plt.title('First min. of recording')

    return df_spikes, cluster_info


def plot_group_clusters_dist(df_spikes, cluster_info, ax=None):
    """
    Plot a bar graph with the number of good, mua and noise clusters
    """

    if ax is None:
        fig, ax = plt.subplots()

    # Get the number of good and mua clusters
    n_clusters = len(cluster_info)
    print(f'Total number of clusters: {n_clusters}')
    n_good_clusters = len(cluster_info[cluster_info['group'] == 'good'])
    print(f'Number of good clusters: {n_good_clusters} ({round(n_good_clusters / n_clusters * 100)}%)')
    n_mua_clusters = len(cluster_info[cluster_info['group'] == 'mua'])
    print(f'Number of mua clusters: {n_mua_clusters} ({round(n_mua_clusters / n_clusters * 100)}%)')
    # n_noise_clusters = len(cluster_info[cluster_info['group'] == 'noise'])
    # print(f'Number of noise clusters: {n_noise_clusters} ({round(n_noise_clusters / n_clusters * 100)}%)')
    print('\n')

    x = ['good', 'mua']
    height = [n_good_clusters, n_mua_clusters]

    colors = ['tab:green', 'tab:orange']
    ax.bar(x, height, label=x, color=colors)
    ax.set_xlabel('Group')
    ax.set_ylabel('Count')
    sns.despine(ax=ax)


def print_timeline(continuous, events, df_behavior, df_spikes):
    """
    Print information about the ephys and behavior data
    """

    Timeline = namedtuple('Timeline', ['y', 'width', 'left', 'ts_edges', 'events_edges'])

    # Ephys session info
    start_aquisition = 0  # Press PLAY in Open Ephys GUI
    first_timestamp = continuous.timestamps[0] / 60  # Press RECORD in Open Ephys GUI
    last_timestamp = continuous.timestamps[-1] / 60
    len_aquisition = last_timestamp - start_aquisition  # For clarity
    len_recording = last_timestamp - first_timestamp

    first_event = events.timestamp.iloc[0] / 60
    last_event = events.timestamp.iloc[-1] / 60
    len_events = last_event - first_event

    first_spike = df_spikes.times.iloc[0] / 60
    last_spike = df_spikes.times.iloc[-1] / 60
    len_spikes = last_spike - first_spike

    # Behavior session info
    start_behavior = df_behavior['TrialStart'].iloc[0] / 60  # Get the first trial start
    end_behavior = df_behavior['TrialEnd'].iloc[-1] / 60  # Get the last trial end
    len_behavior = end_behavior - start_behavior  # Get the total behavior length

    print(f'The aquisition (press PLAY in Open Ephys) started at {start_aquisition} min '
          f'and ended after {round(len_aquisition)} min. '
          f'It includes the time for lowering the probe.')

    print(f'The recording (press REC in Open Ephys) started at {round(first_timestamp)} min of acquisition '
          f'and ended at {round(len_aquisition)} min of acquisition, '
          f'lasting {round(len_recording)} min '
          f'It includes the time for settling the tissue.')

    print(f'The first event happened at {round(first_event)} min of acquisition '
          f'({round((first_event - first_timestamp))} min after recording started) '
          f'and the last event happened at {round(last_event)} min of acquisition '
          f'({round((last_event - first_timestamp))} min after recording started), '
          f'lasting {round(len_events)} min.')

    print(f'The behavior lasted {round(len_behavior)} min.')

    print(f'The first spike was recorded at {round(first_spike) + round(first_timestamp)} min of acquisition '
          f'and the last spike was recorded at {round(last_spike) + round(first_timestamp)} min of acquisition, '
          f'lasting {round(len_spikes)} min.')

    # Check if the length of the behavioral session from behavioral and ephys data match
    assert abs(round(len_events) - round(len_behavior)) <= 1, \
        'Length of behavioral session from behavioral and ephys data do not match. '

    print('\n')

    # Plot timecourse of aquisition, recording, events, behavior and spikes
    y = ['Aquisition', 'Recording', 'Events', 'Behavior', 'Spikes']
    width = [len_aquisition, len_recording, len_events, len_behavior, len_spikes]
    left = [start_aquisition, first_timestamp, first_event, start_behavior + first_event, first_spike + first_timestamp]

    ts_edges = (first_timestamp, last_timestamp)
    events_edges = (first_event, last_event)

    timeline = Timeline(y, width, left, ts_edges, events_edges)

    return timeline


def plot_timeline(timeline, ax=None):
    """
    Plot timecourse of aquisition, recording, events, behavior and spikes
    param y: Labels of the bars
    param width: Width of the bars representing length in time (seconds)
    param left: Coordinates of the left side of the bars in time (seconds)
    """

    if ax is None:
        fig, ax = plt.subplots()

    y, width, left, ts_edges, events_edges = timeline

    color = ['tab:gray', 'tab:red', 'tab:green', 'tab:blue', 'tab:orange']
    ax.barh(y=y, width=width, left=left, color=color)
    ax.axvline(x=ts_edges[0], color='k', linestyle='--')
    ax.axvline(x=ts_edges[1], color='k', linestyle='--')
    ax.axvline(x=events_edges[0], color='k', linestyle='--')
    ax.axvline(x=events_edges[1], color='k', linestyle='--')
    ax.set_xlabel('Time (s)')
    ax.set_title('Timeline')
    sns.despine(ax=ax)


def temp_align(df_ttl, df_behavior, df_spikes, n_decimals=4):
    """
    Temporal alignment of ephys and behavior data.
    :param df_ttl: DataFrame with TTLs
    :param df_behavior: DataFrame with behavior data
    :param df_spikes: DataFrame with spike data
    :param n_decimals: Precision of the timestamps (in decimal places of seconds)
    :return: df_aligned, df_spikes
    """

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
              f'({round((df_aligned.BehaviorStart.max() - df_aligned.BehaviorStart.min()) * 1000)} ms)')

    # Check if the sum of the timestamps of the behavioral session start and the stimulus onset TTLs match
    assert all((df_behavior.TrialStart + df_aligned.BehaviorStart + df_behavior.StimStart).round(n_decimals) ==
               df_ttl.ON.round(n_decimals))

    # Align FSM states to the start of the behavioral session in the ephys clock
    aligned_states = ['TrialStart', 'TrialEnd', 'StimStart', 'StimEnd', 'RespWinStart', 'RespWinEnd']

    for state in aligned_states:
        df_aligned[state] = df_aligned[state] + df_aligned.BehaviorStart

    # Check if the lengths of the states before the alignment match after the alignment
    assert all((df_aligned.TrialEnd - df_aligned.TrialStart).round(n_decimals) == df_aligned.TrialLen.round(n_decimals))
    assert all((df_aligned.StimEnd - df_aligned.StimStart).round(n_decimals) == df_aligned.StimLen.round(n_decimals))
    assert all((df_aligned.RespWinEnd - df_aligned.RespWinStart).round(n_decimals) == df_aligned.RespWinLen.round(n_decimals))
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
    #     condlist = [(df_spikes.times > df_aligned['TrialStart'].iloc[i]) & (df_spikes.times < df_aligned['TrialEnd'].iloc[i]),
    #                 # Spike times within the trial
    #                  (df_spikes.times < df_aligned['TrialStart'].iloc[i]) | (df_spikes.times > df_aligned['TrialEnd'].iloc[i])]
    #                 # Spike times outside the trial
    #
    #     # create a list of the values we want to assign for each condition
    #     choices = [df_aligned['Trial'].iloc[i].astype(int), df_spikes['Trial']]
    #
    #     # create a new column and use np.select to assign values to it using our lists as arguments
    #     df_spikes['Trial'] = np.select(condlist, choices)

    # With np.where (more appropriate as there is only one condition)
    for i, row in df_aligned.iterrows():
        condition = (df_spikes.times > df_aligned['TrialStart'].iloc[i]) & (
                df_spikes.times < df_aligned['TrialEnd'].iloc[i])
        df_spikes['Trial'] = np.where(condition, df_aligned['Trial'].iloc[i].astype(int), df_spikes['Trial'])

    # Drop rows with NaN values in the Trial column  (spikes outside trial times)
    # Would make data lighter if NOT interested in periods before and after the task
    # df_spikes.dropna(inplace=True)

    # Check if the number of trials in behavior and spikes dataframes match
    assert df_spikes.Trial.nunique() == len(df_aligned), \
        'Number of trials in behavior and spikes dataframes do not match'

    # df = pd.merge(df_aligned, df_spikes, on=['Trial'])  # Merge behavior and spikes dataframes. Very heavy
    # Actually only need the trial column in df_spikes. No need to copy the +80 columns from df_aligned for every spike

    # df = pd.merge(df_aligned.Trial, df_spikes, on=['Trial'])  # Merge trials and spikes dataframes.

    return df_aligned, df_spikes


def align_ttl(df_ttl, df_behavior):
    """
    Temporal alignment of ephys and behavior data (from continuous data)
    :param ttl: DataFrame with TTLs
    :param df_behavior: DataFrame with behavior data
    :return ttl_aligned: DataFrame with aligned TTLs
    """

    df_ttl = df_ttl[df_ttl['key'] == 'play'].copy()  # Keep only rows with key == play (stimulus onset, 1 TTL per trial)
    df_ttl['Trial'] = np.arange(len(df_ttl))  # Prepare a column with trial indexes for merging
    df_ttl = df_ttl.iloc[:len(df_behavior)]  # Keep only the first n TTLs (n = number of trials in behavior data)
    df_ttl.reset_index(drop=True, inplace=True)  # Reset index
    assert len(df_ttl) == len(df_behavior), 'Number of stimulus onset TTLs and trials in behavior data do not match'

    # Add Go-cue aligned to stimulus onset (df_ttl.OFF)
    go_cue = df_behavior['RespWinStart'].values - df_behavior['StimStart'].values
    go_cue = df_ttl.OFF + go_cue

    # Add RT (first lick) aligned to stimulus onset (df_ttl.OFF)
    rt = df_behavior.RespWinLen.values
    rt = go_cue + rt

    df_ttl['GoCue'] = go_cue
    df_ttl['RT'] = rt

    return df_ttl


@timer
def preprocess(ephys_id):
    """
    Preprocess the data for a given session ID.
    :param ephys_id: Ephys session ID
    :return: preprocessed data
    """

    # Define the session ID and directory
    subject = ephys_id[:3]
    directory = Path() / 'D:' / subject / ephys_id  # Ephys PC extra SSD HD (C:)
    directory2 = Path.home() / 'Documents/Open Ephys' / ephys_id  # Ephys PC main SSD HD (C:)
    # directory3 = Path('/archive/mouse/Alexis ephys/raw') / subject / ephys_id  # Remote server archive (remote development)
    directory3 = Path('/archive/alexis/ephys/raw') / subject / ephys_id  # Remote server archive (remote development)


    development = dev()

    # Load raw Open Ephys data
    if development == 'local':
        try:
            continuous, events = load_oe_data(directory, sync=True, stream='AP')
        except OSError:
            continuous, events = load_oe_data(directory2, sync=True, stream='AP')
    elif development == 'remote':
        continuous, events = load_oe_data(directory3, sync=True, stream='AP')

    # Get TTLs from continuous or/and event data
    df_ttl = get_ttls(continuous, events)

    # Get the sound filenames and sound orders from TTLs
    df_keys = decode_ttls(df_ttl)

    # Load behavior data
    path_behavior = get_behavior_id(ephys_id)
    df_behavior = parse_v2(path_behavior)

    # Check if the behavior and ephys data match and get the number of trials common to both
    n_trials, sounds_mismatch_index = check_data(df_behavior, df_keys)

    # Load spike sorted data (KS4)
    if development == 'local':
        path_spike_sorting = Path.home() / 'Downloads/spike_sorting' / subject / ephys_id
    elif development == 'remote':
        # path_spike_sorting = Path('/archive/mouse/Alexis ephys/spike_sorting') / subject / ephys_id
        path_spike_sorting = Path('/archive/alexis/ephys/spike_sorting') / subject / ephys_id


    path_ks4 = path_spike_sorting / 'kilosort4'
    path_phy2 = path_spike_sorting / 'phy2'
    df_spikes, cluster_info = load_spike_sorted_data(path_ks4, path_phy2)

    # Print session info
    timeline = print_timeline(continuous, events, df_behavior, df_spikes)

    # Clean redundant TTLs (1 row/trial) for alignment with behavior data
    df_ttl = align_ttl(df_ttl, df_behavior)

    # Temporal alignment of ephys and behavior data (skip for now)
    # df_aligned, df_spikes = temp_align(df_ttl, df_behavior, df_spikes)

    Preprocessed = namedtuple('Preprocessed', [
        'df_ttl', 'df_behavior', 'n_trials', 'df_spikes', 'cluster_info', 'timeline'
    ])

    preprocessed = Preprocessed(
        df_ttl, df_behavior, n_trials, df_spikes, cluster_info, timeline)

    return preprocessed


def check_ephys_sessions_subject(subject):
    """
    Check the ephys sessions for a given subject and recover the ones with errors.
    Errors can arise from the raw ephys data, the spike sorting files, or the behavioral data.
    :param subject: Subject ID
    :return: error_sessions, error_messages
    """

    ephys_ids = np.array(get_ephys_sessions(subject))
    n_sessions = len(ephys_ids)
    load = []
    # exceptions = []
    error_messages = []

    for i, id in enumerate(ephys_ids):
        print(f'Session {i}: {id}')
        try:
            preprocess(id)
            load.append(True)
            error_messages.append(None)  # No error
        except Exception as e:
            print(f'Error in {id}: {e}')
            load.append(False)
            error_messages.append(str(e))

    load = np.array(load, dtype=bool)
    error_sessions = ephys_ids[~load]
    error_messages = np.array(error_messages, dtype=object)[~load]
    n_errors = len(error_sessions)
    print(f'{n_errors}/{n_sessions} error sessions: {error_sessions}')

    for sess, msg in zip(error_sessions, error_messages):
        print(f'{sess}: {msg}')

    return error_sessions, error_messages