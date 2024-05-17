# From https://github.com/open-ephys/open-ephys-python-tools/blob/main/src/open_ephys/analysis/README.md

from pathlib import Path
from open_ephys.analysis import Session
import numpy as np
from my_fun.my_fun import do_sounds_dict
import pandas as pd


# id = 'TEST_2024-05-06_17-18-10'
# id = 'XXX_2024-05-06_16-32-10'
# id = '911_2023-10-13_16-30-38'
# id = '911_2023-10-23_18-39-31'
id = '009_2024-05-08_20-51-26'

# Create Session object
directory = Path.home() / 'Documents' / 'Open Ephys' / id
session = Session(directory)
recordnode = session.recordnodes[0]  # Get the first recordnode
recording = recordnode.recordings[0]  # Get the first recording

# Synchronizing timestamps
recording.add_sync_line(1,           # TTL line number
                       109,          # processor ID
                       'ProbeA-AP',  # stream name
                       main=True)    # use as the main stream

recording.add_sync_line(1,            # TTL line number
                       109,           # processor ID
                       'ProbeA-LFP',  # stream name
                       main=False)    # align to the main stream

recording.compute_global_timestamps()
# recording.compute_global_timestamps(overwrite=True)
recording.events.sort_values('global_timestamp', inplace=True)  # Sort events by global timestamp

########################################################################################################################

# Loading event data
events = recording.events
events_AP = events[events.stream_index == 0]  # Action Potential (AP) stream
events_AP.reset_index(drop=True, inplace=True)
events_LFP = events[events.stream_index == 1]  # Action Potential (AP) stream
events_LFP.reset_index(drop=True, inplace=True)

TTLs = []
for _ in np.arange(0, len(events_AP), 2):
    # TTLs.append(events_AP.global_timestamp[_ + 1] - events_AP.global_timestamp[_])
    TTLs.append(events_AP.global_timestamp[_ + 1] - events_AP.global_timestamp[_])
TTLs = pd.Series(TTLs)

########################################################################################################################

# SOUNDS TTLs

# TTLs pulses durations
load = 0.0085
play = 0.0090
stop = 0.0095
wait = 0.1

sounds_dict = do_sounds_dict(0.0003, 0.0078, 26, 4)  # Create TTL-letter mapping dictionary
sounds_dict_keys = list(sounds_dict.keys())  # Get sounds_dict keys
sounds_dict_values = list(sounds_dict.values())  # Get sounds_dict values

tolerance = 0.0003 / 2  # Half of the step size between TTL lengths

def sounds_dict_inv(x):  # Move function to my_fun
    for i in range(len(sounds_dict)):
        if sounds_dict_values[i] - tolerance < x < sounds_dict_values[i] + tolerance:
            return sounds_dict_keys[i]
        else:  # If the TTL length is not in the dictionary
            pass  # Do nothing


keys = TTLs.apply(sounds_dict_inv).dropna().to_list()  # Get sounds_dict keys given a TTL length as a value
sounds_filenames = [keys[i:i + 3] for i in range(0, len(keys), 3)]
sounds_filenames = [sounds_filenames[i][0] + sounds_filenames[i][1] + sounds_filenames[i][2] for i in range(len(sounds_filenames))]
sounds_filenames = pd.Series(sounds_filenames)

########################################################################################################################

# Loading continuous data
# recording = session.recordnodes[0].recordings[0]
data = recording.continuous[0].get_samples(start_sample_index=0, end_sample_index=len(recording.continuous[0].sample_numbers), selected_channels=[365])
# data = recording.continuous[0].get_samples(start_sample_index=0, end_sample_index=10000, selected_channels=[365])

# Get the differences in data
data_diff = np.diff(data, axis=0)

# Get the indexes of data when there is a change in the data
data_diff = np.where(np.diff(data) != 0)[0]


# Get median of data  (BNC low)
data_median = np.median(data)

# Get indexes where data is different from the median
data_diff_2 = np.where(data != data_median)[0]












# Loading spike data


