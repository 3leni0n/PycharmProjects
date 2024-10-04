import numpy as np
import pandas as pd
import warnings
from neo.core import SpikeTrain
from quantities import ms, s, Hz
from elephant.statistics import time_histogram, instantaneous_rate,  mean_firing_rate
from elephant.kernels import GaussianKernel
from elephant.spike_train_correlation import cross_correlation_histogram
from elephant.conversion import BinnedSpikeTrain
import warnings
warnings.filterwarnings('ignore')

# def find_folder_with_string(path, search_string):
#     """
#     Recursively search for a folder containing a specific string in its name.
#     Parameters:
#         root_dir (str): The root directory to start the search from.
#         search_string (str): The string to search for in folder names.
#     Returns:
#         list: A list of complete paths to folders containing the search string.
#     """
#     matching_folders = []
#     print(f"Starting search in: {path}")
#
#     for dirpath, dirnames, filenames in os.walk(path):
#         for dirname in dirnames:
#             if search_string in dirname:
#                 matching_folders.append(os.path.join(dirpath, dirname))
#
#     return matching_folders


def read_dat_file(file_path, num_channels, dtype=np.int16):
    """Read binary .dat file from Open Ephys."""
    data = np.fromfile(file_path, dtype=dtype)
    data = data.reshape((-1, num_channels)).T  # Reshape and transpose to (channels, samples)
    return data


def get_channel_data(data, channels):
    """Extract data for specific channels."""
    return {channel: data[channel] for channel in channels}


# def read_timestamps(file_path, dtype=np.uint64):
#     """Read timestamps from a binary file."""
#     return np.fromfile(file_path, dtype=dtype)
#
#


def convert_strings_to_lists(df, columns):
    """
    If the csv contains a column that is ',' separated, that column is read as a string.
    We want to convert that string to a list of values. We try to make the list float or string.
    """
    def tolist(stringvalue):
        if isinstance(stringvalue, str):
            try:
                stringvalue = stringvalue.split(sep=',')
                try:
                    val = np.array(stringvalue, dtype=float)
                except:
                    val = np.array(stringvalue)
            except:
                val = np.array([])
        elif np.isnan(stringvalue):
            return np.array([])
        else:
            val = np.array([stringvalue])
        return val.tolist()

    for column in columns:
        df[column] = df[column].apply(tolist)
    return df


def classify_stim_dur_ext(row):
    if row['trial_type'] == 'VG':
        return 0
    elif row['trial_type'] == 'WM_I':
        return row['rw_stim_dur']
    elif row['trial_type'] == 'WM_D':
        return row['wm_stim_dur']
    else:
        return None  # Handle any unexpected trial_type


def find_last_before(row, x, y, z):
    '''Find the last value in a column X before a value of columnn Y, if not take value column Z'''
    colx_times = row[x]
    coly_time = row[y]
    colz_time = row[z]
    try:
        if colx_times and coly_time:
            filtered_values = [time for time in colx_times if time <= coly_time - 0.2] # tail touches
            if filtered_values:
                last_time_x_before_y = max(filtered_values)
            else:
                last_time_x_before_y = max((time for time in colx_times if time <= coly_time), default=colz_time)
            return last_time_x_before_y
        else:
            return np.nan  # Return nan if col_times is empty
    except:
        print('colx or y empty')
        return np.nan




def calculate_firing_rate(unit_df):
    spiketrain = SpikeTrain(unit_df.timestamps_fix.values * 1000 * ms,
                            t_stop=unit_df.timestamps_fix.max() * 1000 * ms,
                            t_start=unit_df.timestamps_fix.min() * 1000 * ms)
    mfr = mean_firing_rate(spiketrain)  # in Hz
    rounded_mfr = np.round(mfr.magnitude * 1000, 2)  # in spikes/s
    return rounded_mfr, spiketrain


def generate_conv_data(spiketrain, bin_size, sdev):
    histogram_rate = time_histogram([spiketrain], bin_size * ms, output='rate')
    gaus_rate = instantaneous_rate(spiketrain, sampling_period=bin_size * ms, kernel=GaussianKernel(sdev * ms))  # s.d of Suzuki & Gottlieb
    conv_times = gaus_rate.times.rescale(s)  # convert to seconds
    conv_firing = gaus_rate.rescale(histogram_rate.dimensionality).magnitude.flatten()
    return pd.DataFrame({'conv_times': conv_times, 'conv_firing': conv_firing*1000})

def classify_time(trial_row, conv_df):
    mask = (conv_df['conv_times'] >= trial_row['STATE_Start_task_START_align']) & \
           (conv_df['conv_times'] <= trial_row['STATE_Exit_END_align'])
    return conv_df[mask]

def generate_autocorr_data(spiketrain, bin_size, autocorr_win):
    binned_spiketrain = BinnedSpikeTrain([spiketrain], bin_size=bin_size * ms)
    autocorr, bins_auto = cross_correlation_histogram(binned_spiketrain, binned_spiketrain, window=[-int(autocorr_win), int(autocorr_win)])
    autocorr_array, bins_auto = np.delete(autocorr.magnitude.flatten(), bins_auto == 0), np.delete(bins_auto, bins_auto == 0)
    return bins_auto, autocorr_array

def change_sign_list(lst):
    return [-x for x in lst]

def select_closest(values_list, ref):
    if len(values_list)==0: #check if there are no values
        closest_value=np.nan
    else:
        closest_value = min(values_list, key=lambda value: abs(value - ref))
    return closest_value
