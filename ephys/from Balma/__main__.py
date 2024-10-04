# IMPORTS
import os
import numpy as np
import pandas as pd
import settings
import utils
import parse
import plots
import warnings
warnings.filterwarnings('ignore')

def main():

    # MAIN LOOP THOUGH THE FOLDERS
    all_items = os.listdir(settings.data_directory) # take all items inside the data_directory
    folders = [item for item in all_items if os.path.isdir(os.path.join(settings.data_directory, item))] # take only the folders
    for folder in folders: # Loop through each folder
        folder_directory = os.path.join(settings.data_directory, folder) + '/'
        save_directory= folder_directory
        # print(folder_directory)

        ######### GENERAL SESSION INFORMATION #########
        session_info = folder_directory.rstrip('/').split('/')[-1]
        session_info_splitted = session_info.split('_') #split by underscores
        animal_hemisphere = session_info_splitted[0]  # Subject name and hemisphere
        animal = animal_hemisphere[:3]
        hemisphere = animal_hemisphere[3:]
        date = session_info_splitted[2]  # Date
        error_flag= 0
        print('')
        print('NEW EPHYS REPORT FOR: ' + animal + ' ' + 'hemisphere:' + hemisphere + ' ' + date)
        print('TOTAL NUMBER OF CHANNELS: ' + str(settings.num_channels) + '    TTLs in:' + str(settings.channels_of_interest))


        ######### GET BEHAVIORAL SESSION #########
        # Assumes 1 behavioral session by recording  (TO DO:  paste multiple behabvior files)
        try:
            for file in os.listdir(folder_directory):
                if file == 'daily_ephys_report.pdf':
                    print('Ephys report already done!')
                    error_flag = 1
                elif file.endswith('.csv'):
                    if file != 'df_ephys_behavior.csv':
                        behavior_file = file

            df_behavior = pd.read_csv(folder_directory + behavior_file, sep=';')
            # Parse the behavioral data
            df_behavior,  starting_time_behavior, states_list_aligned = parse.parse_behavior(df_behavior,settings.states_list)
        except:
            error_flag = 1
            print('ERROR PARSING THE BEHAVIOR'+ '\n' +'')



        # PARSING ONLY IF df_ephys_behavior.csv DOES NOT EXIST
        data_path = os.path.join(folder_directory, 'df_ephys_behavior.csv')
        if not os.path.exists(data_path):

            #########  GET TTLs #########
            print('Parsing behavior, spikes and ttls...')

            try:
                # Samples
                samples_file = folder_directory + 'continuous.dat'
                samples = utils.read_dat_file(samples_file, settings.num_channels)
                # Timestamps
                try:
                    timestamps = np.load(os.path.join(folder_directory + 'timestamps.npy'))  # GUI newer versions timestamps file named sample_numbers
                    if timestamps.shape[0] - samples.shape[1] != 0:  # Lenght missmatch (missing frames)
                        print('WARNING: DIFFERENT NUMBER OF SAMPLES AND TIMESTAMPS')
                        print('Nº of samples: ' + str(samples.shape[1])+ ' | Nº of timestamps: ' + str(timestamps.shape[0]) +'')
                        timestamps = np.arange(0, samples.shape[1],1)  # create an artificial timestamps with the correct lenght
                except:
                    print('WARNING: NO TIMESTAMPS FILE')
                    timestamps = np.arange(0, samples.shape[1], 1) # create an artificial timestamps with the correct lenght

                # Create the ttl DataFrame
                df_ttl, ttl_min_clock, ttl_max_clock = parse.parse_ttls(samples, settings.channels_of_interest, timestamps, settings.bitsVolts, settings.sampling_freq)
            
            except:
                error_flag = 1
                print('ERROR PARSING THE TTLs'+ '\n' +'')


            #########  GET SORTED SPIKES #########
            try:
                spike_times = np.load(folder_directory + 'spike_times.npy')  # Times of the spikes, array of lists
                spike_clusters = np.load(folder_directory + 'spike_clusters.npy')  # cluster number of each of the spikes, same length as before
                df_labels = pd.read_csv(folder_directory + 'cluster_group.tsv', sep='\t')  # Cluster labels (good, noise, mua) for the previous two arrays
                # Create the spikes DataFrame
                df_spikes, spike_min_clock, spike_max_clock = parse.parse_spikes(spike_times, spike_clusters, df_labels, settings.sampling_freq)
            except:
                error_flag = 1
                print('ERROR PARSING THE SPIKES'+ '\n' +'')



            # Continue only if there are no errors parsing
            if error_flag == 0:

                # try:
                ######### EPHYS - BEHAVIOR ALIGNMENT & REMOVE EXTRA TTLS #########
                df_behavior, alignment_diff_c = parse.alignment(df_behavior, df_ttl, states_list_aligned)

                ######### MERGE DATAFRAMES #########
                df_ephys_behavior,  missing_corridor_ttl, missing_response_ttl, missing_delay_ttl = parse.merging(df_behavior, settings.behavior_sorted_columns, df_spikes, hemisphere)

                ######## QUALITY REPORT #########
                save_path = os.path.join(save_directory, 'quality_report.pdf')
                plots.quality_report(df_behavior, df_spikes, ttl_min_clock, ttl_max_clock,  spike_min_clock, spike_max_clock, alignment_diff_c,
                                                                missing_corridor_ttl, missing_response_ttl, missing_delay_ttl, save_path)

                ######### SAVE EPHYS DATAFRAME #########
                save_path = os.path.join(folder_directory, 'df_ephys_behavior.csv')
                df_ephys_behavior.to_csv(save_path, sep=';')
                print('Parsing completed successfully :)')

                # except:
                #     error_flag = 1
                #     print('ERROR MERGINING AND ALIGNING'+ '\n' +'')

        else:
            print('Parsing already done!')
            df_ephys_behavior = pd.read_csv(folder_directory + 'df_ephys_behavior.csv', sep=';')


        ##################### SORTED BEHAVIOR DATAFRAME: DF BY TRIAL WITH TTLS #####################
        if error_flag == 0:
            ######### EPHYS REPORTS #########
            try:
                save_path = os.path.join(save_directory, 'daily_ephys_report.pdf')
                plots.ephys_report(df_ephys_behavior, settings.behavior_sorted_columns, save_path)
            except:
                print('ERROR DOING THE REPORT'+ '\n' +'')



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
