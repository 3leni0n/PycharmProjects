import time
import os
import numpy as np
import pandas as pd
from parse.parse import parse

# To do:
# When a session's file  is corrupted, continue but catch that session's ID


# Define function
def glue_sessions(animal=None, protocol=None, to_csv=False):

    time_start = time.time()

    folder = '/home/alexis/pv_nmdar_eranet/experiments/2AFC/setups/'  # Where the data for all animals is

    if animal == None:

        animals = os.listdir(folder)  # List animals
        animals.sort()  # Sort them by name
        print('Animals: ' + str(animals)[1:-1])  # Remove square brackets
        animal = input('Enter animal')  # Ask user to input animal to glue sessions from

    folder = folder + animal + '/sessions'  # Update folder with selected animal
    sessions = os.listdir(folder)  # List sessions
    sessions.sort()  # Sort them by date

    if protocol == None:

        protocols = []  # Initiate list
        for i, session in enumerate(sessions):
            # print(i, session)
            protocols.append(sessions[i][4:-16])  # Remove animal ID (beginning) and date and time (end)

        print('There are ' + str(len(sessions)) + ' sessions of this animal, ' + str(len(np.unique(protocols))) +
              ' protocols found:')
        for i in range(len(np.unique(protocols))):
            print(i, ' ', np.unique(protocols)[i], ': ', protocols.count(np.unique(protocols)[i]), sep='')

        protocols = list(np.unique(protocols))
        protocol = input('Enter protocol (choose number)')
        protocol = str(protocols[int(protocol)])

    print('Gluing sessions of animal ' + animal + '...\n')
    df = pd.DataFrame()

    for i in range(len(sessions)):

        if protocol in sessions[i]:  # Loop only over sessions with the selected protocol
            # files = os.listdir(folder + '/' + sessions[i] + sessions[i] + '.csv')
            path = folder + '/' + sessions[i] + '/' + sessions[i] + '.csv'  # Get csv file path to input parse.py
            print('Parsing session ' + "'" + sessions[i] + "'" + '...', sep='')
            df_session = parse(path)  # Parse session
            df = pd.concat([df, df_session])  # Add parsed session to the bottom of the DataFrame
        else:
            pass

    if to_csv==True:
        df.to_csv('/home/alexis/PycharmProjects/glue_sessions/' + animal + '.csv')

    time_end = time.time()
    runtime = time_end - time_start
    print('The script took', round(runtime, 2), 'seconds to run')

    return df
