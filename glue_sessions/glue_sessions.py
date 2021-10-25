import time
import os
import numpy as np
import pandas as pd
from parse.parse import parse
import csv


# To do:
# When a session's file  is corrupted, continue but catch that session's ID. Done but save in text file instead of as
# variable?


# Define function
def glue_sessions(animal=None, protocol=None, to_csv=False):

    time_start = time.time()

    folder = '/home/alexis/pv_nmdar_eranet/experiments/2AFC/setups/'  # Where the data for all animals is

    if animal == None:

        animals = os.listdir(folder)  # List animals
        animals.sort()  # Sort them by name

        try:
            animals.remove('Test')  # Usually I don't want to do the daily reports of the Test subject
            animals.remove('.idea')  # Pycharm's archive
        except ValueError:
            pass

        print('Animals: ' + str(animals)[1:-1])  # Remove square brackets
        animal = input('Enter animal')  # Ask user to input animal to glue sessions from

    # # Check if csv from that animal already exist, and if so, import it
    glued_sessions = []  # Initialize empty list so if it's the first time glue all sessions
    glued_animals = os.listdir('/home/alexis/PycharmProjects/glue_sessions')
    glued_animals = [x for x in glued_animals if x.endswith('.csv')]  # Get rid of non csv files

    if animal + '.csv' in glued_animals:
        df = pd.read_csv('/home/alexis/PycharmProjects/glue_sessions/' + animal + '.csv')
        glued_sessions = df.Session.unique().tolist()
    else:
        df = pd.DataFrame()  # Create empty DataFrame if there's no csv yet for that animal

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

    corrupted_sessions = []

    for i in range(len(sessions)):

        # Loop only over sessions with the selected protocol that aren't glued yet
        if protocol in sessions[i] and sessions[i] not in glued_sessions:
            path = folder + '/' + sessions[i] + '/' + sessions[i] + '.csv'  # Get csv file path to input parse.py
            print('Parsing session ' + "'" + sessions[i] + "'" + '...', sep='')

            try:
                df_session = parse(path)  # Parse session
            except IndexError:
                print(f"The session '{sessions[i]}' is corrupted. Adding to corrupted sessions log and continuing with next session...")
                corrupted_sessions.append(sessions[i])

            df = pd.concat([df, df_session])  # Add parsed session to the bottom of the DataFrame
        else:
            pass

    if to_csv == True:
        df.to_csv('/home/alexis/PycharmProjects/glue_sessions/' + animal + '.csv', index=False)
        # index=False to avoid the 'Unmmaed: 0' column

    time_end = time.time()
    runtime = time_end - time_start
    print('The script took', round(runtime, 2), 'seconds to run')
    print('The corrupted sessions are:', *corrupted_sessions, sep='\n')

    column_name = 'corrupted sessions'

    with open('/home/alexis/PycharmProjects/glue_sessions/' + animal + '_corrupted_sessions.csv', 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(corrupted_sessions)

    return df, corrupted_sessions


def update_glued_sessions():

    folder = '/home/alexis/pv_nmdar_eranet/experiments/2AFC/setups/'  # Where the data for all animals is
    animals = os.listdir(folder)  # List animals
    animals.sort()  # Sort them by name

    try:
        animals.remove('Test')  # Usually I don't want to do the daily reports of the Test subject
        animals.remove('.idea')  # Pycharm's archive
    except ValueError:
        pass

    for i in range(len(animals)):
        glue_sessions(animal=animals[i], protocol='stage_training', to_csv=True)
