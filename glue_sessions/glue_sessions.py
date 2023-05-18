import time
import os
import numpy as np
import pandas as pd
from parse.parse import parse
from parse.parse_v2 import parse_v2
import csv

# To do:
# Add training day index column to df


# Define functions
def glue_sessions(animal=None, protocol='stage_training_v4', experiment='2AFC_4', to_csv=False):
    """
    Glue all the sessions of a given animal.
    :param animal: ID number of the animal
    :param protocol: task code version
    :param experiment: batch of the animal
    :param to_csv: if True save data as .csv file
    :return: pandas DataFrame with the data, .csv file with the ID of the corrupted sessions
    """

    time_start = time.time()

    if experiment is None:

        folder_in = '/home/alexis/pv_nmdar_eranet/experiments/'  # Where the data for all animals is
        experiments = os.listdir(folder_in)  # List experiments
        experiments.sort()  # Sort them by name

        try:
            experiments.remove('.idea')  # Pycharm's file
            experiments.remove('Daily check')
            experiments.remove('WaterCalibration')
        except ValueError:
            pass

        print('Experiments: ' + str(experiments)[1:-1])  # Remove square brackets
        experiment = input('Enter experiment name')

    folder_in = '/home/alexis/pv_nmdar_eranet/experiments/' + experiment + '/setups/'  # Where the data for all animals is

    if animal is None:

        animals = os.listdir(folder_in)  # List animals
        animals.sort()  # Sort them by name

        # Usually I don't want Test subject(s)
        animals_to_remove = ['Test', 'Test1', 'Test2', 'Test3', 'Test4', 'Test5', 'Test6', 'Test7', 'Test8',
                             '.idea']  # Pycharm's file

        for i in range(len(animals_to_remove)):
            try:
                animals.remove(animals_to_remove[i])
            except ValueError:
                pass

        print('Animals: ' + str(animals)[1:-1])  # Remove square brackets
        animal = input('Enter animal')  # Ask user to input animal to glue sessions from

    # # Check if csv from that animal already exist, and if so, import it
    glued_sessions = []  # Initialize empty list so if it's the first time glue all sessions

    # Select the output folder and create it if it doesn't exist
    folder_out = '/home/alexis/PycharmProjects/glue_sessions/' + experiment + '/'
    if not os.path.exists(folder_out):
        os.mkdir(folder_out)

    glued_animals = os.listdir(folder_out)
    glued_animals.sort()
    glued_animals = [x for x in glued_animals if x.endswith('.csv')]  # Get rid of non csv files

    if animal + '.csv' in glued_animals:
        df = pd.read_csv('/home/alexis/PycharmProjects/glue_sessions/' + experiment + '/' + animal + '.csv')
        glued_sessions = df.Session.unique().tolist()
    else:
        df = pd.DataFrame()  # Create empty DataFrame if there's no csv yet for that animal

    folder_in = folder_in + animal + '/sessions/'  # Update folder_in with selected animal
    sessions = os.listdir(folder_in)  # List sessions
    sessions.sort()  # Sort them by date

    if protocol is None:

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
            path = folder_in + sessions[i] + '/' + sessions[i] + '.csv'  # Get csv file path to input parse.py
            print('Parsing session ' + "'" + sessions[i] + "'" + '...', sep='')

            try:
                if protocol == 'stage_training':
                    df_session = parse(path)  # Parse session
                elif protocol == 'stage_training_v2' or 'stage_training_v3':
                    df_session = parse_v2(path)  # Parse session
                df = pd.concat([df, df_session])  # Add parsed session to the bottom of the DataFrame
            except (IndexError, ValueError, FileNotFoundError, ZeroDivisionError):  # When passing 2 exceptions it must be in this syntax
                print(
                    f"The session '{sessions[i]}' is corrupted. Adding to corrupted sessions log and continuing with "
                    f"next session...")
                corrupted_sessions.append(sessions[i])

        else:
            pass

    if to_csv:
        df.to_csv(folder_out + animal + '.csv', index=False)  # index=False to avoid the 'Unmmaed: 0' column

    print('The corrupted sessions are:', *corrupted_sessions, '\n', sep='\n')

    if corrupted_sessions:  # If corrupted sessions isn't empty, save them in a .csv file
        # Save corrupted sessions in a separate csv file
        with open('/home/alexis/PycharmProjects/glue_sessions/' + experiment + '/' + animal + '_corrupted_sessions.csv',
                  'w', newline='') as f:
            wr = csv.writer(f)
            wr.writerow(corrupted_sessions)

    time_end = time.time()
    runtime = time_end - time_start
    print('The script took', round(runtime, 2), 'seconds to run')

    return df, corrupted_sessions


def update_glued_sessions(protocol='stage_training_v4', experiment='2AFC_4'):
    """
    Update the glued_sessions .csv files for all animals with the non yet included sessions.
    :param protocol: task code version
    :param experiment: batch of the animals
    :return:
    """

    time_start = time.time()

    if experiment is None:

        folder = '/home/alexis/pv_nmdar_eranet/experiments/'  # Where the data for all animals is
        experiments = os.listdir(folder)  # List experiments
        experiments.sort()  # Sort them by name

        try:
            experiments.remove('.idea')  # Pycharm's archive
            experiments.remove('Daily check')
            experiments.remove('WaterCalibration')
        except ValueError:
            pass

        print('Experiments: ' + str(experiments)[1:-1])  # Remove square brackets
        experiment = input('Enter experiment name')

    folder = '/home/alexis/pv_nmdar_eranet/experiments/' + experiment + '/setups/'  # Where the data for all animals is
    animals = os.listdir(folder)  # List animals
    animals.sort()  # Sort them by name

    # Usually I don't want Test subject(s)
    animals_to_remove = ['Test', 'Test1', 'Test2', 'Test3', 'Test4', 'Test5', 'Test6', 'Test7', 'Test8',
                         '.idea']  # Pycharm's file

    for i in range(len(animals_to_remove)):
        try:
            animals.remove(animals_to_remove[i])
        except ValueError:
            pass

    for i in range(len(animals)):
        print(f'Updating sessions of animal {animals[i]}...')
        glue_sessions(animal=animals[i], protocol=protocol, experiment=experiment, to_csv=True)

    time_end = time.time()
    runtime = time_end - time_start
    print('The script took', round(runtime, 2), 'seconds to run')


def glue_animals(protocol='stage_training_v4', experiment='2AFC_4', to_csv=False):
    """
    Glue all the sessions from all the animals of a given batch.
    :param protocol: task code version
    :param experiment: batch of animals
    :param to_csv: if True save data as .csv file
    :return: pandas DataFrame with the data
    """

    time_start = time.time()

    if experiment is None:

        folder_in = '/home/alexis/PycharmProjects/glue_sessions/'  # Where the data for all animals is
        experiments = os.listdir(folder_in)  # List experiments
        experiments.sort()  # Sort them by name
        experiments = [x for x in experiments if os.path.isdir(folder_in + x)]  # Get rid of non folders

        try:
            experiments.remove('__pycache__')  # Pycharm's file
        except ValueError:
            pass

        print('Experiments: ' + str(experiments)[1:-1])  # Remove square brackets
        experiment = input('Enter experiment name')

    folder_in = '/home/alexis/PycharmProjects/glue_sessions/' + experiment + '/'  # Where the data for all animals is

    update_glued_sessions(protocol=protocol, experiment=experiment)  # Update glued sessions first

    animals = os.listdir(folder_in)  # List animals
    animals.sort()  # Sort them by name

    df = pd.DataFrame()  # Create empty dataframe

    for i in range(len(animals)):
        df_animal = pd.read_csv(folder_in + animals[i])
        df = pd.concat([df, df_animal])  # Add parsed session to the bottom of the DataFrame

    folder_out = folder_in

    if to_csv:
        df.to_csv(folder_out + 'all' + '.csv', index=False)  # index=False to avoid the 'Unnamed: 0' column

    time_end = time.time()
    runtime = time_end - time_start
    print('The script took', round(runtime, 2), 'seconds to run')

    return df


def glue_batches():
    """
    Glue all sessions from all animals from all batches.
    :return: padas Dataframe with the data
    """