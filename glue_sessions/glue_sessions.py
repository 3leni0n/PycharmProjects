import time
import os
import pandas as pd

from parse.parse import parse


# Define function
def glue_sessions():

    time_start = time.time()
    # animal = '902'
    animal = input('Enter animal')
    print('Glueing sessions of: ' + animal)
    folder = '/home/alexis/2AFC/setups/' + animal + '/sessions'
    sessions = os.listdir(folder)
    sessions.sort()  # Sort them by date
    protocol = 'stage_training'
    df = pd.DataFrame()

    for i in range(len(sessions)):

        if protocol in sessions[i]:
            # files = os.listdir(folder + '/' + sessions[i] + sessions[i] + '.csv')
            path = folder + '/' + sessions[i] + '/' + sessions[i] + '.csv'  # Get csv file path to input parse.py
            df_session = parse(path)
            df = pd.concat([df, df_session])
        else:
            pass

    time_end = time.time()
    runtime = time_end - time_start
    print('The script took', round(runtime, 2), 'seconds to run')

    return df
