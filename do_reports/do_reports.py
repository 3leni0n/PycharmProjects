# To do:
# Select what sessions to do the reports
# Choose between plotting together sessions from the same day (with maybe a vertical red line) to separate them, or
# in separate reports

# Line of bash code to sync the cluster data with the local machine:
# rsync -avzP -e 'ssh -p 4022' mouse@neurocomp.fcrb.es:/archive/mouse/pv_nmdar_eranet* ~/

########################################################################################################################

import os
import time
from daily_report.daily_report import daily_report

########################################################################################################################


# Define function
def do_reports():

    time_start = time.time()
    # print('Doing daily reports of: ' + animal)
    folder = '/home/alexis/pv_nmdar_eranet/experiments/2AFC/setups/'
    # animal = input('Enter animal')
    animals = os.listdir(folder)
    animals.sort()

    try:
        animals.remove('Test')  # Usually I don't want to do the daily reports of the Test subject
        animals.remove('.idea')  # Pycharm's archive

        # Remove animals not training
        animals.remove('902')
        animals.remove('904')
        animals.remove('909')
        # animals.remove('915')
    except ValueError:
        pass

    for i in range(len(animals)):

        folder2 = folder + animals[i] + '/sessions/'  # Replace 0 with i in for loop with n = len(animals)
        sessions = os.listdir(folder2)
        sessions.sort()  # Sort them by date
        index = -1  # last session

        sessionID = sessions[index]  # Add scenario in which there are several sessions per day
        date_sessionID = sessionID[
                         -15:-7]  # Indexing from the end because length of date + time won't change, opposite to
        # mice and protocol names
        split_sessions = [s for s in sessions if date_sessionID in s]

        # This block looks if there are more than one session with the same date and do the reports for each if so
        if len(split_sessions) > 1:
            for j in range(len(split_sessions)):
                path = folder2 + split_sessions[j] + '/' + split_sessions[
                    j] + '.csv'  # Get csv file path to input parse.py
                print(""'Doing the daily report(s) of animal ', animals[i], ': ', len(split_sessions),
                      ' sessions found in the same date(s)'"", sep='')
                # print(path)
                daily_report(path)
        else:
            path = folder2 + sessionID + '/' + sessionID + '.csv'  # Get csv file path to input parse.py
            print(""'Doing the daily report(s) of animal ', animals[i], ': ', len(split_sessions),
                  ' sessions found in the same date(s)'"", sep='')
            # print(path)
            daily_report(path)

    time_end = time.time()
    runtime = time_end - time_start
    print('The script took', round(runtime, 2), 'seconds to run')
