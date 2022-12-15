import time
import os
from daily_report.daily_report import *
from daily_report.daily_report_v2 import *
from daily_report.daily_report_v3 import *


def do_daily_reports(version=3, send_slack=False):
    time_start = time.time()
    # print('Doing daily reports of: ' + animal)
    # folder = '/home/alexis/pv_nmdar_eranet/experiments/2AFC/setups/'
    # folder = '/home/alexis/pv_nmdar_eranet/experiments/2AFC_2/setups/'
    folder = '/home/alexis/pv_nmdar_eranet/experiments/2AFC_3/setups/'
    # animal = input('Enter animal')
    animals = os.listdir(folder)
    animals.sort()

    try:
        animals.remove('Test')  # Usually I don't want to do the daily reports of the Test subject
    except ValueError:
        pass
    try:
        animals.remove('.idea')  # Pycharm's hidden file
    except ValueError:
        pass

    # # Remove animals not training
    try:
        animals.remove('387')
    except ValueError:
        pass

    try:
        animals.remove('395')
    except ValueError:
        pass

    try:
        animals.remove('398')
    except ValueError:
        pass

    try:
        animals.remove('419')
    except ValueError:
        pass

    try:
        animals.remove('420')
    except ValueError:
        pass

    # try:
    #     animals.remove('422')
    # except ValueError:
    #     pass

    try:
        animals.remove('501')
    except ValueError:
        pass

    try:
        animals.remove('614')
    except ValueError:
        pass

    try:
        animals.remove('615')
    except ValueError:
        pass
    #
    # try:
    #     animals.remove('616')
    # except ValueError:
    #     pass

    try:
        animals.remove('617')
    except ValueError:
        pass

    try:
        animals.remove('618')
    except ValueError:
        pass

    try:
        animals.remove('619')
    except ValueError:
        pass

    try:
        animals.remove('620')
    except ValueError:
        pass

    # try:
    #     animals.remove('623')
    # except ValueError:
    #     pass

    try:
        animals.remove('625')
    except ValueError:
        pass

    try:
        animals.remove('627')
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
                    j] + '.csv'  # Get csv file path to input parse/parse_v2.py
                print(""'Doing the daily report(s) of animal ', animals[i], ': ', len(split_sessions),
                      ' sessions found in the same date(s)'"", sep='')
                # print(path)
                if version == 1:
                    # daily_report(path, send_slack=send_slack)
                    pass
                elif version == 2:
                    # daily_report_v2(path, send_slack=send_slack)
                    pass
                elif version == 3:
                    daily_report_v3(path, send_slack=send_slack)
        else:
            path = folder2 + sessionID + '/' + sessionID + '.csv'  # Get csv file path to input parse/parse_v2.py
            print(""'Doing the daily report(s) of animal ', animals[i], ': ', len(split_sessions),
                  ' sessions found in the same date(s)'"", sep='')
            # print(path)

            if version == 1:
                # daily_report(path, send_slack=send_slack)
                pass
            elif version == 2:
                # daily_report_v2(path, send_slack=send_slack)
                pass
            elif version == 3:
                daily_report_v3(path, send_slack=send_slack)

    time_end = time.time()
    runtime = time_end - time_start
    print('The script took', round(runtime, 2), 'seconds to run')