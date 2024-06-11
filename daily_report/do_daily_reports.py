import time
from pathlib import Path
import os
from daily_report.daily_report import *
from daily_report.daily_report_v2 import *
from daily_report.daily_report_v3 import *
from daily_report.daily_report_v4 import *
from daily_report.daily_report_v5 import *


def do_daily_reports(version=5, experiment='2AFC_5', index=-1, send_slack=False):
    time_start = time.time()

    if experiment is None:

        # folder_in = '/home/alexis/pv_nmdar_eranet/experiments/'  # Where the data for all animals is
        folder_in = Path.home() / 'pv_nmdar_eranet' / 'experiments'  # Where the data for all animals is

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

    # folder_in = '/home/alexis/pv_nmdar_eranet/experiments/' + experiment + '/setups/'  # Where the data for all animals is
    folder_in = Path(
        Path.home() / 'pv_nmdar_eranet' / 'experiments' / experiment / 'setups')  # Where the data for all animals is

    # Select the output folder for the corrupted sessions and create it if it doesn't exist
    # In case 'glue_sessions.py' hasn't been run yet for this experiment
    # folder_out_corrupted_sessions = '/home/alexis/PycharmProjects/glue_sessions/' + experiment + '/'
    folder_out_corrupted_sessions = Path(Path.home() / 'PycharmProjects' / 'glue_sessions' / experiment)
    if not os.path.exists(folder_out_corrupted_sessions):
        # os.mkdir(folder_out_corrupted_sessions)
        folder_out_corrupted_sessions.mkdir(parents=True, exist_ok=True)

    animals = os.listdir(folder_in)
    animals.sort()

    # Remove test setups and animals not training

    # Remove setups
    test_setups = ['Test', 'Test0', 'Test1', 'Test2', 'Test3', 'Test4', 'Test5', 'Test6', 'Test7', 'Test8']

    # Remove Pycharm folder
    Pycharm_folder = ['.idea']

    # Switched to training in setup2, remove from setup1 to avoid doing reports from old sessions
    # username = os.getlogin()  # 'OSError: [Errno 25] Inappropriate ioctl for device' in ephys PC
    # username = os.environ['USER']
    username = os.path.split(os.path.expanduser('~'))[-1]

    if username == 'setup1':
        remove_from_setup1 = ['000', '001', '005', '008']
    else:
        remove_from_setup1 = []

    # Switched to training in setup1, remove from setup2 to avoid doing reports from old sessions
    if username == 'setup2':
        remove_from_setup2 = ['002', '003', '006', '008']
    else:
        remove_from_setup2 = []

    # Switched to training in setup0, remove from setup2 to avoid doing reports from old sessions
    if username == 'setup0':
        remove_from_setup_ephys = ['420', '422', '619', '623', '876', '907', '909', '911',
                                   '009', '008', '002', '004', '001']  # Batch 5
    else:
        remove_from_setup_ephys = []

    # Animals that died or that didn't learn the task and were retired from training
    not_training = ['001', '002', '003', '005', '006', '008', '009']

    animals_to_remove = test_setups + Pycharm_folder + remove_from_setup1 + remove_from_setup2 + remove_from_setup_ephys \
                        + not_training
    # Usually I don't want to do the daily reports of the 'Test' subject
    # '.idea' is a Pycharm's hidden file
    for i in range(len(animals_to_remove)):
        try:
            animals.remove(animals_to_remove[i])
        except ValueError:
            pass

    print(f'Doing the reports of {len(animals)} animals...')

    for i in range(len(animals)):

        corrupted_sessions = []
        # folder2 = folder_in + animals[i] + '/sessions/'
        folder2 = Path(folder_in / animals[i] / 'sessions')
        sessions = os.listdir(folder2)
        sessions.sort()  # Sort them by date
        sessions = [s for s in sessions if 'stage_training' in s]  # Ignore lick_teaching sessions

        # Do all sessions
        if index == 'all':
            for k in range(len(sessions)):
                session = sessions[k]  # Add scenario in which there are several sessions per day
                date_session = session[-15:-7]  # Indexing from the end because length of date + time won't change
                split_sessions = [s for s in sessions if date_session in s]
                path = Path(folder2 / session / session).with_suffix(
                    '.csv')  # Get csv file path to input parse/parse_v2.py
                print(""'Doing the daily reports of animal ', animals[i], ': ', len(split_sessions),
                      ' sessions found on the date ', date_session, "", sep='')
                try:
                    if version == 1:
                        daily_report(path, send_slack=send_slack)
                    elif version == 2:
                        daily_report_v2(path, send_slack=send_slack)
                    elif version == 3:
                        daily_report_v3(path, send_slack=send_slack)
                    elif version == 4:
                        daily_report_v4(path, send_slack=send_slack)
                    elif version == 5:
                        daily_report_v5(path, send_slack=send_slack)
                except (IndexError, ValueError, FileNotFoundError, KeyError,
                        ZeroDivisionError):  # When passing 2 exceptions it must be in this syntax
                    print(
                        f"The session '{session}' is corrupted. Adding to corrupted sessions log and continuing "
                        f"with next session...")
                    corrupted_sessions.append(session)

        # Do last session
        else:
            # index = -1  # last session
            session = sessions[index]
            date_session = session[
                           -15:-7]  # Indexing from the end because length of date + time won't change, opposite to
            # mice and protocol names
            split_sessions = [s for s in sessions if date_session in s]

            # This block looks if there are more than one session with the same date and do the reports for each if so
            if len(split_sessions) > 1:
                for j in range(len(split_sessions)):
                    session = split_sessions[j]
                    path = Path(folder2 / session / session).with_suffix(
                        '.csv')  # Get csv file path to input parse/parse_vX.py
                    print(""'Doing the daily reports of animal ', animals[i], ': ', len(split_sessions),
                          ' sessions found on the date ', date_session, ' (session ', j + 1, '/', len(split_sessions),
                          ')', "", sep='')
                    try:
                        if version == 1:
                            daily_report(path, send_slack=send_slack)
                        elif version == 2:
                            daily_report_v2(path, send_slack=send_slack)
                        elif version == 3:
                            daily_report_v3(path, send_slack=send_slack)
                        elif version == 4:
                            daily_report_v4(path, send_slack=send_slack)
                        elif version == 5:
                            daily_report_v5(path, send_slack=send_slack)
                    except (IndexError, ValueError, FileNotFoundError, KeyError,
                            ZeroDivisionError):  # When passing 2 exceptions it must be in this syntax
                        print(
                            f"The session '{session}' is corrupted. Adding to corrupted sessions log and continuing "
                            f"with next session...")
                        corrupted_sessions.append(session)
            else:
                path = Path(folder2 / session / session).with_suffix(
                    '.csv')  # Get csv file path to input parse/parse_v2.py
                print(""'Doing the daily reports of animal ', animals[i], ': ', len(split_sessions),
                      ' sessions found on the date ', date_session, "", sep='')
                try:
                    if version == 1:
                        daily_report(path, send_slack=send_slack)
                    elif version == 2:
                        daily_report_v2(path, send_slack=send_slack)
                    elif version == 3:
                        daily_report_v3(path, send_slack=send_slack)
                    elif version == 4:
                        daily_report_v4(path, send_slack=send_slack)
                    elif version == 5:
                        daily_report_v5(path, send_slack=send_slack)
                except (IndexError, ValueError, FileNotFoundError, KeyError,
                        ZeroDivisionError):  # When passing 2 exceptions it must be in this syntax
                    print(
                        f"The session '{sessions}' is corrupted. Adding to corrupted sessions log and continuing "
                        f"with next session...")
                    corrupted_sessions.append(session)

        if corrupted_sessions:  # If corrupted sessions isn't empty, save them in a .csv file
            # Save corrupted sessions in a separate csv file
            # with open(Path(folder_out_corrupted_sessions/animals[i]/'_corrupted_sessions').with_suffix('.csv'),
            #           'w', newline='') as f:
            with open(Path(folder_out_corrupted_sessions / (animals[i] + '_corrupted_sessions')).with_suffix('.csv'),
                      'w', newline='') as f:
                wr = csv.writer(f)
                wr.writerow(corrupted_sessions)

    print('\n')

    time_end = time.time()
    runtime = time_end - time_start
    print('The script took', round(runtime, 2), 'seconds to run')
