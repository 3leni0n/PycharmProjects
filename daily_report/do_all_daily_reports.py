import time
import os
from daily_report import *


def do_all_daily(experiment, animal):

    time_start = time.time()

    folder = '/home/alexis/pv_nmdar_eranet/experiments/' + experiment + '/setups/' + animal + '/sessions/'
    sessions = os.listdir(folder)
    sessions.sort()  # Sort them by date
    sessions = [x for x in sessions if 'stage_training' in x]  # Get rid of non training sessions

    for i in range(len(sessions)):
        path = folder + sessions[i] + '/' + sessions[i] + '.csv'  # Get csv file path to input parse_v2.py
        daily_report_v2(path)

    time_end = time.time()
    runtime = time_end - time_start
    print('The script took', round(runtime, 2), 'seconds to run')