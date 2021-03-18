import pandas as pd
import numpy as np

# Path to csv
path = '/home/alexis/PycharmProjects/parse/Test_stage3_alexis_20210317-173400/Test_stage3_alexis_20210317-173400.csv'

path = '/home/alexis/PycharmProjects/parse/Test_stage3_alexis_20210317-172700/Test_stage3_alexis_20210317-172700.csv'

# Dont take first 6 lines
df = pd.read_csv(path, skiprows=6, sep=';')

index = df[df['TYPE'] == 'TRIAL'].index

for i in range(len(index) - 1):
    band = df[index[i]:index[i+1]]
    print("a")


# Register values:
# filename

# INFO (metadata)
serial_port = df[df.MSG == 'SERIAL-PORT']['+INFO'].iloc[0]
protocol = df[df.MSG == 'PROTOCOL-NAME']['+INFO'].iloc[0]
creator = df[df.MSG == 'CREATOR-NAME']['+INFO'].iloc[0]
project = df[df.MSG == 'PROJECT-NAME']['+INFO'].iloc[0]
experiment = df[df.MSG == 'EXPERIMENT-NAME']['+INFO'].iloc[0]
board = df[df.MSG == 'BOARD-NAME']['+INFO'].iloc[0]
setup = df[df.MSG == 'SETUP-NAME']['+INFO'].iloc[0]
net_port = df[df.MSG == 'NET-PORT']['+INFO'].iloc[0]
subject = df[df.MSG == 'SUBJECT-NAME']['+INFO'].iloc[0]
session = df[df.MSG == 'SESSION-NAME']['+INFO'].iloc[0]
session_started = df[df.MSG == 'SESSION-STARTED']['+INFO'].iloc[0]
session_ended = df[df.MSG == 'SESSION-ENDED']['+INFO'].iloc[0]

creator = creator.split()[0][2:-2]
subject = subject.split()[0][2:-2]
date = session_started.split()[0]
time_session_started = session_started.split()[1]
time_session_ended = session_ended.split()[1]

#VAL
aw = df[df.MSG == 'VAR_AW']['+INFO'].iloc[0]
trial = df[df.MSG == 'VAR_TRIAL']['+INFO'].iloc[0]
timeout = df[df.MSG == 'VAR_TIMEOUT']['+INFO'].iloc[0]
fixation = df[df.MSG == 'VAR_FIXATION']['+INFO'].iloc[0]
stage = df[df.MSG == 'VAR_STAGE']['+INFO'].iloc[0]
substage = df[df.MSG == 'VAR_SUBSTAGE']['+INFO'].iloc[0]
motor = df[df.MSG == 'VAR_MOTOR']['+INFO'].iloc[0]
rec = df[df.MSG == 'VAR_REC']['+INFO'].iloc[0]
reward_side = df[df.MSG == 'REWARD_SIDE']['+INFO'].iloc[0]  # change! to VAR_REWARD_SIDE   Why is 3k instead 1k??????
valve_1 = df[df.MSG == 'VALVE_1:']['+INFO'].iloc[0]  # change! Take out the ':'
valve_2 = df[df.MSG == 'VALVE_2:']['+INFO'].iloc[0]


valves = df[df.MSG.str.startswith("VALVE")]