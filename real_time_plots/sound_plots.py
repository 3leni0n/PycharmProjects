import os
import pandas as pd
import matplotlib.pyplot as plt

username = os.getlogin()

# parameters
data_folder = '/home/' + username + 'setup2/pv_nmdar_eranet/experiments/2AFC_3/setups'
plot_folder = '/home/' + username + '/pv_nmdar_eranet/sound_plots'
csv_path = '/home/' + username + '/pv_nmdar_eranet/sound_plots.csv'
tasks = ['stage_training_v2']
first_day = '20220101'


# read old data
try:
    old_df = pd.read_csv(csv_path)
    filenames = old_df['filename'].unique().tolist()
except:
    old_df = None
    filenames = []


# search for new data
paths = []
dfs = []
bad_files = []

for root, _, file in os.walk(data_folder):
    for f in file:
        if f.endswith('.csv'):
            filename = f.split('.')[-2]
            values = filename.split('_')
            subject = values[0]
            task = [values[i] for i in range(1, len(values) - 1)]
            task = '_'.join(task)
            date = values[-1]

            if task in tasks and filename not in filenames:
                paths.append((os.path.join(root, f), filename, subject, task, date))
                filenames.append(filename)


# read new data
for index, path in enumerate(paths):
    try:
        df = pd.read_csv(path[0], skiprows=6, sep=';')
    except pd.errors.ParserError:
        print('error reading file')
        continue

    index = df[df['TYPE'] == 'TRIAL'].index
    ntrials = len(index) - 1
    box = [df[df.MSG == 'BOARD-NAME']['+INFO'].iloc[0]] * ntrials
    filename = [path[1]] * ntrials
    subject = [path[2]] * ntrials
    task = [path[3]] * ntrials
    date = [path[4]] * ntrials
    trial = []
    sound_left = []
    sound_right = []
    detected_duration = []
    state_duration = []

    for i in range(ntrials):

        trial.append(i)
        band = df[index[i]:index[i + 1]]

        sound_left.append(band[(band['TYPE'] == 'EVENT') & (band['+INFO'] == 'BNC1High')].size)
        sound_right.append(band[(band['TYPE'] == 'EVENT') & (band['+INFO'] == 'BNC2High')].size)

        try:
            detected_first = band[(band['TYPE'] == 'EVENT') & ((band['+INFO'] == 'BNC1High') | (band['+INFO'] == 'BNC2High'))].iloc[0]['BPOD-INITIAL-TIME']
            detected_last = band[(band['TYPE'] == 'EVENT') & ((band['+INFO'] == 'BNC1Low') | (band['+INFO'] == 'BNC2Low'))].iloc[-1]['BPOD-INITIAL-TIME']
            detected_duration.append(detected_last - detected_first)
        except:
            detected_duration.append(0)

        try:
            state_first = band[(band['TYPE'] == 'TRANSITION') & (band['MSG'] == 'StimulusDuration')].iloc[0]['BPOD-INITIAL-TIME']
            state_last = band[(band['TYPE'] == 'TRANSITION') & ((band['MSG'] == 'StimulusStop') | (band['MSG'] == 'Punish'))].iloc[0]['BPOD-INITIAL-TIME']
            value = min(1, state_last - state_first)
            state_duration.append(value)
        except:
            state_duration.append(1)


    df = pd.DataFrame({
        'filename': filename,
        'subject': subject,
        'task': task,
        'date': date,
        'box': box,
        'trial': trial,
        'sound_left': sound_left,
        'sound_right': sound_right,
        'detected_duration': detected_duration,
        'state_duration': state_duration
    })


    dfs.append(df)
    print('adding data from:', path[1])


# create dataframe
if dfs:
    if old_df is None:
        print('creating data from zero')
    else:
        old_df.drop(columns=['sessionNumber'], inplace=True)
        dfs = [old_df] + dfs

    df = pd.concat(dfs)
    df.sort_values(by=['date'], inplace = True)
    session_number = df.groupby('date').ngroup().add(1)
    df.insert(0, 'sessionNumber', session_number)
    df.to_csv(csv_path, index=False)

    print('')
    print('error files:')
    print(bad_files)
    print('')
else:
    df = old_df



# creating new columns and filter date > day
df['sound'] = df['sound_left'] + df['sound_right']
df['side'] = 'red'
df.loc[(df['sound_left'] > df['sound_right']), 'side'] = 'blue'
df.loc[(df['detected_duration'] < 0), 'detected_duration'] = 0
df.loc[(df['state_duration'] < 0), 'state_duration'] = 0
df['diff_duration'] = df['state_duration'] - df['detected_duration']

df = df[df['date'] > first_day]

boxes = df['box'].unique()
boxes.sort()



# ploting number of detections
for b in boxes:

    plots = []
    i = 1

    df1 = df[df['box'] == b]
    sessions = df1['sessionNumber'].unique()
    sessions.sort()
    nsessions = len(sessions)
    fig, ax = plt.subplots(nsessions, 1, figsize=(10, nsessions * 2))

    for (index, s) in enumerate(sessions):
        df2 = df1[df1['sessionNumber'] == s]
        name = df2['filename'].iloc[0]

        ax[index].scatter(df2['trial'], df2['sound'], c = df2['side'])
        ax[index].set_ylim([0, 300])
        ax[index].title.set_text(name)

    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    file_plot = b + '_number_detections.pdf'
    plot_path = os.path.join(plot_folder, file_plot)

    fig.tight_layout()

    plt.savefig(plot_path)




# ploting detected_duration of sound
for b in boxes:

    plots = []
    i = 1

    df1 = df[df['box'] == b]
    sessions = df1['sessionNumber'].unique()
    sessions.sort()
    nsessions = len(sessions)
    fig, ax = plt.subplots(nsessions, 1, figsize=(10, nsessions * 2))

    for (index, s) in enumerate(sessions):
        df2 = df1[df1['sessionNumber'] == s]
        name = df2['filename'].iloc[0]

        ax[index].scatter(df2['trial'], df2['detected_duration'], c = df2['side'])
        ax[index].set_ylim([0, 1.1])
        ax[index].title.set_text(name)

    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    file_plot = b + '_detected_duration.pdf'
    plot_path = os.path.join(plot_folder, file_plot)

    fig.tight_layout()

    plt.savefig(plot_path)



# ploting state_duration of sound
for b in boxes:

    plots = []
    i = 1

    df1 = df[df['box'] == b]
    sessions = df1['sessionNumber'].unique()
    sessions.sort()
    nsessions = len(sessions)
    fig, ax = plt.subplots(nsessions, 1, figsize=(10, nsessions * 2))

    for (index, s) in enumerate(sessions):
        df2 = df1[df1['sessionNumber'] == s]
        name = df2['filename'].iloc[0]

        ax[index].scatter(df2['trial'], df2['state_duration'], c = df2['side'])
        ax[index].set_ylim([0, 1.1])
        ax[index].title.set_text(name)

    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    file_plot = b + '_state_duration.pdf'
    plot_path = os.path.join(plot_folder, file_plot)

    fig.tight_layout()

    plt.savefig(plot_path)


# ploting diff_duration of sound
for b in boxes:

    plots = []
    i = 1

    df1 = df[df['box'] == b]
    sessions = df1['sessionNumber'].unique()
    sessions.sort()
    nsessions = len(sessions)
    fig, ax = plt.subplots(nsessions, 1, figsize=(10, nsessions * 2))

    for (index, s) in enumerate(sessions):
        df2 = df1[df1['sessionNumber'] == s]
        name = df2['filename'].iloc[0]

        ax[index].scatter(df2['trial'], df2['diff_duration'], c = df2['side'])
        ax[index].set_ylim([-1.1, 1.1])
        ax[index].title.set_text(name)

    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    file_plot = b + '_diff_duration.pdf'
    plot_path = os.path.join(plot_folder, file_plot)

    fig.tight_layout()

    plt.savefig(plot_path)
