import sys
import os
import time
from mini_parse import mini_parse
from real_time_plot import real_time_plot
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class File:
    def __init__(self, path):
        self.path = path
        self.skip = 7
        self.df = None
        self.reward_side = None
        self.box = None
        self.axis1 = None
        self.axis2 = None
        self.axis3 = None
        self.axis4 = None


class Parameters:
    def __init__(self, axis, minutes_ago, max_sessions, data_path):
        self.files = []
        self.paths = {}
        self.axis = axis
        self.minutes_ago = minutes_ago
        self.max_sessions = max_sessions
        self.data_path = data_path


def path_generator(path, minutes_ago, max_sessions):
    paths = []
    for root, _, file in os.walk(path):
        for f in file:
            if f.endswith(".csv") and 'lick_teaching' not in f:
                filename = os.path.join(root, f)
                modification = os.path.getmtime(filename)
                if time.time() - modification < minutes_ago * 60:
                    paths.append((filename, modification))
    paths = sorted(paths, key=lambda x: x[1], reverse=True)[:max_sessions]
    return [x[0] for x in paths]


def get_paths(parameters):
    paths = path_generator(parameters.data_path, parameters.minutes_ago, parameters.max_sessions)
    if set(paths) != parameters.paths:
        files = []
        boxes = []
        for path in paths:
            file = File(path)
            mini_parse(file)  # parse the data and creates file.reward_side and file.box
            if file.box not in boxes:
                files.append(file)
                boxes.append(file.box)

        try:
            files = sorted(files, key=lambda x: x.box)

            for index, file in enumerate(files):

                print(file.box)
                if file.box == "Bpod5":
                    j = 0
                elif file.box == "Bpod6":
                    j = 1
                elif file.box == "Bpod7":
                    j = 2
                elif file.box == "Bpod8":
                    j = 3
                else:
                    j = -1

                if j >= 0:
                    print("a ver", j)
                    # print(parameters.axis)
                    file.axis1 = parameters.axis[j * 4]
                    file.axis2 = parameters.axis[j * 4 + 1]
                    file.axis3 = parameters.axis[j * 4 + 2]
                    file.axis4 = parameters.axis[j * 4 + 3]

            parameters.files = files
            parameters.paths = {file.path for file in files}
        except:
            pass


def animate(i, parameters, trials):

    start = time.time()
    print('')
    print('iteration', i)

    if i % 1 == 0:  # How many iterations to wait to look for new files
        get_paths(parameters)
        print('checking for files')
    for file in parameters.files:
        filename = os.path.basename(file.path)
        print('parsing file:', filename)
        mini_parse(file)
        try:
            real_time_plot(file.df, file.box, filename, file.axis1, file.axis2, file.axis3, file.axis4, trials)
            print('OK file:', filename)
        except:
            pass
    print('time:', time.time() - start)


def main():

    data_path = '/home/setup2/pv_nmdar_eranet/experiments/2AFC_2/setups'
    # data_path = '/home/alexis/pv_nmdar_eranet/experiments/2AFC_2/setups'
    minutes_ago = 1  # How much time back look for sessions
    max_sessions = 4  # Max number of boxes at the same time
    interval = 5000  # in ms, the larger the name the less demanding
    trials = 100  # How many trials to show

    try:
        sessions_number = int(sys.argv[1])
    except:
        sessions_number = max_sessions

    if sessions_number < 1 or sessions_number > max_sessions:
        print("number of sessions must be a number from 1 to", max_sessions)
        return

    fig = plt.figure()
    axis = []

    for i in range(sessions_number * 4):
        axis.append(fig.add_subplot(sessions_number * 2, 2, i + 1))

    parameters = Parameters(axis, minutes_ago, sessions_number, data_path)

    ani = FuncAnimation(fig, animate, fargs=(parameters, trials, ), interval=interval)

    plt.subplots_adjust(hspace=.9)
    plt.show()


if __name__ == '__main__':
    main()
