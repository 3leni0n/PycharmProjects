import sys
import os
import time
from mini_parse import mini_parse
from real_time_plot import real_time_plot
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class File:
    def __init__(self, path, axis1, axis2, axis3):
        self.path = path
        self.skip = 7
        self.df = None
        self.axis1 = axis1
        self.axis2 = axis2
        self.axis3 = axis3
        self.reward_side = None
        self.stage = None


class Parameters:
    def __init__(self, axis, minutes_ago, max_sessions, data_path):
        self.files = []
        self.paths = []
        self.axis = axis
        self.minutes_ago = minutes_ago
        self.max_sessions = max_sessions
        self.data_path = data_path


def path_generator(path, minutes_ago, max_sessions):
    paths = []
    for root, _, file in os.walk(path):
        for f in file:
            if f.endswith(".csv"):
                filename = os.path.join(root, f)
                modification = os.path.getmtime(filename)
                if time.time() - modification < minutes_ago * 60:
                    paths.append((filename, modification))
    paths = sorted(paths, key=lambda x: x[1], reverse=True)[:max_sessions]
    return [x[0] for x in paths]


def get_paths(parameters):
    paths = path_generator(parameters.data_path, parameters.minutes_ago, parameters.max_sessions)
    if paths != parameters.paths:
        parameters.paths = paths
        for index, path in enumerate(paths):
            axis1 = parameters.axis[index * 3]
            axis2 = parameters.axis[index * 3 + 1]
            axis3 = parameters.axis[index * 3 + 2]
            file = File(path, axis1, axis2, axis3)
            mini_parse(file)
            parameters.files.append(file)


def animate(i, parameters):

    start0 = time.time()
    print('')
    print('iteration', i)

    if i % 10 == 0:
        start = time.time()
        get_paths(parameters)
        print('checking for files', time.time() - start)
    for file in parameters.files:
        start = time.time()
        mini_parse(file)
        print('parsing file', file.path, time.time() - start)
        start = time.time()
        real_time_plot(file.df, file.axis1, file.axis2, file.axis3)
        print('ploting file', file.path, time.time() - start)
    print('total time', time.time() - start0)


def main():

    data_path = '/home/alexis/2AFC/setups'
    minutes_ago = 60
    max_sessions = 4

    try:
        sessions_number = int(sys.argv[1])
    except:
        sessions_number = max_sessions

    if sessions_number < 1 or sessions_number > max_sessions:
        print("number of sessions must be a number from 1 to", max_sessions)
        return

    fig = plt.figure()
    axis = []

    for i in range(sessions_number * 3):  # n subplots per animal
        axis.append(fig.add_subplot(sessions_number, 3, i + 1))

    parameters = Parameters(axis, minutes_ago, sessions_number, data_path)

    ani = FuncAnimation(fig, animate, fargs=(parameters, ), interval=5000)  # interval is ms
    plt.show()


if __name__ == '__main__':
    main()
