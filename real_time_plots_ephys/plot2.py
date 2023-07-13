import sys
import os
import time
from mini_parse import mini_parse
from real_time_plot import real_time_plot
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from multiprocessing import Process, Queue


def collectData(communicator, data_path, minutes_ago, max_sessions, paths, files):

    while True:
        files = get_paths(data_path, minutes_ago, max_sessions, paths, files)
        communicator.put(files)
        time.sleep(5)



class File:
    def __init__(self, path):
        self.path = path
        self.skip = 7
        self.df = None
        self.reward_side = None
        self.box = None



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



def get_paths(data_path, minutes_ago, max_sessions, paths, files):
    new_paths = path_generator(data_path, minutes_ago, max_sessions)
    if set(new_paths) != paths:
        files = []
        boxes = []
        for path in new_paths:
            file = File(path)
            mini_parse(file)  # parse the data and creates file.reward_side and file.box
            if file.box not in boxes:
                files.append(file)
                boxes.append(file.box)

        try:
            files = sorted(files, key=lambda x: x.box)
        except:
            pass
    else:
        for file in files:
            mini_parse(file)
    return files




def animate(i, trials, axis, communicator):

    start = time.time()
    print('')
    print('iteration', i)

    files = communicator.get()

    print("a")
    print(files)
    print("a")
    for file in files:
        filename = os.path.basename(file.path)
        print('parsing file:', filename)
        print(file.box)
        try:
            if file.box == "Bpod5":
                real_time_plot(file.df, file.box, filename, axis[0], axis[1], axis[2], axis[3], trials)
                print(axis[0])
            elif file.box == "Bpod6":
                real_time_plot(file.df, file.box, filename, axis[4], axis[5], axis[6], axis[7], trials)
            elif file.box == "Bpod7":
                real_time_plot(file.df, file.box, filename, axis[8], axis[9], axis[10], axis[11], trials)
            elif file.box == "Bpod8":
                real_time_plot(file.df, file.box, filename, axis[12], axis[13], axis[14], axis[15], trials)
            print('OK file:', filename)
        except:
            pass
    print('time:', time.time() - start)


def main():

    data_path = '/home/setup2/pv_nmdar_eranet/experiments/2AFC_2/setups'
    # data_path = '/home/alexis/pv_nmdar_eranet/experiments/2AFC_2/setups'
    minutes_ago = 1  # How much time back look for sessions
    max_sessions = 4  # Max number of boxes at the same time
    interval = 1000  # in ms, the larger the name the less demanding
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
    paths = []
    files = []

    for i in range(sessions_number * 4):
        axis.append(fig.add_subplot(sessions_number * 2, 2, i + 1))

    communicator = Queue()
    duta = Process(target=collectData, args=(communicator, data_path, minutes_ago, max_sessions, paths, files,))
    duta.start()

    ani = FuncAnimation(fig, animate, fargs=(trials, axis, communicator), interval=interval)

    plt.subplots_adjust(hspace=.9)
    plt.show()

    duta.join()


if __name__ == '__main__':
    main()
