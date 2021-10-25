import os
import time
from intersession.intersession import intersession

########################################################################################################################


def do_intersessions():

    time_start = time.time()
    # print('Doing intersession reports of: ' + animal)
    folder = '/home/alexis/PycharmProjects/glue_sessions/'
    # animal = input('Enter animal')
    animals = os.listdir(folder)
    animals = [animals for animals in animals if animals.endswith('.csv')]
    animals.sort()

    for i in range(len(animals)):
        path = folder + animals[i]
        intersession_test(path)

    time_end = time.time()
    runtime = time_end - time_start
    print('The script took', round(runtime, 2), 'seconds to run')