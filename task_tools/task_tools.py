import serial
import time
from queue import Queue
from threading import Thread
import os
import csv
from string import ascii_lowercase
import numpy as np


# Arduino reader
class ArduinoReader:
    def __init__(self, address):
        self.arduino = serial.Serial(address, 9600, timeout=1.0)
        self.arduino.setDTR(False)
        time.sleep(1)
        self.arduino.flushInput()
        self.arduino.setDTR(True)
        self.queue = Queue()
        self.queue2 = Queue()
        self.t = Thread(target=self.read_serial, daemon=True)
        self.t.start()

    def read_serial(self):
        while True:
            line = self.arduino.readline()
            try:
                line = str(line.decode('utf-8'))
                if len(line) == 9:
                    self.queue.put(line)
                elif len(line) > 20:
                    self.queue2.put(line)
            except UnicodeDecodeError:
                pass

    def read_value(self):
        val = ''
        while True:
            try:
                val = self.queue.get_nowait()
            except:
                break
        return val

    def read_value2(self):
        val = ''
        while True:
            try:
                val = self.queue2.get_nowait()
            except:
                break
        return val


# Get calibration of the valves (from UtilsR)
def getWaterCalib(board, ports):
    log_name = os.path.expanduser("~/pluginsr-for-pybpod/water-calibration-plugin/DATA/water_calibration.csv")
    with open(log_name, 'r') as log:
        calibration_data = csv.DictReader(log, delimiter=';')
        latest_row = None
        for row in calibration_data:
            if row['board'] == board:
                latest_row = dict(row)
        if latest_row is None:
            raise Exception("Water calibration data not found.")
        else:
            results = latest_row['pulse_duration'].split("-")
            return [float(results[i - 1]) for i in ports]


def do_sounds_dict(start, stop, num, n_decimals):
    """Dictionary letter: TTL pulses. Need to be in line with Arduino's code"""
    if num > 26:
        raise ValueError("'num' cannot be higher than abc's length (26)")
    chars = list(ascii_lowercase[:num])  # Make a list of all the lowercase letters as long as num
    pulses = np.around(np.linspace(start, stop, num), n_decimals)  # Make evenly spaced TTL pulses rounded to round2
    return dict(zip(chars, pulses))


def select_ilds(ilds, p, side):
    # r = random.random()  # Generate random float between 0 and 1. PyBpod missing random library
    r = np.random.random(1)[0]  # Generate random float between 0 and 1
    if r > p:
        if side == 0:  # Left
            return ilds.min()  # Sound left only
        else:  # Right
            return ilds.max()  # Sound right only
    else:
        if side == 0:  # Left
            options = ilds[ilds <= 0]  # Left ILDs
            options = np.repeat(options, 2)  # Repeat each element of the vector
            options = options[:-1]  # Exclude one of the 0s from the vector, so it has 1/2 p than the rest
        else:  # Right
            options = ilds[ilds >= 0]  # Right ILDs
            options = np.repeat(options, 2)  # Repeat each element of the vector
            options = options[1:]  # Exclude one of the 0s from the vector, so it has 1/2 p than the rest
    # selected_ild = random.choice(options)
    selected_ild = np.random.choice(options)
    return selected_ild
