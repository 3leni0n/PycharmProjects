import serial
import time
from queue import Queue
from threading import Thread
import os
import csv
from string import ascii_lowercase
import numpy as np
print(os.getcwd())
from toolsR import VideoR  # For video streaming and/or recording
import user_settings as conf


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
    log_name = os.path.expanduser("~/pybpod_plugins/water-calibration-plugin/DATA/water_calibration.csv")
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


def make_sounds_dict(start, stop, num, n_decimals):
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


def open_cam(n_cams=1, cam_sync=None, states_on_sync=False, mask=False):
    """
    This function is a wrapper for the code block opening the camera(s) and recording the video. Its main function is to
    avoid copying the same code across tasks.
    """

    # Check if video directory already exist, else create it
    video_folder = os.path.expanduser('~/Videos/' + conf.PYBPOD_SUBJECTS[0][2:5] + '/')
    if not os.path.exists(os.path.expanduser(video_folder)):
        os.makedirs(os.path.expanduser(video_folder))

    # Select automatically the cam depending on the PC (only for n_cams=1)
    username = os.getlogin()
    if username == 'setup0':  # Ephys PC
        indx_or_path = '/dev/video-cam01'  # Front cam
        # indx_or_path = '/dev/video-cam02'  #  Side cam
    elif username == 'setup1' or username == 'setup2':  # Boxes 1-8
        indx_or_path = '/dev/video-cam' + conf.PYBPOD_BOARD[-1]

    if n_cams == 1:  # For when a 1 cam is needed (all the test tasks, boxes 1-8)
        try:
            # Front camera
            cam = VideoR(indx_or_path=indx_or_path + '_front',
                         name_video=conf.PYBPOD_SESSION + '_front' + '.avi',
                         path=video_folder,
                         fps=60,
                         # codec_cam='MJPG',  # Commented for video compression
                         # codec_video='MJPG',  # Commented for video compression
                         title=conf.PYBPOD_BOARD + '_front',
                         cam_sync=cam_sync,
                         states_on_sync=states_on_sync,
                         mask=mask
                         )
            cam.play()
            if int(conf.VAR_REC) > 0:
                cam.record()
        except:
            print(
                "Could not open device. This may happen because either it's already in use or wrong device index number was provided")

    elif n_cams == 2:  # For when 2 cams are needed (ephys box only, 'cam_double' and 'stage_training' tasks)
        try:
            # Front camera
            cam1 = VideoR(indx_or_path=indx_or_path + '_front',
                          name_video=conf.PYBPOD_SESSION + '_front' + '.avi',  # Different file names to not overwrite each other
                          path=video_folder,
                          fps=60,
                          # codec_cam='MJPG',  # Commented for video compression
                          # codec_video='MJPG'  # Commented for video compression
                          title=conf.PYBPOD_BOARD + '_front',
                          cam_sync=cam_sync,
                          states_on_sync=states_on_sync,
                          mask=mask
                          )
            # Side camera
            cam2 = VideoR(indx_or_path=indx_or_path + '_side',
                          name_video=conf.PYBPOD_SESSION + '_front' + '.avi',  # Different file names to not overwrite each other
                          path=video_folder,
                          fps=60,
                          # codec_cam='MJPG',  # Commented for video compression
                          # codec_video='MJPG'  # Commented for video compression
                          title=conf.PYBPOD_BOARD + '_side',
                          cam_sync=cam_sync,
                          states_on_sync=states_on_sync,
                          mask=mask
                          )
            cam1.play()
            cam2.play()
            if int(conf.VAR_REC) > 0:
                cam1.record()
                cam2.record()
        except:
            print(
                "Could not open device. This may happen because either it's already in use or wrong device index number was provided")

    return cam


def close_cam(cam, n_cams=1):
    """
    This function is a wrapper for the code block closing the camera(s) and recording the video. Its main function is to
    avoid copying the same code across tasks.
    """

    if n_cams == 1:  # For when a 1 cam is needed (all the test tasks, boxes 1-8)
        cam.stop()

    elif n_cams == 2:  # For when 2 cams are needed (ephys box only, 'cam_double' and 'stage_training' tasks)
        cam1.stop()
        cam2.stop()
