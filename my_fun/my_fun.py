# Import libraries (need to reduce scipy's entries)
import numpy as np
from scipy.signal import firwin, lfilter  # For white_noise
from scipy.stats import beta  # Important! If using this, can't call any variable 'beta'!
import pandas as pd
from string import ascii_lowercase
# from matplotlib import pyplot as plt
# from sympy import symbols, Eq, log, nsolve  # Not installed in setup1 and setup2 PCs
# import slack
import os
import csv
import random
from matplotlib import pyplot as plt
from pathlib import Path

# For compute_psych_curve
from scipy import stats
from scipy.optimize import minimize
from collections import namedtuple


########################################################################################################################


def white_noise(fs=44100, cutoff=[2000, 20000], amp=1, dur=1, fn=10000,
                normalize=True):  # Adapted from UtilsR's 'whiteNoiseGen'
    """Create 'white noise' (between quotes as the signal is actually being band pass filtered).
    Note: if it takes too long try reducing the sampling rate or the filter length.
    :param fs: Sampling frequency
    :param cutoff: Low and high frequency band edges. Should be positive and monotonically increasing
    :param amp: Amplitude
    :param dur: Duration in seconds
    :param fn: Filter length
    :param normalized: If True normalize signal
    """
    mean = 0
    std = 1
    nyq = fs / 2  # Nyquist frequency (also found as fs * 0.5)
    normalized_cutoff = [cutoff[0] / nyq, cutoff[1] / nyq]  # Normalize by Nyquist frequency
    noise = amp * np.random.normal(mean, std, int(fs * dur * 2))  # * 2 as there is an artifact at the beginning of the
    # signal when applying the filter. So create double length and then later trim out the beginning
    band_pass = firwin(fn, normalized_cutoff, pass_zero=False)  # FIR filter with window method
    band_noise = lfilter(band_pass, 1, noise)  # Filter data with the FIR filter

    # Remove the first part as there is an artifact from the filter (resulting in 1 less frame than intended)
    # Plot to understand
    trim_length = int(fs * dur)
    band_noise = band_noise[-trim_length:]

    if normalize:
        band_noise = band_noise / np.max(abs(band_noise))
    return band_noise


def sine_wave(length=1, fs=44100, cycles=10, amp=1, phase=0, v_shift=0, plot=False):
    """Function that returns a sine wave (https://en.wikipedia.org/wiki/Sine_wave).
    :param length: In seconds
    :param fs: Sampling frequency
    :param cycles: Number of oscillations
    :param amp: Amplitude (peak deviation of the function from 0)
    :param phase: Phase (φ -phi-) or horizontal shift (where in its cycle the oscillation is at t = 0 in rad/s)
    :param v_shift: Vertical shift
    """

    x = np.arange(0, length, 1 / fs)  # Time vector of 'length' seconds and 'fs' points
    f = cycles / length  # Ordinary frequency: number of oscillations (cycles) that occur each second of time (Hz)
    ang_freq = 2 * np.pi * f  # Angular frequency (ω -omega-): rate of change of the function in rad/s
    y = amp * np.sin(ang_freq * x + phase) + v_shift  # Sine wave function

    if plot:
        plt.plot(x, y)
        plt.title('Sine wave')
        plt.xlabel('$\it{t}$ (s)')
        plt.xticks(np.linspace(0, length, cycles + 1))
        plt.ylabel('$\it{y}$ (t)')

    return x, y


def envelope(noise, coh, fs=44100, amp=1, dur=1, n_frames=10, var=0.015, paired=False):
    """
    Modulate a white noise sound with a sine wave and wrap it with an envelope according to stimulus coherence.
    :param noise: white noise vector
    :param coh: coherence [0=left, 1=right]
    :param fs: sampling frequency (needs to match the fs of white noise and sine wave)
    :param amp: amplitude
    :param dur: duration (in seconds)
    :param n_frames: number of frames
    :param var: variance of the beta distribution
    :param paired: if True, the sum of both sides = 1
    :return: sound left, sound right, stairs left (* n_frames), stairs_right (* n_frames)
    """

    # noise = white_noise(fs=fs, cutoff=[2000, 20000], amp=amp, dur=dur, fn=10000)
    n_points = dur * fs  # Should be an integer

    if len(noise) != n_points:
        raise ValueError('whitenoise and n_points need to  be the same length')

    x, mod_wave = sine_wave(length=dur, fs=fs, cycles=n_frames, amp=0.5, phase=-np.pi / 2, v_shift=0.5, plot=False)
    # amp = 0.5 so the length of y domain is 1 (-0.5, 0.5) instead of 2 (-1, 1)
    # phase = -np.pi/2 so the function starts at its minimum
    # v_shift = amp = 0.5 so the function y domain starts at 0 and is positive

    if coh < 0 or coh > 1:
        raise ValueError(f'{coh} is an invalid coherence, it must be within the range [0, 1]')

    elif coh == 0 or coh == 1:
        # If coh == 0, stairsR = 0 and stairsL = -1; elif coh == 1, stairsR == 1 and stairsL == 0
        envelope_R = np.repeat(coh, n_points)
        envelope_L = envelope_R - 1
        sound_R = envelope_R * noise * mod_wave * amp
        sound_L = envelope_L * noise * mod_wave * amp  # Change svec for white_noise
        stairs_R = np.repeat(coh, n_frames)
        stairs_L = np.repeat(coh - 1, n_frames)  # Change name to envelope
        return sound_L, sound_R, stairs_L, stairs_R

    # Don't understand this if block (fix it or remove)
    elif coh <= var * 1.1 or coh >= 1 - var * 1.1:  # Why this variance (0.015) and why 1.1??
        raise ValueError(
            'Invalid coherences for given variance or viceversa (if coherences!=0|1, 1.1*var<coherences<1-var*1.1)')

    else:
        # Resources to understand the beta distribution (the core of the envelope function):
        # https://en.wikipedia.org/wiki/Beta_distribution
        # https://stats.stackexchange.com/questions/12232/calculating-the-parameters-of-a-beta-distribution-using-the-mean-and-variance
        # http://varianceexplained.org/statistics/beta_distribution_and_baseball/
        # https://www.youtube.com/watch?v=juF3r12nM5A
        a, b = get_alpha_beta(coh, var)
        # a = ((1 - coh) / var - 1 / coh) * coh ** 2  # 2: Substituting solved beta in variance formula
        # b = a * (1 / coh - 1)  # 1: Solving beta in mean formula (given mean -coh- and variance are known)
        stairs_R = np.random.beta(a, b, size=n_frames)  # Draw samples from a Beta distribution
        stairs_L = stairs_R - 1  # This line is what makes it 'paired'
        # (stairs_envelopeR + abs(stairs_envelopeL) = 1 * n_frames
        envelope_R = np.repeat(stairs_R, int(n_points / n_frames))
        envelope_L = envelope_R - 1
        sound_R = envelope_R * noise * mod_wave * amp

        if not paired:
            stairs_L = np.random.beta(a, b, size=n_frames) - 1
            envelope_L = np.repeat(stairs_L, int(n_points / n_frames))  # When 'paired=False' it draws it from
            # the beta distro
            sound_L = envelope_L * noise * mod_wave * amp
            return sound_L, sound_R, stairs_L, stairs_R

        sound_L = envelope_L * noise * mod_wave * amp

        return sound_L, sound_R, stairs_L, stairs_R


def do_envelope_dB_normal(noise, dB_left, dB_right, max_vol, fs=44100, amp=1, dur=1, n_frames=10, sigma=1):
    """
    Modulate a white noise sound with a sine wave and wrap it with an envelope according to stimulus coherence
    :param noise: white noise vector
    :param dB_left: np array with left dB
    :param dB_right: np array with right dB
    :param max_vol: maximum volume (i.e. calibration value)
    :param fs: sampling frequency (needs to match the fs of white noise and sine wave)
    :param amp: amplitude
    :param dur: duration (in seconds)
    :param n_frames: number of frames
    :param sigma: standard deviation of the normal distribution
    :param paired: if True, the sum of both sides = 1
    :return: sound left, sound right, stairs left (* n_frames), stairs_right (* n_frames)
    """

    # noise = white_noise(fs=fs, cutoff=[2000, 20000], amp=amp, dur=dur, fn=10000)
    n_points = dur * fs  # Should be an integer

    if len(noise) != n_points:
        raise ValueError('whitenoise and n_points need to  be the same length')

    x, mod_wave = sine_wave(length=dur, fs=fs, cycles=n_frames, amp=0.5, phase=-np.pi / 2, v_shift=0.5, plot=False)
    # amp = 0.5 so the length of y domain is 1 (-0.5, 0.5) instead of 2 (-1, 1)
    # phase = -np.pi/2 so the function starts at its minimum
    # v_shift = amp = 0.5 so the function y domain starts at 0 and is positive

    if dB_left < 0 or dB_left > max_vol:
        raise ValueError(f'{dB_left} is an invalid coherence, it must be within the range [0, 1]')

    elif dB_left == 0 or dB_left == max_vol:

        coh_left = get_amp_from_dB(dB_left, max_vol)
        coh_right = get_amp_from_dB(dB_right, max_vol)

        # If coh == 0, stairsR = 0 and stairsL = -1; elif coh == 1, stairsR == 1 and stairsL == 0
        envelope_L = np.repeat(coh_left, n_points)
        envelope_R = np.repeat(coh_right, n_points)

        sound_L = envelope_L * noise * mod_wave * amp  # Change svec for white_noise
        sound_R = envelope_R * noise * mod_wave * amp
        stairs_L = np.repeat(dB_left, n_frames)  # Change name to envelope
        stairs_R = np.repeat(dB_right, n_frames)
        return sound_L, sound_R, stairs_L, stairs_R

    else:
        stairs_L = np.random.normal(dB_left, sigma, size=n_frames)  # Draw samples from a Beta distribution
        stairs_R = np.random.normal(dB_right, sigma, size=n_frames)  # Draw samples from a Beta distribution

        stairs_L_amp = get_amp_from_dB(stairs_L, max_vol)
        stairs_R_amp = get_amp_from_dB(stairs_R, max_vol)

        envelope_L = np.repeat(stairs_L_amp, int(n_points / n_frames))
        envelope_R = np.repeat(stairs_R_amp, int(n_points / n_frames))

        sound_L = envelope_L * noise * mod_wave * amp
        sound_R = envelope_R * noise * mod_wave * amp

        return sound_L, sound_R, stairs_L, stairs_R


def get_alpha_beta(mean, var, plot=False):
    """
    Get the alpha and beta parameters of a beta distribution given the mean and variance. Optional plot of the distro.
    """

    a = ((1 - mean) / var - 1 / mean) * mean ** 2  # 2: Substituting solved beta in variance formula
    b = a * (1 / mean - 1)  # 1: Solving beta in mean formula (given mean -coh- and variance are known)
    # Note: can't call it 'beta' as it would overwrite the scipy's 'beta' object imported

    # Both alpha and beta parameters must be real positive numbers (>0)
    if a <= 0:
        raise ValueError(f'alpha = {a}  <= 0')
    elif b <= 0:
        raise ValueError(f'beta = {b} <=0')
    else:
        if plot:  # Plot the beta distribution with parameters a, b
            mean, var, skew, kurt = beta.stats(a, b, moments='mvsk')
            x = np.linspace(beta.ppf(0.01, a, b), beta.ppf(0.99, a, b), 100)
            plt.plot(x, beta.pdf(x, a, b), 'r-', lw=5, alpha=0.6, label='beta pdf')
        return a, b


def get_beta_var_range(mean, num=1000, size=10):
    """
    Get the range of valid variances of a beta distribution for a given mean.
    :param mean: coherence [0, 1]. A coherence returns the same variance ranges regardless of its sign (-/+)
    :param num: number of samples of the variances vector
    :param size: number of samples to be drawn from the beta distribution
    :return: range of valid variances
    """

    vars = np.linspace(0.01, 1, num)  # Variance can't start at 0 (otherwise ZeroDivisionError)
    samples = []  # Initiate empty lists
    valid_vars = []
    a_list = []
    b_list = []

    for i in range(len(vars)):
        try:
            a, b = get_alpha_beta(mean, vars[i])
            samples.append(np.random.beta(a, b, size=size))
            valid_vars.append(vars[i])
            a_list.append(a)
            b_list.append(b)
        except ValueError:
            print(f'iteration {i}: alpha or beta <= 0')
    return samples, valid_vars, a_list, b_list


def getWaterCalib(board, ports):  # From UtilsR
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


def my_select_evidence_old(trial_type, evidences):  # Adapted from UtilsR
    """
    Reduce the prob of 0 evidence to 1/2 as it is part of both left and right trials. This function would be equivalent
    to repeat each evidence in the array except for 0
    trial_type: int, 0=left, 1=right
    evidences: np.array with all possible evidences
    returns: a randomly selected evidence from the available ones according to trial_type and withdrawn with equal prob
    """
    evidences = np.array(evidences)
    if trial_type == 0:
        available = evidences[evidences <= 0]  # evidences corresponding to the left
    else:
        available = evidences[evidences >= 0]  # evidences corresponding to the right
    if 0 not in evidences:  # just pick one randomly
        selected_evidence = np.random.choice(available)
    else:  # find it and set its prob of being taken by np.random.choice by 1/2 of the rest
        zero_loc = np.where(available == 0)[0][0]  # index of 0 in our vector available
        prob = 1 / len(evidences)  # prob of any particular evidence
        prob_vec = np.repeat(prob * 2, len(available))  # Make them all double
        prob_vec[zero_loc] = prob  # Set prob for evidence 0 to 1/2 of the rest so it appears with the same prob to
        # other evidences if added both reward sides
        selected_evidence = np.random.choice(available, p=prob_vec)
    return selected_evidence  # UtilsR one returns coherence


def my_select_evidence_new(trial_type, evidences, p=None):  # Adapted from UtilsR
    # def my_select_evidence(trial_type, evidences, shape='uniform'):  # Adapted from UtilsR
    """
    Reduce the prob of 0 evidence to 1/2 as it is part of both left and right trials. This function would be equivalent
    to repeat each evidence in the array except for 0
    trial_type: int, 0=left, 1=right
    evidences: np.array with all possible evidences
    returns: a randomly selected evidence from the available ones according to trial_type and withdrawn with equal prob
    NOTE, uniform works only without evidence 0
    """

    evidences = np.array(evidences)

    if trial_type == 0:
        available = evidences[evidences <= 0]  # Evidences corresponding to the left
        if p is not None:
            p = p
        # if shape == 'u-shape':
        # p = [((1 / 3) + (1 / 3 / 3)), (1 / 3), ((1 / 3) - (1 / 3 / 3))]
        # elif shape == 'uniform':
        # p = list(np.repeat(1 / len(available), len(available)))
    elif trial_type == 1:
        available = evidences[evidences >= 0]  # Evidences corresponding to the right
        if p is not None:
            # p.reverse()
            p = p[::-1]
        # if shape == 'u-shape':
        # p = [((1 / 3) - (1 / 3 / 3)), (1 / 3), ((1 / 3) + (1 / 3 / 3))]
        # elif shape == 'uniform':
        # p = list(np.repeat(1 / len(available), len(available)))

    if 0 not in evidences:  # just pick one randomly
        # selected_evidence = np.random.choice(available)
        selected_evidence = np.random.choice(available, p=p)
    else:  # find it and set its prob of being taken by np.random.choice by 1/2 of the rest
        zero_loc = np.where(available == 0)[0][0]  # index of 0 in our vector available
        prob = 1 / len(evidences)  # prob of any particular evidence
        prob_vec = np.repeat(prob * 2, len(available))  # Make them all double
        prob_vec[zero_loc] = prob  # Set prob for evidence 0 to 1/2 of the rest so it appears with the same prob to
        # other evidences if added both reward sides

        if p is not None:
            zero_loc = np.where(available == 0)[0][0]
            p_corr = p.copy()  # Very important to copy() the original list, otherwise (p_corr = p) both would be linked
            p_corr[zero_loc] = p_corr[zero_loc] / 2  # Make p0 the half of it
            rest = 1 - sum(p_corr)  # Get the remainder half of p0
            non_zero_loc = np.where(available != 0)[0]  # Find the indexes of non zero evidences in available vector
            for i in range(len(non_zero_loc)):
                p_corr[non_zero_loc[i]] = p_corr[non_zero_loc[i]] + rest / len(
                    non_zero_loc)  # Sum the other half of p0 to the p of the non zero
                # evidences (so the whole sums 1)
            selected_evidence = np.random.choice(available, p=p_corr)
        else:
            selected_evidence = np.random.choice(available, p=prob_vec)

    return selected_evidence  # UtilsR one returns coherence


def select_ilds(ilds, p, side):
    """
    :param ilds: array with the ilds to select from
    :param p: probability of difficult ilds (non-maximum evidence). Between 0 and 1
    :param side: Side from where to select the ilds from. 0=left, 1=right
    :return: randomly selected ild from ilds for a given side
    """
    # r = random.random()  # Generate random float between 0 and 1. PyBpod missing random library
    r = np.random.random(1)[0]  # Return random floats in the half-open interval [0.0, 1.0)
    if r > p:  # Easiest ilds (maximum evidence)
        if side == 0:  # Left
            return ilds.min()  # Sound left only
        else:  # Right
            return ilds.max()  # Sound right only
    else:  # Rest of the ilds (non maximum evidence)
        if side == 0:  # Left
            options = ilds[ilds <= 0]  # Left ILDs
            options = np.repeat(options, 2)  # Repeat each element of the vector
            options = options[:-1]  # Exclude one of the 0s from the vector, so it has 1/2 p than the rest
        else:  # Right
            options = ilds[ilds >= 0]  # Right ILDs
            options = np.repeat(options, 2)  # Repeat each element of the vector
            options = options[1:]  # Exclude one of the 0s from the vector, so it has 1/2 p than the rest
        selected_ild = np.random.choice(options)  # Generates a random sample from a given 1-D array
    return selected_ild


def enterthematrix(filepath):
    """Create sounds matrix"""
    # Import sounds DataFrame but only the behavior_filenames
    # filepath = '/home/alexis/PycharmProjects/create_sounds/sounds.csv'  # My laptop
    # filepath_setup2 = '/home/setup2/'  # setup2 pc
    # behavior_filenames = pd.read_csv(filepath, usecols=['filename'])  # Alternatively usecols=[0], to import only 'filename' column
    df = pd.read_csv(filepath)  # Import all columns
    # Create relevant arrays to build DataFrame columns
    evidences = np.array([-1, -0.9, -0.8, -0.75, -0.6, -0.5, -0.4, -0.3, -0.25, -0.1,
                          0, 0.1, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 0.8, 0.9, 1])
    coherences = (evidences + 1) / 2
    difficulties = np.array(['zero', 'ez', 'ez', 'ez', 'mid', 'mid', 'mid', 'hard', 'hard', 'hard',
                             'hero', 'hard', 'hard', 'hard', 'mid', 'mid', 'mid', 'ez', 'ez', 'ez', 'zero'])
    # stages = np.array([0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 3, 3, 3, 2, 2, 2, 1, 1, 1, 0])
    # substages = np.array([0, 1, 2, 3, 1, 2, 3, 1, 2, 3, 0, 3, 2, 1, 3, 2, 1, 3, 2, 1, 0])
    substages = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
    # Create new DataFrame columns and fill them with the arrays created above
    df.insert(1, 'evidence',
              np.repeat(evidences, len(evidences) ** 2))  # Repeat each evidence per n sounds with that evidence
    df.insert(2, 'coherence', np.repeat(coherences, len(coherences) ** 2))
    df.insert(3, 'difficulty', np.repeat(difficulties, len(difficulties) ** 2))
    # behavior_filenames.insert(4, 'stage', np.repeat(stages, len(stages) ** 2))
    df.insert(4, 'substage', np.repeat(substages, len(substages) ** 2))

    # behavior_filenames['evidence'] = np.repeat(evidences, len(evidences) ** 2)  # Repeat each evidence per n sounds with that evidence
    # behavior_filenames['coherence'] = np.repeat(coherences, len(coherences) ** 2)
    # behavior_filenames['difficulty'] = np.repeat(difficulties, len(difficulties) ** 2)
    # # behavior_filenames['stage'] = np.repeat(stages, len(stages) ** 2)
    # behavior_filenames['substage'] = np.repeat(substages, len(substages) ** 2)
    return df


def do_sounds_dict(start, stop, num, n_decimals):
    """Dictionary letter: TTL pulses. Need to be in line with Arduino's code"""
    if num > 26:
        raise ValueError("'num' cannot be higher than abc's length (26)")
    chars = list(ascii_lowercase[:num])  # Make a list of all the lowercase letters as long as num
    pulses = np.around(np.linspace(start, stop, num), n_decimals)  # Make evenly spaced TTL pulses rounded to round2
    return dict(zip(chars, pulses))


def floating_points(x):
    float2str = str(x)
    str_len = len(float2str)
    if '.' in float2str:
        str_len = str_len - 2
    else:
        str_len = 0  # 0 better than None as Python's build in function 'round' accepts both but np's 'around' only 0
    return str_len


def evi2coh(evi):
    """Transform evidence (-1=left, 1=right) to coherence (0=left, 1=right)"""
    coh = (evi + 1) / 2
    return coh


def coh2evi(coh):
    """Transform coherence (0=left, 1=right) into evidence (-1=left, 1=right)"""
    evi = 2 * coh - 1
    return evi


def find_power_dB_par(amp_ref=0.00002, dB_cal=73, ambient_noise=33):
    """Find the parameters to input the amplitude to dB transformation function so it returns values matching reality
    (calibration value and ambient noise in dB)"""

    # Define equations symbols
    x, y = symbols('x y')

    # amp_ref = 0.00002  # The commonly used reference sound pressure in air is 20 µPa
    # dB_cal = 73  # Calibration value of the speakers in dB
    # ambient_noise = 33  # Ambient noise in the behavioral box measured with the microphone

    # Define system of nonlinear equations
    eq1 = Eq(x * log((1 + y) / amp_ref, 10) - dB_cal, 0)  # --> x * np.log10((1 + y) / 0.00002) = 73
    eq2 = Eq(x * log((0 + y) / amp_ref, 10) - ambient_noise, 0)  # --> x * np.log10((0 + y) / 0.00002) = 33

    # Solve equations numerically
    sol = np.array(nsolve((eq1, eq2), (x, y), (20, 0.01))).astype(float)

    x = float(sol[0])
    y = float(sol[1])

    return x, y


def power_dB(amp):
    """Transform amplitude into decibels (dB)"""
    amp_ref = 0.00002  # The commonly used reference sound pressure in air is 20 µPa
    dB = 20 * np.log10(amp / amp_ref)
    # x, y = find_power_dB_par()
    # dB = 15.535 * np.log10((amp + 0.00267) / amp_ref)
    # dB = x * np.log10((amp + y) / amp_ref)
    return dB


def find_dB_evi0(max_vol):
    """Find dB for evidence 0 (how much volume when evidence is 0?)"""
    amp_ref = 0.00002  # The commonly used reference sound pressure in air is 20 µPa
    dB = 20 * np.log10(10 ** (max_vol / 20) / 2)
    return dB


########################################################################################################################

# Set of functions with Rafa on December 14th 2021 to create sounds sampling in the dB space


def find_constant(max_vol):
    """Find the constant that produces max_vol dB when amplitude is maximum (1). Depends on speaker, amplifier,
    calibation, etc"""
    constant = (1 / 10 ** (max_vol / 20))
    return constant


def get_dB_from_amp(amp, max_vol):
    """Transform amplitude into decibels (dB). A reduction of amplitude in half = -6dB
    https://stackoverflow.com/questions/6571894/calculate-decibel-from-amplitude-android-media-recorder
    Minimum amp is 0.001 = 10dB; 0.0001 = -10dB"""
    if amp < 0.001:
        dB = 0
        return dB
    if amp > 1:
        amp = 1
    constant = find_constant(max_vol)
    dB = 20 * np.log10(amp / constant)
    if dB < 0:
        dB = 0
    return dB


def get_amp_from_dB(dB, max_vol):
    """Transform amplitude into decibels (dB). A reduction of amplitude in half = -6dB"""
    constant = find_constant(max_vol)
    amp = constant * (10 ** (dB / 20))
    # if amp < 0.001:
    #     return 0.001
    return amp


def get_complementary_amp(amp):
    complementary_amp = 1 - amp
    return complementary_amp


def get_complementary_dB(dB, max_vol):
    amp = get_amp_from_dB(dB, max_vol)
    complementary_amp = get_complementary_amp(amp)
    complementary_dB = get_dB_from_amp(complementary_amp, max_vol)
    return complementary_dB


def get_diff_amp(amp):
    complementary_amp = get_complementary_amp(amp)
    diff = amp - complementary_amp
    return diff


def get_diff_dB(dB, max_vol):
    complementary_dB = get_complementary_dB(dB, max_vol)
    diff = dB - complementary_dB
    return diff


def get_amps_from_diff(diff):
    val1 = 0.5 - diff / 2
    val2 = 0.5 + diff / 2
    return val1, val2


def get_dBs_from_diff(diff, max_vol):
    """Find the parameters to input the amplitude to dB transformation function so it returns values matching reality
    (calibration value and ambient noise in dB)"""

    # Define equations symbols
    x, y = symbols('x y')

    constant = find_constant(max_vol)

    # Define system of nonlinear equations
    eq1 = Eq(constant * 10 ** (x / 20) + constant * 10 ** (y / 20) - 1, 0)
    eq2 = Eq(x - y - diff, 0)

    # Solve equations numerically
    sol = np.array(nsolve((eq1, eq2), (x, y), (40, 40))).astype(float)

    x = float(sol[0])
    y = float(sol[1])

    return x, y


def get_dBs_and_amps_from_diff(diff, max_vol):
    x, y = get_dBs_from_diff(diff, max_vol)
    x2 = get_amp_from_dB(x, max_vol)
    y2 = get_amp_from_dB(y, max_vol)
    return x, y, x2, y2


########################################################################################################################

def ild():
    """Get the inter aural level difference (ild) of a sound given its evidence (-1=left, 1=right).
    The input should be a csv file to convert to DataFrame. Only for sounds.csv (batch 1)
    """
    path = '/home/alexis/PycharmProjects/create_sounds/sounds.csv'  # My laptop
    df = pd.read_csv(path)
    # behavior_filenames = pd.read_csv(path).drop('filename', 1)  # Import csv as DataFrame dropping the column 'filename'
    df_dB = df  # Copy DataFrame
    df_dB.iloc[:, 1:21] = power_dB(abs(df.iloc[:, 1:21]))  # Apply the function to entire DataFrame except 'filename'
    # column. abs because can't do log10 of negative number. To retrieve the negative sign for left the ILD will be
    # computed as right - left later
    df_dB_left = df_dB.iloc[:, 1:11]  # Index left skipping 'filename'
    df_dB_left.columns = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']  # DataFrames needs to have BOTH the same
    # row and column indices in order to perform an element-wise subtraction
    df_dB_right = df_dB.iloc[:, 11:21]  # Index right
    df_dB_right.columns = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    df_ild = df_dB_right - df_dB_left  # Interaural level difference. Right minus left so we get negative for left and
    # positive for right
    df_ild.columns = ['ILD0', 'ILD1', 'ILD2', 'ILD3', 'ILD4', 'ILD5', 'ILD6', 'ILD7', 'ILD8',
                      'ILD9']  # Change column labels
    df_ild['Mean'] = df_ild.mean(axis=1)  # Add mean ILD per sound at the end of the DataFrame
    df_ild.insert(0, 'Filename', df.filename)  # Insert in 'filename' i the first column
    evidences = np.array([-1, -0.9, -0.8, -0.75, -0.6, -0.5, -0.4, -0.3, -0.25, -0.1,
                          0, 0.1, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 0.8, 0.9, 1])  # Define evidences
    df_ild.insert(1, 'Evidence',
                  np.repeat(evidences, len(evidences) ** 2))  # Repeat each evidence per n sounds with that evidence
    # df_ild_summary = behavior_filenames.groupby('Evidence', as_index=False).mean()  # SQL-style index
    df_ild_summary = df_ild.groupby('Evidence').mean()  # Group labels asn index
    return df_ild


def get_ild(n_frames):
    # Load sounds
    sounds_path = Path.home() / 'PycharmProjects' / 'create_sounds' / 'sounds_2.csv'
    sounds = pd.read_csv(sounds_path)
    # n_frames = 10

    # Left frames
    left_frames_column_names = [f'EL{n:01}' for n in range(n_frames)]
    frames_left = sounds[left_frames_column_names].values

    # Right frames
    right_frames_column_names = [f'ER{n:01}' for n in range(n_frames)]
    frames_right = sounds[right_frames_column_names].values

    # Frames ILD (elementwise substraction)
    frames_ild = frames_right - frames_left
    frames_ild = pd.DataFrame(frames_ild)
    frames_ild.insert(0, column='filename', value=sounds.filename)  # Insert behavior_filenames in first column

    return frames_ild


def compute_window(data, runningwindow):
    """
    Computes a rolling average with a length of runningwindow samples.
    """
    performance = []
    for i in range(len(data)):
        if i < runningwindow:
            performance.append(round(np.mean(data[0:i + 1]), 2))
        else:
            performance.append(round(np.mean(data[i - runningwindow:i]), 2))
    return performance


def compute_psych_curve(x, y, n_points=100):
    """Computes a psychometric function."""

    # https://psychology.stackexchange.com/questions/13347/how-can-i-fit-a-psychometric-function-such-that-the-minimum-is-50-chance-level

    def sigmoid_mme(fit_params: tuple):
        k, x0, b, p = fit_params

        # k = weight (slope)
        # x0 = bias
        # b, p = lapses

        # Function to fit:
        y_pred = b + (1 - b - p) / (1 + np.exp(-k * (xdata - x0)))

        # Calculate negative log likelihood:
        ll = - np.sum(stats.norm.logpdf(ydata, loc=y_pred))

        return ll

    coherence_dataframe = pd.DataFrame({'r_resp': y, 'evidence': x})

    info = coherence_dataframe.groupby(['evidence'])['r_resp'].mean()
    ydata = [np.around(elem, 3) for elem in info.values]
    xdata = info.index.values
    fit_error = [np.around(elem, 3) for elem in coherence_dataframe.groupby(['evidence'])['r_resp'].sem().values]

    initial_guess = np.array([1, 1, 0, 0])

    # Run the minimizer:
    ll = minimize(sigmoid_mme, initial_guess)

    # Fit parameters:
    k, x0, b, p = [np.around(param, 2) for param in ll['x']]

    # Compute the fit with n_points number of points:
    fit = b + (1 - b - p) / (1 + np.exp(-k * (np.linspace(np.min(x), np.max(x), n_points) - x0)))
    fit = [np.around(elem, 3) for elem in fit]

    psych_curve = namedtuple('psych_curve',
                             ['xdata',
                              'ydata',
                              'fit',
                              'params',
                              'fit_error'])

    if len(ydata) == 0:
        return psych_curve(xdata=[np.nan],
                           ydata=[np.nan],
                           fit=[np.nan] * n_points,
                           params=[np.nan] * 4,
                           fit_error=[np.nan])
    else:
        return psych_curve(xdata=xdata,
                           ydata=ydata,
                           fit=fit,
                           params=[k, x0, b, p],
                           fit_error=fit_error)


def pc_lapses0(x, k, x0):
    """
    Psychometric function when lapses = 0
    :param x: value to predict
    :param k: slope
    :param x0: bias
    :return: value of y for the input value of x
    """
    return 1 / (1 + np.exp(-k * (x - x0)))


def pc_lapses0_x0(k, x0):
    """
    Psychometric function when lapses = 0 and x = 0 (special case of the above)
    :param k: slope
    :param x0: bias
    :return: value of y for x = 0
    """
    return 1 / (1 + np.exp(k * x0))


def pc_bias0(x, b, p, k):
    """
    Psychometric function when bias = 0
    :param x: value to predict
    :param b: lower lapse
    :param p: upper lapse
    :param k: slope
    :return: value of y for the input value of x
    """
    return b + (1 - b - p) / (1 + np.exp(-k * x))


def pc_bias0_x0(b, p):
    """
    Psychometric function when bias = 0 and x = 0 (special case of the above)
    :param b: lower lapse
    :param p: upper lapse
    :return: value of y for x = 0
    """
    return b + (1 - b - p) / 2


def slack_spam(msg='Hey buddy!', filepath=None, userid='U01DDHH7LLX'):  # Adapted from UtilsR (Jordi's)
    """This sends msgs through the bot. Avoid spamming too much else it will get banned or timed-out. Atm not possible
    to update several files at the same time (https://github.com/slackapi/python-slack-sdk/issues/442)
    """

    ids_dic = {
        'alexis': 'U01DDHH7LLX',
        'jaime': 'U7UTKNN0P',
        'carles': 'UPZPM32UC',
        'my_channel': '#pv_nmdar_eranet',
        'reports': '#pv_nmdar_eranet_reports'
    }

    if (userid[0] != 'U') and (userid[0] != '#'):  # Assumes it is a first name
        try:
            userid = ids_dic[userid.lower()]
        except:
            raise ValueError('Double-check slack channel ID (receiver)')

    token = os.environ.get('SLACK_BOT_TOKEN')

    if token is None:
        print('No SLACK_BOT_TOKEN in environ')
        raise EnvironmentError('no SLACK_BOT_TOKEN in environ')
    else:
        try:
            client = slack.WebClient(token=token)
            if filepath is None:
                response = client.chat_postMessage(
                    channel=userid,
                    text=msg)
            elif os.path.exists(filepath):
                response = client.files_upload(
                    channels=userid,
                    file=filepath,
                    initial_comment=msg)
            else:
                print(f"filepath '{filepath}' doesn't exist")
        except Exception as e:
            print(e)  # Perhaps prints are caught by pybpod


def check_date_exist(date, dates):
    """
    Check if a string date exist in a list or Series of string dates
    :param date: date as string
    :param dates: dates as list or pd.Series of strings
    :return:
    """
    date = str(date)  # Ensure date is in string format
    if type(dates) is list:  # Check if iterable of dates is a list
        # print('Is list')
        if date in dates:
            print(f'Date {date} exists')
            return True
        else:
            print(f'Date {date} doesnt exist')
            return False
    elif type(dates) is pd.core.series.Series:  # Check if iterable of dates is a pandas Series
        # print('Is pandas Series')
        if dates.str.contains(date).any():
            print(f'Date {date} exists')
            return True
        else:
            print(f'Date {date} doesnt exist')
            return False


# The following 2 functions are under testing were developed for kernels. Will need to adapt to make them work for
# other cases
def get_experiment(experiment=None, session='glue_sessions'):
    """
    Get experiment
    :param experiment: If not None, experiment=experiment. Else, show possible experiments and ask for user input.
    :param session: if glue_sessions look for individual sessions, elif intersession look for intersessions
    :return: experiment
    """

    if experiment is None:

        # folder_in = '/home/alexis/PycharmProjects/glue_sessions/'  # Where the data for all animals is
        folder = Path.home() / 'PycharmProjects' / session  # Where the data for all animals is
        experiments = os.listdir(folder)  # List experiments
        experiments.sort()  # Sort them by name
        # experiments = [x for x in experiments if os.path.isdir(folder_in + x)]  # Get rid of non folders
        experiments = [x for x in experiments if Path(folder / x).is_dir()]  # Get rid of non folders

        try:
            experiments.remove('__pycache__')  # Pycharm's file
        except ValueError:
            pass

        print('Experiments: ' + str(experiments)[1:-1])  # Remove square brackets
        experiment = input('Enter experiment name')
    else:
        folder = Path.home() / 'PycharmProjects' / session / experiment

    return experiment, folder


def get_animal(experiment, session='glue_sessions', animal=None):
    """
    Get animal
    :param experiment: If not None, experiment=experiment. Else, show possible experiments and ask for user input.
    :param session: if glue_sessions look for individual sessions, elif intersession look for intersessions
    :param animal: If not None, animal=animal. Else, show possible animals and ask for user input.
    :return: animal
    """

    if experiment is None:
        experiment = get_experiment(experiment, session)

    folder_in = Path.home() / 'PycharmProjects' / session / experiment

    if animal is None:
        animals = os.listdir(folder_in)  # List animals
        animals.sort()  # Sort them by name
        animals = [x[:-4] for x in animals]  # Get rid of .csv extension
        animals = [i for i in animals if '_corrupted_sessions' not in i]  # Remove '_corrupted_sessions'.csv files

        print('Animals: ' + str(animals))  # Remove square brackets
        animal = input('Enter animal')  # Ask user to input animal to glue sessions from

    return animal


def save_fig(folder_out, filename):
    """
    Save figure twice, one in png with white background and another one in svg with transparent background.
    :param folder_out: Folder where to save the figures
    :param filename: Name of the figure
    :return:
    """
    if not folder_out.exists():
        folder_out.mkdir(parents=True, exist_ok=True)
    os.chdir(folder_out)
    plt.savefig(Path(folder_out / (filename + '.' + 'png')), format='png', transparent=False)
    plt.savefig(Path(folder_out / (filename + '.' + 'svg')), format='svg', transparent=True)
