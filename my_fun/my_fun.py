import numpy as np
from scipy.signal import firwin, lfilter  # Filters
import pandas as pd
import string
from matplotlib import pyplot as plt


# Function to generate white noise
def my_whiteNoiseGen():  # Adapted from UtilsR
    fs = 44100  # sampling frequency
    cutoff = [2000, 20000]  # low and high band edges. Must be positive and monotonically increasing between 0 and fs/2
    nyq = fs / 2  # Nyquist frequency (also found as fs * 0.5)
    normalized_cutoff = [cutoff[0] / nyq, cutoff[1] / nyq]  # Normalize by Nyquist frequency
    amp = 1  # amplitude
    duration = 1  # secs
    white_noise = amp * np.random.normal(0, 1, (fs * (duration + 1)))  # +1 to make the length double and then chop the
    # the first half
    fil_len = 10000
    band_pass = firwin(fil_len, normalized_cutoff, pass_zero=False)  # FIR filter with window method
    band_noise = lfilter(band_pass, 1, white_noise)  # Filter data with the FIR filter
    signal = band_noise[fs:int(fs * (duration + 1))]  # Indexing from fs to fs * 2, taking the second half of band_noise
    # (plot to understand)
    return signal


def envelope(coh, whitenoise, dur, nframes, samplingR=192000, variance=0.015, randomized=False, paired=True, LAmp=1.0,
             RAmp=1.0, oldbug=True, randgen=None):  # From UtilsR
    """
    coherences: coherences from 0(left only)to 1(right). ! var < coherences < (1-var). Else this wont work
    whitenoise: vec containing sound (not necessarily whitenoise)
    dur: total duration of the stimulus (secs)
    nframes: total frames in the whole stimulus
    samplingR: soundcard sampling rate (ie 96000). Need to match with EVERYTHING
    variance: fixed var
    randomized: shuffles noise vec
    paired: each instantaneous evidence is paired with its counterpart so their sum = 1
    randgen: np.random.RandomState instance to sample from
    returns: left noise vec, right noise vec, left coherences stairs, right coherences stairs [being them all 1d-arrays]

    From conver with Jordi: Hay varios argumentos que no tienen sentido (randomized, oldbug), siguen ahí por
    compatibilidad y que no peten las tareas del bpod, pero estarían borrados ya. El oldbug genera el doble de
    "envelopes" aunque la información cambia cada 2. El randomized corta y pega el vector de ruido para que no sea
    exactamente el mismo, lo que no es bueno (petardea) y se va a la     porra el filtro de tu rango de Hz. Siempre
    tienen que estar en False.

    Lout y Rout son los vectores de broadband noise  modulados por  'modwave' y la staircase (escalera) esto pesa mucho
    porque hay muchos puntos por segundo en las sesiones de bpod solo guardamos la escalera (20 valores) estos 20
    puntos/(sacados de la distribuci'on beta) son los stairs_envelope
    """
    if randgen is None:
        randgen = np.random
    totpoints = dur * samplingR  # should be an integer
    if len(whitenoise) < totpoints:
        raise ValueError('whitenoise is shorter than expected')

    # If True, this block basically takes whitenoise vector, reshape it in length/10 rows and 10 columns, shuffle it and then
    # flatten it again in a 1-D array (like it was at the beginning). While for noise this might not be relevant
    # (because the sound is sampled randomly from a distribution anyways), it would be necessary to make the function
    # generic to other sounds like pure tones
    if randomized == True:
        svec = whitenoise[:int(totpoints)]  # Yield the exact same variable than whitenoise (i.e. no change a all)
        svec = svec.reshape(int(len(svec) / 10), 10)
        randgen.shuffle(svec)
        svec = svec.flatten()
    else:
        svec = whitenoise[:int(totpoints)]
    modfreq = nframes / dur
    if oldbug:  # when freq was doubled, maintaining it because of compatibility issues. #envs = #stairs*2
        modwave = 1 * np.sin(2 * np.pi * (modfreq) * np.arange(0, dur, step=1 / samplingR) + np.pi)
    else:  # bug fixed, stairs paired with envelopes (#envs = #stairs)
        modwave = 0.5 * (
                np.sin(2 * np.pi * (modfreq) * np.arange(0, dur, step=1 / samplingR) - np.pi / 2) + 1)  # Ask Jordi

    if coh < 0 or coh > 1:
        raise ValueError(f'{coh} is an invalid coherences, it must fall w/i range 0 ~ 1')

    elif coh == 0 or coh == 1:
        staircaseR = np.repeat(coh, dur * samplingR)
        staircaseL = staircaseR - 1
        Lout = staircaseL * svec * modwave * LAmp
        Rout = staircaseR * svec * modwave * RAmp
        return Lout, Rout, np.repeat(coh - 1, nframes), np.repeat(coh, nframes)
    elif coh <= (variance * 1.1) or coh >= 1 - variance * 1.1:
        raise ValueError(
            'invalid coherences for given variance or viceversa (if coherences!=0|1, 1.1*var<coherences<1-var*1.1)')
    else:
        alpha = ((1 - coh) / variance - 1 / coh) * coh ** 2
        beta = alpha * (1 / coh - 1)
        stairs_envelopeR = randgen.beta(alpha, beta, size=nframes)
        staircaseR = np.repeat(stairs_envelopeR, int(totpoints / nframes))
        staircaseL = staircaseR - 1
        Rout = staircaseR * svec * modwave * RAmp
        if paired == False:
            stairs_envelopeL = randgen.beta(alpha, beta, size=nframes) - 1
            staircaseL = np.repeat(stairs_envelopeL, int(totpoints / nframes))
            Lout = staircaseL * svec * modwave * LAmp
            return Lout, Rout, stairs_envelopeL, stairs_envelopeR
        Lout = staircaseL * svec * modwave * LAmp
        return Lout, Rout, stairs_envelopeR - 1, stairs_envelopeR


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


def my_select_evidence(trial_type, evidences):  # Adapted from UtilsR
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


def enterthematrix(filepath):
    """Create sounds matrix"""
    # Import sounds DataFrame but only the filenames
    # filepath = '/home/alexis/PycharmProjects/create_sounds/sounds.csv'  # My laptop
    # filepath_setup2 = '/home/setup2/'  # setup2 pc
    df = pd.read_csv(filepath, usecols=['filename'])  # Alternatively usecols=[0]
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
    df['evidence'] = np.repeat(evidences, len(evidences) ** 2)  # Repeat each evidence per n sounds with that evidence
    df['coherence'] = np.repeat(coherences, len(coherences) ** 2)
    df['difficulty'] = np.repeat(difficulties, len(difficulties) ** 2)
    # df['stage'] = np.repeat(stages, len(stages) ** 2)
    df['substage'] = np.repeat(substages, len(substages) ** 2)
    return df


def sounds_dict(start, stop, num, decimals):
    """Dictionary letter: TTL pulses. Need to be in line with Arduino's code"""
    if num > 26:
        raise ValueError("'num' cannot be higher than abc's length (26)")
    chars = list(string.ascii_lowercase[:num])  # Make a list of all the lowercase letters as long as num
    pulses = np.around(np.linspace(start, stop, num), decimals)  # Make evenly spaced TTL pulses rounded to round2
    return dict(zip(chars, pulses))


def floatingpoints(x):
    float2str = str(x)
    str_len = len(float2str)
    if '.' in float2str:
        str_len = str_len - 2
    else:
        str_len = 0  # 0 better than None as Python's build in function 'round' accepts both but np's 'around' only 0
    return str_len


def evi2coh(x):
    return (x + 1) / 2


def coh2evi(x):
    return 2 * x - 1


# Under development

# 1 Convert to coherence
# 2 Convert to db
# 3 Compute difference
filepath = '/home/alexis/PycharmProjects/create_sounds/sounds.csv'  # My laptop
df = pd.read_csv(filepath)
# df = pd.read_csv(filepath).drop('filename', 1)
df_coh = evi2coh(df.loc[:, df.columns != 'filename'])  # Take all rows from all columns but 'filename'
df_db = power_dB(df_coh)

# Create DataFrame column labels
columns = ['ILD0', 'ILD1', 'ILD2', 'ILD3', 'ILD4', 'ILD5', 'ILD6', 'ILD7', 'ILD8', 'ILD9']
df_ild = pd.DataFrame(data=None, columns=columns)  # ILD =  # Interaural Level Difference. Difference between the volume
# (amplitude) of the sounds from both sides

# Think this si wrong. This is first calculating the amplitude diff and from there the ILD, but I think it must be the
# opposite (first calculate dB), then diff between them?
for i in range(len(df)):
    row = df.iloc[i].drop('filename')  # Take a row (series) without the filename (kkk)
    row = row.values  # From series to ndarray
    col = np.split(row, 2)  # Split by 2, but return list
    col = col[0] + col[1]  # Same as np.add(x[0], x[1]). As EL is negative and ER is positive, adding them = difference
    col = abs(col)
    df2_idl = pd.DataFrame([col], index=[i], columns=columns)
    df2_idl = power_dB(df2_idl)-error
    df_ild = df_ild.append(df2_idl)


def power_dB(amp):
    amp_ref = 0.00002  # The commonly used reference sound pressure in air is 20 µPa
    return 20 * np.log10(amp / amp_ref)


# Do the sexy plot
evidences = np.array([-1, -0.9, -0.8, -0.75, -0.6, -0.5, -0.4, -0.3, -0.25, -0.1,
                      0, 0.1, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 0.8, 0.9, 1])
coherences = evi2coh(evidences)

emp_left_dB = np.array([33, 43.3, 53.9, 56.9, 59.2, 62.2, 64.5, 64.9, 65.4, 66.4, 68.6, 69.8, 70.7, 70.5, 70.85, 70.35,
                    70.7, 71.0, 71.0, 71.0, 71.0])  # Registered values in dB recorded with micro from left speaker
# of box 8 with Rafa on March 3rd 2021
exp_right_dB = np.flip(emp_left_dB)

theor_left_dB = power_dB(coherences)
error = theor_left_dB - emp_left_dB
error = np.mean(error[1:])  # Exclude -Inf

# Plot left
x = np.linspace(0, 1, 1000)
y = power_dB(x) - error
plt.plot(x, y, 'g', label='theoretical left')
plt.plot(coherences, emp_left_dB, 'go', markerfacecolor='None', label='empirical left')

# Plot right
plt.plot(np.flip(x), y, 'm', label='theoretical right')
plt.plot(coherences, exp_right_dB, 'mo', markerfacecolor='None', label='expected right')

plt.xlabel('Amplitude')
plt.ylabel('dB')
plt.legend()
plt.title('SPL')
plt.savefig('SPL.png')
