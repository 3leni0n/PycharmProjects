"""
Short programming assessment

In order to make sure you have the minimal coding skills required to follow the pace of tutorial sessions, please
complete the following programming assessment, which is about plotting a psychometric curve. Read the instructions
first and then complete all the steps, either in Matlab or Python. It is important that you measure how long it takes
you to perform all the steps. You should be able to complete it under 15 minutes. Please do it in real conditions.
If you needed more than 15 minutes, then we advise you not to apply this time to the summer school, as you would have
difficulties following the pace of tutorial sessions.

1- Download the data (either in .mat or .csv format) at https://bit.ly/bhvdata
2- Open the data in Matlab or Python. The data corresponds to behavioral variables from a subject performing a
two-alternative forced-choice discrimination task. There are two vectors in the data: 'stimulus_evidence' describes the
strength of evidence of each stimulus in favour of of either response, from -1 (very clear stimulus associated with
lefward response) to +1 (very clear stimulus associated with rightward response); 'response' is the response of the
participant (0 for leftward response, 1 for rightward response).
3- Plot the psychometric curve. This is the proportion of rightward responses as a function of stimulus evidence.
Provide labels for the axes. (You do not need to fit the psychometric curve, just plot the proportion of rightward
responses for each value of stimulus evidence)
4- Copy your code below.
"""

# Import modules
import pandas as pd
from scipy.optimize import curve_fit
import numpy as np
from matplotlib import pyplot as plt

# Import data
df = pd.read_csv(r'C:\Users\alexi\Downloads\behavioral_data\behavioral_data.csv')  # r to evaluate string as 'raw' and
# scape backslashes (\)

# Plot mean and sem responses per stimulus evidence
ydata = df.groupby(['stimulus_evidence'])['response'].mean()  # Mean response per stimulus evidence
error_data = df.groupby(['stimulus_evidence'])['response'].sem().values  # sem response per stimulus evidence
xdata = ydata.index.values  # Stimulus evidences
plt.errorbar(xdata, ydata, error_data, color='tab:blue', fmt='o', mfc='none', label='mean ± sem')  # Plot raw data


# Define sigmoid function
def sigmoid(slope, bias, lapse1, lapse2):
    y = lapse1 + (1 - lapse1 - lapse2) / (1 + np.exp(slope * (xdata - bias)))
    return y


# Fit sigmoid function
popt, pcov = curve_fit(sigmoid, xdata, ydata)

# Plot psychometric curve fit
x = np.linspace(-1, 1, len(xdata))
y = sigmoid(x, *popt)
plt.plot(x, y, color='tab:blue', label='sigmoid fit')
plt.title('Psychometric curve')
plt.xlabel('Stimulus evidence')
plt.ylabel('Probability choose right')
plt.legend()

# Plot horizontal and vertical lines
plt.axhline(0.5, color='tab:gray', ls='--')
plt.axvline(0, color='tab:gray', ls='--')