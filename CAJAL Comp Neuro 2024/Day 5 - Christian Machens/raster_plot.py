from scipy.io import loadmat
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

data = loadmat(r'C:\Users\alexi\PycharmProjects\CAJAL Comp Neuro 2024\Day 5 - Christian Machens\R14001_001.mat', squeeze_me=True)['result']  # loadmat is a function in scipy.io
# result = data['result'].esqueeze()  # esqueeze is a method of the result object

df = pd.DataFrame(data, columns=[data[0, :]])
spikes = df.spikes[1:]

for j in range(len(spikes)):  # Trials
    n_spikes = len(spikes.iloc[j][0][0])
    spikes_per_electrode = spikes.iloc[j][0][0]
    for i in range(n_spikes):
        plt.plot(spikes_per_electrode[i], j, marker='o', color='k')

