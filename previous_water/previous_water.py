from pathlib import Path
import os
import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
matplotlib.use('Qt5Agg')
import statsmodels.api as sm


path = r'C:\Users\Usuario\PycharmProjects\intersession\2AFC_5\0_intersession.csv'

# Make variable with all the files in the parent folder of current path
parent_folder = Path.home() / 'PycahrmProjects' / 'intersession' / '2AFC_5'

# Load data
df = pd.read_csv(path)

# Add a new column to df that is the water drank the previous session
df['PreviousWater'] = df.Water.shift(1)

# Pearson correlation coefficient between these 2 variables
correlation = df['PreviousWater'].corr(df['Accuracy'], method='pearson')

# Plot with pandas previous water vs accuracy
df.plot(x='PreviousWater', y='Accuracy', xlim=(0, 2000), ylim=(0.5, 1), style='o', xlabel='Previous Water', ylabel='Accuracy',
        title=f'Subject {df.Subject.unique()[0]}, Pearson corr. coef.: {round(correlation, 2)}', legend=False)

# Fit an OLS model
endog = df['Accuracy'][1:]  # Skip the first row because it has a nan value
exog = df['PreviousWater'][1:]  # Skip the first row because it has a nan value
exog = sm.add_constant(exog)
mod = sm.OLS(endog, exog)
res = mod.fit()
print(res.summary())

# Plot model prediction
plt.plot(exog.iloc[:, 1], res.predict(exog), label='OLS')  # iloc[:, 1] to get the second column of the exog matrix
# (excluding the constant)


