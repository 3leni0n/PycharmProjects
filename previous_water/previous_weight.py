from pathlib import Path
import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('Qt5Agg')
import statsmodels.api as sm


# Load baseline weights data
path = Path.home() / 'Downloads' / 'Mice - Training batch 5.csv'
df = pd.read_csv(path)

dates = df.Date

# Function to transform date format from Google Sheets to PyBpod's format
def transform_date_format(dates):
    transformed_dates = []
    for date in dates:
        try:
            # Parse the date string into a datetime object
            date_obj = datetime.strptime(date, '%d/%m/%Y')
            # Format the datetime object into the desired format
            formatted_date = date_obj.strftime('%Y-%m-%d')
            transformed_dates.append(formatted_date)
        except ValueError as e:
            print(f"Error converting date {date}: {e}")
    return transformed_dates


# Transform the date strings
dates = transform_date_format(dates)

# Substitute the original dates in df ('Date' column) with the transformed ones
df['Date'] = dates

# Make a pandas DataFrame in which each row is one of the dates and each column is one of the mice baseline weights
mice = df.Mouse.unique()
dates = df.Date.unique()

baseline_weights = []

for date in dates:
    mouse_baseline_weights_per_date = []
    for mouse in mice:
        # mouse_df = df[(df['Mouse'] == mouse) & (df['Date'] == date)]
        # mouse_baseline_weights_per_date.append(mouse_df['Relative weight (%)'].iloc[0])
        try:
            mouse_baseline_weights_per_date.append(
                float(df[(df['Mouse'] == mouse) & (df['Date'] == date)]['Relative weight (%)'].values[0]))
        except ValueError:  # If the value is not a number, append a nan
            mouse_baseline_weights_per_date.append(np.nan)
    baseline_weights.append(mouse_baseline_weights_per_date)

baseline_weights = pd.DataFrame(baseline_weights, columns=mice, index=dates)

# Normalize the baseline weights (percentage) to be between 0 and 1 (same as Accuracy)
baseline_weights = baseline_weights / 100

########################################################################################################################

# Load intersession data
path_intersession = r'C:\Users\Usuario\PycharmProjects\intersession\2AFC_5\7_intersession.csv'
df_intersession = pd.read_csv(path_intersession)

# Get the subject of the intersession data
subject = df_intersession.Subject.unique()[0]

# Get the baseline weights of the subject
baseline_weights_subject = baseline_weights[subject]

# Add a new column to df_intersession that is the baseline weight of the subject only on the dates that matches
# intersession data
df_intersession['BaselineWeight'] = baseline_weights_subject.loc[df_intersession.Dates].values

# Drop the rows with nan values in the 'PreviousBaseWeight' column
df_intersession.dropna(subset=['BaselineWeight'], inplace=True)

########################################################################################################################

# Pearson correlation coefficient between these 2 variables
correlation = df_intersession['BaselineWeight'].corr(df_intersession['Accuracy'], method='pearson')

# Find the minimum of the baseline weights and accuracy
lim = min(df_intersession.BaselineWeight.min(), df_intersession.Accuracy.min())

# Plot with pandas previous water vs accuracy
df_intersession.plot(x='BaselineWeight', y='Accuracy', style='o',
                     xlabel='Baseline weight', ylabel='Accuracy',
                     title=f'Subject {df_intersession.Subject.unique()[0]}, Pearson corr. coef.: {round(correlation, 2)}',
                     legend=False)

# Fit an OLS model
endog = df_intersession['Accuracy']
exog = df_intersession['BaselineWeight']
exog = sm.add_constant(exog)
mod = sm.OLS(endog, exog)
res = mod.fit()
print(res.summary())

# Plot model prediction
plt.plot(exog.iloc[:, 1], res.predict(exog), label='OLS')  # iloc[:, 1] to get the second column of the exog matrix
# (excluding the constant)

plt.show()

