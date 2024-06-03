from pathlib import Path
import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

matplotlib.use('Qt5Agg')
import statsmodels.api as sm


# Define functions
def get_bl_weights(path: Path) -> pd.DataFrame:
    """
    Load the baseline weights data from a csv file and return a pandas DataFrame with the baseline weights normalized
    :param path: Path to the csv file with the baseline weights data
    :return: pandas DataFrame with the baseline weights normalized
    """
    df = pd.read_csv(path)
    dates = df.Date

    # Define function to transform date format from Google Sheets to PyBpod's format
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
            try:
                mouse_baseline_weights_per_date.append(
                    float(df[(df['Mouse'] == mouse) & (df['Date'] == date)]['Relative weight (%)'].values[0]))
            except ValueError:  # If the value is not a number, append a nan
                mouse_baseline_weights_per_date.append(np.nan)
        baseline_weights.append(mouse_baseline_weights_per_date)

    # Create a pandas DataFrame with the baseline weights
    baseline_weights = pd.DataFrame(baseline_weights, columns=mice, index=dates)

    # Normalize the baseline weights (%) to be between 0 and 1 (same as Accuracy)
    baseline_weights = baseline_weights / 100

    return baseline_weights


def add_bs_weight_2_intersession(mouse: str, path: Path, baseline_weights: pd.DataFrame) -> pd.DataFrame:
    """
    Add the baseline weights to the intersession data of a mouse
    :param mouse: mouse number
    :param path: path to the intersession data
    :param baseline_weights: pandas DataFrame with the baseline weights
    :return: pandas DataFrame with the intersession data of the mouse with the baseline weights added
    """
    filename = mouse + '_intersession.csv'
    path = path / filename
    df_intersession = pd.read_csv(path)

    # Get the subject of the intersession data
    subject = df_intersession.Subject.unique()[0]

    # Get the baseline weights of the subject
    baseline_weights_subject = baseline_weights[subject]

    # Add a new column to df_intersession that is the baseline weight of the subject only on the dates that matches
    # intersession data
    df_intersession['BaselineWeight'] = baseline_weights_subject.loc[df_intersession.Dates].values

    # Drop the rows with nan values in the 'PreviousBaseWeight' column
    df_intersession.dropna(subset=['BaselineWeight'], inplace=True)

    return df_intersession


########################################################################################################################

# Load baseline weights data
path_bl_weights = Path.home() / 'Downloads' / 'Mice - Training batch 5.csv'
baseline_weights = get_bl_weights(path_bl_weights)

# Add baseline weights to intersession data
path_intersession = Path.home() / 'PycharmProjects' / 'intersession' / '2AFC_5'
df_intersession = add_bs_weight_2_intersession('0', path_intersession, baseline_weights)

# Add baseline weights to all intersessions:
df_intersessions_all = pd.DataFrame()
for mouse in range(10):
    df_intersession = add_bs_weight_2_intersession(str(mouse), path_intersession, baseline_weights)
    df_intersessions_all = pd.concat([df_intersessions_all, df_intersession])

# Pearson correlation coefficient between these 2 variables
correlation = df_intersession['BaselineWeight'].corr(df_intersession['Accuracy'], method='pearson')

# Plot accuracy vs baseline weight for all mice (raw data)
df_intersessions_all.plot(x='BaselineWeight', y='Accuracy', style='o', xlabel='Baseline weight', ylabel='Accuracy',
                            title=f'All mice (Pearson corr. coef.: {correlation})', legend=False)

# Find the min and max of baseline weights of all mice
min_bl_weight = df_intersessions_all['BaselineWeight'].min()
max_bl_weight = df_intersessions_all['BaselineWeight'].max()

# Make 10 equal bins between the min and max baseline weights
n_bins = 10
bins_step = 0.05
bins = np.linspace(min_bl_weight, max_bl_weight, n_bins + 1)


# Compute the average accuracy for n_bins of baseline weights
df_intersessions_all['BaselineWeightBins'] = pd.cut(df_intersessions_all['BaselineWeight'], bins=bins)


# # If the number of sessions (rows) in the bin is less than a threshold, drop the sessions of that bin
# threshold = 3
# for bin, group in df_intersessions_all.groupby('BaselineWeightBins'):
#     if len(group) < threshold:
#         # Print the bin that will be dropped
#         print(f'Dropping bin {bin}')
#         df_intersessions_all = df_intersessions_all[df_intersessions_all['BaselineWeightBins'] != bin]
#
# # Re-compute the BaselineWeightBin taking into account the bins that were dropped. Take the min and max of the
# # remaining bins and make n_bins equal bins between them
# min_bl_weight = df_intersessions_all['BaselineWeight'].min()
# max_bl_weight = df_intersessions_all['BaselineWeight'].max()
# bins = np.linspace(min_bl_weight, max_bl_weight, n_bins + 1)
# df_intersessions_all['BaselineWeightBins'] = pd.cut(df_intersessions_all['BaselineWeight'], bins=bins)
#

# Plot the average accuracy for each bin of baseline weight for all mice with error bars
df_intersessions_all.groupby(['BaselineWeightBins'])['Accuracy'].agg(['mean', 'sem']).plot(y='mean', yerr='sem', style='o',
    color='k', xlabel='Baseline weight', ylabel='Accuracy', title='All mice', legend=False)

# Place the xticks in the middle of the bins
plt.xticks(np.arange(n_bins), [f'{round(bins[i], 2)}-{round(bins[i + 1], 2)}' for i in range(n_bins)])

# Change the xticks labels to the mean of the bins
plt.xticks(np.arange(n_bins), [round((bins[i] + bins[i + 1]) / 2, 2) for i in range(n_bins)])


# Plot the average accuracy for each bin of baseline weight for all mice with different colors for each mouse
df_intersessions_all.groupby(['BaselineWeightBins', 'Subject'])['Accuracy'].mean().unstack().plot(style='o-',
    xlabel='Baseline weight', ylabel='Accuracy', title='All mice')














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
