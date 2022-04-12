import time
import pandas as pd
import datetime
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from my_fun.my_fun import compute_psych_curve, slack_spam

########################################################################################################################

# Register time
time_start_total = time.time()

df_332 = pd.read_csv('/home/alexis/PycharmProjects/intersession/332_intersession.csv')
df_333 = pd.read_csv('/home/alexis/PycharmProjects/intersession/333_intersession.csv')

dates_332 = df_332.Dates.tolist()
dates_333 = df_333.Dates.tolist()
dates_all = np.unique(dates_332 + dates_333).tolist()

df_332_format_date = pd.to_datetime(df_332.Dates)
df_333_format_date = pd.to_datetime(df_333.Dates)

plt.plot(df_332_format_date, df_332.Accuracy)
plt.plot(df_333_format_date, df_333.Accuracy)


plt.plot(df_332.Dates, df_332.Accuracy)
plt.plot(df_333.Dates, df_333.Accuracy)

dates_all.isin(dates_332)
dates_all.isin(dates_333)


date_exist = []

for i in range(len(dates_all)):
    date_exist.append(check_date_exist(dates_all[i], dates_332))






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


datetime.datetime.strptime('2022-01-17', "%Y-%m-%d")
