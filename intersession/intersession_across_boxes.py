import pandas as pd
from matplotlib import pyplot as plt


df = pd.read_csv('/home/alexis/PycharmProjects/intersession/2AFC_3/all_intersessions.csv')

plt.plot(df.Dates, df.Accuracy)