import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from parse.parse import parse
from glue_sessions.glue_sessions import glue_sessions

# Objectives:
# - Track accuracy (general first, then per side)
# - Track misses (general first, then per side)
# - Track errors (general first, then per side)
# - Track bias (general first, then per side)

# ACCURACY:
# Example with the last 2 sessions of 915:
df_915_1 = parse('/home/alexis/pv_nmdar_eranet/experiments/2AFC/setups/915/sessions/915_stage_training_20210728-155222/915_stage_training_20210728-155222.csv')
df_915_2 = parse('/home/alexis/pv_nmdar_eranet/experiments/2AFC/setups/915/sessions/915_stage_training_20210729-191156/915_stage_training_20210729-191156.csv')
df_915_1and2 = pd.concat([df_915_1, df_915_2])
df_grouped = df_915_1and2.groupby('Session')
df_grouped.Hit.sum()
df_grouped.Response.sum()

df_915_all_sessions = glue_sessions()
df_915_all_sessions_grouped = df_915_all_sessions.groupby('Session')
df_915_all_sessions_grouped.Hit.sum()
df_915_all_sessions_grouped.Response.sum()
plt.plot(df_915_all_sessions_grouped.Hit.sum() / df_915_all_sessions_grouped.Response.sum())  # y axis
df_915_all_sessions_grouped.ngroup().unique()  # x axis

# Accuracy in general intersession
plt.plot(df_915_all_sessions_grouped.ngroup().unique(), df_915_all_sessions_grouped.Hit.sum() / df_915_all_sessions_grouped.Response.sum(), marker='o')