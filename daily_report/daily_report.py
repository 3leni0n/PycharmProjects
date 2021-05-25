# To do:
# Check using GridSpec instead of plt.subplot2grid as suggested by matplotlib doc
# (https://matplotlib.org/stable/gallery/userdemo/demo_gridspec01.html)
# Create and keep a different axis for every subplot (ax1, ax2, ax3, etc) instead of overwriting a single axis 'ax'
# Psychometric curves for repeating and alternating evidence
# Clean, dense to read!

########################################################################################################################

import matplotlib.pyplot as plt
from matplotlib.patches import Patch  # For custom legend
from matplotlib.lines import Line2D  # For custom legend
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import numpy as np
import pandas as pd
from my_fun.my_fun import *

########################################################################################################################

# SUMMARY VARIABLES
# Trials
trials = len(df_session)
trials_left = df_session.Side.value_counts()[0]
trials_right = df_session.Side.value_counts()[1]

# Hits
hits = df_session.Hit.sum().astype(int)
hits_left = df_session.Hit[df_session.Side == 0].sum().astype(int)
hits_right = df_session.Hit[df_session.Side == 1].sum().astype(int)

# Performance
performance = hits / trials
performance_left = hits_left / trials_left
performance_right = hits_right / trials_right

# Responses (valid trials)
responses = df_session.Response.sum()
responses_left = df_session.Response[df_session.Side == 0].sum()
responses_right = df_session.Response[df_session.Side == 1].sum()

# Accuracy (hit rate)
accuracy = hits / responses
accuracy_left = hits_left / responses_left
accuracy_right = hits_right / responses_right

# Misses (invalid trials)
misses = df_session.Miss.sum()
misses_left = df_session.Miss[df_session.Side == 0].sum()
misses_right = df_session.Miss[df_session.Side == 1].sum()

# Miss rate
miss_rate = misses / trials
miss_rate_left = misses_left / trials_left
miss_rate_right = misses_right / trials_right

# Reward
rewards = df_session.Reward.sum()
rewards_left = df_session.Reward[df_session.Side == 0].sum()
rewards_right = df_session.Reward[df_session.Side == 1].sum()

# Water
reward_size = 2.5  # μL
water = rewards * reward_size
water_left = rewards_left * reward_size
water_right = rewards_right * reward_size

########################################################################################################################

fig = plt.figure(figsize=(8.27, 11.69))  # A4 size in inches portrait
# fig = plt.figure(figsize=(11.69, 8.27))  # A4 size in inches landscape

########################################################################################################################

# SUMMARY TEXT

s1 = ('Date: ' + df_session.Date.unique()[0] + ', ' +
      'Time: ' + df_session.SessionStart.unique()[0][0:-7] + ' - ' + df_session.SessionEnd.unique()[0][0:-7] + '\n')
# [0:-7] to get rid of the floating numbers in the seconds

s2 = ('Subject: ' + df_session.Subject.unique()[0] + ', ' +
      'Box: ' + df_session.Board.unique()[0][4] + ', ' +
      'Stage: ' + str(df_session.Stage.unique()[0]) + ', ' +
      'Substage: ' + str(df_session.Substage.unique()[0]) + ', ' +
      'Fixation: ' + str(df_session.Fixation.unique()[0]) + ', ' +
      'Timeout: ' + str(df_session.Timeout.unique()[0]) + ', ' +
      # 'Switch: ' + tr(df_session.Switch.unique()[0]) + ', ' +
      'Motor: ' + str(df_session.Motor.unique()[0]) + '\n')

s3 = ('Total trials: ' + str(trials) + ', ' +
      'Performance: ' + str(round(performance * 100)) + '%' + ', ' +
      'Hits left:' + str(hits_left) + ' (' + str(round(performance_left * 100)) + '%)' + ', ' +
      'Hits right: ' + str(hits_right) + ' (' + str(round(performance_right * 100)) + '%)' + '\n')

s4 = ('Responses: ' + str(responses) + ', ' +
      'Accuracy: ' + str(round(accuracy * 100)) + '%' + ', ' +
      'Hits left: ' + str(hits_left) + ' (' + str(round(accuracy_left * 100)) + '%)' + ', ' +
      'Hits right: ' + str(hits_right) + ' (' + str(round(accuracy_right * 100)) + '%)' + '\n')

s5 = ('Misses: ' + str(misses) + ' (' + str(round(miss_rate * 100, 1)) + '%)' + ', ' +
      'Miss left: ' + str(misses_left) + ' (' + str(round(miss_rate_left * 100)) + '%)' + ', ' +
      'Miss right: ' + str(misses_right) + ' (' + str(round(miss_rate_right * 100)) + '%)' + '\n')

s6 = ('Water: ' + str(water) + 'μL' + ', ' +
      'Water left: ' + str(water_left) + 'μL' + ', ' +
      'Water right: ' + str(water_right) + 'μL' + ', ' +
      'AW: ' + str(df_session.AW.unique()[0]) + 'μL' + '\n')

# plt.text(0.1, 0.90, s1 + s2 + s3 + s4 + s5 + s6, fontsize=8, transform=plt.gcf().transFigure)
# plt.text(0, 1, s1 + s2 + s3 + s4 + s5 + s6)

########################################################################################################################

# fig = plt.figure()

# PLOT 1: ACCURACY PER SIDE

# Compute accuracy rolling average
ra_total = compute_window(df_session.Hit[df_session.Miss == 0], 20)  # All valid trials
ra_left = compute_window(df_session.Hit[(df_session.Miss == 0) & (df_session.Side == 0)], 20)  # Left valid trials
ra_right = compute_window(df_session.Hit[(df_session.Miss == 0) & (df_session.Side == 1)], 20)  # Right valid trials

# Prepares the grid for the plots
ax1 = plt.subplot2grid((16, 4), (0, 0), rowspan=2, colspan=3)
# ax1 = plt.subplot2grid((4, 1), (0, 0))

# Plot horizontal lines
ax1.axhline(0.5, color='tab:gray', linestyle='--')  # Chance level
ax1.axhline(0.25, color='tab:gray', linestyle=':')  # Accuracy 0.25
ax1.axhline(0.75, color='tab:gray', linestyle=':')  # Accuracy 0.75

# Plot accuracy rolling average
ax1.plot(df_session.Hit[df_session.Miss == 0].index, ra_total, marker='o', color='black', label='Total')
ax1.plot(df_session.Hit[(df_session.Miss == 0) & (df_session.Side == 0)].index, ra_left, marker='o', color='tab:blue',
        label='Left')
ax1.plot(df_session.Hit[(df_session.Miss == 0) & (df_session.Side == 1)].index, ra_right, marker='o', color='tab:orange',
        label='Right')

ax1.set_xlim([1, len(df_session)])  # 1 to not plot trial 0
ax1.set_xticklabels([])
ax1.set_ylabel('Acc.\n(%)')
ax1.set_ylim([0, 1.1])
ax1.set_yticks(list(np.arange(0, 1.1, 0.1)))
ax1.set_yticklabels(['0', '', '', '', '', '50', '', '', '', '', '100'])
# ax1.set_yticklabels(['', '', '', '', '', '', '', '', '', '', ''])
ax1.legend(loc='lower right', fontsize='xx-small', frameon=True)
ax1.spines['top'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
#ax1.spines['right'].set_visible(False)

# Instantiate a second axes that shares the same x-axis
ax1_twin = ax1.twinx()
ax1_twin.set_ylim([0, 1.1])
ax1_twin.set_yticks(list(np.arange(0, 1.1, 0.1)))
ax1_twin.set_yticklabels(['', '', '', '', '', '', '', '', '', '', ''])
ax1_twin.spines['top'].set_visible(False)
ax1_twin.spines['bottom'].set_visible(False)

# Plot text
ax1.text(0, 1, s1 + s2 + s3 + s4 + s5 + s6)

########################################################################################################################

# PLOT 2: REPEATING VS ALTERNATING ACCURACY

# Compute accuracy rolling average for repeating vs alternating trials
ra_rep = compute_window(df_session.Hit[(df_session.Miss == 0) & (df_session.RepTrial == 1)], 20)
ra_alt = compute_window(df_session.Hit[(df_session.Miss == 0) & (df_session.RepTrial == 0)], 20)

# Prepares the grid for the plots
ax2 = plt.subplot2grid((16, 4), (2, 0), rowspan=2, colspan=3)
# ax = plt.subplot2grid((4, 1), (1, 0))

# Plot horizontal lines
ax2.axhline(0.5, color='tab:gray', linestyle='--')  # Chance level
ax2.axhline(0.25, color='tab:gray', linestyle=':')  # Accuracy 0.25
ax2.axhline(0.75, color='tab:gray', linestyle=':')  # Accuracy 0.75

# Plot accuracy rolling average
ax2.plot(df_session.Hit[(df_session.Miss == 0) & (df_session.RepTrial == 0)].index, ra_alt, marker='o',
        color='tab:purple', label='Alt')
ax2.plot(df_session.Hit[(df_session.Miss == 0) & (df_session.RepTrial == 1)].index, ra_rep, marker='o',
        color='tab:brown', label='Rep')

ax2.set_xlim([1, len(df_session)])  # 1 to not plot trial 0
ax2.set_xticklabels([])
ax2.set_ylabel('Acc.\n(%)')
ax2.set_ylim([0, 1.1])
ax2.set_yticks(list(np.arange(0, 1.1, 0.1)))
ax2.set_yticklabels(['0', '', '', '', '', '50', '', '', '', '', '100'])
# ax2.set_yticklabels(['', '', '', '', '', '', '', '', '', '', ''])
ax2.legend(loc='lower right', fontsize='xx-small', frameon=True)
ax2.spines['top'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
#ax2.spines['right'].set_visible(False)

# Instantiate a second axes that shares the same x-axis
ax2_twin = ax2.twinx()
ax2_twin.set_ylim([0, 1.1])
ax2_twin.set_yticks(list(np.arange(0, 1.1, 0.1)))
ax2_twin.set_yticklabels(['', '', '', '', '', '', '', '', '', '', ''])
ax2_twin.spines['top'].set_visible(False)
ax2_twin.spines['bottom'].set_visible(False)

########################################################################################################################

# PLOT 3: MISSES

# Compute accuracy rolling average
ra_total_miss = compute_window(df_session.Miss, 20)  # All valid trials
ra_left_miss = compute_window(df_session.Miss[df_session.Side == 0], 20)  # Left valid trials
ra_right_miss = compute_window(df_session.Miss[df_session.Side == 1], 20)  # Right valid trials

# Prepares the grid for the plots
ax3 = plt.subplot2grid((16, 4), (4, 0), rowspan=2, colspan=3)
#ax3 = plt.subplot2grid((4, 1), (2, 0))

# Plot horizontal lines
ax3.axhline(0.5, color='tab:gray', linestyle='--')  # Chance level
ax3.axhline(0.25, color='tab:gray', linestyle=':')  # Accuracy 0.25
ax3.axhline(0.75, color='tab:gray', linestyle=':')  # Accuracy 0.75

# Plot misses rolling average
ax3.plot(df_session.index, ra_total_miss, marker='o', color='black', label='Total')
ax3.plot(df_session[df_session.Side == 0].index, ra_left_miss, marker='o', color='tab:blue', label='Left')
ax3.plot(df_session[df_session.Side == 1].index, ra_right_miss, marker='o', color='tab:orange', label='Right')

ax3.set_xlim([1, len(df_session)])  # 1 to not plot trial 0
ax3.set_xticklabels([])

ax3.set_ylim([0, 1.1])
ax3.set_ylabel('Miss\n(%)')
ax3.set_yticks(list(np.arange(0, 1.1, 0.1)))
ax3.set_yticklabels(['0', '', '', '', '', '50', '', '', '', '', '100'])
# ax3.set_yticklabels(['', '', '', '', '', '', '', '', '', '', ''])
ax3.legend(loc='lower right', fontsize='xx-small', frameon=True)
ax3.spines['top'].set_visible(False)
ax3.spines['bottom'].set_visible(False)
#ax3.spines['right'].set_visible(False)

# Instantiate a second axes that shares the same x-axis
ax3_twin = ax3.twinx()
ax3_twin.set_ylim([0, 1.1])
ax3_twin.set_yticks(list(np.arange(0, 1.1, 0.1)))
ax3_twin.set_yticklabels(['', '', '', '', '', '', '', '', '', '', ''])
ax3_twin.spines['top'].set_visible(False)
ax3_twin.spines['bottom'].set_visible(False)

########################################################################################################################

# PLOT 4: HIT SCATTER PLOT

# Prepares the grid for the plots
ax4 = plt.subplot2grid((16, 4), (6, 0), rowspan=2, colspan=3)
# ax4 = plt.subplot2grid((4, 1), (3, 0))

palette = ['tab:red', 'tab:green', 'grey']
hue = ['Error' if i == 0 else 'Hit' if i == 1 else 'Miss' for i in df_session.Hit]
hue_order = ['Error', 'Hit', 'Miss']

if df_session.Stage.unique()[0] <= 3:  # No coherences, plot sides
    scatter = sns.scatterplot(df_session.index, df_session.Side, hue=hue, palette=palette, hue_order=hue_order)
    ax4.set_ylim(-0.8, 1.8)
    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(['L', 'R'])
    ax4.set_ylabel('Sides')

else:  # Plot coherences
    scatter = sns.scatterplot(df_session.index, df_session.Evidence, hue=hue, palette=palette, hue_order=hue_order)
    # Plot horizontal lines
    ax4.axhline(0, color='tab:gray', linestyle='--')  # Evidence 0
    ax4.axhline(-0.5, color='tab:gray', linestyle=':')  # Evidence -0.5
    ax4.axhline(0.5, color='tab:gray', linestyle=':')  # Evidence 0.5
    ax4.set_ylim(-1.1, 1.1)  # Evidences
    # ax.set_ylim(0, 1)  # Coherences
    ax4.set_yticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    ax4.set_yticklabels(['L', '', '', '', '0', '', '', '', 'R'])  # Evidences
    # ax.set_yticklabels(['Left', '', '', '', '0.5', '', '',  '', 'Right'])  # Coherences
    ax4.set_ylabel('Evi.')
    # ax.set_ylabel('Coherence')

ax4.set_xlim([1, len(df_session)])  # 1 to not plot trial 0
ax4.set_xlabel('Trial')
# scatter.legend(bbox_to_anchor=(1, 1))
scatter.legend(loc='lower right', fontsize='xx-small', frameon=True)
# scatter.get_legend().remove()
ax4.spines['top'].set_visible(False)
#ax4.spines['right'].set_visible(False)

# Instantiate a second axes that shares the same x-axis
ax4_twin = ax4.twinx()
ax4_twin.set_ylim(-1.1, 1.1)  # Evidences
ax4_twin.set_yticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
ax4_twin.set_yticklabels(['', '', '', '', '', '', '', '', ''])
ax4_twin.spines['top'].set_visible(False)

########################################################################################################################

# PLOT 5: PERISTIMULUS LICK RASTER

#fig = plt.figure()
xlim = [[], []]  # Initialize empty list to store left and right xlim

for k in range(len(df_session.Side.unique())):  # k=0 left trials and k=1 right trials

    if k == 0:  # Left subplot: left trials
        # ax = plt.subplot2grid((1, 2), (0, 0))
        ax = plt.subplot2grid((16, 4), (9, 0), rowspan=4, colspan=2)
        stim_color = 'tab:blue'
        ax.set_title('Left trials')
        # ax.set_xlabel('Time (s)')
        ax.set_xticklabels([])
        ax.set_ylabel('Trial')
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
    else:  # Right subplot: right trials
        # ax = plt.subplot2grid((1, 2), (0, 1))
        ax = plt.subplot2grid((16, 4), (9, 2), rowspan=4, colspan=2)
        stim_color = 'tab:orange'
        ax.set_title('Right trials')
        # ax.set_xlabel('Time (s)')
        ax.set_xticklabels([])
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)

    for j in range(len(df_session[df_session.Side == k])):  # n trials

        # Plot stimulus length
        ax.barh(df_session[df_session.Side == k].reset_index().index,
                df_session[df_session.Side == k].reset_index().StimLen,
                left=df_session[df_session.Side == k].reset_index().StimStart[j] -
                     df_session[df_session.Side == k].reset_index().StimStart[j],
                color=stim_color, label='Stim')

        if df_session[df_session.Side == k].reset_index().Hit[j] == 0.0:
            resp_win_color = 'tab:red'
        elif df_session[df_session.Side == k].reset_index().Hit[j] == 1.0:
            resp_win_color = 'tab:green'
        elif np.isnan(df_session[df_session.Side == k].reset_index().Hit[j]):
            resp_win_color = 'tab:gray'

        # Plot response window length
        ax.barh(df_session[df_session.Side == k].reset_index().index.values[j],
                df_session[df_session.Side == k].reset_index().RespWinLen[j],
                left=df_session[df_session.Side == k].reset_index().RespWinStart[j] -
                     df_session[df_session.Side == k].reset_index().StimStart[j],
                color=resp_win_color)

        # Left licks
        for i in range(len(df_session[df_session.Side == k].reset_index().Port1In[j])):  # n licks
            if df_session[df_session.Side == k].reset_index().Port1In[j] == []:
                # if not df_session.Port1In[j]:  # Equivalent
                pass
            else:
                ax.plot(df_session[df_session.Side == k].reset_index().Port1In[j][i] -
                        df_session[df_session.Side == k].reset_index().StimStart[j],
                        df_session[df_session.Side == k].Port1In.reset_index().index[j], marker='o', color='tab:blue',
                        markersize=200/len(df_session.Side == 0))
                        # markersize = ax.containers[1][0].get_height()

        # Right licks
        for i in range(len(df_session[df_session.Side == k].reset_index().Port2In[j])):  # n licks
            if df_session[df_session.Side == k].reset_index().Port2In[j] == []:
                # if not df_session.Port1In[j]:  # Equivalent
                pass
            else:
                ax.plot(df_session[df_session.Side == k].reset_index().Port2In[j][i] -
                        df_session[df_session.Side == k].reset_index().StimStart[j],
                        df_session[df_session.Side == k].reset_index().Port2In.index[j], marker='o', color='tab:orange',
                        markersize=200/len(df_session.Side == 1))
                        # markersize = ax.containers[1][0].get_height()

    xlim[k] = [ax.get_xlim()]  # Store xlim from left and right plots

# Custom legend
legend_elements = [Patch(facecolor='tab:blue', label='Stim. left'),
                   Patch(facecolor='tab:orange', label='Stim. right'),
                   Patch(facecolor='tab:green', label='Correct'),
                   Patch(facecolor='tab:red', label='Error'),
                   Patch(facecolor='tab:gray', label='Miss'),
                   Line2D([0], [0], marker='o', color='w', label='Left licks', markerfacecolor='tab:blue'),
                   Line2D([0], [0], marker='o', color='w', label='Right licks', markerfacecolor='tab:orange')]

ax.legend(handles=legend_elements, loc='upper right', fontsize='xx-small', frameon=True)

########################################################################################################################

# PLOT 6: PERISTIMULUS LICK HISTOGRAM

# fig = plt.figure()

bin_size = 0.1

for k in range(len(df_session.Side.unique())):  # k=0 left trials and k=1 right trials

    histcounts_L = []
    histcounts_R = []

    if k == 0:  # Left subplot: left trials
        # ax = plt.subplot2grid((1, 2), (0, 0))
        ax = plt.subplot2grid((16, 4), (13, 0), rowspan=1, colspan=2)
        # ax.set_title('Left trials')
        #ax.set_xlabel('Time (s)')
        ax.set_xlim(xlim[k][0])  # Use the same xlim that left raster
        # ax.set_xticklabels([])
        ax.set_ylabel('All licks\n(licks/s)')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    else:  # Right subplot: right trials
        # ax = plt.subplot2grid((1, 2), (0, 1))
        ax = plt.subplot2grid((16, 4), (13, 2), rowspan=1, colspan=2)
        # ax.set_title('Right trials')
        #ax.set_xlabel('Time (s)')
        ax.set_xlim(xlim[k][0])  # Use the same xlim that right raster
        # ax.set_xticklabels([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    for j in range(len(df_session[df_session.Side == k])):  # n trials

        # Left licks
        for i in range(len(df_session[df_session.Side == k].reset_index().Port1In[j])):  # n licks
            if df_session[df_session.Side == k].reset_index().Port1In[j] == []:
                # if not df_session.Port1In[j]:  # Equivalent
                pass
            else:
                histcounts_L.append(df_session[df_session.Side == k].reset_index().Port1In[j][i] -
                                    df_session[df_session.Side == k].reset_index().StimStart[j])

        # Right licks
        for i in range(len(df_session[df_session.Side == k].reset_index().Port2In[j])):  # n licks
            if df_session[df_session.Side == k].reset_index().Port2In[j] == []:
                # if not df_session.Port1In[j]:  # Equivalent
                pass
            else:
                histcounts_R.append(df_session[df_session.Side == k].reset_index().Port2In[j][i] -
                                    df_session[df_session.Side == k].reset_index().StimStart[j])

    # ax.hist(histcounts_L, density=True, histtype='step', color='tab:blue', label='Left licks')
    # ax.hist(histcounts_R, density=True, histtype='step', color='tab:orange', label='Right licks')

    ax.hist(histcounts_L, histtype='step', color='tab:blue', label='Left licks', bins=np.arange(0, 4, bin_size),
            weights=np.repeat((1 / len(df_session[(df_session.Miss == 0) & (df_session.Side == 0)])) / bin_size,
                              len(histcounts_L)))
    ax.hist(histcounts_R, histtype='step', color='tab:orange', label='Right licks', bins=np.arange(0, 4, bin_size),
            weights=np.repeat((1 / len(df_session[(df_session.Miss == 0) & (df_session.Side == 1)])) / bin_size,
                              len(histcounts_R)))

ax.legend(loc='upper right', fontsize='xx-small', frameon=True)

########################################################################################################################

# PLOT 7: PERISTIMULUS FIRST LICK HISTOGRAM

# fig = plt.figure()

for k in range(len(df_session.Side.unique())):  # k=0 left trials and k=1 right trials

    first_lick_L = []
    first_lick_R = []

    if k == 0:  # Left subplot: left trials
        # ax = plt.subplot2grid((1, 2), (0, 0))
        ax = plt.subplot2grid((16, 4), (15, 0), rowspan=1, colspan=2)
        # ax.set_title('Left trials')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('First lick\n(licks/s)')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        #ax.patch.set_facecolor('none')
    else:  # Right subplot: right trials
        # ax = plt.subplot2grid((1, 2), (0, 1))
        ax = plt.subplot2grid((16, 4), (15, 2), rowspan=1, colspan=2)
        # ax.set_title('Right trials')
        ax.set_xlabel('Time (s)')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        #ax.patch.set_facecolor('none')

    for j in range(len(df_session[df_session.Side == k])):  # n trials

        # Left licks
        for i in range(len(df_session[df_session.Side == k].reset_index().Port1In[j])):  # n licks
            if df_session[df_session.Side == k].reset_index().Port1In[j] == []:
                # if not df_session.Port1In[j]:  # Equivalent
                pass
            else:
                if df_session[df_session.Side == k].reset_index().Port1In[j][i] - df_session[df_session.Side == k].reset_index().StimStart[j] < df_session.RespWinStart[j] - df_session[df_session.Side == k].reset_index().StimStart[j]:
                    first_lick_L.append(df_session[df_session.Side == k].reset_index().Port1In[j][i] - df_session[df_session.Side == k].reset_index().StimStart[j])
                elif df_session[df_session.Side == k].reset_index().Port1In[j][i] - df_session[df_session.Side == k].reset_index().StimStart[j] > df_session.RespWinStart[j] - df_session[df_session.Side == k].reset_index().StimStart[j]:
                    first_lick_L.append(df_session[df_session.Side == k].reset_index().Port1In[j][i] - df_session[df_session.Side == k].reset_index().StimStart[j])
                    break

        # Right licks
        for i in range(len(df_session[df_session.Side == k].reset_index().Port2In[j])):  # n licks
            if df_session[df_session.Side == k].reset_index().Port2In[j] == []:
                # if not df_session.Port1In[j]:  # Equivalent
                pass
            else:
                if df_session[df_session.Side == k].reset_index().Port2In[j][i] - df_session[df_session.Side == k].reset_index().StimStart[j] < df_session.RespWinStart[j] - df_session[df_session.Side == k].reset_index().StimStart[j]:
                    first_lick_R.append(df_session[df_session.Side == k].reset_index().Port2In[j][i] - df_session[df_session.Side == k].reset_index().StimStart[j])
                elif df_session[df_session.Side == k].reset_index().Port2In[j][i] - df_session[df_session.Side == k].reset_index().StimStart[j] > df_session.RespWinStart[j] - df_session[df_session.Side == k].reset_index().StimStart[j]:
                    first_lick_R.append(df_session[df_session.Side == k].reset_index().Port2In[j][i] - df_session[df_session.Side == k].reset_index().StimStart[j])
                    break

    # ax.hist(first_lick_L, density=True, histtype='step', color='tab:blue', label='Left')
    # ax.hist(first_lick_R, density=True, histtype='step', color='tab:orange', label='Right')

    ax.hist(first_lick_L, histtype='step', color='tab:blue', label='Left licks', bins=np.arange(0, 4, bin_size),
            weights=np.repeat((1 / len(df_session[(df_session.Miss == 0) & (df_session.Side == 0)])) / bin_size,
                              len(first_lick_L)))
    ax.hist(first_lick_R, histtype='step', color='tab:orange', label='Right licks', bins=np.arange(0, 4, bin_size),
            weights=np.repeat((1 / len(df_session[(df_session.Miss == 0) & (df_session.Side == 1)])) / bin_size,
                              len(first_lick_R)))

    ax.patch.set_facecolor('none')  # Make axes transparent so the xaxes labels from the upper plot are visible
# ax.legend(loc='upper right')

########################################################################################################################

# PLOT 8: PSYCOMETRIC CURVE

# fig = plt.figure()

ax11 = plt.subplot2grid((16, 4), (3, 3), rowspan=3, colspan=1)

# Compute psychometric curves
psych_curve = compute_psych_curve(df_session.Evidence[df_session.Miss == 0], df_session.Choice[df_session.Miss == 0])
psych_curve_rep = compute_psych_curve(df_session.EviRep, df_session.RepChoice)

# Plot horizontal and vertical lines
ax11.axhline(0.5, color='tab:gray', ls='--')
ax11.axvline(0., color='tab:gray', ls='--')

# Plot psychometric curve and errorbars
ax11.plot(np.linspace(-1, 1, 30), psych_curve.fit, color='k', label='L-R')
ax11.errorbar(psych_curve.xdata, psych_curve.ydata, yerr=psych_curve.fit_error, fmt='ko', markerfacecolor='none')

ax11.plot(np.linspace(-1, 1, 30), psych_curve_rep.fit, color='g', label='Alt-Rep')
ax11.errorbar(psych_curve_rep.xdata, psych_curve_rep.ydata, yerr=psych_curve_rep.fit_error, fmt='go', markerfacecolor='none')

ax11.set_xlabel('Evi.')
ax11.set_xlim([-1.05, 1.05])
ax11.set_ylabel('Prob. right')
ax11.set_ylim([-0.025, 1.025])
#ax11.set_yticks(np.arange(0, 1.1, step=0.1))
ax11.legend(loc="lower right", frameon=False)

ax11_right_yaxis = ax11.twinx()  # instantiate a second axes that shares the same x-axis
#ax11_right_yaxis.set_ylabel('Prob. right')
ax11.set_yticklabels([])  # Remove left yticklabels
ax11.set_yticks([])  # Remove left yticks

ax11.spines['top'].set_visible(False)
ax11.spines['left'].set_visible(False)
ax11_right_yaxis.spines['top'].set_visible(False)
ax11_right_yaxis.spines['left'].set_visible(False)

ax11.annotate(str(round(psych_curve.ydata[0], 2)), xy=(psych_curve.xdata[0], psych_curve.ydata[0]),
              xytext=(psych_curve.xdata[0], psych_curve.ydata[0]), color='tab:red')
ax11.annotate(str(round(psych_curve.ydata[-1], 2)), xy=(psych_curve.xdata[-1], psych_curve.ydata[-1]),
              xytext=(psych_curve.xdata[-1], psych_curve.ydata[-1]), color='tab:red')

sensitivity, bias, lr_left, lr_right = psych_curve.params

ax11.annotate("S=" + str(round(sensitivity, 2)) + "\n" +  # Sensitivity
              "B=" + str(round(bias, 2)) + "\n" +  # Bias
              "LR_L=" + str(round(lr_left, 2)) + "\n" +  # Left lapse rate
              "LR_R=" + str(round(lr_right, 2)), xy=(0, 0), xytext=(-1, 0.5), fontsize='xx-small')  # Right lapse rate

########################################################################################################################

# PLOT 9: INTERSESSION

# fig = plt.figure()

ax12 = plt.subplot2grid((16, 4), (0, 3), rowspan=3, colspan=1)
ax12.set_xticks([])
ax12.set_xticklabels([])
ax12.set_yticks([])
ax12.set_yticklabels([])
ax12.text(0.5, 0.5, 'Intersession\ndata will go here', horizontalalignment='center', verticalalignment='center', transform=ax12.transAxes)