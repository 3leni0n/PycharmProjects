# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 16:17:55 2019

@author: Tiffany
"""

import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from datahandler import Utils
import logging
import warnings
import math

#if os.environ.get('DISPLAY', '') == '':
#    mpl.use('Agg')

import matplotlib.dates as mdates

logger = logging.getLogger(__name__)

# Plot parameters:
COLORLEFT = '#31A2AC'
COLORRIGHT = '#FF8D3F'
DELAY_H = 'black'
DELAY_M = 'purple'
DELAY_L = 'crimson'
            
BINSIZE = 0.1

mpl.rcParams['lines.linewidth'] = 0.7
mpl.rcParams['lines.markersize'] = 1
mpl.rcParams['font.size'] = 7
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
label_kwargs = {'fontsize': 9, 'fontweight': 'roman'}

trials_path ='C:/Users/user/Google Drive (tiffany.ona@gmail.com)/WORKING_MEMORY/EXPERIMENTS/2B/clean/C09/C09_StageTraining_2B_V1_20190212-124958_trials.csv'
params_path ='C:/Users/user/Google Drive (tiffany.ona@gmail.com)/WORKING_MEMORY/EXPERIMENTS/2B/clean/C09/C09_StageTraining_2B_V1_20190212-124958_params.csv'
daily_report_path ='C:/Users/user/Google Drive (tiffany.ona@gmail.com)/WORKING_MEMORY/EXPERIMENTS/2B/C09'

# trials_path ='C:/Users/Tiffany/Google Drive/WORKING_MEMORY/EXPERIMENTS/4B/clean/N04/N04_StageTraining_4B_V1_20191216-093527_trials.csv'
# params_path ='C:/Users/Tiffany/Google Drive/WORKING_MEMORY/EXPERIMENTS/4B/clean/N04/N04_StageTraining_4B_V1_20191216-093527_params.csv'
# daily_report_path ='C:/Users/Tiffany//WORKING_MEMORY/EXPERIMENTS/4B/N01'

def create_duration_states(df,params):
    ''' Creates new columns with duration of certain states '''
    state_list = params.state_list.split(',')
    
    for state in state_list:
        df[state + '_duration'] = df[state + '_end'] - df[state + '_start']

    return df

def alignment(df,params,zero):
        ''' Substract to timestamps of a dataframe a specific value to set the 0 in different positions 
            Takes:
                df_ild = Dataframe of trials
                params = list with overall session values
                zero = State at from which reset timing
        '''
        
        state_list = params.state_list.split(',')
        
        df = Utils.convert_strings_to_lists(df, ['Fixation_end','Fixation_start','Fixation_Break_start','Fixation_Break_end','L_s', 'C_s','C_e','L_e'])
        
        if params.lick == 2:
            df = Utils.convert_strings_to_lists(df, ['ResponseWindow_start','ResponseWindow_end','SecondWrongLick_start','SecondWrongLick_end','SecondCorrectLick_start', 'SecondCorrectLick_end'])
            
        # Shifts the alignment to StimulusTrigger
        alignment = []
        for j in df[zero]:
            alignment.append(float(j[-1]))
            
        for state in state_list:
            try:
                df[state+'_start'] = df[state+'_start']-alignment
                df[state+'_end'] = df[state+'_end']-alignment
            except:
                pass
                
        df['L_s'] = df['L_s'] - alignment
        df['L_e'] = df['L_e'] - alignment
        df['C_s'] = df['C_s'] - alignment
        df['C_e'] = df['C_e'] - alignment
                                 
        return df
    
def wm_daily_report(trials_path, params_path, daily_report_path):

    try:
        df = pd.read_csv(trials_path, sep=';')
    except FileNotFoundError:
        logger.critical("Error: Reading trials CSV file. Exiting...")
        raise
    
    try:
        params = pd.read_csv(params_path, sep=';')
    except FileNotFoundError:
        logger.critical("Error: Reading params CSV file. Exiting...")
        raise
    
    df.index = df.trials
    df = df[:-1]
    
    # transform params df_ild that only contains one row in a Series object
    params = params.iloc[0]

    df = Utils.convert_strings_to_floats(df, ['ResponseWindow_start','ResponseWindow_end', 'AW_start','StimulusDuration_start',
                                                  'StimulusDuration_end','StimulusTrigger_end', 'StimulusTrigger_start',
                                                  'Reward_start','Miss_start','Punish_start','WrongLick_start',
                                                 'AW_start', 'StimulusTrigger_end', 
                                                 ])     
    if params.timeout != 0:
        df = Utils.convert_strings_to_floats(df, ['TimeOut_end', 'TimeOut_start'])    

    if params.motor >= 6:
        df = Utils.convert_strings_to_floats(df, ['Delay_end', 'Delay_start'])    
                
    if  params.fixation >= 0.1:
        try:
            df = alignment(df,params,'Fixation_end')
        except KeyError:
            params.fixation = 0
            df = Utils.convert_strings_to_lists(df, ['L_s', 'C_s','C_e','L_e'])       
    else:
        df = Utils.convert_strings_to_lists(df, ['L_s', 'C_s','C_e','L_e'])

    # EXPANDED DATAFRAMES
    pokes_df_left = Utils.unnesting(df, ['L_s', 'L_e'])
    pokes_df_right = Utils.unnesting(df, ['C_s', 'C_e'])    
     
    #Duration of states for faster raster plot
    #df_ild = create_duration_states(df_ild,params)
    
    # SUB-DATAFRAMES    
    left_trials_df = df[df.reward_side == 0]
    right_trials_df = df[df.reward_side == 1]
    
    valid_trials_df = df[df.misshistory == False]
    left_valid_trials_df = valid_trials_df[valid_trials_df.reward_side == 0]
    right_valid_trials_df = valid_trials_df[valid_trials_df.reward_side == 1]

    if params.motor == 6:    
        if params.delay_progression_value == 0 and params.delay_l != params.delay_h:
            df_l = df[df['delay_times'] == sorted(df.delay_times.unique())[0]]
            df_m = df[df['delay_times'] ==  sorted(df.delay_times.unique())[1]]
            df_h = df[df['delay_times'] ==  sorted(df.delay_times.unique())[-1]]
        else:
            df_l = df[df['delay_types'] == 'delay_l']
            df_m = df[df['delay_types'] == 'delay_m']
            df_h = df[df['delay_types'] == 'delay_h']
                
    # USEFUL VALUES
    number_of_trials = df.shape[0]
    
    # Measure accuracy of the session assuming that wronglicks were punishes. 
    if params.stage_number == 1:
        correct_licks = 'wronglickhistory'
        
    else:   
        correct_licks = 'hithistory'
        
    corrects_left = left_trials_df[left_trials_df[correct_licks] == 1].shape[0]
    total_left = left_trials_df.shape[0]
    per_left = corrects_left / total_left
    
    corrects_right = right_trials_df[right_trials_df[correct_licks] == 1].shape[0]
    total_right = right_trials_df.shape[0]
    try:
        per_right = corrects_right / total_right
    except:
        per_right = 0

    per_accuracy = df[df[correct_licks]==1].shape[0] / params.valid_trials
    per_invalids = params.invalid_trials / params.total_trials
    per_performance = params.correct_trials / params.total_trials
        
    valid_left = left_valid_trials_df.shape[0]
    try:
        acc_left = corrects_left / valid_left
    except:
        acc_left = 0
        
    valid_right = right_valid_trials_df.shape[0]
    try:
        acc_right = corrects_right / valid_right
    except:
        acc_right = 0
    
    miss_left = left_trials_df[left_trials_df.misshistory == True].shape[0]
    try:
        m_left = miss_left / total_left
    except:
        m_left = 0
        
    miss_right = right_trials_df[right_trials_df.misshistory == True].shape[0]
    try:
        m_right = miss_right / total_right 
    except:
        m_right= 0
    AWwater = df.loc[(df.AWhistory == True)]['trials'].count()

    with PdfPages(daily_report_path) as pdf:
        plt.figure(figsize=(8.5, 13.7))        
       
# ---------------------- FIRST PLOT: rolling averages of accuracy -------------------------------------            
        print("Let's do the trend")
        # Prepares the grid for the plots
        axes = plt.subplot2grid((16,2), (0,0), colspan=2, rowspan=3)
        
        if params.stage_number == 1:
            ra_total = Utils.compute_window(valid_trials_df.wronglickhistory, 20)
            ra_right = Utils.compute_window(right_valid_trials_df.wronglickhistory, 20)
            ra_left = Utils.compute_window(left_valid_trials_df.wronglickhistory, 20)
    
            axes.plot(valid_trials_df.trials, ra_total, marker='o', color='black')
            axes.plot(left_valid_trials_df.trials, ra_left, marker='o', color=COLORLEFT)
            axes.plot(right_valid_trials_df.trials, ra_right, marker='o', color=COLORRIGHT)
    
            axes.set_ylabel('Categorization'+'\n'+' Accuracy [%]' + '\n' + '(w/o WrongLick)')
            
        elif params.stage_number != 1:
            ra_total = Utils.compute_window(valid_trials_df.hithistory, 20)
            ra_right = Utils.compute_window(right_valid_trials_df.hithistory, 20)
            ra_left = Utils.compute_window(left_valid_trials_df.hithistory, 20)
    
            axes.plot(valid_trials_df.trials, ra_total, marker='o', color='black')
            axes.plot(left_valid_trials_df.trials, ra_left, marker='o', color=COLORLEFT)
            axes.plot(right_valid_trials_df.trials, ra_right, marker='o', color=COLORRIGHT)

            axes.set_ylabel('Accuracy [%]')        
            
        axes.set_ylim([0, 1.1])
        axes.set_yticks(list(np.arange(0, 1.1, 0.1)))
        axes.set_yticklabels(['0', '', '', '', '', '50', '', '', '', '', '100'])

        axes.set_xlim([1, number_of_trials + 1])
        axes.set_xlabel('')
        axes.yaxis.set_tick_params()
        axes.xaxis.set_tick_params()
        axes.hlines(xmin=0, xmax=number_of_trials + 1, y=0.5, linewidth=0.2, color='gray',linestyle = ':')

        legend_elements = [Line2D([0], [0], color='black', label='Total'),
                           Line2D([0], [0], color=COLORLEFT, label='Left'),
                           Line2D([0], [0], color=COLORRIGHT, label='Right')]
        axes.legend(loc="lower right", handles=legend_elements, ncol=1, prop={'size': 8})

# _______________________ UPPER TEXT _________________________________________
        print('Let"s do the summary')
        s1 = ('Date: ' + str(params.day) + ' ' + str(params.time) + '\n')
        s2 = ('Subject name: ' + str(params.subject_name) +
              ' / Box name: ' + str(params.box) +
              ' / Stage: ' + str(params.stage_number)+  
              ' / Substage: ' + str(params.substage) + 
              ' / Fixation: '  + str(params.fixation) + 
              ' / Timeout: ' + str(params.timeout) + 
              ' / Switch: ' + str(params.switch) + 
              ' / Motor: ' + str(params.motor) +
              ' / Lick:' + '\n' )       
        
        s3 = ('Total trials: ' + str(params.total_trials) +
              ' / Performance: ' + str(round(per_performance * 100, 1)) + ' %'
              ' / Corrects on left: ' + str(corrects_left) + ' (' + str(round(per_left * 100, 1)) + ' %)'
              ' / Corrects on right: ' + str(corrects_right) + ' (' + str(round(per_right * 100, 1)) + ' %)' + '\n')

        s4 = ('Valid trials: ' + str(params.valid_trials) +
              ' / Accuracy: ' + str(round(per_accuracy * 100, 1)) + ' %'
              ' / Corrects on left: ' + str(corrects_left) + ' (' + str(round(acc_left * 100, 1)) + ' %)'
              ' / Corrects on right: ' + str(corrects_right) + ' (' + str(round(acc_right * 100, 1)) + ' %)'
              + '\n')

        s5 = ('Invalid trials: ' + str(params.invalid_trials) + ' (' + str(round(per_invalids * 100, 1)) + ' %)'
              ' / Miss on left: ' + str(miss_left) + ' (' + str(round(m_left * 100, 1)) + ' %)'
              ' / Miss on right: ' + str(miss_right) + ' (' + str(round(m_right * 100, 1)) + ' %)' + '\n')
        
        if params.motor == 6:
            s6 = ('Water: ' + str(params.correct_trials * 2.5) + 'mL ' + 
            '/ AW: ' + str(AWwater * 2.5) + 'mL' +
            '    / Delay H: ' + str(round(max(df.delay_times_h),2)) + ', Delay M: ' + str(round(max(df.delay_times_m),2)))
        else:
            s6 = ('Water: ' + str(params.correct_trials * 2.5) + 'mL ' + 
            '/ AW: ' + str(AWwater * 2.5) + 'mL' )

        plt.text(0.1, 0.90, s1 + s2 + s3 + s4 + s5 + s6, fontsize=8, transform=plt.gcf().transFigure)

# ________________ SECOND PLOT: MISS PLOT ______________________________________
        print("Let's do the misses")
        # Prepares the grid for the plots
        axes = plt.subplot2grid((16,2), (3,0), colspan=2, rowspan=1)

        ra_total_miss = Utils.compute_window(df.misshistory, 20)
        ra_right_miss = Utils.compute_window(right_trials_df.misshistory, 20)
        ra_left_miss = Utils.compute_window(left_trials_df.misshistory, 20)
            
        # Plot the percentage of misses in a running window of 20. 
        axes.plot(df.trials, ra_total_miss, marker = 'o', color='black')
        axes.plot(left_trials_df.trials, ra_left_miss, marker = 'o', color=COLORLEFT)
        axes.plot(right_trials_df.trials, ra_right_miss, marker = 'o', color=COLORRIGHT)
        
        axes.set_xlim([0, number_of_trials + 1])
        axes.set_xticklabels([])  
        
        axes.set_ylim([0,1.1])   
        axes.set_ylabel('Miss (%)')   
        axes.set_yticklabels(['0', '', '', '', '', '50', '', '', '', '', '100'])
        axes.hlines(xmin=0, xmax=number_of_trials + 1, y=0.5, linewidth=0.2, color='gray',linestyle = ':')

# ________________ THIRD PLOT: SCATTER PLOT________________________________
        print("Let's do the scatter")
        conditions = [df.hithistory == 0, df.hithistory == 1, df.hithistory == -3, df.wronglickhistory == True]# Compute accuracy rolling average
ra_total = compute_window(df_session.Hit[df_session.Miss == 0], 20)  # All valid trials
ra_left = compute_window(df_session.Hit[(df_session.Miss == 0) & (df_session.Side == 0)], 20)  # Left valid trials
ra_right = compute_window(df_session.Hit[(df_session.Miss == 0) & (df_session.Side == 1)], 20)  # Right valid trials

# Prepares the grid for the plots
ax = plt.subplot2grid((16, 2), (0, 0), colspan=2, rowspan=3)

# Plot horizontal line at chance level
ax.axhline(0.5, color='tab:gray', linestyle='--')

# Plot accuracy rolling average
ax.plot(df_session.Hit[df_session.Miss == 0].index, ra_total, marker='o', color='black', label='Total')
ax.plot(df_session.Hit[(df_session.Miss == 0) & (df_session.Side == 0)].index, ra_left, marker='o', color='tab:blue',
        label='Left')
ax.plot(df_session.Hit[(df_session.Miss == 0) & (df_session.Side == 1)].index, ra_right, marker='o', color='tab:orange',
        label='Right')

# Needed? I integrated everything in Hit
if session_val['stage'] == '1':
    ax.set_ylabel('Categorization' + '\n' + ' Accuracy [%]' + '\n' + '(w/o WrongLick)')
else:
    ax.set_ylabel('Accuracy [%]')

ax.set_xlim([1, len(df_session)])  # 1 to not plot trial 0
ax.set_ylim([0, 1.1])
ax.set_yticks(list(np.arange(0, 1.1, 0.1)))
ax.set_yticklabels(['0', '', '', '', '', '50', '', '', '', '', '100'])
ax.legend()

########################################################################################################################

# PLOT 2: MISSES

# Prepares the grid for the plots
ax = plt.subplot2grid((16, 2), (3, 0), colspan=2, rowspan=1)

# Compute accuracy rolling average
ra_total_miss = compute_window(df_session.Miss, 20)  # All valid trials
ra_left_miss = compute_window(df_session.Miss[df_session.Side == 0], 20)  # Left valid trials
ra_right_miss = compute_window(df_session.Miss[df_session.Side == 1], 20)  # Right valid trials

# Prepares the grid for the plots
ax = plt.subplot2grid((16, 2), (3, 0), colspan=2, rowspan=1)

# Plot horizontal line at chance level
ax.axhline(0.5, color='tab:gray', linestyle='--')

# Plot misses rolling average
ax.plot(df_session.index, ra_total_miss, marker='o', color='black', label='Total')
ax.plot(df_session[df_session.Side == 0].index, ra_left_miss, marker='o', color='tab:blue', label='Left')
ax.plot(df_session[df_session.Side == 1].index, ra_right_miss, marker='o', color='tab:orange', label='Right')

ax.set_xlim([1, len(df_session)])  # 1 to not plot trial 0
ax.set_xticklabels([])

ax.set_ylim([0, 1.1])
ax.set_ylabel('Miss (%)')
ax.set_yticklabels(['0', '', '', '', '', '50', '', '', '', '', '100'])
        choices = ['red', 'green', 'black', 'grey']

        df['color'] = np.select(conditions, choices)

        num_of_colors = len(df.color.unique())
        palette = choices[:num_of_colors]
        
        # 
        if params.stage_number <= 3:
            axes = plt.subplot2grid((16,2), (4,0), colspan=2, rowspan=1)
            sns.scatterplot(df.trials, df.reward_side, hue=df.hithistory, palette=palette, s=8, ax=axes, linewidth=0.1)
            axes.set_ylim(-0.8, 1.8)
            axes.set_yticks([0, 1])
            axes.set_yticklabels(['Left', 'Right'])
            axes.set_ylabel('Sides')
        
        # 
        else:
            axes = plt.subplot2grid((16,2), (4,0), colspan=2, rowspan=2)
            sns.scatterplot(df.trials, df.coherences, hue=df.hithistory, palette=palette, s=12, ax=axes, linewidth=0.1)
            axes.set_ylim(-26, 26)
            axes.set_yticks([-24,-16,-8,0,8,16,24])
            axes.set_ylabel('Coherences')
            
        axes.set_xlim([1, number_of_trials + 1])        
        axes.set_xticklabels([])  
        axes.get_legend().remove()

# -----------------------------  THIRD PLOT: DELAY DURATIONS  ---------------------------------------------------                
        # 
        if params.motor == 6:
            axes = plt.subplot2grid((16,2), (5,0), colspan=2, rowspan=2)
    
            axes.hlines(y=np.nanmax(df.delay_times), xmin=0, xmax=df.trials, linewidth=0.2, color='gray',linestyle = ':')
                
            if params.delay_progression_value == 1:                        
                axes.plot(df.trials,df.delay_times_m)
                axes.plot(df.trials,df.delay_times_h)
                axes.plot(df.trials,df.delay_times_l)
                
                sns.scatterplot(df.trials, df.delay_times, hue=df.hithistory, palette=palette, s=12, ax=axes)
        #
            elif float(params.delay_progression_value) == 0:                    
                sns.scatterplot(df.trials, df.delay_times, hue=df.hithistory, palette=palette, s=12, ax=axes)
                #axes.set_yticks([str(df_ild.delay_times.unique())])
            
            axes.set_ylabel('Delay (ms)')     
            axes.set_xlabel('Trials')          
            axes.set_xlim([1, len(df) + 1])
            axes.set_ylim(-0.1, np.around(np.nanmax(df.delay_times)+0.5,decimals=1)) 
                       
# ____________________ FORTH PLOT: RASTER PLOT ________________________________________________
        print("Let's do the raster")
        # Plotting a square marker for relevant trial conclusion. Had to remove the last trial because of reward list errors. 
        if params.stage_number == 1:
            colors = ['g','r','y','k','grey']
            states = ['Reward_start','Punish_start','AW_start','Miss_start','WrongLick_start']

        else:
            colors = ['g','r','y','k']
            states = ['Reward_start','Punish_start','AW_start','Miss_start']
            
        if params.motor < 6 or params.motor == 6 and params.delay_progression_value == 1 or math.isnan(params.motor):     
            axes1 = plt.subplot2grid((16,2), (6,0), colspan=1, rowspan=7)
            axes2 = plt.subplot2grid((16,2),(13,0), colspan=1, rowspan=2) 
            axes3 = plt.subplot2grid((16,2), (6,1), colspan=1, rowspan=7)
            axes4 = plt.subplot2grid((16,2),(13,1), colspan=1, rowspan=2) 
            axes5 = plt.subplot2grid((16,2), (15,0), colspan=1, rowspan=1)
            axes6 = plt.subplot2grid((16,2), (15,1), colspan=1, rowspan=1)
    
            axes1.set_title('Left trials')
            axes3.set_title('Right trials')
               
            for color, state in zip(colors,states):
                axes1.scatter(df.loc[df.reward_side == 0][state][:-1],range(len(df.index[df.reward_side == 0][:-1])), marker = 's', color=color, alpha = 0.8, zorder=2)
                axes3.scatter(df.loc[df.reward_side == 1][state][:-1],range(len(df.index[df.reward_side == 1][:-1])), marker = 's', color=color, alpha = 0.8, zorder=2)
                  
            col = 0
            truevalue = 0
            colors = [COLORLEFT,COLORRIGHT]
                        
            for columns in df[['L_s','C_s']]:
                j=0
                color = colors[col]
                col+=1
                histogram = []
                for sublist in df.loc[df.reward_side == 0, str(columns)]:
                    axes1.plot(np.array(sublist),np.repeat(j, len(sublist),axis =0), '.', color=color, zorder=1)
                    j+=1                        
                    for item in sublist:
                        histogram.append(item)
                    
                axes2.hist(pd.Series(histogram).dropna(),bins=np.arange(-4, 7, BINSIZE),align = 'right', linewidth=1, edgecolor=color, histtype='step',alpha=0.7, color=color,weights=np.repeat((1/len(left_valid_trials_df))/0.1,len(pd.Series(histogram).dropna()))) 
            
                j=0
                histogram = []
                for sublist in df.loc[df.reward_side == 1, str(columns)]:
                    axes3.plot(np.array(sublist),np.repeat(j, len(sublist),axis =0), '.',color=color, zorder=1)
                    j+=1
                    for item in sublist:
                        histogram.append(item)
#                        if item > 5:
#                            truevalue -= item - df_ild[df_ild.reward_side == 1]['ITI_end'][j]
#                            axes3.plot(truevalue,j+1, '.',color=color, zorder=1)
#                            histogram.append(truevalue)                            
                                          
                axes4.hist(pd.Series(histogram).dropna(),bins=np.arange(-4, 7, BINSIZE),align = 'right', linewidth=1, edgecolor=color, histtype='step',alpha=0.7, color=color,weights=np.repeat((1/len(right_valid_trials_df))/0.1,len(pd.Series(histogram).dropna()))) 

#            axes2.hist(pokes_df_left.loc[(pokes_df_left['reward_side'] == 0)]['L_s'].dropna(),bins=np.arange(-4, 7, BINSIZE),align = 'right', linewidth=1, edgecolor=COLORLEFT, histtype='step', alpha=0.7, color=color, weights=np.repeat((1/len(left_valid_trials_df))/0.1,len(pokes_df_left.loc[(pokes_df_left['reward_side'] == 0)].dropna()))) 
#            axes2.hist(pokes_df_left.loc[(pokes_df_left['reward_side'] == 1)]['L_s'].dropna(),bins=np.arange(-4, 7, BINSIZE),align = 'right', linewidth=1, edgecolor=COLORLEFT, histtype='step', alpha=0.7, color=color, weights=np.repeat((1/len(left_valid_trials_df))/0.1,len(pokes_df_left.loc[(pokes_df_left['reward_side'] == 1)].dropna())))
#
#            axes4.hist(pokes_df_right.loc[(pokes_df_right['reward_side'] == 0)]['C_s'].dropna(),bins=np.arange(-4, 7, BINSIZE),align = 'right', linewidth=1, edgecolor=COLORRIGHT, histtype='step', alpha=0.7, color=color, weights=np.repeat((1/len(right_valid_trials_df))/0.1,len(pokes_df_right.loc[(pokes_df_right['reward_side'] == 0)].dropna()))) 
#            axes4.hist(pokes_df_right.loc[(pokes_df_right['reward_side'] == 1)]['C_s'].dropna(),bins=np.arange(-4, 7, BINSIZE),align = 'right', linewidth=1, edgecolor=COLORRIGHT, histtype='step', alpha=0.7, color=color, weights=np.repeat((1/len(right_valid_trials_df))/0.1,len(pokes_df_right.loc[(pokes_df_right['reward_side'] == 1)].dropna()))) 
       
            for side in [0,1]:
                if side == 0:
                    axes = axes1
                else:
                    axes = axes3
                    
                for i, j in zip(range(len(df.reward_side == side)),df.loc[(df['reward_side'] == side)]['trials']):
                     try:
                         axes.barh(i, df['TimeOut_end'][j]-df['TimeOut_start'][j],left = df['TimeOut_start'].tolist()[j], color='black', alpha=0.2) 
                     except:
                         pass
                     
                     try:
                         axes.barh(i, df['Delay_end'].tolist()[j]-df['Delay_start'].tolist()[j],left = df['Delay_start'].tolist()[j], color='black', alpha=0.2) 
                     except:
                         pass
                     
                     try:
                         axes.barh(i, -df['Fixation_start'].tolist()[j][0], left = df['Fixation_start'].tolist()[j][0], color='grey', alpha=0.3) 
                     except:
                         pass
                     
                     if params.motor <= 6 or math.isnan(params.motor):
                         try:
                             axes.barh(i, df['ResponseWindow_end'].tolist()[j]-df['StimulusDuration_start'].tolist()[j],left = df['StimulusDuration_start'].tolist()[j],color='blue', alpha=0.2) 
                         except:
                             axes.barh(i, df['ResponseWindow_end'][j]-df['StimulusDuration_start'][j],left = df['StimulusDuration_start'][j],color='blue', alpha=0.2) 
                             
                     else:
                         axes.barh(i, df['StimulusDuration_end'][j]-df['StimulusDuration_start'].tolist()[j],left = df['StimulusDuration_start'].tolist()[j],color='blue', alpha=0.3) 
                         axes.barh(i, df['ResponseWindow_end'][i]-df['Delay_end'].tolist()[i],left = df['Delay_end'].tolist()[j],color='blue', alpha=0.2) 
                                        
            # Adjust the axis
            axes1.set_ylabel('Trials (nยบ)')
            axes2.set_ylabel('Lick rate\n (licks/s)')
            axes4.set_ylabel('Lick rate\n (licks/s)')
            axes5.set_xlabel('Time (s)')
            axes6.set_xlabel('Time (s)')
    
    #        axes5.axvspan(df_ild['StimulusTrigger_end'][1],df_ild['StimulusDuration_end'][1],color ='c',alpha=0.3)
    #        axes7.axvspan(df_ild['StimulusTrigger_end'][1],df_ild['StimulusDuration_end'][1],color ='c',alpha=0.3)
    
            ITI_DURATION = df['ITI_end'].median()-df['ITI_start'].median()            
            limitx = [-params.fixation-2,ITI_DURATION+3]
            
            axes1.set_xlim(limitx)
            axes2.set_xlim(limitx)
            axes3.set_xlim(limitx)
            axes4.set_xlim(limitx)
            
# ---------------- Measure first lick within the response window ------------------------------------------------------------------------------        
            
            i=0
            histogramLL = []
            histogramRL = []        
            histogramLR = []
            histogramRR = []  

            try:            
                histogramLL = pokes_df_left.loc[(pokes_df_left['L_s'] > pokes_df_left['StimulusTrigger_start']) & (pokes_df_left['L_s'] < pokes_df_left['ResponseWindow_end'] + 0.1) & (pokes_df_left['reward_side'] == 0)]['L_s']            
                histogramLR = pokes_df_left.loc[(pokes_df_left['L_s'] > pokes_df_left['StimulusTrigger_start']) & (pokes_df_left['L_s'] < pokes_df_left['ResponseWindow_end'] + 0.1 ) & (pokes_df_left['reward_side'] == 1)]['L_s'] 
                
                histogramRL = pokes_df_right.loc[(pokes_df_right['C_s'] > pokes_df_right['StimulusTrigger_start']) & (pokes_df_right['C_s'] < pokes_df_right['ResponseWindow_end'] + 0.1 ) & (pokes_df_right['reward_side'] == 0)]['C_s']            
                histogramRR = pokes_df_right.loc[(pokes_df_right['C_s'] > pokes_df_right['StimulusTrigger_start']) & (pokes_df_right['C_s'] < pokes_df_right['ResponseWindow_end'] + 0.1 ) & (pokes_df_right['reward_side'] == 1)]['C_s'] 

            except:
                histogramLL = pokes_df_left.loc[(pokes_df_left['L_s'] > pokes_df_left['StimulusTrigger_start']) & (pokes_df_left['L_s'] < float(pokes_df_left['ResponseWindow_end'][0]) + 0.1) & (pokes_df_left['reward_side'] == 0)]['L_s']            
                histogramLR = pokes_df_left.loc[(pokes_df_left['L_s'] > pokes_df_left['StimulusTrigger_start']) & (pokes_df_left['L_s'] < float(pokes_df_left['ResponseWindow_end'][0]) + 0.1 ) & (pokes_df_left['reward_side'] == 1)]['L_s'] 
                
                histogramRL = pokes_df_right.loc[(pokes_df_right['C_s'] > pokes_df_right['StimulusTrigger_start']) & (pokes_df_right['C_s'] < float(pokes_df_right['ResponseWindow_end'][0])+ 0.1 ) & (pokes_df_right['reward_side'] == 0)]['C_s']            
                histogramRR = pokes_df_right.loc[(pokes_df_right['C_s'] > pokes_df_right['StimulusTrigger_start']) & (pokes_df_right['C_s'] < float(pokes_df_right['ResponseWindow_end'][0]) + 0.1 ) & (pokes_df_right['reward_side'] == 1)]['C_s'] 

            #Plot histogram for left trials first licks
            axes5.hist(histogramLL.dropna(),bins=np.arange(df['StimulusDuration_start'][0], df['StimulusDuration_start'][0]+2, 0.05),align = 'mid', linewidth=1, edgecolor=COLORLEFT, histtype='step',
                       alpha=0.8,  weights=np.repeat((1/len(left_trials_df))/0.05,len(pd.Series(histogramLL).dropna()))) 
            
            axes5.hist(histogramRL.dropna(),bins=np.arange(df['StimulusDuration_start'][0], df['StimulusDuration_start'][0]+2, 0.05),align = 'mid', linewidth=1, edgecolor=COLORRIGHT, histtype='step', 
                       alpha=0.8,  weights=np.repeat((1/len(left_trials_df))/0.05,len(pd.Series(histogramRL).dropna())))


            #Plot histogram of first lick for right trials
            axes6.hist(histogramLR.dropna(),bins=np.arange(df['StimulusDuration_start'][0], df['StimulusDuration_start'][0]+2, 0.05),align = 'mid', linewidth=1, edgecolor=COLORLEFT, histtype='step',
                       alpha=0.8,  weights=np.repeat((1/len(right_trials_df))/0.05,len(histogramLR.dropna())))
            
            axes6.hist(histogramRR.dropna(),bins=np.arange(df['StimulusDuration_start'][0], df['StimulusDuration_start'][0]+2, 0.05),align = 'mid', linewidth=1, edgecolor=COLORRIGHT, histtype='step', 
                       alpha=0.8,  weights=np.repeat((1/len(right_trials_df))/0.05,len(histogramRR.dropna())))
                                                             
            axes5.set_xlim(0,2)
            axes6.set_xlim(0,2)

            axes1.axvspan(df['StimulusTrigger_end'].mean(),df['StimulusDuration_end'].mean(),color ='c',alpha=0.3)
            axes3.axvspan(df['StimulusTrigger_end'].mean(),df['StimulusDuration_end'].mean(),color ='c',alpha=0.3)                      
            axes2.axvspan(df['StimulusTrigger_end'].mean(),df['StimulusDuration_end'].mean(),color ='c',alpha=0.3)
            axes4.axvspan(df['StimulusTrigger_end'].mean(),df['StimulusDuration_end'].mean(),color ='c',alpha=0.3)
            axes5.axvspan(df['StimulusTrigger_end'].mean(),df['StimulusDuration_end'].mean(),color ='c',alpha=0.3)
            axes6.axvspan(df['StimulusTrigger_end'].mean(),df['StimulusDuration_end'].mean(),color ='c',alpha=0.3)
                                           
        else:
            
            # Prepares the grid for the plots            
            axes1 = plt.subplot2grid((16,2), (8,0), colspan=1, rowspan=2)
            axes2 = plt.subplot2grid((16,2),(8,1), colspan=1, rowspan=2) 
            axes3 = plt.subplot2grid((16,2), (10,0), colspan=1, rowspan=2)
            axes4 = plt.subplot2grid((16,2),(10,1), colspan=1, rowspan=2) 
            axes5 = plt.subplot2grid((16,2), (12,0), colspan=1, rowspan=2)
            axes6 = plt.subplot2grid((16,2), (12,1), colspan=1, rowspan=2)
            axes7 = plt.subplot2grid((16,2), (14,0), colspan=1, rowspan=2)
            axes8 = plt.subplot2grid((16,2), (14,1), colspan=1, rowspan=2)
                                                 
            for color, state in zip(colors,states):
                axes1.scatter(df_l.loc[df_l.reward_side == 0][state].tolist(),range(len(df_l.index[df_l.reward_side == 0])), marker = 's', color=color, alpha = 0.8, zorder=2)
                axes2.scatter(df_l.loc[df_l.reward_side == 1][state].tolist(),range(len(df_l.index[df_l.reward_side == 1])), marker = 's', color=color, alpha = 0.8, zorder=2)
                axes3.scatter(df_m.loc[df_m.reward_side == 0][state].tolist(),range(len(df_m.index[df_m.reward_side == 0])), marker = 's', color=color, alpha = 0.8, zorder=2)
                axes4.scatter(df_m.loc[df_m.reward_side == 1][state].tolist(),range(len(df_m.index[df_m.reward_side == 1])), marker = 's', color=color, alpha = 0.8, zorder=2)
                axes5.scatter(df_h.loc[df_h.reward_side == 0][state].tolist(),range(len(df_h.index[df_h.reward_side == 0])), marker = 's', color=color, alpha = 0.8, zorder=2)
                axes6.scatter(df_h.loc[df_h.reward_side == 1][state].tolist(),range(len(df_h.index[df_h.reward_side == 1])), marker = 's', color=color, alpha = 0.8, zorder=2)
            
            # Prints the states in box types of trials.      
            for axes,df,line_type in zip([axes1,axes3,axes5,],[df_l,df_m,df_h],[':','-','--']):
                for i in range(len(df.loc[df.reward_side == 0])): 
                    axes.barh(i, df.loc[df.reward_side == 0]['StimulusDuration_end'].tolist()[i]-df.loc[df.reward_side == 0]['StimulusDuration_start'].tolist()[i],left = df.loc[df.reward_side == 0]['StimulusDuration_start'].tolist()[i],color='blue', alpha=0.6) 
                    axes.barh(i, float(df.loc[df.reward_side == 0]['TimeOut_end'].tolist()[i])-df.loc[df.reward_side == 0]['TimeOut_start'].tolist()[i],left = df.loc[df.reward_side == 0]['TimeOut_start'].tolist()[i],color='yellow', alpha=0.3) 
                    axes.barh(i, float(df.loc[df.reward_side == 0]['ResponseWindow_start'].tolist()[i])-df.loc[df.reward_side == 0]['Delay_start'].tolist()[i],left = df.loc[df.reward_side == 0]['Delay_start'].tolist()[i],color='grey', alpha=0.3) 
            
                #Change axis
                if axes == axes1:
                    axes = axes2
                elif axes == axes3:
                    axes = axes4
                elif axes == axes5:
                    axes = axes6        
                    
                for i in range(len(df.loc[df.reward_side == 1])): 
                    axes.barh(i, df.loc[df.reward_side == 1]['StimulusDuration_end'].tolist()[i]-df.loc[df.reward_side == 1]['StimulusDuration_start'].tolist()[i],left = df.loc[df.reward_side == 1]['StimulusDuration_start'].tolist()[i],color='blue', alpha=0.6) 
                    axes.barh(i, float(df.loc[df.reward_side == 1]['TimeOut_end'].tolist()[i])-df.loc[df.reward_side == 1]['TimeOut_start'].tolist()[i],left = df.loc[df.reward_side == 1]['TimeOut_start'].tolist()[i],color='yellow', alpha=0.3) 
                    axes.barh(i, float(df.loc[df.reward_side == 1]['ResponseWindow_start'].tolist()[i])-df.loc[df.reward_side == 1]['Delay_start'].tolist()[i],left = df.loc[df.reward_side == 1]['Delay_start'].tolist()[i],color='grey', alpha=0.3)
            
            for axes,df,line_type in zip([axes1,axes3,axes5,],[df_l,df_m,df_h],[':','-','--']):    
                col = 0
                colors = [COLORLEFT,COLORRIGHT]
                for columns in df[['L_s','C_s']]:
                    #Change axis
                    if axes == axes2:
                        axes = axes1
                    elif axes == axes4:
                        axes = axes3
                    elif axes == axes6:
                        axes = axes5  
                        
                    j=0
                    color = colors[col]
                    col+=1
                    histogram = []
                    for sublist in df.loc[df.reward_side == 0, str(columns)]:
                        axes.plot(np.array(sublist),np.repeat(j, len(sublist),axis =0), '.', color=color, zorder=1)
                        j+=1                        
                        for item in sublist:
                            histogram.append(item)
            
                    axes7.hist(pd.Series(histogram).dropna(),bins=np.arange(-4, 7, BINSIZE),align = 'right', linestyle = line_type,edgecolor=color, histtype='step',alpha=0.7, color=color,weights=np.repeat((1/len(left_trials_df))/0.1,len(pd.Series(histogram).dropna()))) 
            
                    #Change axis
                    if axes == axes1:
                        axes = axes2
                    elif axes == axes3:
                        axes = axes4
                    elif axes == axes5:
                        axes = axes6     
                    
                    j=0
                    histogram = []
                        
                    for sublist in df.loc[df.reward_side == 1, str(columns)]:
                        axes.plot(np.array(sublist),np.repeat(j, len(sublist),axis =0), '.',color=color, zorder=1)
                        j+=1
                        for item in sublist:
                            histogram.append(item)
            
                    axes8.hist(pd.Series(histogram).dropna(),bins=np.arange(-4, 7, BINSIZE),align = 'right', linestyle = line_type, edgecolor=color, histtype='step',alpha=0.7, color=color,weights=np.repeat((1/len(right_trials_df))/0.1,len(pd.Series(histogram).dropna()))) 
            
                axes1.set_title('Left trials')
                axes2.set_title('Right trials')
                axes1.set_ylabel('Delay low')
                axes3.set_ylabel('Delay mid')
                axes5.set_ylabel('Delay high')
                axes7.set_xlim(1,6)
                axes8.set_xlim(1,6)                
                
        # Custom legend:
        legend_elements = [Line2D([0], [0], color=COLORLEFT, label='Left'),
        Line2D([0], [0], color=COLORRIGHT, label='Right'),
        Line2D([0], [0], color='cyan', label='Stimulus', alpha = 0.3),
        Line2D([0], [0], color='gray', label='Fixation', alpha = 0.3),
        Line2D([0], [0], color='green', label='Reward', marker ='s', alpha = 1),
        Line2D([0], [0], color='red', label='Punish', marker ='s', alpha = 1),
        Line2D([0], [0], color='black', label='Miss', marker ='s', alpha = 1)]
        leg = plt.legend(loc="lower right", handles=legend_elements, ncol=1, prop={'size': 8})
        leg.get_frame().set_alpha(0.5)         

        pdf.savefig(plt.gcf(), transparent=True)  #Saves the current figure into a pdf page
        plt.close()          
# ---------------------- SECOND PAGE   ---------------------------------------------------
#_______________________ DELAY DETAILS ___________________________________________________
        if params.stage_number >= 3 and params.motor > 5:    
            fig = plt.figure(figsize=(8.5, 13.7))             
            #USEFUL VALUES
            acc_low = df_l['hithistory'].mean()
            acc_medium = df_m['hithistory'].mean()
            acc_high = df_h['hithistory'].mean()
            
            miss_low = df_l['misshistory'].mean()
            miss_medium = df_m['misshistory'].mean()
            miss_high = df_h['misshistory'].mean()            

            try:
                df = pd.read_csv(trials_path, sep=';')
            except FileNotFoundError:
                logger.critical("Error: Reading trials CSV file. Exiting...")
                raise
            
            try:
                params = pd.read_csv(params_path, sep=';')
            except FileNotFoundError:
                logger.critical("Error: Reading params CSV file. Exiting...")
                raise
            
            df.index = df.trials
            params = params.iloc[0]  
                   
            axes = plt.subplot2grid((16,2), (0,0), colspan=1, rowspan=3)  
            axes1 = plt.subplot2grid((16,2), (0,1), colspan=1, rowspan=3)  
                          
            if params.delay_progression_value == 1:
                sns.pointplot('delay_types', 'hithistory', data=left_valid_trials_df.loc[(df['probabilities'] =='Random')], order=['delay_l','delay_m','delay_h'], ax=axes,color = COLORLEFT)
                sns.pointplot('delay_types', 'hithistory', data=right_valid_trials_df.loc[(df['probabilities'] =='Random')], ax=axes,color = COLORRIGHT, order=['delay_l','delay_m','delay_h'])
                sns.pointplot('delay_types', 'hithistory', data=valid_trials_df.loc[(df['probabilities'] =='Random')], ax=axes1, order=['delay_l','delay_m','delay_h'], alpha=0.5,color = 'black')
                sns.pointplot('delay_types', 'hithistory', data=df, ax=axes1, order=['delay_l','delay_m','delay_h'], color='grey')
               
            else:
                sns.pointplot('delay_times', 'hithistory', data=left_valid_trials_df.loc[(df['probabilities'] =='Random')], ax=axes, color = COLORLEFT)
                sns.pointplot('delay_times', 'hithistory', data=right_valid_trials_df.loc[(df['probabilities'] =='Random')], ax=axes, color= COLORRIGHT)
                sns.pointplot('delay_times', 'hithistory', data=valid_trials_df.loc[(valid_trials_df['probabilities'] =='Random')], ax=axes1,color = 'black')
                sns.pointplot('delay_times', 'hithistory', data=df, ax=axes1,color = 'grey')
                        
            legend_elements = [Line2D([0], [0], color=COLORLEFT, label='Left'),
            Line2D([0], [0], color=COLORRIGHT, label='Right'),
            Line2D([0], [0], color='black', label='Valids Random'),
            Line2D([0], [0], color='grey', label='Total')]
            leg = plt.legend(loc="lower right", handles=legend_elements, ncol=2, prop={'size': 8})
            leg.get_frame().set_alpha(0.5)  
            
            axes.set_xlabel('Delay (s)')
            axes1.set_xlabel('Delay (s)')
            axes.set_ylabel('Fraction of correct')  
            axes1.set_ylabel('Fraction of correct')  
            axes.set_ylim(0.5,1)
            axes1.set_ylim(0.5,1)

#____________________________ DELAY ACCURACY RUNNING WINDOW ___________________________
                   
            # Plot accuracy according to type of delay
            axes = plt.subplot2grid((16,2), (4,0), colspan=2, rowspan=4)
        
            ra_delay_l = Utils.compute_window(df_l.hithistory, 20)
            ra_delay_m = Utils.compute_window(df_m.hithistory, 20)
            ra_delay_h = Utils.compute_window(df_h.hithistory, 20)
                
            # Plot the percentage of misses in a running window of 20. 
            axes.plot(df_h.trials, ra_delay_h, marker = 'o', color=DELAY_H)
            axes.plot(df_m.trials, ra_delay_m, marker = 'o', color=DELAY_M)
            axes.plot(df_l.trials, ra_delay_l, marker = 'o', color=DELAY_L)
            
            axes.set_xlim([1, number_of_trials + 1])
            axes.set_ylim([0,1.1])
            axes.set_ylabel('Categorization'+'\n'+ 'Accuracy (%)')   
            axes.set_yticklabels(['0', '', '', '', '', '50', '', '', '', '', '100'])
            
            axes.hlines(xmin=0, xmax=number_of_trials + 1, y=0.5, linewidth=0.2, color='gray',linestyle = ':')
        
            axes.set_xlabel('Trials') 
    
            legend_elements = [
            Line2D([0], [0], color=DELAY_L, label='Delay short',  alpha = 1),
            Line2D([0], [0], color=DELAY_M, label='Delay medium',  alpha = 1),
            Line2D([0], [0], color = DELAY_H, label='Delay long',  alpha = 1)]
            leg = plt.legend(handles=legend_elements, ncol=1, prop={'size': 9})
            leg.get_frame().set_alpha(0.5)  
            
            pdf.savefig()
            plt.close()        
#______________________ FIFTH PLOT: PSYCOMETRIC _______________________________________________
        
#        axes = plt.subplot2grid((16,4), (5,2), colspan=2, rowspan=4)
#    
#        coherences = valid_trials_df.coherences
#        response = valid_trials_df.response
#    
#        # we need coherences from -1 to 1, response from 0 to 1
#        coherences = coherences * 2 - 1
#    
#        psych_curve = Utils.compute_psych_curve(coherences, response)
#    
#        axes.plot([0, 0], [0, 1], 'k-', lw=1, linestyle=':')
#        axes.plot([-1, 1], [0.5, 0.5], 'k-', lw=1, linestyle=':')
#    
#        axes.errorbar(psych_curve.xdata, psych_curve.ydata,
#                      yerr=psych_curve.fit_error, fmt='ro', elinewidth=1, markersize=3)
#        
#        axes.plot(np.linspace(-1, 1, 30), psych_curve.fit, color='black', linewidth=1)
#        axes.set_yticks(np.arange(0, 1.1, step=0.1))
#        axes.set_xlabel('Evidence')
#        axes.set_ylabel('Probability of right')
#        axes.set_xlim([-1.05, 1.05])
#        axes.yaxis.set_tick_params(labelsize=9)
#        axes.xaxis.set_tick_params(labelsize=9)
#        axes.tick_params(labelsize=9)
#        axes.annotate(str(round(psych_curve.ydata[0], 2)), xy=(psych_curve.xdata[0], psych_curve.ydata[0]),
#                      xytext=(psych_curve.xdata[0] - 0.03, psych_curve.ydata[0] + 0.05), fontsize=8)
#        axes.annotate(str(round(psych_curve.ydata[-1], 2)), xy=(psych_curve.xdata[-1], psych_curve.ydata[-1]),
#                      xytext=(psych_curve.xdata[-1] - 0.1, psych_curve.ydata[-1] - 0.08), fontsize=8)
#        s, b, lr_r, lr_l = psych_curve.params
#    
#        axes.annotate("S = " + str(round(s, 2)) + "\n" + "B = " +
#                      str(round(b, 2)) + "\n" + "LR_L = " +
#                      str(round(lr_r, 2)) + "\n" + "LR_R = " +
#                      str(round(lr_l, 2)), xy=(0, 0), xytext=(-1, 0.85), fontsize=7)
            
            
#        df_ild = Utils.unnesting(df_ild, ['Fixation_start', 'Fixation_end'])
#        df_ild = Utils.unnesting(df_ild, ['ResponseWindow_start', 'ResponseWindow_end'])
#        
#        df_ild['StimulusDuration'] = df_ild['StimulusDuration_end'] - df_ild['StimulusDuration_start']
#        if params.fixation != 0:
#            df_ild['Fixation'] = df_ild['Fixation_end'] - df_ild['Fixation_start']
#        else:
#            df_ild['Fixation'] = 0
#       
#        df_ild['ResponseWindow'] = df_ild['ResponseWindow_end'] - df_ild['ResponseWindow_start']
#        df_ild['ITI'] = df_ild['ITI_end'] - df_ild['ITI_start']
#        
#        if params.stage_number == 3:
#            df_ild['Delay'] = df_ild['Delay_end'] - df_ild['Delay_start']
#        else:
#            df_ild['Delay'] = 0
            
        
        params['performance'] = per_performance
        params['accuracy'] = per_accuracy
        params['miss'] = per_invalids
        params['miss_left'] = m_left
        params['miss_right'] = m_right
        params['accuracy_left'] = acc_left
        params['accuracy_right'] = acc_right
        
        if params.stage_number >= 3 and params.motor > 5:
            params['miss_low'] = miss_low
            params['miss_medium'] = miss_medium
            params['miss_high'] = miss_high             
            
            params['accuracy_low'] = acc_low
            params['accuracy_medium'] = acc_medium
            params['accuracy_high'] = acc_high      
        
        return params
        

        