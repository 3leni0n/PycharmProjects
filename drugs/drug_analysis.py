import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.stats.anova import AnovaRM
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Mel's code snippet for poster
sns.set_theme()
sns.set_style('white')
sns.set_style('ticks')
sns.set_context('poster')
# sns.despine()

# Based  on MPL's Grouped bar chart with labels example
# (https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html)


# def drug_vs_saline_bar_chart(ylabel, path='/home/alexis/PycharmProjects/intersession/337_intersession.csv', nrows=1, ncols=1, index=1):
#     """
#     :param ylabel: DataFrame columm label of the intersession .csv with the variable to plot (str)
#     :return: Grouped bar chart for the variable of interest between saline and drug sessions
#     """
#
#     # path = '/home/alexis/PycharmProjects/intersession/332_intersession.csv'
#     df = pd.read_csv(path)
#     subject = df.Subject.unique()[0]
#     drug_session_index = df.Drug[df.Drug == 'MK801'].index
#     saline_pre_drug_session_index = drug_session_index - 1
#     assert df.Drug.iloc[saline_pre_drug_session_index].unique()[0] == 'saline'
#
#     labels = list(df.Dates.iloc[drug_session_index])
#     saline_y = df[ylabel].iloc[saline_pre_drug_session_index]
#     drug_y = df[ylabel].iloc[drug_session_index]
#
#     x = np.arange(len(labels))  # the label locations
#     width = 0.35  # the width of the bars
#
#     # fig, ax = plt.subplots()
#     fig = plt.figure()
#     ax = fig.add_subplot(nrows, ncols, index)
#
#
#     rects1 = ax.bar(x - width / 2, saline_y, width, label='Saline', color='tab:gray')
#     rects2 = ax.bar(x + width / 2, drug_y, width, label='MK801', color='tab:pink')
#
#     # Add some text for labels, title and custom x-axis tick labels, etc.
#     ax.set_xlabel('Session')
#     ax.set_ylabel(ylabel)
#     # ax.set_title(f'Mouse {subject}: {ylabel} by sessions and injection')
#     ax.set_xticks(x)
#
#     ax.legend()
#
#     ax.bar_label(rects1, padding=3)
#     ax.bar_label(rects2, padding=3)
#
#     fig.tight_layout()
#     plt.show()
#
#
# ylabel = 'Accuracy'
# drug_vs_saline_bar_chart(ylabel, nrows=2, ncols=2, index=1)
# ylabel = 'Responses'
# drug_vs_saline_bar_chart(ylabel, nrows=2, ncols=2, index=2)
# ylabel = 'LateralBias'
# drug_vs_saline_bar_chart(ylabel, nrows=2, ncols=2, index=3)
# ylabel = 'CorrRepBias'
# drug_vs_saline_bar_chart(ylabel, nrows=2, ncols=2, index=4)
# plt.suptitle('337')
# plt.savefig('/home/alexis/Escritorio/337.png')


# def pussy_plot(x, y):
df = pd.read_csv('/home/alexis/PycharmProjects/intersession/all_intersessions.csv')
df = df[df.Drug.notna()]

# Find outliers based on accuracy and drop them
min_saline = df[df.Drug == 'saline'].Accuracy.min()
min_saline_index = df[df.Accuracy == min_saline].index[0]
min_saline_date = df[df.Accuracy == min_saline].Dates
min_drug = df[df.Drug == 'MK801'].Accuracy.min()
min_drug_index = df[df.Accuracy == min_drug].index[0]
min_drug_date = df[df.Accuracy == min_drug].Dates
min_rest = df[df.Drug == 'rest'].Accuracy.min()
min_rest_index = df[df.Accuracy == min_rest].index[0]
min_rest_date = df[df.Accuracy == min_rest].Dates
df.drop(index=[min_saline_index, min_drug_index, min_rest_index], inplace=True)
# Find outlier based on misses and drop it
min_rest2 = df[df.Drug == 'rest'].Misses.max()
min_rest_index2 = df[df.Misses == min_rest2].index[0]
min_rest_date2 = df[df.Misses == min_rest2].Dates
min_saline2 = df[df.Drug == 'saline'].MissRate.max()
min_saline_index2 = df[df.MissRate == min_saline2].index[0]
min_saline_date2 = df[df.MissRate == min_saline2].Dates
df.drop(index=[min_rest_index2, min_saline_index2], inplace=True)


def cat_drug_2_drug_binary():
    # Brake down categorical drug data into binary variables
    drug = df.Drug.to_list()
    saline = []
    MK801 = []
    rest = []
    for i in range(len(drug)):
        print(drug[i])

        if drug[i] == 'saline':
            saline.append(1)
            MK801.append(0)
            rest.append(0)

        elif drug[i] == 'MK801':
            MK801.append(1)
            saline.append(0)
            rest.append(0)

        elif drug[i] == 'rest':
            rest.append(1)
            saline.append(0)
            MK801.append(0)

    df['Saline'] = saline
    df['MK801'] = MK801
    df['Rest'] = rest


cat_drug_2_drug_binary()


x = 'Drug'
y = 'CorrRepBias'
hue = 'Subject'
order = ['saline', 'MK801', 'rest']
palette = {'saline': 'tab:gray', 'MK801': 'tab:pink', 'rest': 'tab:gray'}  # Gotta be pink

# # Pinky palettes (https://medium.com/@morganjonesartist/color-guide-to-seaborn-palettes-da849406d44f)
# palette="PuRd"
# palette="RdPu"
# palette="spring"
# palette="spring_r"

means = df.groupby(x)[y].mean().reindex(order)
sems = df.groupby(x)[y].sem().reindex(order)
stds = df.groupby(x)[y].std().reindex(order)

# # Violin plot with swarmplot inside
# palette = {'saline': 'tab:gray', 'MK801': 'tab:pink', 'rest': 'tab:gray'}
# g = sns.catplot(x=x, y=y, order=order, kind='violin', inner=None, palette=palette, data=df)
# sns.swarmplot(x=x, y=y, order=order, color='k', data=df, ax=g.ax)
# # sns.scatterplot(x=x, y=y, hue=hue, style=hue, hue_order=order, data=df)  # For different markers
# # sns.pointplot(x=x, y=y, order=order, linestyles='', data=df)  # For CI or std, not sem
# g.set(xlabel=None)

# Box plot with swarmplot inside and hue by subject
palette = {'saline': 'tab:gray', 'MK801': 'tab:pink', 'rest': 'tab:gray'}
g = sns.catplot(x=x, y=y, order=order, kind='box', palette=palette, data=df)
sns.swarmplot(x=x, y=y, order=order, color='k', data=df, ax=g.ax)
# sns.scatterplot(x=x, y=y, hue=hue, style=hue, hue_order=order, data=df)  # For different markers
# sns.pointplot(x=x, y=y, order=order, linestyles='', data=df)  # For CI or std, not sem
g.set(xlabel=None)

# Add mean and sem errorbars
plt.scatter(x=range(len(means)), y=means, c='none', edgecolors='w', zorder=5, alpha=0.75)
plt.errorbar(x=range(len(means)), y=means, yerr=sems, c='w', fmt='none', zorder=5, alpha=0.75)

# # Repeated Measures ANOVA
# # https://www.statsmodels.org/dev/generated/statsmodels.stats.anova.AnovaRM.html
# results = AnovaRM(data=df, depvar='Accuracy', subject='Subject', within=['Drug'], aggregate_func='mean')
# fit = results.fit()
# print(fit)

# Linear Mixed Effects Models
# https://www.statsmodels.org/stable/mixed_linear.html
# md = smf.mixedlm("Accuracy ~ Drug", df, groups=df.Subject)
# mdf = md.fit()
# print(mdf.summary())

md = smf.mixedlm("CorrRepBias ~ MK801 + Saline", df, groups=df.Subject)
mdf = md.fit()
print(mdf.summary())

# md = smf.mixedlm("Accuracy ~ MK801 + Rest", df, groups=df.Subject)
# mdf = md.fit()
# print(mdf.summary())
#
# md = smf.mixedlm("Accuracy ~ Saline + Rest", df, groups=df.Subject)
# mdf = md.fit()
# print(mdf.summary())

# Stats annotation
for i in range(len(df.Drug.unique()) - 1):

    color = 'k'

    if i == 0:  # Saline, 1st variable in the pussy plot
        p = mdf.pvalues[2]
        if p <= 0.0001:
            symbol = '****'
        elif p <= 0.001:
            symbol = '***'
        elif p <= 0.01:
            symbol = '**'
        elif p <= 0.05:
            symbol = '*'
        elif p > 0.05:
            symbol = 'ns'
        x1, x2 = 0, 2  # x coordinates of the 2 variables to compare
        y1 = plt.gca().get_ylim()[1]  # Upper ylim
        y2 = plt.gca().get_ylim()[1] * 1 / 100  # 1% of the upper ylim
        plt.plot([x1, x1, x2, x2], [y1, y1 + y2, y1 + y2, y1], lw=1.5, c=color)
        plt.text((x1 + x2) * .5, y1 + y2, symbol, ha='center', va='bottom', c=color)

    else:  # Rest, 1st variable in the pussy plot
        p = mdf.pvalues[1]
        if p <= 0.0001:
            symbol = '****'
        elif p <= 0.001:
            symbol = '***'
        elif p <= 0.01:
            symbol = '**'
        elif p <= 0.05:
            symbol = '*'
        elif p > 0.05:
            symbol = 'ns'
        x1, x2 = 1, 2  # x coordinates of the 2 variables to compare
        y1 = plt.gca().get_ylim()[1]  # Upper ylim
        y2 = plt.gca().get_ylim()[1] * 1 / 100  # 1% of the upper ylim
        y3 = plt.gca().get_ylim()[1] * 5 / 100  # 5% of the upper ylim, spacing between significant bars
        plt.plot([x1, x1, x2, x2], [y1 + y3, y1 + y2 + y3, y1 + y2 + y3, y1 + y3], lw=1.5, c=color)
        plt.text((x1 + x2) * .5, y1 + y3, symbol, ha='center', va='bottom', c=color)

# plt.ylim(plt.gca().get_ylim()[0], plt.gca().get_ylim()[1] + plt.gca().get_ylim()[1] * 5 / 100)  # Increase ylim 5%
# plt.title(y + ' vs ' + x)
# plt.savefig('/home/alexis/Escritorio/' + y + ' vs ' + x + '_violin_plot_intercept=rest.png')
plt.savefig('/home/alexis/Escritorio/' + y + ' vs ' + x + '_box_plot_intercept=rest.svg', transparent=True)

########################################################################################################################
########################################################################################################################

# using the variable axs for multiple Axes

y_list_right = ['SensitivityPCRight', 'BiasPCRight', 'LapseLeft', 'LapseRight']
y_list_rep = ['SensitivityPCRep', 'BiasPCRep', 'LapseAlt', 'LapseRep']

x = 'Drug'
y = 'SensitivityPCRep'
hue = 'Subject'
order = ['saline', 'MK801', 'rest']
palette = {'saline': 'tab:gray', 'MK801': 'tab:pink', 'rest': 'tab:gray'}  # Gotta be pink

means = df.groupby(x)[y].mean().reindex(order)
sems = df.groupby(x)[y].sem().reindex(order)
stds = df.groupby(x)[y].std().reindex(order)

# fig, axs = plt.subplots(2, 2)
# sns.boxplot(x=x, y=y, order=order, palette=palette, data=df, ax=axs[0, 0])
# sns.swarmplot(x=x, y=y, order=order, color='k', data=df, ax=axs[0, 0])

plt.figure(constrained_layout=True)

g = sns.catplot(x=x, y=y, order=order, kind='box', palette=palette, data=df)
sns.swarmplot(x=x, y=y, order=order, color='k', data=df, ax=g.ax)
g.set(xlabel=None)

# Add mean and sem errorbars
plt.scatter(x=range(len(means)), y=means, c='none', edgecolors='w', zorder=5, alpha=0.75)
plt.errorbar(x=range(len(means)), y=means, yerr=sems, c='w', fmt='none', zorder=5, alpha=0.75)

md = smf.mixedlm("SensitivityPCRep ~ MK801 + Saline", df, groups=df.Subject)
mdf = md.fit()
print(mdf.summary())

# Stats annotation
for i in range(len(df.Drug.unique()) - 1):

    color = 'k'

    if i == 0:  # Saline, 1st variable in the pussy plot
        p = mdf.pvalues[2]
        if p <= 0.0001:
            symbol = '****'
        elif p <= 0.001:
            symbol = '***'
        elif p <= 0.01:
            symbol = '**'
        elif p <= 0.05:
            symbol = '*'
        elif p > 0.05:
            symbol = 'ns'
        x1, x2 = 0, 2  # x coordinates of the 2 variables to compare
        y1 = plt.gca().get_ylim()[1]  # Upper ylim
        y2 = plt.gca().get_ylim()[1] * 1 / 100  # 1% of the upper ylim
        plt.plot([x1, x1, x2, x2], [y1, y1 + y2, y1 + y2, y1], lw=1.5, c=color)
        plt.text((x1 + x2) * .5, y1 + y2, symbol, ha='center', va='bottom', c=color)

    else:  # Rest, 1st variable in the pussy plot
        p = mdf.pvalues[1]
        if p <= 0.0001:
            symbol = '****'
        elif p <= 0.001:
            symbol = '***'
        elif p <= 0.01:
            symbol = '**'
        elif p <= 0.05:
            symbol = '*'
        elif p > 0.05:
            symbol = 'ns'
        x1, x2 = 1, 2  # x coordinates of the 2 variables to compare
        y1 = plt.gca().get_ylim()[1]  # Upper ylim
        y2 = plt.gca().get_ylim()[1] * 1 / 100  # 1% of the upper ylim
        y3 = plt.gca().get_ylim()[1] * 5 / 100  # 5% of the upper ylim, spacing between significant bars
        plt.plot([x1, x1, x2, x2], [y1 + y3, y1 + y2 + y3, y1 + y2 + y3, y1 + y3], lw=1.5, c=color)
        plt.text((x1 + x2) * .5, y1 + y3, symbol, ha='center', va='bottom', c=color)

plt.savefig('/home/alexis/Escritorio/' + y + ' vs ' + x + '_box_plot_intercept=rest.svg', transparent=True)

########################################################################################################################

df_saline = df[df.Drug == 'saline']
df_MK801 = df[df.Drug == 'MK801']
df_rest = df[df.Drug == 'rest']

ilds = [-70, -8, -4, -2, 0, 2, 4, 8, 70]
n_points = 100
plt.figure()

# PC Right analysis
# Saline
s_right_saline = df_saline.SensitivityPCRight.mean()
b_right_saline = df_saline.BiasPCRight.mean()
lapse_left_saline = df_saline.LapseLeft.mean()
lapse_right_saline = df_saline.LapseRight.mean()
fit_right_saline = lapse_right_saline + (1 - lapse_right_saline - lapse_left_saline) / (1 + np.exp(-s_right_saline * (np.linspace(np.min(ilds), np.max(ilds), n_points) - b_right_saline)))  # PC function
plt.plot(np.linspace(np.min(ilds), np.max(ilds), n_points), fit_right_saline, color='tab:gray', mfc='tab:gray', label='')


# MK801
s_right_MK801 = df_MK801.SensitivityPCRight.mean()
b_right_MK801 = df_MK801.BiasPCRight.mean()
lapse_left_MK801 = df_MK801.LapseLeft.mean()
lapse_right_MK801 = df_MK801.LapseRight.mean()
fit_right_MK801 = lapse_right_MK801 + (1 - lapse_right_MK801 - lapse_left_MK801) / (1 + np.exp(-s_right_MK801 * (np.linspace(np.min(ilds), np.max(ilds), n_points) - b_right_MK801)))
plt.plot(np.linspace(np.min(ilds), np.max(ilds), n_points), fit_right_MK801, color='tab:orange', mfc='tab:orange', label='')

# Rest
s_right_rest = df_rest.SensitivityPCRight.mean()
b_right_rest = df_rest.BiasPCRight.mean()
lapse_left_rest = df_rest.LapseLeft.mean()
lapse_right_rest = df_rest.LapseRight.mean()
fit_right_rest = lapse_right_rest + (1 - lapse_right_rest - lapse_left_rest) / (1 + np.exp(-s_right_rest * (np.linspace(np.min(ilds), np.max(ilds), n_points) - b_right_rest)))
plt.plot(np.linspace(np.min(ilds), np.max(ilds), n_points), fit_right_rest, color='tab:blue', mfc='tab:blue', label='')

plt.figure()

# PC Rep analysis
# Saline
s_rep_saline = df_saline.SensitivityPCRep.mean()
b_rep_saline = df_saline.BiasPCRep.mean()
lapse_alt_saline = df_saline.LapseAlt.mean()
lapse_rep_saline = df_saline.LapseRep.mean()

fit_rep_saline = lapse_rep_saline + (1 - lapse_rep_saline - lapse_alt_saline) / (1 + np.exp(-s_rep_saline * (np.linspace(np.min(ilds), np.max(ilds), n_points) - b_rep_saline)))  # PC function
plt.plot(np.linspace(np.min(ilds), np.max(ilds), n_points), fit_rep_saline, color='tab:gray', mfc='tab:gray', label='')

# MK801
s_rep_MK801 = df_MK801.SensitivityPCRep.mean()
b_rep_MK801 = df_MK801.BiasPCRep.mean()
lapse_alt_MK801 = df_MK801.LapseAlt.mean()
lapse_rep_MK801 = df_MK801.LapseRep.mean()
fit_rep_MK801 = lapse_rep_MK801 + (1 - lapse_rep_MK801 - lapse_alt_MK801) / (1 + np.exp(-s_rep_MK801 * (np.linspace(np.min(ilds), np.max(ilds), n_points) - b_rep_MK801)))  # PC function
plt.plot(np.linspace(np.min(ilds), np.max(ilds), n_points), fit_rep_MK801, color='tab:orange', mfc='tab:orange', label='')

# Rest
s_rep_rest = df_rest.SensitivityPCRep.mean()
b_rep_rest = df_rest.BiasPCRep.mean()
lapse_alt_rest = df_rest.LapseAlt.mean()
lapse_rep_rest = df_rest.LapseRep.mean()
fit_rep_rest = lapse_rep_rest + (1 - lapse_rep_rest - lapse_alt_rest) / (1 + np.exp(-s_rep_rest * (np.linspace(np.min(ilds), np.max(ilds), n_points) - b_rep_rest)))  # PC function
plt.plot(np.linspace(np.min(ilds), np.max(ilds), n_points), fit_rep_rest, color='tab:blue', mfc='tab:blue', label='')