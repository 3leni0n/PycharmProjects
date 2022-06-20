import time
import os
import pandas as pd
from my_fun.my_fun import *
from scipy import stats
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

# Mel's code snippet for poster
sns.set_theme()
sns.set_style("white")
sns.set_context("poster")


def plot_pc(experiment='2AFC_2', animal=None, kind='prob_right', save=False, format='svg', transparent=False):
    """Plot psychometric curve"""

    time_start = time.time()

    if experiment is None:

        folder_in = '/home/alexis/PycharmProjects/glue_sessions/'  # Where the data for all animals is
        experiments = os.listdir(folder_in)  # List experiments
        experiments.sort()  # Sort them by name
        experiments = [x for x in experiments if os.path.isdir(folder_in + x)]  # Get rid of non folders

        try:
            experiments.remove('__pycache__')  # Pycharm's archive
        except ValueError:
            pass

        print('Experiments: ' + str(experiments)[1:-1])  # Remove square brackets
        experiment = input('Enter experiment name')

    folder_in = '/home/alexis/PycharmProjects/glue_sessions/' + experiment + '/'  # Where the data for all animals is

    if animal is None:

        animals = os.listdir(folder_in)  # List animals
        animals.sort()  # Sort them by name
        animals = [x[:-4] for x in animals]  # Get rid of .csv extension

        print('Animals: ' + str(animals))  # Remove square brackets
        animal = input('Enter animal')  # Ask user to input animal to glue sessions from

    folder_in = folder_in + animal + '.csv'
    df = pd.read_csv(folder_in)  # Read behavioral data
    df = df[df.P > 0]  # Only those sessions with ilds
    # df = df[df.Stage == 4]  # Only those sessions in stage 4
    # Only sessions with accuracy > X threshold?

    # Compute psychometric curves
    n_points = 100
    # evidences = np.sort(df.evidence.unique())  # Pilot batch
    ilds = np.sort(df.ILD.unique())

    # Plot psychometric curves
    plt.figure()

    if kind == 'prob_right':

        # Compute left-right psychometric curve
        # psych_curve_right = compute_psych_curve(df.Evidence, df.Choice)  # Pilot batch
        psych_curve = compute_psych_curve(df.ILD, df.Choice, n_points)  # No need to filter out the misses

        # Plot params
        color = 'tab:orange'
        label = 'Prob. right'
        plt.xlabel('Stimulus ILD (dB)')
        plt.ylabel('Prob. choose right')

        # Annotation params
        lower_lapse = "LR_R="
        upper_lapse = "LR_L="
        # xy = (ilds[0], 1)
        # xytext = (ilds[0], 1)
        xy = (-20, 1)
        xytext = (-20, 1)
        va = 'top'
        ha = 'left'
        fontsize = 'medium'

        filename = '_PC_prob_right'

    elif kind == 'prob_rep':

        # Compute rep-alt psychometric curve
        # psych_curve_rep = compute_psych_curve(df.EviRep, df.RepChoice)  # Pilot batch
        psych_curve = compute_psych_curve(df.ILDRep, df.RepChoice, n_points)

        # Plot params
        color = 'tab:brown'
        label = 'Prob. repeat'
        plt.xlabel('Repeating stimulus ILD (dB)')
        plt.ylabel('Prob. choose repeat')

        # Annotate params
        lower_lapse = "LR_Rep="
        upper_lapse = "LR_Alt="
        # xy = (ilds[-1], 0)
        # xytext = (ilds[-1], 0)
        xy = (20, 0)
        xytext = (20, 0)
        color = 'tab:brown'
        va = 'bottom'
        ha = 'right'
        fontsize = 'medium'

        filename = '_PC_prob_rep'

    # elif kind == 'prob_rep_after':
    #
    #     # Plot rep-alt psychometric curve and errorbars for after hits
    #     df_after_hit = df[df.AfterHit == 1]
    #     psych_curve_after_hit = compute_psych_curve(df_after_hit.ILDRep, df_after_hit.RepChoice, n_points)
    #     plt.plot(np.linspace(np.min(ilds), np.max(ilds), n_points), psych_curve_after_hit.fit, color='tab:green', label='After hit')
    #     plt.errorbar(psych_curve_after_hit.xdata, psych_curve_after_hit.ydata, yerr=psych_curve_after_hit.fit_error, color='tab:green',
    #                  fmt='o', markerfacecolor='none')
    #     sensitivity_after_hit, bias_after_hit, lr_left_after_hit, lr_right_after_hit = psych_curve_after_hit.params
    #     plt.annotate("S=" + str(round(sensitivity_after_hit, 2)) + "\n" +  # Sensitivity
    #                  "B=" + str(round(bias_after_hit, 2)) + "\n" +  # Bias
    #                  "LR_Rep=" + str(round(lr_left_after_hit, 2)) + "\n" +  # Left lapse rate
    #                  "LR_Alt=" + str(round(lr_right_after_hit, 2)),
    #                  xy=(ilds[0], 1), xytext=(ilds[0], 1), color='tab:green',
    #                  va='top', ha='left', fontsize='medium')
    #
    #     # Plot rep-alt psychometric curve and errorbars for after errors
    #     df_after_error = df[df.AfterHit == 0]
    #     psych_curve_after_error = compute_psych_curve(df_after_error.ILDRep, df_after_error.RepChoice, n_points)
    #     plt.plot(np.linspace(np.min(ilds), np.max(ilds), n_points), psych_curve_after_error.fit, color='tab:red', label='After error')
    #     plt.errorbar(psych_curve_after_error.xdata, psych_curve_after_error.ydata, yerr=psych_curve_after_error.fit_error, color='tab:red',
    #                  fmt='o', markerfacecolor='none')
    #     sensitivity_after_error, bias_after_error, lr_left_after_error, lr_right_after_error = psych_curve_after_error.params
    #     plt.annotate("S=" + str(round(sensitivity_after_error, 2)) + "\n" +  # Sensitivity
    #                  "B=" + str(round(bias_after_error, 2)) + "\n" +  # Bias
    #                  "LR_Rep=" + str(round(lr_left_after_error, 2)) + "\n" +  # Left lapse rate
    #                  "LR_Alt=" + str(round(lr_right_after_error, 2)),
    #                  xy=(ilds[-1], 0), xytext=(ilds[-1], 0), color='tab:red',
    #                  va='bottom', ha='right', fontsize='medium')
    #     plt.xlabel('Repeating ILD')
    #     plt.ylabel('Prob. repeat')
    #     filename = ' PC_prob_after.png'

    # Plot psychometric curve and errorbars
    plt.plot(np.linspace(np.min(ilds), np.max(ilds), n_points), psych_curve.fit, color=color, mfc=color,
             label='')

    # Move extreme datapoints closer to the center to zoom in
    psych_curve.xdata[0] = -20
    psych_curve.xdata[-1] = 20

    plt.errorbar(psych_curve.xdata, psych_curve.ydata, yerr=psych_curve.fit_error, color=color,
                 fmt='o', mfc=color)

    sensitivity, bias, lr_lower, lr_upper = psych_curve.params
    plt.annotate("S=" + str(round(sensitivity, 2)) + "\n" +  # Sensitivity
                 "B=" + str(round(bias, 2)) + "\n" +  # Bias
                 lower_lapse + str(round(lr_lower, 2)) + "\n" +  # Upper lapse rate
                 upper_lapse + str(round(lr_upper, 2)),  # Lower lapse rate
                 xy=xy, xytext=xytext, color=color,
                 va=va, ha=ha, fontsize=fontsize)

    # plt.xscale('symlog', linthreshx=20)  # Set symmetric logarithmic spacing to zoom in the middle
    # plt.minorticks_off()  # Remove minor ticks
    plt.title(f'Mouse {df.Setup.unique()[0]}, {len(df)} trials')
    # plt.title(f'Mouse {len(df.Setup.unique())}, {len(df)} trials')
    plt.axhline(0.5, color='tab:gray', ls='--')

    if kind == 'prob_rep':  # Plot only for 'prob_rep'
        pc0_bias0 = lr_lower + (1 - lr_lower - lr_upper) / 2  # Value of the PC for x = 0 when bias = 0
        plt.axhline(pc0_bias0, color='tab:blue', ls=':', label='y(x=0)|B=0')
        pc0_lapses0 = 1 / (1 + np.exp(sensitivity * bias))  # Value of the PC for x = 0 when lapses = 0
        plt.axhline(pc0_lapses0, color='tab:orange', ls=':', label='y(x=0)|LR=0')
        plt.legend(loc='upper left', fontsize='xx-small', frameon=False)

    plt.axvline(0., color='tab:gray', ls='--')
    # plt.xlim([ilds[0]-7, ilds[-1]+7]
    plt.xlim([-21, 21])  # To chop the extreme values
    ilds[0] = -20
    ilds[-1] = 20
    plt.xticks(ilds)
    plt.gca().set_xticklabels(['-70', '-8', '', '', '0', '', '', '8', '70'])
    plt.ylim([-0.025, 1.025])
    plt.yticks([0, 0.5, 1])
    # plt.legend(loc='lower center')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    if save:
        folder_out = '/home/alexis/Documentos/psychometric curves/'
        if not os.path.exists(folder_out):
            os.mkdir(folder_out)
        os.chdir(folder_out)
        plt.savefig(folder_out + animal + filename, format=format, transparent=transparent)
        plt.close()

    return psych_curve


def do_pcs(experiment='2AFC_2', animals=['325', '327', '329', '330', '332', '333', '335', '337'], kind='prob_right',
           save=False, format='svg', transparent=False):
    """Do the kernels for all animals of a given batch (experiment)"""

    time_start = time.time()

    if experiment is None:

        folder = '/home/alexis/PycharmProjects/glue_sessions/'  # Where the data for all animals is
        experiments = os.listdir(folder)  # List experiments
        experiments.sort()  # Sort them by name
        experiments = [x for x in experiments if os.path.isdir(folder + x)]  # Get rid of non folders

        try:
            experiments.remove('__pycache__')  # Pycharm's archive
        except ValueError:
            pass

        print('Experiments: ' + str(experiments)[1:-1])  # Remove square brackets
        experiment = input('Enter experiment name')

    folder = '/home/alexis/PycharmProjects/glue_sessions/' + experiment + '/'  # Where the data for all animals is

    for i in range(len(animals)):
        path = folder + animals[i]
        print(path)
        plot_pc(experiment=experiment, animal=animals[i], kind=kind, save=save, format=format, transparent=transparent)

    time_end = time.time()
    runtime = time_end - time_start
    print('The script took', round(runtime, 2), 'seconds to run')


def plot_pc_across_animals(experiment='2AFC_2', animals=['325', '327', '329', '330', '332', '333', '335', '337'],
                           save=False, kind='prob_right', format='svg', transparent=False):

    time_start = time.time()

    if experiment is None:

        folder_in = '/home/alexis/PycharmProjects/glue_sessions/'  # Where the data for all animals is
        experiments = os.listdir(folder_in)  # List experiments
        experiments.sort()  # Sort them by name
        experiments = [x for x in experiments if os.path.isdir(folder_in + x)]  # Get rid of non folders

        try:
            experiments.remove('__pycache__')  # Pycharm's archive
        except ValueError:
            pass

        print('Experiments: ' + str(experiments)[1:-1])  # Remove square brackets
        experiment = input('Enter experiment name')

    folder_in = '/home/alexis/PycharmProjects/glue_sessions/' + experiment + '/'  # Where the data for all animals is

    fit = []
    fit_error = []
    params = []
    xdata = []
    ydata = []
    trials = 0

    n_points = 100

    for i in range(len(animals)):
        print(folder_in + animals[i] + '.csv')
        df = pd.read_csv(folder_in + animals[i] + '.csv')
        df = df[df.P > 0]
        ilds = np.sort(df.ILD.unique())

        if kind == 'prob_right':
            psych_curve = compute_psych_curve(df.ILD, df.Choice, n_points)
        elif kind == 'prob_rep':
            psych_curve = compute_psych_curve(df.ILDRep, df.RepChoice, n_points)

        plt.plot(np.linspace(np.min(ilds), np.max(ilds), n_points), psych_curve.fit, color='tab:grey', marker=None,
                 mfc='none', mec='none', mew=0, ms=0, alpha=0.25)
        # mec='none, mew=0 and ms=0 to not plot markers in individual animals
        # plt.errorbar(psych_curve.xdata, psych_curve.ydata, yerr=psych_curve.fit_error, color='tab:grey', fmt='o',
        #              mfc='none', alpha=0.25)  # Don't plot the errorbars in individual animals

        fit.append(psych_curve.fit)
        fit_error.append(psych_curve.fit_error)
        params.append(psych_curve.params)
        xdata.append(psych_curve.xdata)
        ydata.append(psych_curve.ydata)
        trials += len(df)

    xdata = np.array(xdata).flatten()
    ydata = np.array(ydata).flatten()
    psych_curve = compute_psych_curve(xdata, ydata, n_points)

    if kind == 'prob_right':

        # Plot params
        color = 'tab:orange'
        label = 'Prob. right'
        xlabel = 'Stimulus ILD (dB)'
        ylabel = 'Prob. choose right'

        # Annotation params
        lower_lapse = "LR_L="
        upper_lapse = "LR_R="
        # xy = (ilds[0], 1)
        # xytext = (ilds[0], 1)
        xy = (-20, 1)
        xytext = (-20, 1)
        va = 'top'
        ha = 'left'
        fontsize = 'medium'

        filename = ' PC_prob_right_all_animals'

    elif kind == 'prob_rep':

        # Plot params
        color = 'tab:brown'
        label = 'Prob. repeat'
        xlabel = 'Repeating stimulus ILD (dB)'
        ylabel = 'Prob. choose repeat'

        # Annotate params
        lower_lapse = "LR_Alt="
        upper_lapse = "LR_Rep="
        # xy = (ilds[-1], 0)
        # xytext = (ilds[-1], 0)
        xy = (20, 0)
        xytext = (20, 0)
        color = 'tab:brown'
        va = 'bottom'
        ha = 'right'
        fontsize = 'medium'

        filename = ' PC_prob_rep_all_animals'

    plt.plot(np.linspace(np.min(ilds), np.max(ilds), n_points), psych_curve.fit, color=color, mfc=color,
             label=label)

    # Move extreme datapoints closer to the center to zoom in
    psych_curve.xdata[0] = -20
    psych_curve.xdata[-1] = 20

    plt.errorbar(psych_curve.xdata, psych_curve.ydata, yerr=psych_curve.fit_error, color=color, fmt='o',
                 mfc=color)

    # plt.xscale('symlog', linthreshx=20)  # Set symmetric logarithmic spacing to zoom in the middle
    # plt.minorticks_off()  # Remove minor ticks
    plt.axhline(0.5, color='tab:gray', ls='--')
    plt.axvline(0., color='tab:gray', ls='--')
    plt.title(f'N={len(animals)}, {trials} trials')
    # plt.title(f'N={len(df.Setup.unique())}, {len(df)} trials')
    # plt.xlim([ilds[0] - 7, ilds[-1] + 7])
    plt.xlim([-21, 21])  # To chop the extreme values
    ilds[0] = -20
    ilds[-1] = 20
    plt.xticks(ilds)
    plt.gca().set_xticklabels(['-70', '-8', '', '', '0', '', '', '8', '70'])
    plt.ylim([-0.025, 1.025])
    plt.yticks([0, 0.5, 1])
    # plt.legend(loc='lower center')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    sensitivity, bias, lr_left, lr_right = psych_curve.params
    plt.annotate("S=" + str(round(sensitivity, 2)) + "\n" +  # Sensitivity
                 "B=" + str(round(bias, 2)) + "\n" +  # Bias
                 upper_lapse + str(round(lr_left, 2)) + "\n" +  # Upper lapse rate
                 lower_lapse + str(round(lr_right, 2)),  # Lower lapse rate
                 xy=xy, xytext=xytext, color=color,
                 va=va, ha=ha, fontsize=fontsize)

    if save:
        folder_out = '/home/alexis/Documentos/psychometric curves/'
        if not os.path.exists(folder_out):
            os.mkdir(folder_out)
        os.chdir(folder_out)
        plt.savefig(folder_out + filename, format=format, transparent=transparent)
        plt.close()

    return np.array(params)


def test_lapses(experiment='2AFC_2', animals=['325', '327', '329', '330', '332', '333', '335', '337'], kind='prob_rep',
                save=True, format=format, transparent=False):

    params = plot_pc_across_animals(experiment=experiment, animals=animals, kind=kind)
    lower_lapses = params[:, 2]
    upper_lapses = params[:, 3]
    plt.figure()
    plt.scatter(upper_lapses, lower_lapses, color='k')
    x = np.linspace(0, 0.5)
    y = x
    plt.plot(x, y, 'r')
    plt.axis('square')
    plt.title('Rep. vs Alt. lapses')
    # plt.xlim(0, 0.5)
    plt.xlim(0, np.round(np.max(lower_lapses), 1))
    plt.xlabel('Alternating lapses')
    plt.xticks(np.linspace(0, np.round(np.max(lower_lapses), 1), 3))
    # plt.ylim(0, 0.5)
    plt.ylim(0, np.round(np.max(lower_lapses), 1))
    plt.ylabel('Repeating lapses')
    plt.yticks(np.linspace(0, np.round(np.max(lower_lapses), 1), 3))

    # Paired-samples t-test
    t_test = stats.ttest_rel(lower_lapses, upper_lapses)  # Two-sided
    # t_test = stats.ttest_rel(lower_lapses, upper_lapses, alternative='greater')  # One-sided (you have a priori hypothesis of
    # why one mean would be larger than the other one)

    plt.annotate('Rep. lapse = ' + str(np.round(np.mean(lower_lapses), 2)) + ' ± ' +
                 str(np.round(stats.sem(lower_lapses), 2)), xy=(0.15, 0.04), color='k', va='bottom', ha='center',
                 fontsize='xx-small')
    plt.annotate('Alt. lapse = ' + str(np.round(np.mean(upper_lapses), 2)) + ' ± ' +
                 str(np.round(stats.sem(upper_lapses), 2)),
                 xy=(0.15, 0.02), color='k', va='bottom', ha='center', fontsize='xx-small')
    plt.annotate('p = ' + str(np.round(t_test.pvalue, 4)), xy=(0.15, 0), color='k', va='bottom', ha='center',
                 fontsize='xx-small')

    if save:
        filename = 'Rep_Alt_lapses'
        folder_out = '/home/alexis/Documentos/psychometric curves/'
        plt.savefig(folder_out + filename, format=format, transparent=transparent)
