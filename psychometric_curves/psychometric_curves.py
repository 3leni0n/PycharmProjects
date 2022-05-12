import time
import os
import pandas as pd
from my_fun.my_fun import *
from scipy import stats
from matplotlib import pyplot as plt
import numpy as np


def plot_pc(experiment='2AFC_2', animal=None, kind='prob_right', save=False):
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
        plt.xlabel('ILD')
        plt.ylabel('Prob. right')

        # Annotation params
        xy = (ilds[0], 1)
        xytext = (ilds[0], 1)
        va = 'top'
        ha = 'left'
        fontsize = 'medium'

        filename = '_PC_prob_right.png'

    elif kind == 'prob_rep':

        # Compute rep-alt psychometric curve
        # psych_curve_rep = compute_psych_curve(df.EviRep, df.RepChoice)  # Pilot batch
        psych_curve = compute_psych_curve(df.ILDRep, df.RepChoice, n_points)

        # Plot params
        color = 'tab:brown'
        label = 'Prob. repeat'
        plt.xlabel('Repeating ILD')
        plt.ylabel('Prob. repeat')

        # Annotate params
        xy = (ilds[-1], 0)
        xytext = (ilds[-1], 0)
        color = 'tab:brown'
        va = 'bottom'
        ha = 'right'
        fontsize = 'medium'

        filename = '_PC_prob_rep.png'

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
    plt.plot(np.linspace(np.min(ilds), np.max(ilds), n_points), psych_curve.fit, color=color, label=label)
    plt.errorbar(psych_curve.xdata, psych_curve.ydata, yerr=psych_curve.fit_error, color=color,
                 fmt='o', markerfacecolor='none')

    sensitivity, bias, lr_left, lr_right = psych_curve.params
    plt.annotate("S=" + str(round(sensitivity, 2)) + "\n" +  # Sensitivity
                 "B=" + str(round(bias, 2)) + "\n" +  # Bias
                 "LR_Rep=" + str(round(lr_left, 2)) + "\n" +  # Left lapse rate
                 "LR_Alt=" + str(round(lr_right, 2)),  # Left lapse rate
                 xy=xy, xytext=xytext, color=color,
                 va=va, ha=ha, fontsize=fontsize)

    plt.xscale('symlog', linthreshx=20)  # Set symmetric logarithmic spacing to zoom in the middle
    plt.minorticks_off()  # Remove minor ticks
    plt.title(f'Psychometric curve, animal {df.Setup.unique()[0]}, {len(df)} trials')
    # plt.title(f'Psychometric curve, {len(df.Setup.unique())} animals, {len(df)} trials')
    plt.axhline(0.5, color='tab:gray', ls='--')
    plt.axvline(0., color='tab:gray', ls='--')
    plt.xlim([ilds[0]-7, ilds[-1]+7])
    plt.xticks(ilds)
    plt.gca().set_xticklabels(['-70', '-8', '', '', '0', '', '', '8', '70'])
    plt.ylim([-0.025, 1.025])
    # plt.legend(loc='lower center')

    if save:
        folder_out = '/home/alexis/Documentos/psychometric curves/'
        if not os.path.exists(folder_out):
            os.mkdir(folder_out)
        os.chdir(folder_out)
        plt.savefig(folder_out + animal + filename)
        plt.close()

    return psych_curve


def do_pcs(experiment='2AFC_2', animals=['325', '327', '329', '330', '332', '333', '335', '337'], kind='prob_right'):
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
        plot_pc(experiment=experiment, animal=animals[i], kind=kind, save=True)

    time_end = time.time()
    runtime = time_end - time_start
    print('The script took', round(runtime, 2), 'seconds to run')


def plot_pc_across_animals(experiment='2AFC_2', animals=['325', '327', '329', '330', '332', '333', '335', '337'],
                           kind='prob_right'):

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

        plt.plot(np.linspace(np.min(ilds), np.max(ilds), n_points), psych_curve.fit, color='tab:grey', alpha=0.25)
        plt.errorbar(psych_curve.xdata, psych_curve.ydata, yerr=psych_curve.fit_error, color='tab:grey', fmt='o',
                     markerfacecolor='none', alpha=0.25)

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
        xlabel = 'ILD'
        ylabel = 'Prob. right'

        # Annotation params
        xy = (ilds[0], 1)
        xytext = (ilds[0], 1)
        va = 'top'
        ha = 'left'
        fontsize = 'medium'

        filename = ' PC_prob_right_all_animals.png'

    elif kind == 'prob_rep':

        # Plot params
        color = 'tab:brown'
        label = 'Prob. repeat'
        xlabel = 'Repeating ILD'
        ylabel = 'Prob. repeat'

        # Annotate params
        xy = (ilds[-1], 0)
        xytext = (ilds[-1], 0)
        color = 'tab:brown'
        va = 'bottom'
        ha = 'right'
        fontsize = 'medium'

        filename = ' PC_prob_rep_all_animals.png'

    plt.plot(np.linspace(np.min(ilds), np.max(ilds), n_points), psych_curve.fit, color=color, label=label)
    plt.errorbar(psych_curve.xdata, psych_curve.ydata, yerr=psych_curve.fit_error, color=color, fmt='o',
                 markerfacecolor='none')

    plt.axhline(0.5, color='tab:gray', ls='--')
    plt.axvline(0., color='tab:gray', ls='--')
    plt.title(f'Psychometric curve, {len(animals)} animals, {trials} trials')
    # plt.title(f'Psychometric curve, {len(df.Setup.unique())} animals, {len(df)} trials')
    plt.xlim([ilds[0] - 7, ilds[-1] + 7])
    plt.xticks(ilds)
    plt.gca().set_xticklabels(['-70', '-8', '', '', '0', '', '', '8', '70'])
    plt.ylim([-0.025, 1.025])
    plt.legend(loc='lower center')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    folder_out = '/home/alexis/Documentos/psychometric curves/'

    sensitivity, bias, lr_left, lr_right = psych_curve.params
    plt.annotate("S=" + str(round(sensitivity, 2)) + "\n" +  # Sensitivity
                 "B=" + str(round(bias, 2)) + "\n" +  # Bias
                 "LR_Rep=" + str(round(lr_left, 2)) + "\n" +  # Left lapse rate
                 "LR_Alt=" + str(round(lr_right, 2)),  # Left lapse rate
                 xy=xy, xytext=xytext, color=color,
                 va=va, ha=ha, fontsize=fontsize)

    plt.savefig(folder_out + filename)

    return np.array(params)


def test_lapses(experiment='2AFC_2', animals=['325', '327', '329', '330', '332', '333', '335', '337'], kind='prob_rep',
                save=True):
    params = plot_pc_across_animals(experiment=experiment, animals=animals, kind='prob_rep')
    rep_lapses = params[:, 2]
    alt_lapses = params[:, 3]
    plt.figure()
    plt.scatter(alt_lapses, rep_lapses, color='k')
    x = np.linspace(0, 0.5)
    y = x
    plt.plot(x, y, 'r')
    plt.axis('square')
    plt.title('Repeating vs alteranting lapses')
    plt.xlim(0, 0.5)
    plt.xlabel('Alternating lapses')
    plt.ylim(0, 0.5)
    plt.ylabel('Repeating lapses')

    # Paired-samples t-test
    t_test = stats.ttest_rel(rep_lapses, alt_lapses)  # Two-sided
    # t_test = stats.ttest_rel(rep_lapses, alt_lapses, alternative='greater')  # One-sided (you have a priori hypothesis of
    # why one mean would be larger than the other one)

    print('Rep. lapse = ' + str(np.round(np.mean(rep_lapses), 2)) + ' ± ' +
                 str(np.round(stats.sem(rep_lapses), 2)) + ', ' + 'p = ' + str(np.round(t_test.pvalue, 2)), sep='')
    print('Alt. lapse = ' + str(np.round(np.mean(alt_lapses), 2)) + ' ± ' +
                 str(np.round(stats.sem(alt_lapses), 2)) + ', ' + 'p = ' + str(np.round(t_test.pvalue, 2)), sep='')

    plt.annotate('Rep. lapse = ' + str(np.round(np.mean(rep_lapses), 2)) + ' ± ' +
                 str(np.round(stats.sem(rep_lapses), 2)) + ', ' + 'p = ' + str(np.round(t_test.pvalue, 2)),
                 xy=(0.25, 0.45), color='k', va='bottom', ha='center', fontsize='medium')
    plt.annotate('Alt. lapse = ' + str(np.round(np.mean(alt_lapses), 2)) + ' ± ' +
                 str(np.round(stats.sem(alt_lapses), 2)) + ', ' + 'p = ' + str(np.round(t_test.pvalue, 2)),
                 xy=(0.25, 0.43), color='k', va='bottom', ha='center', fontsize='medium')

    if save:
        filename = 'Rep_Alt_lapses.png'
        folder_out = '/home/alexis/Documentos/psychometric curves/'
        plt.savefig(folder_out + filename)
