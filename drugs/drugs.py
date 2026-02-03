import pandas as pd
from scipy.stats import ttest_rel
from my_fun.my_fun import add_star_between
from matplotlib import pyplot as plt
import seaborn as sns


# Define function to compare behavioral metrics between saline and drug conditions
def plot_var_drug(df_intersessions, var_name='Accuracy', **kwargs):
    """
    Plot a variable (e.g., accuracy) for each subject and drug condition together with the mean and sem across subjects, and run a t-test between saline and drug conditions.
    :param df_intersessions: DataFrame containing intersession data from all animals
    :param var_name: Name of the variable to plot (e.g., 'Accuracy')
    :return: None
    """

    var = df_intersessions.groupby(['Subject', 'Drug'])[var_name].mean().reset_index()
    var_mean = var.groupby('Drug')[var_name].mean()
    var_sem = var.groupby('Drug')[var_name].sem()
    n_subjects = var.Subject.nunique()

    # Plot paired lines
    plt.figure(constrained_layout=True, **kwargs)
    sns.boxplot(data=var, x='Drug', y=var_name, palette=['tab:gray', 'tab:pink'], showfliers=False, showcaps=False,
                fill=False)
    sns.lineplot(data=var, x='Drug', y=var_name, hue='Subject', marker='', palette=['k'] * n_subjects, alpha=0.1, legend=False)
    # Plot mean and SEM
    # plt.errorbar(x=var_mean.index, y=var_mean.values, yerr=var_sem.values, fmt='-o', color=color)
    # plt.title(var_name)
    plt.xticks([0, 1], ['Saline', 'Drug'])
    plt.xlabel('')
    # plt.ylim([0.5, 1])
    plt.ylabel(var_name)
    # plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))  # Set a global formatter for y-axis ticks for equal sized plots
    sns.despine()

    # Run t-test
    saline = var[var['Drug'] == 0][var_name]
    drug = var[var['Drug'] == 1][var_name]
    t_stat, p_val = ttest_rel(saline, drug)
    print(f'{var_name}: t = {t_stat:.3f}, p = {p_val:.3f}')
    add_star_between(p_val)