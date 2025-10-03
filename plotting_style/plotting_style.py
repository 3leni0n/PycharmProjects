from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

sns.set_style('ticks')
sns.set_context('notebook')


def fig_size(n_cols=1, ratio=None):

    if ratio is None:
        default_figsize = np.array(plt.rcParams['figure.figsize'])
        default_ratio = default_figsize[0] / default_figsize[1]
        ratio = default_ratio

    # All measurements are in inches
    A4_size = np.array((8.27, 11.69))  # A4 measurements
    margins = 1  # On all sides
    effective_size = A4_size - 2*margins  # Effective size after margins removal (2 per dimension)
    effective_width = effective_size[0]
    effective_height = effective_size[1]

    fig_width = effective_width / n_cols
    fig_height = fig_width / ratio
    figsize = (fig_width, fig_height)

    return figsize