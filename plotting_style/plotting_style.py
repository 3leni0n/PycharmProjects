from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

sns.set_style('ticks')
sns.set_context('notebook')


def fig_size(n_cols=1, ratio=None):

    if ratio is None:
        default_figsize = np.array(plt.rcParams['figure.figsize'])
        default_ratio = default_figsize[0] / default_figsize[1]
        ratio = default_ratio  # 4:3

    # All measurements are in inches
    A4_size = np.array((8.27, 11.69))  # A4 measurements
    margins = 2  # On each dimension
    size = A4_size - margins  # Effective size after margins removal (2 per dimension)
    width = size[0]
    height = size[1]

    # Full page (minus margins)
    if n_cols == 0:
        if ratio == 1:  # Square
            size = (size[0], size[0])
        return size

    else:
        fig_width = width / n_cols
        fig_height = fig_width / ratio
        figsize = (fig_width, fig_height)
        return figsize