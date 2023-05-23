import time
from pathlib import Path
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from matplotlib import pyplot as plt
from scipy import stats
import seaborn as sns
from collections import namedtuple
from my_fun.my_fun import get_experiment, get_animal, get_ild
from kernels.kernels_copy import *

# Plotting parameters
sns.set_theme()
sns.set_style('white')
sns.set_style('ticks')
sns.set_context('poster')

# Kernel parameters
experiment = '2AFC_2'
# animal = '333'
# animals = ['325', '326', '327', '329', '330', '332', '333', '334', '335', '337']  # Bach 2 (with ILDs)
# animals = ['325', '326', '327', '329', '330', '332', '333', '335', '337']  # Bach 2 (with ILDs) -334
# animals = ['419', '420', '422', '616', '617', '619', '620', '623']  # Batch 3 (with ILDs)
animals = ['332', '333', '337']
animal = animals
library = 'sm'
target_ilds = [-8, -4, -2, 0, 2, 4, 8]
# drug = None
residuals = False
zscore = True
control = None
n_mean_frames = 2
iterations = 1000
save = False
format = 'svg'
transparent = False

# Get mean PKs
pk_rest = get_mean_pk(experiment=experiment, animals=animals, library=library, target_ilds=target_ilds,
                 drug='rest', residuals=residuals, zscore=zscore, control=control, n_mean_frames=n_mean_frames,
                      iterations=iterations)

pk_saline = get_mean_pk(experiment=experiment, animals=animals, library=library, target_ilds=target_ilds,
                 drug='saline', residuals=residuals, zscore=zscore, control=control, n_mean_frames=n_mean_frames,
                        iterations=iterations)

pk_mk801 = get_mean_pk(experiment=experiment, animals=animals, library=library, target_ilds=target_ilds,
                 drug='MK801', residuals=residuals, zscore=zscore, control=control, n_mean_frames=n_mean_frames,
                       iterations=iterations)

# Plot mean PKs
plot_pk(experiment=experiment, animal=animal, library=library, target_ilds=target_ilds, drug='rest',
        residuals=residuals, zscore=zscore, control=control, n_mean_frames=n_mean_frames, iterations=iterations,
        save=False, format='svg', transparent=False)

plot_pk(experiment=experiment, animal=animal, library=library, target_ilds=target_ilds, drug='saline',
        residuals=residuals, zscore=zscore, control=control, n_mean_frames=n_mean_frames, iterations=iterations,
        save=False, format='svg', transparent=False)

plot_pk(experiment=experiment, animal=animal, library=library, target_ilds=target_ilds, drug='MK801',
        residuals=residuals, zscore=zscore, control=control, n_mean_frames=n_mean_frames, iterations=iterations,
        save=False, format='svg', transparent=False)


# Normalized PK Area
npka_rest = normalized_pi_pk_area(pk_rest)
npka_saline = normalized_pi_pk_area(pk_saline)
npka_mk801 = normalized_pi_pk_area(pk_mk801)

print(f'\n The Normalized PK area are: \n'
      f'rest: {round(npka_rest, 3)}\n'
      f'saline: {round(npka_saline, 3)}\n'
      f'MK801: {round(npka_mk801, 3)}')

# Normalized PK slope
npks_rest = normalized_pk_slope(pk_rest)
npks_saline = normalized_pk_slope(pk_saline)
npks_mk801 = normalized_pk_slope(pk_mk801)

print(f'\n The Normalized PK slope are: \n'
      f'rest: {round(npks_rest, 3)}\n'
      f'saline: {round(npks_saline, 3)}\n'
      f'MK801: {round(npks_mk801, 3)}')

# Primacy-recency index
pri_rest = primacy_recency_index(pk_rest)
pri_saline = primacy_recency_index(pk_saline)
pri_mk801 = primacy_recency_index(pk_mk801)

print(f'\n The Primacy-recency index are: \n'
      f'rest: {round(pri_rest, 3)}\n'
      f'saline: {round(pri_saline, 3)}\n'   
      f'MK801: {round(pri_mk801, 3)}')
