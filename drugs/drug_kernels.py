from pathlib import Path
from matplotlib import pyplot as plt
from scipy import stats
import seaborn as sns

from kernels.kernels_copy import *

# Plotting parameters
# sns.set_theme()
# sns.set_style('white')
# sns.set_style('ticks')
# sns.set_context('poster')

# Kernel parameters
experiment = '2AFC_2'
animals = ['332', '333', '337']
animal = animals
library = 'sm'
target_ilds = [-8, -4, -2, 0, 2, 4, 8]
# drug = None
residuals = False
zscore = False
control = None
n_mean_frames = 2
iterations = 10
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
# npka_rest = normalized_pi_pk_area(pk_rest)
# npka_saline = normalized_pi_pk_area(pk_saline)
# npka_mk801 = normalized_pi_pk_area(pk_mk801)
#
# print(f'\nThe Normalized PK areas are: \n'
#       f'rest: {round(npka_rest, 3)}\n'
#       f'saline: {round(npka_saline, 3)}\n'
#       f'MK801: {round(npka_mk801, 3)}')

# Normalized PK slope
npks_rest = normalized_pk_slope(pk_rest)
npks_saline = normalized_pk_slope(pk_saline)
npks_mk801 = normalized_pk_slope(pk_mk801)

print(f'\nThe Normalized PK slopes are: \n'
      f'rest: {round(npks_rest, 3)}\n'
      f'saline: {round(npks_saline, 3)}\n'
      f'MK801: {round(npks_mk801, 3)}')

# Primacy-recency index
pri_rest = primacy_recency_index(pk_rest)
pri_saline = primacy_recency_index(pk_saline)
pri_mk801 = primacy_recency_index(pk_mk801)

print(f'\nThe Primacy-recency indexes are: \n'
      f'rest: {round(pri_rest, 3)}\n'
      f'saline: {round(pri_saline, 3)}\n'   
      f'MK801: {round(pri_mk801, 3)}')


labels = ['rest', 'saline', 'MK801']
# Plotting
# plt.figure(constrained_layout=True)

# # ax1 = plt.subplot(1, 3, 1)
# plt.figure(constrained_layout=True)
# plt.plot([npka_rest, npka_saline, npka_mk801], marker='o', linestyle='-', color='k')
# plt.title('Normalized PK area')
# plt.xlabel('Drug')
# plt.ylabel('NPKA')
# plt.xticks([0, 1, 2], labels)
# sns.despine()

# ax2 = plt.subplot(1, 3, 2)
plt.figure(constrained_layout=True)
plt.plot([npks_rest, npks_saline, npks_mk801], marker='o', linestyle='-', color='k')
plt.title('Normalized PK slope')
plt.xlabel('Drug')
plt.ylabel('NPKS')
plt.xticks([0, 1, 2], labels)
sns.despine()

# ax3 = plt.subplot(1, 3, 3)
plt.figure(constrained_layout=True)
plt.plot([pri_rest, pri_saline, pri_mk801], marker='o', linestyle='-', color='k')
plt.title('Primacy-recency index')
plt.xlabel('Drug')
plt.ylabel('PRI')
plt.xticks([0, 1, 2], labels)
sns.despine()
