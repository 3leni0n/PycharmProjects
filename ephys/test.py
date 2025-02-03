import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

from my_fun.my_fun import pval_to_star, add_stars


cortex_clusters = cluster_info[cluster_info.depth <= 1500].cluster_id
deep_clusters = cluster_info[cluster_info.depth > 1500].cluster_id

sync_cortex = get_sync(df_spikes[df_spikes.cluster.isin(cortex_clusters)], df_ttl, time_win=[-1.5, -0.5], bin_size=0.02,
                       method='anal', smooth=False)
sync_deep = get_sync(df_spikes[df_spikes.cluster.isin(deep_clusters)], df_ttl, time_win=[-1.5, -0.5], bin_size=0.02,
                     method='anal', smooth=False)

fig, ax = plt.subplots(figsize=figsize)
ax.plot(sync_cortex, color='tab:gray', label='cortex')
ax.plot(sync_cortex.argmin(), sync_cortex.min(), marker='o', color='k')
ax.plot(sync_cortex.argmax(), sync_cortex.max(), marker='o', color='k')
print(sync_cortex.argmin(), sync_cortex.min())
print(sync_cortex.argmax(), sync_cortex.max())
ax.plot(sync_deep, color='tab:pink', label='deep')
ax.plot(sync_deep.argmin(), sync_deep.min(), marker='o', color='tab:red')
ax.plot(sync_deep.argmax(), sync_deep.max(), marker='o', color='tab:red')
print(sync_deep.argmin(), sync_deep.min())
print(sync_deep.argmax(), sync_deep.max())
ax.set_xlabel('deeps)')
ax.set_ylabel('Sync')
ax.set_title(f'{df_behavior.Session.unique()[0]} ({method} method)')
ax.legend(frameon=False)



res = pearsonr(sync_cortex, sync_deep)
plt.figure()
plt.scatter(sync_cortex, sync_deep)
plt.gca().get_xlim()[1]
plt.gca().get_ylim()[1]
ax_min = np.min((plt.gca().get_xlim()[0], plt.gca().get_ylim()[0]))
ax_max = np.max((plt.gca().get_xlim()[1], plt.gca().get_ylim()[1]))
plt.xlim(ax_min, ax_max)
plt.ylim(ax_min, ax_max)
plt.xlabel('Cortex')
plt.ylabel('Deep')
plt.gca().set_aspect('equal')
# Plot equity line
plt.plot([ax_min, ax_max], [ax_min, ax_max], color='black')
plt.title(f'R = {round(res.statistic, 2)}, p = {round(res.pvalue, 2)})')
# Set the same ticks on both axes with only integers
plt.xticks(np.arange(round(ax_min), int(ax_max) + 1, 1))
plt.yticks(np.arange(round(ax_min), int(ax_max) + 1, 1))



cch_cortex, lags_cortex = plot_autocorrelogram(df_spikes[df_spikes.cluster.isin(cortex_clusters)], bin_size=0.001,
                                               window=[-1000, 1000], cross_corr_coeff=True, ax=None)
plt.title('Cortex')
cch_deep, lags_deep = plot_autocorrelogram(df_spikes[df_spikes.cluster.isin(deep_clusters)], bin_size=0.001,
                                           window=[-1000, 1000], cross_corr_coeff=True, ax=None)
plt.title('Deep')

plt.figure()
plt.plot(lags_cortex, cch_cortex, color='tab:gray', label='cortex')
plt.plot(lags_deep, cch_deep, color='tab:pink', label='deep')
plt.title('Autocorrelogram')
plt.xlabel('Time lag (ms)')
plt.ylabel('Correlation')
plt.legend(frameon=False)





# MAke a figure with 2 subplots and plot syn2 raw and sync smooth
fig, ax = plt.subplots(2, 1, constrained_layout=True)
ax[0].plot(sync_smooth, c='k')
ax[1].plot(sync_raw, c='k')
ax[0].set_ylabel('Sync (smooth)')
ax[1].set_ylabel('Sync (raw)')
ax[1].set_xlabel('Trial')
plt.suptitle(f'{df_behavior.Session.unique()[0]}')
