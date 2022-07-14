# Permutation test
from scipy.stats import permutation_test

shuffled_var = []

for _ in range(20):

    # bs = np.random.choice(indexes, size=len(indexes), replace=True)
    # df_bs = df.sample(frac=1, replace=True)  # Bootstrapped df

    # choices = choices.sample(frac=1).reset_index(drop=True)
    df_shuffled = stim_strength.sample(frac=1).reset_index(drop=True)
    # model = sm.Logit(choices, stim_strength)  # Discrete Logit model
    model = sm.GLM(choices, df_shuffled,
                   family=sm.families.Binomial())  # GLM with Binomial family and Logit link
    results = model.fit()
    var = results.params

    shuffled_var.append(var)
    plt.plot(np.arange(1, len(var)), var.iloc[1:11], color='tab:gray', marker=None, mfc='none', mec='none', mew=0,
             ms=0, label=label, alpha=0.5)

percentiles = np.percentile(shuffled_var, 95, axis=0)  # Get upper 5 percentile of the shuffled_var
shuffled_var = np.array(shuffled_var)



plt.plot(np.arange(1, len(params)), params.iloc[1:11], color=color, marker='o', label=label)
plt.errorbar(np.arange(1, len(params)), params.iloc[1:11], yerr=beta_std_err.iloc[1:11], color=color,
             marker='o', fmt='none', mec='none', ms=0)  # Without constant (bias)

mean_shuffles = np.mean(shuffled_var, axis=0)[1:11]
plt.plot(np.arange(1, len(params)), mean_shuffles, color='tab:gray', ls='--', zorder=1.8)
plt.plot(np.arange(1, len(params)), percentiles[1:11], color='tab:red', ls=':', zorder=1.9)









plt.figure(constrained_layout=True)

# Plot all shuffled_var
for i in range(len(shuffled_var)):
    plt.plot(np.arange(1, len(params)), shuffled_var[i][1:], color='tab:gray', marker=None, mfc='none', mec='none', mew=0,
             ms=0, label=label, alpha=0.1)

# Get the shuffled_var that are above the 5 percentile
upper_shuffles = [shuffled_var[np.where(shuffled_var[:, i] > percentiles[i])][:, i] for i in range(len(shuffled_var[0]))]
upper_shuffles = np.array(upper_shuffles)
upper_shuffles = upper_shuffles.T

# Plot upper 5% shuffled_var
for i in range(len(upper_shuffles) - 1):
    plt.plot(np.arange(1, len(params)), upper_shuffles[i][1:], color='tab:gray', marker=None, mfc='none', mec='none', mew=0,
             ms=0, label=label, alpha=0.5)

# Plot kernel
plt.plot(np.arange(1, len(params)), params_test.iloc[1:11], color='k', marker=None, mfc='none', mec='none', mew=0,
         ms=0, label=label)

plt.errorbar(np.arange(1, len(params)), params_test.iloc[1:11], yerr=beta_std_err.iloc[1:11], color=color,
             marker=None, fmt='none', mfc='none', mec='none', ms=0, capsize=10, zorder=3)  # Without constant (bias)

mean_upper_shuffles = np.mean(upper_shuffles, axis=0)
std_upper_shuffles = np.std(upper_shuffles, axis=0)

plt.plot(np.arange(1, len(params)), mean_upper_shuffles[1:], color='tab:pink')
plt.errorbar(np.arange(1, len(params)), mean_upper_shuffles[1:11], yerr=std_upper_shuffles[1:11], color='tab:pink',
             marker=None, fmt='none', mfc='none', mec='none', ms=0, capsize=10, zorder=3)  # Without constant (bias)

plt.title(f'Mouse {df.Setup.unique()[0]}, {len(df)} trials')
plt.xlabel('Stimulus frame')
plt.ylabel('GLM weight (z-scored)')
# plt.legend()
yticks = plt.gca().get_yticks()  # Get current axis yticks for the significance annotations
