"""
Notes

Bootstrapping uses brute computational force to simulate replicating an experiment, which is expensive and
time-consuming. It is an alternative to statistical inference based on parametric assumptions. Given a dataset of
size N samples, we randomly draw the same N samples with replacement (drawing without replacement will lead to the
exact same dataset from which we are drawing samples, which is pointless) to get a new, bootstrapped dataset. This
new bootstrapped dataset gives us a hint of what would look like if we would repeat the experiment. We can now
calculate statistics (mean, median, std, sem). Repeat the process many times, like 1-10k. This is bootstrapping.
Bootstrapping can be used to estimate the underlying distribution from a sample, by treating that sample like the
population from which to draw samples from. If we calculate the mean of every bootstrapped dataset and save it,
the std of that bootstrapped distribution is the same as the sem from the original dataset, and a 95% confidence
interval (CI) is just an interval that covers 95% of the bootstrapped means. If the 95% CI covers 0, we can't reject
H0 (hypothesis testing). We can apply bootstrapping to any statistc, not only the mean, without using formulas.

For calculating p-values, first calculate the mean (or any other statistic) of the dataset if the null hypothesis (H0)
was true. Then shift all the values of the dataset the H0 mean units so that the mean of the shifted data is the same as
the mean if H0 was true. Use bootstrapping to see how the mean varies under H0 varies. Make a histogram of the means and
use this distribution to get p-values.

4 steps:
1. Make a bootstrapped dataset
2. Calculate something
3. Keep track of that calculation
4 Repeat  steps 1-3 a bunch of times

Resources:
Bootstrapping Main Ideas!!! - StatQuest: https://www.youtube.com/watch?v=Xz0x-8-cgaQ
Using Bootstrapping to Calculate p-values!!! - StatQuest: https://www.youtube.com/watch?v=N4ZQQqyIf6k&t=8s
https://en.wikipedia.org/wiki/Bootstrapping_%28statistics%29
http://allendowney.blogspot.com/2011/05/there-is-only-one-test.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html
http://pillowlab.princeton.edu/teaching/mathtools16/slides/lec21_Bootstrap.pdf
"""


from scipy.stats import bootstrap

indexes = df.reset_index().index.values

bootstrap_samples = []
for _ in range(10000):
    x = np.random.choice(indexes, size=len(indexes), replace=True)
    bootstrap_samples.append(x.mean())

test = df.sample(replace=False)


import statsmodels.stats.api as sms
sms.DescrStatsW(bootstrap_samples).tconfint_mean()
