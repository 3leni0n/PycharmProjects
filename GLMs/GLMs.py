import statsmodels.formula.api as smf

# This fits y based on the regressors a,b individually and the interaction a:b for the dataframe df_dat. So y,a,b are
# the names of the columns you want to relate. summary gives you all the stats in a... well.. summary (edited)
M = smf.ols(formula='y~a*b', data=df_data).fit()
print(M.summary())

