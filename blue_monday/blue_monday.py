import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

path = '/home/alexis/Documentos/intersession reports/2AFC_2/333_intersession.csv'
df = pd.read_csv(path)

# Create DataFrame per day of the week
df_mondays = df[df.DoW == 0]
df_tuesdays = df[df.DoW == 1]
df_wednesdays = df[df.DoW == 2]
df_thursdays = df[df.DoW == 3]
df_fridays = df[df.DoW == 4]
df_saturdays = df[df.DoW == 5]
df_sundays = df[df.DoW == 6]

# Trials per day of the week
trials_mondays = df_mondays.Trials.mean()
trials_tuesdays = df_tuesdays.Trials.mean()
trials_wednesdays = df_wednesdays.Trials.mean()
trials_thursdays = df_thursdays.Trials.mean()
trials_fridays = df_fridays.Trials.mean()
trials_saturdays = df_saturdays.Trials.mean()
trials_sundays = df_sundays.Trials.mean()

# Water per day of the week
water_mondays = df_mondays.Water.mean()
water_tuesdays = df_tuesdays.Water.mean()
water_wednesdays = df_wednesdays.Water.mean()
water_thursdays = df_thursdays.Water.mean()
water_fridays = df_fridays.Water.mean()
water_saturdays = df_saturdays.Water.mean()
water_sundays = df_sundays.Water.mean()

# Accuracy per day of the week
acc_mondays = df_mondays.Accuracy.mean()
acc_tuesdays = df_tuesdays.Accuracy.mean()
acc_wednesdays = df_wednesdays.Accuracy.mean()
acc_thursdays = df_thursdays.Accuracy.mean()
acc_fridays = df_fridays.Accuracy.mean()
acc_saturdays = df_saturdays.Accuracy.mean()
acc_sundays = df_sundays.Accuracy.mean()

# endog, exog, what’s that? (https://www.statsmodels.org/stable/endog_exog.html)
# endog: y variable, left hand side (LHS), dependent variable, regressand, outcome, response variable
# exog: x variable, right hand side (RHS), independent variable, regressors, design, explanatory variable

# Load data
# endog = df.Water
endog = df.Trials
# endog = df.Accuracy
exog = df.DoW
exog = sm.add_constant(exog)

# Fit and summarize OLS model
# mod = sm.OLS(endog, exog)  # Need to add constant
mod = smf.ols(formula='Water~DoW', data=df)  # Adds intercept by default, but skip it if there was one
# mod = smf.ols(formula='Water~C(DoW)', data=df)  # C() operator for integer variables that we want to treat specifically
# as categorical (https://www.statsmodels.org/stable/example_formulas.html)
# mod = smf.ols(formula='Water~C(DoW) - 1', data=df)  # -1 to remove the intercept from the model
# mod = smf.ols(formula='df.Water~df.DoW*df.Setup', data=df)  # If across animals include the interaction
mod = smf.ols(formula='Trials~DoW', data=df)  # Adds intercept by default, but skip it if there was one
mod = smf.ols(formula='Accuracy~DoW', data=df)  # Adds intercept by default, but skip it if there was one

res = mod.fit()
print(res.summary())
