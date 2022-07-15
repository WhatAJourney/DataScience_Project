#%%
import pandas
import statsmodels.api as sm
pandas.set_option('display.max_columns', 10)
pandas.set_option('display.width', 350)
#read from google drive
data = pandas.read_csv('https://drive.google.com/uc?export=download&id=1PXjbqSMu__d_ppEv92i_Gnx3kKgfvhFk')
# make dummy variables from categorical ones. Using one-hot encoding and drop_first=True 
data = pandas.get_dummies(data, drop_first=True)
#add intercept
data['intercept'] = 1
#drop the label
train_cols = data.drop('clicked', axis=1)
#Build LR
logit = sm.Logit(data['clicked'], train_cols)
output = logit.fit()

#%%
output_table = pandas.DataFrame(dict(coefficients = output.params, SE = output.bse, z = output.tvalues, p_values = output.pvalues))
print(output_table.iloc[[1,14]])


#%%
import pandas
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
#group by hour and estimate click rate and total count of emails sent per hour
data_hour = data.groupby('hour')['clicked'].agg(['count', 'mean']).reset_index()
data_hour.plot(x='hour', y='mean')
plt.ylabel("click_rate")
plt.show()

print(data_hour)

#%%
#add new rule-based variable
data['middle_day'] = np.where((data['hour']<18)&(data['hour']>6), 1, 0)
# rebuild the regression
data = pandas.get_dummies(data, drop_first=True)
data['intercept'] = 1
train_cols = data.drop('clicked', axis=1)
logit = sm.Logit(data['clicked'], train_cols)
output = logit.fit()

#%%
output_table = pandas.DataFrame(dict(coefficients = output.params, SE = output.bse, z = output.tvalues, p_values = output.pvalues))
#reorder the output
print(output_table.iloc[[15,1, 14]].append(output_table.drop(output_table.index[[15,1, 14]])))
#%%