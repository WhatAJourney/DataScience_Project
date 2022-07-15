#%%

import pandas
import statsmodels.api as sm
pandas.set_option('display.max_columns', 10)
pandas.set_option('display.width', 350)
  
#Read from google drive. This is the same dataset described in the previous section
data = pandas.read_csv('https://drive.google.com/uc?export=download&id=1PXjbqSMu__d_ppEv92i_Gnx3kKgfvhFk')
  
#Before building the regression, we need to know which ones are the reference levels for the categorical variables
#only keep categorical variables
data_categorical = data.select_dtypes(['object']).astype("category") 
#find reference level, i.e. the first level
print(data_categorical.apply(lambda x: x.cat.categories[0]))
# %%
#make dummy variables from categorical ones. Using one-hot encoding and drop_first=True 
data = pandas.get_dummies(data, drop_first=True)
  
#add intercept
data['intercept'] = 1
#drop the label
train_cols = data.drop('clicked', axis=1)
  
#Build Logistic Regression
logit = sm.Logit(data['clicked'], train_cols)
output = logit.fit()
# %%
output_table = pandas.DataFrame(dict(coefficients = output.params, SE = output.bse, z = output.tvalues, p_values = output.pvalues))
#get coefficients and pvalues
print(output_table)
# %%
#only keep significant variables and order results by coefficient value
print(output_table.loc[output_table['p_values'] < 0.05].sort_values("coefficients", ascending=False))
  
# %%
 