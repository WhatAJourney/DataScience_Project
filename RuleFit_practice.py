#%%
import pandas as pandas
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression
from rulefit import RuleFit
  
pandas.set_option('display.max_columns', 10)
pandas.set_option('display.width', 350)
np.random.seed(4684)
  
#read from google drive
data = pandas.read_csv('https://drive.google.com/uc?export=download&id=1PXjbqSMu__d_ppEv92i_Gnx3kKgfvhFk')
# make dummy variables from categorical ones. Using one-hot encoding and drop_first=True 
data = pandas.get_dummies(data, drop_first=True)
#drop the label
train_cols = data.drop('clicked', axis=1)
  
#Extract rules from Random Forest
#set tree forest parameters
rf=RandomForestClassifier(max_depth=2, n_estimators=10, class_weight={0:0.05,1:0.95})
  
#set RuleFit parameters. We are keeping RuleFit pretty small here to make it faster. Increasing max_depth, n_estimators, and setting exp_rand_tree_size = True will generate way more rules and make it somewhat more reliable. As always, there is a trade-off between accuracy and processing time, which should be considered on a case-by-case basis. Note that RuleFit is really slow, so this trade-off is pretty significant here.     
rufi=RuleFit(rfmode="classify", tree_generator=rf, exp_rand_tree_size=False, lin_standardise=False)
#fit RuleFit
rufi.fit(train_cols.values, data['clicked'].values, feature_names = train_cols.columns)
print("We have extracted", rufi.transform(train_cols.values).shape[1], "rules")

#%%
#These are a few of the rules we have extracted
output=rufi.get_rules()
print(output[output['type']=="rule"]['rule'].head().values)

#%%

#X_concat is the new dataset given by the original variables (train_cols.values)
#as well as the new rules extracted from the trees (rufi.transform(train_cols.values))
X_concat = np.concatenate((train_cols, rufi.transform(train_cols.values)), axis=1)
#Build the logistic regression with penalty. This will set low coefficients to zero, so only the relevant ones will survive
log = LogisticRegression(penalty='l1', solver='liblinear', C=0.1)
log.fit(X_concat, data['clicked'])
#get the full output with variables, coefficients, and support
output.iloc[:,2] = np.transpose(log.coef_)
output[output['coef']!=0].sort_values('coef', ascending = False )
# %%
