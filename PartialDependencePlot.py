#%%
import pandas 
import numpy as np
from pdpbox import pdp, info_plots
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
pandas.set_option('display.max_columns', 10)
pandas.set_option('display.width', 350)
#%%
np.random.seed(4684)
  
#read data from google drive. It is the same dataset we used in the previous sections
data = pandas.read_csv('https://drive.google.com/uc?export=download&id=1PXjbqSMu__d_ppEv92i_Gnx3kKgfvhFk')
  
#prepare the data for the model
data = pandas.get_dummies(data, drop_first=True)
train_cols = data.drop('clicked', axis=1)

#%%
#Build the model. Here we just care about showing how to extract insights. 
# Optimizing the model is beyond the scope of this section. 
# Only thing we do beside default options is 
# (a) increasing the weight of minority class via class_weight to make sure the trees split on something and 
# (b) increase the number of trees
rf = RandomForestClassifier(class_weight={0:0.05,1:0.95}, n_estimators=50)
rf.fit(train_cols, data['clicked'])
  
#Let's build the PDP for the variable email_version. It is going to be fast since it has just two unique values. But the entire procedure would be exactly the same for any other categorical or numerical variable, would just be more computationally expensive
pdp_version = pdp.pdp_isolate( model=rf, dataset=train_cols, model_features=list(train_cols), feature='email_version_personalized')
  
#Let's get the actual values.
print(pdp_version.pdp)
#%%

fig, axes = pdp.pdp_plot(pdp_version, 'email_version_personalized', center=False)
_ = axes['pdp_ax'].set_xticklabels(['generic', 'personalized'])
plt.show()# %%

# %%


