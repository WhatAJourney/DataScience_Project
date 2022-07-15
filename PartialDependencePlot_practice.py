#%%

#Same steps as in the previous section to get the dataset ready
import pandas 
import numpy as np
from pdpbox import pdp, info_plots
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from matplotlib import rcParams
pandas.set_option('display.max_columns', 10)
rcParams.update({'figure.autolayout': True})
  
np.random.seed(4684)
data = pandas.read_csv('https://drive.google.com/uc?export=download&id=1PXjbqSMu__d_ppEv92i_Gnx3kKgfvhFk')
  
#prepare the data for the model
data_dummy = pandas.get_dummies(data, drop_first=True)
train_cols = data_dummy.drop('clicked', axis=1)
  
#build the model. Again, here we just care about showing how to extract insights. Optimizing the model is beyond the scope of this section. Only thing we do beside default options is increasing the weight of minority class via classwt to make sure the trees split on something and choose 50 as tree number
rf = RandomForestClassifier(class_weight={0:0.05,1:0.95}, n_estimators=50)
rf.fit(train_cols, data_dummy['clicked'])
  
#estimate variable importance
feat_importances = pandas.Series(rf.feature_importances_, index=train_cols.columns)
feat_importances.sort_values(ascending=True).plot(kind='barh')
plt.show()
# %%

#Let's plot all of them. Note that pdpbox allows to group together dummy variables coming from the same categorical variable. For this we need to get the original variable names.
feat_original = data.columns.drop('clicked')
  
#%%
#plot all variables with a for loop. If a variable is categorical make sure to plot all levels together.
for i in range(len(feat_original)):
    #get all variables that include the name in feat_original. So, if it is numeric, just take that variable. If it is categorical, take all dummies belonging to the same categorical variable. Since for dummies their name is variable_level, this is easy. Just pick all variables that start with the original variable name. I.e. to get all weekday dummies (weekday_Monday, weekday_Tuesday, etc.), we just look for all variables starting with "weekday"
    #variables to plot
    plot_variable = [e for e in list(train_cols) if e.startswith(feat_original[i])]
      
    #numeric variables or dummy with just 1 level
    if len(plot_variable) == 1:
       pdp_iso = pdp.pdp_isolate( model=rf, dataset=train_cols, model_features=list(train_cols), feature=plot_variable[0], num_grid_points=50)
       pdp_dataset = pandas.Series(pdp_iso.pdp, index=pdp_iso.feature_grids)
       #pdpbox has several options if you want to use their built-in plots. I personally prefer just using .plot. It is totally subjective obviously.
       pdp_dataset.plot(title=feat_original[i])
       plt.show()
         
    #categorical variables with several levels
    else:
       pdp_iso = pdp.pdp_isolate( model=rf, dataset=train_cols, model_features=list(train_cols),  feature=plot_variable, num_grid_points=50)
       pdp_dataset = pandas.Series(pdp_iso.pdp, index=pdp_iso.display_columns)
       pdp_dataset.sort_values(ascending=False).plot(kind='bar', title=feat_original[i])
       plt.show()
    plt.close()   
# %%
