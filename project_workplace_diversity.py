#%%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 350)
os.chdir("c:/Users/Mr.Goldss/Desktop/datamasked/diversity")

print(os.getcwd())

company=pd.read_csv("company_hierarchy.csv")
employee=pd.read_csv("employee.csv")

# %%
#review the current data for missing and mistake
print(company.describe(include='all'))
print()
print(employee.describe(include='all'))
print()
print(company.isnull().sum())
#the one boss id which is meissing shuold be the CEO (make sense)
print()
print(employee.isnull().sum())



#%%
#Q1: identify level.

#At first, we set everyone as IC, then we keep updating this vector
company['level'] = "IC"
  
#identify the CEO
company.loc[company.dept == "CEO", "level"] = "CEO"
  
#assign everyone else based on their boss level
#company_levels = ["CEO","E"]
company_levels = ["CEO", "E", "VP", "D", "MM"]
#%%
  
for i in range(1,len(company_levels)):
  #identify IDs of the boss. This is employee ID of the level above
  boss_id = company.loc[company.level == company_levels[i-1], 'employee_id']
  company.loc[company.boss_id.isin(boss_id), "level"] = company_levels[i]
  

#frequency
print(company.level.value_counts())


#%%


#At first, we set 0. This will be true for ICs, then we keep updating this vector for all others
#At first, we set 0. This will be true for ICs, then we keep updating this vector for all others
company['num_reports'] = 0
  
#same as before, but now we start from the bottom
company_levels = ["IC", "MM", "D", "VP", "E"]
  
i=0
while i<len(company_levels):
  #this is the count of direct reports + the prior count so we take into account direct reports of direct reports, etc.
  level_count=company.loc[company.level == company_levels[i]].groupby('boss_id')['num_reports'].agg(lambda x: x.count() + x.sum())
                                         
  #join to the main table to get the new report count for the bosses from the step above
  company=pd.merge(left=company, right=level_count, how='left', left_on="employee_id", right_on="boss_id", suffixes=('', '_updated'))
  company['num_reports'] = company.num_reports_updated.combine_first(company.num_reports)
  #we can delete this now that we have updated the count
  del company['num_reports_updated']
  i+=1

  
print(company.sort_values(by=['num_reports'], ascending=False).head(6))
print()
print(company.sort_values(by=['num_reports'], ascending=False).tail(6))

# %%

#Let's join the two datasets
data=pd.merge(left=company, right=employee)
#And get rid of the columns we don't care about, i.e. the IDs and CEO, which is just one row so hardly useful
data=data.query("dept!=\"CEO\"").drop(["employee_id","boss_id"], axis=1)
  
#Level and degree are actually conceptually ordered. So in this case it can make sense to replace them with numbers that represent the rank
codes = {"High_School":1, "Bachelor":2, "Master":3, "PhD":4}
data['degree_level'] = data['degree_level'].map(codes)
  
codes = {"IC":1, "MM":2, "D":3,"VP":4, "E":5}
data['level'] = data['level'].map(codes)
  
#Let's make sex numerical manually
data['is_male'] = np.where(data['sex']=='M', 1,  0)
del data['sex']

# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.model_selection import train_test_split
np.random.seed(1234)
  
#dummy variables for the categorical one, i.e. dept. We keep all levels here. It is not going to affect RF badly, after all it does well with correlated variables and it is just one variable with few levels. And this is going to make insights much cleaner
data_dummy = pd.get_dummies(data)
  
#split into train and test to avoid overfitting
train, test = train_test_split(data_dummy, test_size = 0.34)  
  
#Will will use standard params here. They are likely close to optimal and changing them won't make much of a different in terms of insights, which is our goal here
rf = RandomForestRegressor(oob_score=True, n_estimators = 100)
rf.fit(train.drop('salary', axis=1), train['salary'])  
# %%
print("MSE on OOB is", metrics.mean_squared_error(train['salary'], rf.oob_prediction_).round(), "and on test set is", metrics.mean_squared_error(test['salary'], rf.predict(test.drop('salary', axis=1))).round())
  
print("% explained variance on OOB is", metrics.explained_variance_score(train['salary'], rf.oob_prediction_).round(2), "and on test set is", metrics.explained_variance_score(test['salary'], rf.predict(test.drop('salary', axis=1))).round(2))
# %%
#deciles
print(np.percentile(train['salary'], np.arange(0, 100, 10))) 
# %%
#Let's check variable importance
feat_importances = pd.Series(rf.feature_importances_, index=train.drop('salary', axis=1).columns)
feat_importances.sort_values().plot(kind='barh')
plt.show()
# %%
#Let's check partial dependence plots of the top 2 variables: dept and yrs_experience, as well as sex
from pdpbox import pdp, info_plots
  
#dept
pdp_iso = pdp.pdp_isolate(model=rf, 
                          dataset=train.drop('salary', axis=1),      
                          model_features=list(train.drop('salary', axis=1)), 
                          feature=['dept_HR', 'dept_engineering', 'dept_marketing', 'dept_sales'], 
                          num_grid_points=50)
pdp_dataset = pd.Series(pdp_iso.pdp, index=pdp_iso.display_columns)
pdp_dataset.sort_values(ascending=False).plot(kind='bar', title='Partial Plot for Dept')
plt.show()
# %%
#yrs_experience
pdp_iso = pdp.pdp_isolate(model=rf, 
                          dataset=train.drop('salary', axis=1),      
                          model_features=list(train.drop('salary', axis=1)), 
                          feature='yrs_experience', 
                          num_grid_points=50)
pdp_dataset = pd.Series(pdp_iso.pdp, index=pdp_iso.feature_grids)
pdp_dataset.plot(title='Partial Plot for Years of Experience')
plt.show()

#is_male
pdp_iso = pdp.pdp_isolate(model=rf, 
                          dataset=train.drop('salary', axis=1),      
                          model_features=list(train.drop('salary', axis=1)), 
                          feature='is_male', 
                          num_grid_points=50)
pdp_dataset = pd.Series(pdp_iso.pdp, index=pdp_iso.feature_grids)
pdp_dataset.sort_values(ascending=False).plot(kind='bar',title='Partial Plot for is_male')
plt.show()

#%%

#avg salary males vs females
print(data.groupby('is_male')['salary'].mean())

#%%#avg salary males vs females by dept
print(data.groupby(['dept','is_male'])['salary'].agg({'mean', 'count'}))

#%%