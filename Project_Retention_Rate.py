#%%
import pandas
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pandas.set_option('display.max_columns', 10)
pandas.set_option('display.width', 350)
  
#read from google drive
data=pandas.read_csv("https://drive.google.com/uc?export=download&id=1uH3Gpzam1pAfhdHIJd9UqraBS009wDer")
  
print(data.head())
# %%
#firstly get count by billing cycle and price point
#size() which counts the number of entries/rows in each group
retention_rate_data_points = data.groupby(['subscription_monthly_cost', 'billing_cycles']).size().reset_index(name = 'tmp_count_billing')

#%%
#then estimate retention rate: cumulative sum by billing cycle divided by total users gives churn rate
#i.e. 1 - churn_rate is actual retention rate
retention_rate_data_points['retention_rate'] = retention_rate_data_points.groupby('subscription_monthly_cost')['tmp_count_billing'].transform(
                                               lambda x: 1-x.cumsum()/x.sum()
                                               )

#%%
# remove tmp_count_billing column and billing cycle == 8 because those are still active
retention_rate_data_points = retention_rate_data_points.drop('tmp_count_billing', 1).query('billing_cycles < 8')
#%%
#plot them
sns.lineplot(x="billing_cycles", y="retention_rate", 
             hue="subscription_monthly_cost", data=retention_rate_data_points, legend="full"
             )
plt.show()
#%%
# create new feature using logs
retention_rate_data_points["billing_log"] = np.log(retention_rate_data_points["billing_cycles"])

retention_rate_data_points["retention_rate_log"] = np.log(retention_rate_data_points["retention_rate"])

#plot 
sns.lineplot(x="billing_log", y="retention_rate_log", 
             hue="subscription_monthly_cost", data=retention_rate_data_points, legend="full"
             )
plt.show()
# %%
import statsmodels.api as sm
  
#build linear model for price 29
y29 = retention_rate_data_points.query('subscription_monthly_cost==29')['retention_rate_log']
X29 = retention_rate_data_points.query('subscription_monthly_cost==29')['billing_log']
#%%
X29 = sm.add_constant(X29)
  

# %%
model29 = sm.OLS(y29, X29)
results29 = model29.fit()
print(results29.summary())
print()
print("R Squared for 29$ is:", round(results29.rsquared,2))
# %%
#build linear model for price 49
y49 = retention_rate_data_points.query('subscription_monthly_cost==49')['retention_rate_log']
X49 = retention_rate_data_points.query('subscription_monthly_cost==49')['billing_log']
X49 = sm.add_constant(X49)
  
model49 = sm.OLS(y49, X49)
results49 = model49.fit()
print("R Squared for 49$ is:", round(results49.rsquared,2))
# %%
#build linear model for price 99
y99 = retention_rate_data_points.query('subscription_monthly_cost==99')['retention_rate_log']
X99 = retention_rate_data_points.query('subscription_monthly_cost==99')['billing_log']
X99 = sm.add_constant(X99)
  
model99 = sm.OLS(y99, X99)
results99 = model99.fit()
print("R Squared for 99$ is:", round(results99.rsquared,2))
# %%

#make the predictions
Xpred = np.log(range(8,13))
Xpred = sm.add_constant(Xpred)
print(results29.predict(Xpred))
# %%
#However, our regression is actually predicting log(retention_rate). To get retention rate, we need to take the exponential of those predictions
print(np.exp(results29.predict(Xpred)))
# %%
print("For the 29$ price point, after 1 year, we will retain", 
      round(np.exp(results29.predict(Xpred))[4], 3), 
     "of our customers")
# %%
#Do the same for the other price points.
print("For the 49$ price point, after 1 year, we will retain", 
      round(np.exp(results49.predict(Xpred))[4], 3), 
     "of our customers")
# %%
print("For the 99$ price point, after 1 year, we will retain", 
      round(np.exp(results99.predict(Xpred))[4], 3), 
     "of our customers")
# %%
#our models work well enough that we can directly use their predictions for all 12 billing cycle to estimate customer LTV.
  
Xpred = np.log(range(1,13))
Xpred = sm.add_constant(Xpred)
#Firstly get retention rate for each billing cycle
retention_rate_tmp = np.exp(results29.predict(Xpred))
#Then get overall expected value by multiplying each probability * corresponding money at each billing cycle.
LTV_29 = 29 + sum(retention_rate_tmp * 29)
print("Avg LTV of customers paying 29$ a month is:", round(LTV_29), "dollars")
# %%
#do it for other price points
retention_rate_tmp = np.exp(results49.predict(Xpred))
LTV_49 = 49 + sum(retention_rate_tmp * 49)
print("Avg LTV of customers paying 49$ a month is:", round(LTV_49), "dollars")
# %%
retention_rate_tmp = np.exp(results99.predict(Xpred))
LTV_99 = 99 + sum(retention_rate_tmp * 99)
print("Avg LTV of customers paying 99$ a month is:", round(LTV_99), "dollars")
# %%


#group by country and price and check retention rate as defined above
country_retention = data.groupby(['country', 'subscription_monthly_cost']).apply(
                    lambda x: pandas.Series({
                             # active users
                            'retention_rate': x['is_active'].mean(),
                             # total count by price and country
                            'count': x['is_active'].count(),
                            # avg revenue by price and country
                            'revenue' : (x['subscription_monthly_cost'] * x['billing_cycles']).mean()
  })
).reset_index()
# %%
#Let's plot retention rate by country and price
sns.barplot(x="country", hue="subscription_monthly_cost", y="retention_rate", data=country_retention)
plt.show()
# %%
#Let's plot user count by country and price
sns.barplot(x="country", hue="subscription_monthly_cost", y="count", data=country_retention)
plt.show()
# %%

#Let's find out for each country avg user revenue within the first 8 months
sns.barplot(x="country", hue="subscription_monthly_cost", y="revenue", data=country_retention)
plt.show()
# %%


#group by source and price and check retention rate as defined above
source_retention = data.groupby(['source', 'subscription_monthly_cost']).apply(
                    lambda x: pandas.Series({
                             # active users
                            'retention_rate': x['is_active'].mean(),
                             # total count by price and source
                            'count': x['is_active'].count(),
                            # avg revenue by price and source
                            'revenue' : (x['subscription_monthly_cost'] * x['billing_cycles']).mean()
  })
).reset_index()
      
#Let's plot retention rate by source and price
sns.barplot(x="source", hue="subscription_monthly_cost", y="retention_rate", data=source_retention)
plt.show()
# %%
#Let's plot user count by source and price
sns.barplot(x="source", hue="subscription_monthly_cost", y="count", data=source_retention)
plt.show()
# %%

#Let's find out for each source avg user revenue within the first 8 months
sns.barplot(x="source", hue="subscription_monthly_cost", y="revenue", data=source_retention)
plt.show()

#%%