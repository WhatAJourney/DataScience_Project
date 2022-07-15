#%%
import pandas
#read from google drive
data = pandas.read_csv("https://drive.google.com/uc?export=download&id=1E8aYgxZpYXO_HxLgQPAGMUS1_fbfTW5d")
print(data.head())
# %%
from scipy import stats
#check conversion rate for both groups
print(data.groupby('test')['conversion'].mean())
# %%
#perform a statistical test between the two groups
test = stats.ttest_ind(data.loc[data['test'] == 1]['conversion'], data.loc[data['test'] == 0]['conversion'], equal_var=False)
#t statistics
print(test.statistic)
# %%
#p-value
print(test.pvalue)
# %%
#print test results
if (test.pvalue>0.05):
  print ("Non-significant results")
elif (test.statistic>0):
  print ("Statistically better results")
else:
  print ("Statistically worse results")
# %%
