#%%
import pandas
pandas.set_option('display.max_columns', 20)
pandas.set_option('display.width', 350)
  
#read from google drive
data= pandas.read_csv("https://drive.google.com/uc?export=download&id=10LkHByquDZAf7K7krvqHZfYOQanChVjF")
print(data.head())
# %%
from scipy import stats
#t-test of test vs control for our target metric 
test = stats.ttest_ind(data.loc[data['test'] == 1]['pages_visited'], data.loc[data['test'] == 0]['pages_visited'], equal_var=False)
  
#t statistics
print(test.statistic)

#p-value
print(test.pvalue) 

#print test results
if (test.pvalue>0.05):
  print ("Non-significant results")
elif (test.statistic>0):
  print ("Statistically better results")
else:
  print ("Statistically worse results")
# %%
