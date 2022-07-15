#%%
import pandas
pandas.set_option('display.max_columns', 20)
pandas.set_option('display.width', 350)
#read from google drive
data = pandas.read_csv("https://drive.google.com/uc?export=download&id=1jYFe4qjaQ1ZZZrqJ2R8Nu-eRBjhAfsoL")
data.head()

# %%
#let's group by source and estimate relative frequencies
data_grouped_source = data.groupby("source")["test"].\
    agg({lambda x: len(x[x==0]), lambda x: len(x[x==1])} )
data_grouped_source=pandas.DataFrame(data_grouped_source)  
data_grouped_source = data_grouped_source.rename(columns={"<lambda_0>": "frequency_test_0","<lambda_1>": "frequency_test_1"})
#get relative frequencies
# %%
print(data_grouped_source/data_grouped_source.sum())
# %%
import graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from graphviz import Source
  
#drop user_id, not needed
data = data.drop(['user_id'], axis=1)
#make dummy vars. Don't drop one level here, keep them all. You don't want to risk dropping the one level that actually creates problems with the randomization
data_dummy = pandas.get_dummies(data)
#model features, test is the label and conversion is not needed here
train_cols = data_dummy.drop(['test', 'conversion'], axis=1)
  
tree=DecisionTreeClassifier(
    #change weights. Our data set is now perfectly balanced. It makes easier to look at tree output
    class_weight="balanced",
    #only split if if it's worthwhile. The default value of 0 means always split no matter what if you can increase overall performance, which creates tons of noisy and irrelevant splits
    min_impurity_decrease = 0.001
    )
tree.fit(train_cols,data_dummy['test'])
  
export_graphviz(tree, out_file="tree_test.dot", feature_names=train_cols.columns, proportion=True, rotate=True)
s = Source.from_file("tree_test.dot")
s.view()
# %%
print(data_dummy.groupby("test")[["country_Argentina", "country_Uruguay"]].mean())
# %%
from scipy import stats
  
#this is the test results using the orginal dataset
original_data = stats.ttest_ind(data_dummy.loc[data['test'] == 1]['conversion'], 
                                data_dummy.loc[data['test'] == 0]['conversion'], 
                                equal_var=False)
  
#this is after removing Argentina and Uruguay
data_no_AR_UR = stats.ttest_ind(data_dummy.loc[(data['test'] == 1) & (data_dummy['country_Argentina'] ==  0) & (data_dummy['country_Uruguay'] ==  0)]['conversion'], 
                                data_dummy.loc[(data['test'] == 0) & (data_dummy['country_Argentina'] ==  0) & (data_dummy['country_Uruguay'] ==  0)]['conversion'], 
                                equal_var=False)
  
print(pandas.DataFrame( {"data_type" : ["Full", "Removed_Argentina_Uruguay"], 
                         "p_value" : [original_data.pvalue, data_no_AR_UR.pvalue],
                         "t_statistic" : [original_data.statistic, data_no_AR_UR.statistic]
                         }))
# %%
