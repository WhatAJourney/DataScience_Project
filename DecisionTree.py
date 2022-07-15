#%%
import pandas 
import graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from graphviz import Source
#%%  
#Read from google drive. Always the same dataset.
data = pandas.read_csv('https://drive.google.com/uc?export=download&id=1PXjbqSMu__d_ppEv92i_Gnx3kKgfvhFk')
  
#prepare the data for the model by creating dummy vars and removing the label
data_dummy = pandas.get_dummies(data, drop_first=True)
train_cols = data_dummy.drop('clicked', axis=1)
  
#build the tree
tree=DecisionTreeClassifier(
    #set max tree dept at 4. Bigger than that it just becomes too messy
    max_depth=4,
    #change weights given that we have unbalanced classes. Our data set is now perfectly balanced. It makes easier to look at tree output
    class_weight="balanced",
    #only split if it's worthwhile. The default value of 0 means always split no matter what if you can increase overall performance, which might lead to irrelevant splits
    min_impurity_decrease = 0.001
    )
tree.fit(train_cols,data_dummy['clicked'])
#%%
#visualize it
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz/bin'

export_graphviz(tree, out_file="tree.dot", feature_names=train_cols.columns, proportion=True, rotate=True)
s = Source.from_file("tree.dot")
s.view()

# %%
