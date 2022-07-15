#%%
#pip install --trusted-host pypi.python.org --trusted-host pypi.org --trusted-host files.pythonhosted.org {package name}
import pandas
import numpy as np
import graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from graphviz import Source
from scipy import stats
  
pandas.set_option('display.max_columns', 20)
pandas.set_option('display.width', 350)
#%%
#set seed to be able to reproduce the results
np.random.seed(4684)
#read from google drive. Again, it is always the email dataset
data = pandas.read_csv('https://drive.google.com/uc?export=download&id=1PXjbqSMu__d_ppEv92i_Gnx3kKgfvhFk')
  
#get dummy variables from categorical ones
data_dummy = pandas.get_dummies(data, drop_first=True).drop('email_id', axis=1)
  
#split into train and test to avoid overfitting
train, test = train_test_split(data_dummy, test_size = 0.34)
  
#build the model. We choose a single decision tree here, but this issue might apply to all models.
#we have a split if impurity (think about it as loss) improves by at least 0.001, which is a very low number, forcing the tree to split pretty much always except for clearly useless splits
tree=DecisionTreeClassifier(min_impurity_decrease=0.001)
tree.fit(train.drop('clicked', axis=1),train['clicked'])

#%%
#visualize it
export_graphviz(tree, out_file="tree.dot", feature_names=train.drop('clicked', axis=1).columns, proportion=True, rotate=True)
with open("tree.dot") as f:
    dot_graph = f.read()
s = Source.from_file("tree.dot")
s.view()
# %%
tree_probabilities = tree.predict_proba(test.drop('clicked', axis=1))[:,1]
print(stats.describe(tree_probabilities))
# %%
# RF weights can be passed as class weights inside RandomForestClassifier. 
#Then, for each weight configuration, we save class errors and accuracy. 
#Finally, pick the best combination of both class errors 
  
#Build 20 RF models with different weights
class0_error = []
class1_error = []
accuracy = []
  
#apply weights from 10 to 200 with 10 as a step
for i in range(10,210,10) :
               rf = RandomForestClassifier(n_estimators=50, oob_score=True, class_weight={0:1,1:i})
               rf.fit(train.drop('clicked', axis=1), train['clicked'])
               #let's get confusion matrix
               conf_matrix = pandas.DataFrame(confusion_matrix(test['clicked'], rf.predict(test.drop('clicked', axis=1)), labels=[0, 1]))
               #class0/1 errors are 1 -  (correctly classified events/total events belonging to that class)
               class0_error.append( 1 - conf_matrix.loc[0,0]/(conf_matrix.loc[0,0]+conf_matrix.loc[0,1]))
               class1_error.append( 1 - conf_matrix.loc[1,1]/(conf_matrix.loc[1,0]+conf_matrix.loc[1,1]))
               accuracy.append((conf_matrix.loc[1,1]+conf_matrix.loc[0,0])/conf_matrix.values.sum())
  
dataset_weights = pandas.DataFrame ({'minority_class_weight': pandas.Series(range(10,210,10)),
                                     'class0_error': pandas.Series(class0_error),
                                     'class1_error': pandas.Series(class1_error),
                                     'accuracy':     pandas.Series(accuracy)
                                   })
  
print(dataset_weights)
# %%
#Calculate trade-ff between class errors
dataset_weights['optimal_value'] = 1 - dataset_weights['class1_error'] - dataset_weights['class0_error']
  
#Order by optimal_value and pick the first row
print(dataset_weights.sort_values('optimal_value', ascending=False).head(1))

