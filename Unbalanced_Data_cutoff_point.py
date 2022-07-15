#%%
import pandas
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
pandas.set_option('display.max_columns', 20)
pandas.set_option('display.width', 350)
  
#set seed to be able to reproduce the results
np.random.seed(4684)
#read from google drive. Again, it is always the email dataset
data = pandas.read_csv('https://drive.google.com/uc?export=download&id=1PXjbqSMu__d_ppEv92i_Gnx3kKgfvhFk')
  
#get dummy variables from categorical ones
data_dummy = pandas.get_dummies(data, drop_first=True).drop('email_id', axis=1)
  
#split into train and test to avoid overfitting
train, test = train_test_split(data_dummy, test_size = 0.34)
  
#build the model. We choose a RF, but this issues applies to pretty much all models
rf = RandomForestClassifier(n_estimators=50, oob_score=True)
rf.fit(train.drop('clicked', axis=1), train['clicked'])
#let's print OOB confusion matrix
print(pandas.DataFrame(confusion_matrix(train['clicked'], rf.oob_decision_function_[:,1].round(), labels=[0, 1])))

# %%
#and let's print test set confusion matrix
print(pandas.DataFrame(confusion_matrix(test['clicked'], rf.predict(test.drop('clicked', axis=1)), labels=[0, 1])))
# %%
#confusion matrix test set
conf_matrix = pandas.DataFrame(confusion_matrix(test['clicked'], rf.predict(test.drop('clicked', axis=1)), labels=[0, 1]))
#class0/1 errors are 1 -  (correctly classified events/total events belonging to that class)
class0_error = 1 - conf_matrix.loc[0,0]/(conf_matrix.loc[0,0]+conf_matrix.loc[0,1])
class1_error = 1 - conf_matrix.loc[1,1]/(conf_matrix.loc[1,0]+conf_matrix.loc[1,1])
  
print(pandas.DataFrame( {'test_class0_error':[class0_error],
                        'test_class1_error':[class1_error]
}))
# %%

from sklearn.metrics import roc_curve
  
#get test set predictions as a probability
pred_prob=rf.predict_proba(test.drop('clicked', axis=1))[:,1]
#get false positive rate and true positive rate, for different cut-off points
#and let's save them in a dataset. 
fpr, tpr, thresholds = roc_curve(test['clicked'],pred_prob)
#For consistency with R, we will focus on class errors, defined as
# class0_error = fpr and class1_error = 1 - tpr
error_cutoff=pandas.DataFrame({'cutoff':pandas.Series(thresholds),
                               'class0_error':pandas.Series(fpr),
                               'class1_error': 1 - pandas.Series(tpr)
                                })
#let's also add accuracy to the dataset, i.e. overall correctly classified events.
#This is: (tpr * positives samples in the test set + tnr * positive samples in the dataset)/total_events_in_the_data_set
error_cutoff['accuracy']=((1-error_cutoff['class0_error'])*sum(test['clicked']==0)+(1-error_cutoff['class1_error'])*sum(test['clicked']==1))/len(test['clicked'])
  
print(error_cutoff)
# %%
#let's check best combination of class0 and class1 errors
error_cutoff['optimal_value'] = 1 - error_cutoff['class1_error'] - error_cutoff['class0_error']
  
print(error_cutoff.sort_values('optimal_value', ascending=False).head(20))
# %%
#we already have predicted probabilities from the previous step, i.e.
#pred_prob=rf.predict_proba(test.drop('clicked', axis=1))[:,1]
  
#let's create a 0/1 vector according to the 0.002 cutoff
best_cutoff = error_cutoff.sort_values('optimal_value', ascending=False)['cutoff'].values[0]
predictions=np.where(pred_prob>=best_cutoff,1,0)
#get confusion matrix for those predictions
#confusion matrix test set
conf_matrix_new = pandas.DataFrame(confusion_matrix(test['clicked'], predictions, labels=[0, 1]))
print(conf_matrix_new)
# %%
class0_error = 1 - conf_matrix_new.loc[0,0]/(conf_matrix_new.loc[0,0]+conf_matrix_new.loc[0,1])
class1_error = 1 - conf_matrix_new.loc[1,1]/(conf_matrix_new.loc[1,0]+conf_matrix_new.loc[1,1])
print(pandas.DataFrame( {'cutoff':[best_cutoff],
                         'test_class0_error_new':[class0_error],
                         'test_class1_error_new':[class1_error]
}))