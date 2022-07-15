#%%
import pandas
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
pandas.set_option('display.max_columns', 20)
pandas.set_option('display.width', 350)
#%%
  
#set seed to be able to reproduce the results
np.random.seed(4684)
#read data 
data = pandas.read_csv('https://drive.google.com/uc?export=download&id=1PXjbqSMu__d_ppEv92i_Gnx3kKgfvhFk')
  
#Bin the variables accoding to the rules described above
#Hour
data['hour_binned']=pandas.cut(data['hour'], bins=[1,5, 13, 21, 24], include_lowest=True, labels=['night', 'morning', 'afternoon', 'night2'])
#replace night2 with night
data['hour_binned']=data['hour_binned'].replace('night2', 'night').cat.remove_unused_categories()
  
#Bin purchases
data['purchase_binned']=pandas.cut(data['user_past_purchases'], bins=[0,1, 4, 8, 23], include_lowest=True, right=False, labels=['None', 'Low', 'Medium', 'High'])
  
#prepare the data for the model
data_dummy = pandas.get_dummies(data, drop_first=True).drop(['email_id', 'hour', 'user_past_purchases'], axis=1)
  
#split into train and test to avoid overfitting
train, test = train_test_split(data_dummy, test_size = 0.34)
  
#build the model. We choose a RF, but this personalization approach works with any kinds of models
rf = RandomForestClassifier(class_weight={0:0.05,1:0.95}, n_estimators=50, oob_score=True)
rf.fit(train.drop('clicked', axis=1), train['clicked'])
  
#let's print OOB confusion matrix
print(pandas.DataFrame(confusion_matrix(train['clicked'], rf.oob_decision_function_[:,1].round(), labels=[0, 1])))
# %%
#and let's print test set confusion matrix
print(pandas.DataFrame(confusion_matrix(test['clicked'], rf.predict(test.drop('clicked', axis=1)), labels=[0, 1])))
# %%
#We remove the label, we don't need it here
data_unique = data_dummy.drop(['clicked'], axis=1)
  
#We create all unique combinations of our features
data_unique = data_unique.drop_duplicates()
  
#Now we feed this into our model and get a prediction for each row
predictions = rf.predict_proba(data_unique)
  
#Finally, we add these predictions to the dataset
data_unique['prediction'] = [x[1] for x in predictions]
  
#And this is how it looks like
data_unique.head(3)
# %%
#Sort by prediction. This way highest predictions will be at the top of the dataset 
data_unique = data_unique.sort_values('prediction', ascending=False)
  
#Remove duplicates for country and purchase binned. This way, for each unique combination of country and purchase,
#we will only have the top 1 value, which means the highest prediction
best_segment = data_unique.drop_duplicates(subset=['user_country_FR', 'user_country_UK', 'user_country_US', 
                                         'purchase_binned_Low', 'purchase_binned_Medium', 'purchase_binned_High'
                                         ]).copy()
                                             
                                ################### "Unbin"" - Start  ######################
#This is not strictly needed. However, it is pretty hard to read that dataset cause we have all the dummy variables
#So let's reconstruct manually the original categorical varibles. It will be so much clearer that way
  
#Country
best_segment['user_country'] = np.where(best_segment['user_country_UK'] == 1, "UK", 
                                   np.where(best_segment['user_country_US'] == 1, "US", 
                                      np.where(best_segment['user_country_FR'] == 1, "FR",
                                     "ES"
)))
best_segment = best_segment.drop([e for e in list(data_unique) if e.startswith('user_country_')], axis=1)
  
#Number_purchases
best_segment['purchase_binned'] = np.where(best_segment['purchase_binned_High'] == 1, "High", 
                                   np.where(best_segment['purchase_binned_Medium'] == 1, "Medium", 
                                    np.where(best_segment['purchase_binned_Low'] == 1, "Low",
                                     "None"
)))
best_segment = best_segment.drop([e for e in list(data_unique) if e.startswith('purchase_binned_')], axis=1)
  
#Email Text
best_segment['email_text'] = np.where(best_segment['email_text_short_email'] == 1, "short_email", "long_email")
best_segment = best_segment.drop('email_text_short_email', axis=1)
  
#Email version
best_segment['email_version'] = np.where(best_segment['email_version_personalized'] == 1, "personalized", "generic")
best_segment = best_segment.drop('email_version_personalized', axis=1)
  
#Weekday
best_segment['weekday'] = np.where(best_segment['weekday_Monday'] == 1, "Monday", 
                                    np.where(best_segment['weekday_Saturday'] == 1, "Saturday", 
                                       np.where(best_segment['weekday_Sunday'] == 1, "Sunday",
                                          np.where(best_segment['weekday_Thursday'] == 1, "Thursday", 
                                              np.where(best_segment['weekday_Tuesday'] == 1, "Tuesday",
                                                   np.where(best_segment['weekday_Wednesday'] == 1, "Wednesday",
                                                      "Friday"
))))))
best_segment = best_segment.drop([e for e in list(data_unique) if e.startswith('weekday_')], axis=1)      
  
#Hour
best_segment['hour_binned'] = np.where(best_segment['hour_binned_afternoon'] == 1, "afternoon", 
                                   np.where(best_segment['hour_binned_morning'] == 1, "morning", 
                                     "night"
))
best_segment = best_segment.drop([e for e in list(data_unique) if e.startswith('hour_binned_')], axis=1) 
                              ################### "Unbin"" - End ######################
best_segment  
# %%
#Firstly let's get count by group. We need this for the weighted average at the end
count_segment = data[['user_country','purchase_binned']].groupby(['user_country','purchase_binned']).size().reset_index(name='counts')
  
#Get the proportion instead of the counts. Just easier to deal with to later get weighted average
count_segment['weight'] = count_segment['counts'].div(count_segment['counts'].sum())
  
#Merge it, so in our final dataset we also have weight
best_segment = pandas.merge(best_segment, count_segment).sort_values('prediction',ascending=False)
  
best_segment
# %%

#Now let's add class1 and class 0 errors to the dataset. We will take it from the test error confusion matrix
conf_matrix = pandas.DataFrame(confusion_matrix(test['clicked'], rf.predict(test.drop('clicked', axis=1)), labels=[0, 1]))
  
#We define positive predictive value (ppv) as the proportion of times the model is right when it predicts 1, this is also called precision 
ppv = conf_matrix.loc[1,1]/(conf_matrix.loc[1,1]+conf_matrix.loc[0,1])
  
#We also need false omission rate (FOR). Indeed, those are actual clicks (the model is mistakenly predicting non-click, but it is actually a click)
forate = conf_matrix.loc[1,0]/(conf_matrix.loc[1,0]+conf_matrix.loc[0,0])
  
#Adjusted predicted click-rate for each segment
best_segment['adjusted_prediction'] = best_segment['prediction'] * ppv + (1-best_segment['prediction']) * forate
  
#Finally, let's multiply this by the weight of each segment in the dataset and compare it with the starting click-rate
CTR_comparison = pandas.DataFrame( {'predicted_click_rate':[(best_segment['adjusted_prediction']*best_segment['weight']).sum()],
                                    'old_click_rate':[data['clicked'].mean()]
                                    })
# %%
