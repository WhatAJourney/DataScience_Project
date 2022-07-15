#%%
import pandas
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib import rcParams
os.chdir("c:/Users/Mr.Goldss/Desktop/datamasked/Funnel_analysis")
os.getcwd()
# %%

rcParams.update({'figure.autolayout': True})
pandas.set_option('display.max_columns', 10)
pandas.set_option('display.width', 350)
  
#read the files
user = pandas.read_csv("user_table.csv")
print(user.head())
# %%
home_page = pandas.read_csv("home_page_table.csv")
print(home_page.head())

# %%
search_page = pandas.read_csv("search_page_table.csv")
print(search_page.head())
# %%
payment_page = pandas.read_csv("payment_page_table.csv")
print(payment_page.head())
# %%
payment_confirmation = pandas.read_csv("payment_confirmation_table.csv")
print(payment_confirmation.head())

# %%
data=pandas.merge(user,home_page,how='left',on='user_id')
data=pandas.merge(data,search_page,how='left',on='user_id',suffixes=('_home','_search'))
data=pandas.merge(data,payment_page,how='left',on='user_id')
data=pandas.merge(data,payment_confirmation,how='left',on='user_id',suffixes=('_payment','_confirmation'))
# %%
#Replace NAs with 0 and non-NAs with 1
for i in range(4,8):
    data.iloc[:,i] = np.where(data.iloc[:,i].isna(), 0, 1)

      
#Make it a date
data['date'] = pandas.to_datetime(data['date'])
print(data.describe())
# %%
#Check data reliability. Are user_ids unique?
print(data['user_id'].nunique()==len(data))
# %%
#Anyone who made it to a given step of the funnel also made it to the prior step(s)? I.e., if a user got to the payment_page, were they also in the search_page? They should! 
#check payment_page vs search_page
print((data['page_search'] >= data['page_payment']).unique())
# %%
#check confirmation_page vs payment_page
print((data['page_payment'] >= data['page_confirmation']).unique())
# %%
#Firstly, let's get overall conversion rate for web and mobile
#conversion rate significantly lower than 1% is pretty low for most on-line businesses
print(data.groupby('device')['page_confirmation'].agg({'mean', 'count'}))

# %%
#Let's check the entire funnel
#50% of bouncers is a pretty common number
print(data.groupby('device').apply(
                    lambda x: pandas.Series({
                            'to_search' : x['page_search'].mean()/x['page_home'].mean(),
                            'to_payment': x['page_payment'].mean()/x['page_search'].mean(),
                            'to_confirmation': x['page_confirmation'].mean()/x['page_payment'].mean()
  }))
)
# %%
#Conversion rate as a time series
cr_ts = data.groupby(['date', 'device'])['page_confirmation'].mean().reset_index(name='conversion_rate')
  
g = sns.lineplot(x="date", hue="device", y="conversion_rate", data=cr_ts)  

g.xaxis.set_major_locator(mdates.MonthLocator())
g.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.show()
# %%
#funnel as a time series
funnel = data.groupby(['date', 'device']).apply(
                    lambda x: pandas.Series({
                            'to_search' : x['page_search'].mean()/x['page_home'].mean(),
                            'to_payment': x['page_payment'].mean()/x['page_search'].mean(),
                            'to_confirmation': x['page_confirmation'].mean()/x['page_payment'].mean()
  })).reset_index()
# %%
funnel = pandas.melt(funnel,id_vars=['date', 'device'])
# %%
#Plot it
g = sns.FacetGrid(funnel, hue="device", row="variable", aspect=4, sharey=False)
g.map(sns.lineplot, "date", "value")
g.set_axis_labels("", "Conversion Rate")
g.set(xticks=funnel.date[2::30].unique())
plt.xticks(rotation=45)
plt.legend(loc='upper right')
plt.show()
# %%
# %%
#Overall conversion rate for variable sex
print(data.groupby('sex')['page_confirmation'].agg({'mean', 'count'}))

# %%
from scipy import stats
#Statisitical test on conversion rate and sex
test = stats.ttest_ind(data.loc[data['sex'] == 'Female']['page_confirmation'], data.loc[data['sex'] != 'Female']['page_confirmation'], equal_var=False)
  
#t statistics
print(test.statistic)
print(test.pvalue)
# %%
#interaction of sex and device
print(data.groupby(['device', 'sex'])['page_confirmation'].agg({'mean', 'count'}))
# %%