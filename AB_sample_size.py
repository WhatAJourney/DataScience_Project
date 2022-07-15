#%%
import statsmodels.stats.api as sms
#Firstly, we need to define the two conversion rates via proportion_effectsize. 
#The first element here (0.1) is simply the conversion rate of the site prior to running the test. Aka control conversion rate
#The second one (0.11) is the minimum conversion rate of the test that would make it worth it to make the change
p1_and_p2 = sms.proportion_effectsize(0.1, 0.11)
#Now we can run the function that after passing the two conversion rates above + power and significance, returns sample size
sample_size = sms.NormalIndPower().solve_power(p1_and_p2, power=0.8, alpha=0.05)
print("The required sample size per group is ~", round(sample_size))
# %%
import numpy as np
import matplotlib.pyplot as plt
#Possible p2 values. We choose from 10.5% to 15% with 0.5% increments
possible_p2 = np.arange(.105, .155, .005)
print(possible_p2)
#now let's estimate sample size for all those values and plot them

sample_size = []
for i in possible_p2:
   p1_and_p2 = sms.proportion_effectsize(0.1, i)
   sample_size.append(sms.NormalIndPower().solve_power(p1_and_p2, power=0.8, alpha=0.05))
plt.plot(sample_size, possible_p2)
plt.title("Sample size vs Minimum Effect size")
plt.xlabel("Sample Size")
plt.ylabel("Minimum Test Conversion rate")
plt.show()
# %%
    