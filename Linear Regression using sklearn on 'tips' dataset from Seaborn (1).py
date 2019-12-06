#!/usr/bin/env python
# coding: utf-8

# In[50]:


import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
import statsmodels.api as sm
from sklearn import linear_model
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 


# In[ ]:


# USING SKLEARN TO PERFORM LINEAR REGRESSION ON TIPS DATASET TO FIND OUT RELATIONSHIP BETWEEN TOTAL AMOUNT AND TIPS
# ON SEABORN TIPS DATASET


# In[ ]:


# First the dataset must be imported


# In[71]:


tips = sns.load_dataset('tips')
print(tips)


# In[72]:


tips.describe()


# In[73]:


# Here we can see where our values fall within the data
# There are 244 lines
# the maximum tip is 10, the minimum 1 and the mean around 3
# the maximum total bill is 50, the minimum 3 and the mean around 8.90


# In[26]:


tips.plot(x='total_bill', y='tip', style='o')  
plt.title('total bill by tip')  
plt.xlabel('total_bill')  
plt.ylabel('tip')  
plt.show()


# In[ ]:


# From the basic plot is appears that there is a linear relationship between these two variables, although there are some
# anomalous results


# In[31]:


x = tips['total_bill']
y = tips['tip']


# In[32]:


model = sm.OLS(y, x)
visual = model.fit()
print(visual.summary())


# In[6]:


# INTREPRETTING THE RESULTS

# The Coefficient value here tells us that as the total_bill increases by 1 dollar, the tip increases by a factor of 0.1437 
# From the R squared value, we can see that total_bill accounts for 89.2% of the increase in the tip amount
# We can also say this is statsically significant as the p value is far below 0.05
# the low standard error tells us most results lie close to the mean


# In[ ]:


# PERFORMING THE REGRESSION


# In[41]:


X = tips['total_bill'].values.reshape(-1,1)
y = tips['tip'].values.reshape(-1,1)


# In[47]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 


# In[ ]:


# Here with have split the data into test data and training data


# In[51]:


ln = LinearRegression() 


# In[54]:


ln.fit(X_train, y_train) 
print(ln)


# In[53]:


# Our model has now been trained


# In[55]:


print(ln.score(X_test, y_test)) 


# In[ ]:


# This shows our model has an accuracy score of approximately 0.533, perhaps this data is not so well suited to a linear
# regression model, suggesting some of the other factors such as sex, time or day may have had a larger impact


# In[65]:


prediction = ln.predict(X_test) 
plt.scatter(X_test, y_test, color ='black')   
plt.plot(X_test, prediction, color ='red') 
plt.show() 


# In[58]:


# However in our result we can see our model has produced a positive linear relationship which appears to match well
# with our test data


# In[ ]:





# In[ ]:




