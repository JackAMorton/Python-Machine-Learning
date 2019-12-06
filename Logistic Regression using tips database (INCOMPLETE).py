#!/usr/bin/env python
# coding: utf-8

# In[49]:


import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size= 16)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns


# In[ ]:


# LOGISTIC REGRESSION


# In[ ]:


# To keep things simple I will be using the tips database that was used in linear regression
# However first these values will need to be converted to numeric values


# In[41]:


tips = sns.load_dataset('tips')
tips.head()


# In[39]:


tips["sex"] = tips["sex"].apply(lambda sex:1 if sex=="male" else 0)
tips["smoker"] = tips["smoker"].apply(lambda smoker: 1 if smoker == "Yes" else 0)
tips["time"] = tips["time"].apply(lambda time: 1 if time == "Dinner" else 0)
mapping = {"Thur": 1, "Fri": 2, "Sat": 3, "Sun": 4}
tips.replace({"day": mapping})


# In[54]:


test = tips
train = tips


# In[40]:


# This has now converted any string values into numeric values so our regression can work


# In[ ]:




