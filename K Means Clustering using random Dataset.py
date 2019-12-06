#!/usr/bin/env python
# coding: utf-8

# In[53]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# DATA CLUSTERING USING A RANDOM DATASET


# In[3]:


# First the dataset must be created


# In[5]:


df_1 = pd.DataFrame({
    'x': [9, 11, 38, 49, 8, 7, 56, 78, 81, 44, 66, 71, 55, 95, 24, 56, 17, 5, 38],
    'y': [2, 39, 21, 58, 88, 76, 61, 44, 14, 15, 91, 7, 13, 52, 14, 8, 11, 90, 91]})


# In[28]:


np.random.seed(100)


# In[ ]:


# ITERATION 1


# In[ ]:


# STAGE 1 INITIATION


# In[7]:


# To create three groups of clusters we must set the 'K' value as three, this means the clustering will begin with three
# randomly generated centroids from which the measure the distance to the data points


# In[10]:


k = 3


# In[16]:


centroids = {
    i+1: [np.random.randint(0, 100), np.random.randint(0, 100)]
    for i in range(k)}


# In[17]:


# Our centroids will now be a random number between 0 and 100


# In[27]:


s_p = plt.figure(figsize=(8, 8))
# Here we have defined the size of the scatter plot below
plt.scatter(df['x'], df['y'], color='black')
colmap = {1: 'green', 2: 'blue', 3: 'red'}
# The datapoints are now coloured black, with the centroids red, blue and green
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.show()
# The for loop with plot a coloured point for each centroid in the list, while i is equal to out cluster limit of 3 


# In[ ]:


# The "INITIALISATION" stage has taken place and the random centroids have been placed


# In[ ]:


# ASSIGNMENT STAGE 
# This stage assigns each of the data points a cluster based on their nearest centroid


# In[40]:


def assignment(df, centroids):
    for i in centroids.keys():
        df['distance_from_{}'.format(i)] = (
            np.sqrt(
                (df['x'] - centroids[i][0]) ** 2
                + (df['y'] - centroids[i][1]) ** 2))
    centroid_distance_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]
    df['closest'] = df.loc[:, centroid_distance_cols].idxmin(axis=1)
    df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))
    df['color'] = df['closest'].map(lambda x: colmap[x])
    return df


# In[50]:


df1 = assignment(df, centroids)
fig = plt.figure(figsize=(8, 8))
plt.scatter(df1['x'], df1['y'], color=df1['color'], alpha=0.3, edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.show()


# In[52]:


# The code has now discerned the closest centroid to each data point and coloured appropriately


# In[54]:


old_centroids = copy.deepcopy(centroids)

def update(k):
    for i in centroids.keys():
        centroids[i][0] = np.mean(df[df['closest'] == i]['x'])
        centroids[i][1] = np.mean(df[df['closest'] == i]['y'])
    return k

centroids = update(centroids)


# In[62]:


fig = plt.figure(figsize=(8, 8))
ax = plt.axes()
plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.3, edgecolor='black')
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.xlim(0, 100)
plt.ylim(0, 100)
for i in old_centroids.keys():
    old_x = old_centroids[i][0]
    old_y = old_centroids[i][1]
    dx = (centroids[i][0] - old_centroids[i][0]) 
    dy = (centroids[i][1] - old_centroids[i][1])
    ax.arrow(old_x, old_y, dx, dy, head_width=2, head_length=2, fc=colmap[i], ec=colmap[i])
plt.show()


# In[ ]:


# The centroids have now moved to refelect the means of their respective clusters


# In[ ]:


# ASSIGNMENT STAGE 2


# In[63]:


df = assignment(df, centroids)
fig = plt.figure(figsize=(8, 8))
plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='black')
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.show()


# In[64]:


# The colours of the datapoints have now been updated to reflect their distance to the new centroids


# In[ ]:


# UPDATE STAGE 2


# In[65]:


old_centroids = copy.deepcopy(centroids)

def update(k):
    for i in centroids.keys():
        centroids[i][0] = np.mean(df[df['closest'] == i]['x'])
        centroids[i][1] = np.mean(df[df['closest'] == i]['y'])
    return k

centroids = update(centroids)


# In[66]:


fig = plt.figure(figsize=(8, 8))
ax = plt.axes()
plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.3, edgecolor='black')
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.xlim(0, 100)
plt.ylim(0, 100)
for i in old_centroids.keys():
    old_x = old_centroids[i][0]
    old_y = old_centroids[i][1]
    dx = (centroids[i][0] - old_centroids[i][0]) 
    dy = (centroids[i][1] - old_centroids[i][1])
    ax.arrow(old_x, old_y, dx, dy, head_width=2, head_length=2, fc=colmap[i], ec=colmap[i])
plt.show()


# In[67]:


# The new means for the clusters means the centroids have moved again, however by a much smaller amount
# In this iteration the datapoints have stayed within the same cluster so more iterations are unccessary and we 
# can say that we have our three clusters as shown above


# In[68]:


df = assignment(df, centroids)
fig = plt.figure(figsize=(8, 8))
plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='black')
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.show()


# In[69]:


# Here we have our final clusters


# In[ ]:




