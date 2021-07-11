#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd

data = pd.read_csv('worldbank_data') #function to upload the data (in brackets use tab button)

#data # just printing the name results in a cool table

data.sort_values('avg_income', inplace=True) 
#I want my DataFrame to change as a result of sorting it whereas, otherwise, it will just return the sorted version. 
#I'm going to pass this argument here ‘inplace’ and set it to True. 
#So, that means the data itself is going to be affected and we're not just going to get a copy of it.


# In[3]:


import numpy as np

richest = data[data['avg_income'] > 15000] #all the countries with the income > 15000
richest.iloc[0] # to pull out a row

rich_mean = np.mean(richest['avg_income'])
all_mean = np.mean(data['avg_income'])


plt.scatter(richest['avg_income'], richest['happyScore'])
for k, row in richest.iterrows():
    plt.text(row['avg_income'], row['happyScore'], row['country'])






# In[4]:


happy = data['happyScore'] # data.happyScore
income = data['avg_income']
print(happy, income)


# In[8]:


import matplotlib.pyplot as plt

plt.xlabel('income, USD')
plt.ylabel('happy')
plt.scatter(income, happy, s = 50, alpha = 0.25)


# In[30]:


# K-means analysis

from sklearn.cluster import KMeans 
import numpy as np # to work with the data

income_happy = np.column_stack((income, happy)) #as in k-means library there is no way to insert 2 separate datasets
# to determine the clusters, so one need to join them using _slack
#print(income_happy[:5])

kmeans = KMeans(n_clusters=3).fit(income_happy) # to fit the data in 3 clusters
kmeans.cluster_centers_ # to determine the coordintes of the centres

y_kmeans = kmeans.predict(income_happy) # predict the group for each pair income-happy
plt.xlabel('average income, USD')
plt.ylabel('happiness')
plt.scatter(income_happy[:, 0], income_happy[:, 1],c = y_kmeans, s = 100, alpha = 0.25)


# In[100]:


import pandas as pd
data = pd.read_csv('worldbank_data') #function to upload the data (in brackets use tab button)
data
import numpy as np
data_draw = data[['GDP', 'adjusted_satisfaction', 'country']].sort_values('GDP')

last = data_draw.iloc[-1] #max GDP
first = data_draw.iloc[0] #min GDP

plt.scatter(data_draw['GDP'], data_draw['adjusted_satisfaction'], alpha = 0.5) #initial plot with the satis = f(GDP)
plt.xlabel('GDP, bln USD')
plt.ylabel('Satisfaction')
# it appeared that level of satisfaction as expected proportional to the level of GDP
plt.text(last[0], last[1], last[2], c= 'green') # max
plt.text(first[0], first[1], first[2], c = 'red') # min

plt.show()

# 1) the choice of the columns was arbitrary (I thought, that the wealther country sholud have higher satisfaction)
# 2) the data was sorted by GDP, so from min to max 
# 3) then the lowest and the highest GDP countries were marked with text => there is a clear pattern as expected, 
# however, the poorest country is not the most unsatisfied


# In[102]:


gdp = data['GDP']
satis = data['adjusted_satisfaction']

from sklearn.cluster import KMeans 
import numpy as np # to work with the data

satis_gdp = np.column_stack((gdp, satis)) #as in k-means library there is no way to insert 2 separate datasets
# to determine the clusters, so one need to join them using _slack
#print(income_happy[:5])

kmeans = KMeans(n_clusters=3).fit(satis_gdp) # to fit the data in 3 clusters
kmeans.cluster_centers_ # to determine the coordintes of the centres

y_kmeans = kmeans.predict(satis_gdp) # predict the group for each pair income-happy
plt.xlabel('GDP, bln USD')
plt.ylabel('Satisfaction')
plt.scatter(satis_gdp[:, 0], satis_gdp[:, 1],c = y_kmeans, s = 100, alpha = 0.25)
plt.text(last[0], last[1], last[2], c= 'green') # max
plt.text(first[0], first[1], first[2], c = 'red') # min

plt.show()


# # Banknotes project 

# In[ ]:


import numpy as np #for data
import pandas as pd # for statistics
import matplotlib.pyplot as plt # for the chart 
import matplotlib.patches as patches # to lay over another chart (like oval of the std in this case)

data = pd.read_csv('Banknote-authentication-dataset-.csv') 
data
# V1 - variance of the transformed image (deviation from the mean)
# V2 - skewness of the transformed image (how far and where is the peak shifted)
len(data) #1372 observations

## statistical features ##
# mean
multidim_mean = np.mean(data, 0) #0 for column mean and 1 for the each row mean
# std
multidim_std = np.std(data, 0)
print(round(multidim_mean, 3), round(multidim_std, 3))
## ellipse

ellipse_1 = patches.Ellipse([multidim_mean[0], multidim_mean[1]], multidim_std[0]*2, multidim_std[1]*2, alpha = 0.4, color = 'red')
ellipse_2 = patches.Ellipse([multidim_mean[0], multidim_mean[1]], multidim_std[0]*4, multidim_std[1]*4, alpha = 0.4, color = 'purple')
fig, graph = plt.subplots()

## plot the V1 = f(V2) ##
plt.xlabel('V1')
plt.ylabel('V2')
plt.scatter(data['V1'], data['V2'], alpha = 0.25) #only by plotting this chart one may distinguish like 3 clusters
                                                    # if counted horizontally (like layers)
#add std_dev oval
plt.scatter(multidim_mean[0], multidim_mean[1]) # centre of the oval
#graph.add_patch(ellipse_1)
#graph.add_patch(ellipse_2)



####
df = pd.DataFrame({'V1': data['V1'], 'V2': data['V2']})
df[(df.V1 > 2*multidim_std[0]) & (df.V2 > 2*multidim_std[1])].count()
df[df.V1 > 2*multidim_std[0]].count() 
df[df.V2 > 2*multidim_std[1]].count()


# In[ ]:


## k-means clustering ##

from sklearn.cluster import KMeans 
import numpy as np # to work with the data

#v1_v2 = np.column_stack(data['V1'], data['V2']) #as in k-means library there is no way to insert 2 separate datasets
# to determine the clusters, so one need to join them using _slack

kmeans = KMeans(n_clusters=2, n_init = 2, max_iter = 20).fit(data) # to fit the data in 3 clusters
clusters = kmeans.cluster_centers_ # to determine the coordintes of the centres
n_iterations = kmeans.n_iter_
labels = kmeans.labels_ #with 2 clusters lable 0 or 1 is assigned

y_kmeans = kmeans.predict(data) # predict the group for each pair income-happy

plt.xlabel('V1, standard deviation')
plt.ylabel('V2, skewness')
plt.scatter(data['V1'], data['V2'], c = y_kmeans, s = 50, alpha = 0.2)
plt.scatter(clusters[:, 0], clusters[:, 1], c = 'blue', s = 70, alpha = 0.8)
plt.show()
n_iterations # number of iterations before assigning to the cluster
clusters


# In[ ]:


from sklearn.metrics import accuracy_score
data_full = pd.read_csv('full_banknotes.csv') 
labels2 = labels + 1 # as in the data 'Class' has either 1 or 2 lables

#final = np.column_stack((data_full['Class'], labels2))

score = accuracy_score(data_full['Class'], labels2)
print(round(score, 3))

