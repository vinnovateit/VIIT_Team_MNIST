#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis on the California Housing Prices
# <img></img>

# ### Importing Libraries

# In[19]:


import os
import tarfile
import urllib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

import seaborn as sns
sns.set_style('darkgrid')
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split



# ### Importing the Dataset on California Housing Prices

# In[21]:



dataset_path = os.path.join("dataset")
download_url="https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.tgz"


# In[23]:


def fetch_data(download_url= download_url, dataset_path= dataset_path):
    os.makedirs(dataset_path, exist_ok= True)
    tgz_path = os.path.join(dataset_path,"housing.tgz")
    urllib.request.urlretrieve(download_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path= dataset_path)
    housing_tgz.close()


# In[25]:


fetch_data()


# In[27]:


def load_data(dataset_path= dataset_path):
    csv_path = os.path.join(dataset_path, "housing.csv")
    return pd.read_csv(csv_path)


# In[29]:


df = load_data()


# ### Glance of the DataSet

# In[31]:


df


# ### Inference:
# We can observe that the dataset consists of 20,639 households across 10 different attributes

# ### Cleaning and Filtering the data

# In[33]:


df = df.drop_duplicates() 
df.duplicated().values.any()  #Finding any duplicates


# In[ ]:


df = df.fillna(method="ffill")
pd.isnull(df).any()  # Checking for Null Values


# ### Feature Engineering

# In[11]:


df['avgRooms'] = df['total_rooms'] / df['households']
df['avgBedrooms'] = df['total_bedrooms'] / df['households']
df['pop_per_household'] = df['population'] / df['households']


# I have added more number of features which can help with the proper distribution and predicting better values in our model
# <li>Average Rooms per House</li>
# <li>Average Bedrooms per House</li>
# <li>Number of people per household</li>

# In[12]:



dum = pd.get_dummies(df.ocean_proximity)


# In[326]:


merged_df = pd.concat([df, dum], axis = 'columns')


# In[330]:


merged_df = merged_df.drop(['ocean_proximity', 'ISLAND'], axis= 'columns')
merged_df.head()


# In[334]:


X = merged_df.drop('median_house_value', axis= 'columns')
y = merged_df['median_house_value']


# ## Correlation between the Target and the Different Features

# ### Correlation
# ### $$ \rho _{XY} = corr(X,Y)$$
# ### $$ -1.0 \leq \rho _{XY} \leq +1.0$$

# In[294]:



prices = merged_df['median_house_value']
features = merged_df.drop('median_house_value', axis = 1)

X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size = 0.2)

#len(X_train)/len(features)


# In[377]:


regr = LinearRegression()
regr.fit(X_train, y_train)

print('Intercept', regr.intercept_)
pd.DataFrame(data = regr.coef_, index=X_train.columns, columns = ['Coef'])


# In[378]:


regr.score(X_train, y_train)


# In[1]:


pickle.dump(regr, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))


# In[ ]:





# In[381]:


regr.score(X_test, y_test)


