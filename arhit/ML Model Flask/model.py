import os
import tarfile
import urllib

import numpy as np
import pandas as pd
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("housing.csv")
df['avgRooms'] = df['total_rooms'] / df['households']
df['avgBedrooms'] = df['total_bedrooms'] / df['households']
df['pop_per_household'] = df['population'] / df['households']

dum = pd.get_dummies(df.ocean_proximity)
merged_df = pd.concat([df, dum], axis = 'columns')
merged_df = merged_df.drop(['ocean_proximity', 'ISLAND'], axis= 'columns')
merged_df.head()
X = merged_df.drop('median_house_value', axis= 'columns')
y = merged_df['median_house_value']

prices = merged_df['median_house_value']
features = merged_df.drop('median_house_value', axis = 1)

X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size = 0.2)

regr = LinearRegression()
regr.fit(X_train, y_train)

pickle.dump(regr, open('model.pkl','wb'))
pickle.load(open('model.pkl','rb'))