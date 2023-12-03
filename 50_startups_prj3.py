#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[4]:


data_set = pd.read_csv("D:/Python Scripts/50_startups.csv")


# In[6]:


data_set


# In[5]:


x =data_set.iloc[:,:-1].values


# In[9]:


print(x)


# In[6]:


y=data_set.iloc[:,4].values


# In[7]:


print(y)


# In[28]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer


# In[33]:


labelencoder_x = LabelEncoder()
x[:,3]=labelencoder_x.fit_transform(x[:,3])  
onehotencoder= OneHotEncoder(categories= 'auto')     
x= onehotencoder.fit_transform(x).toarray()


# In[34]:


x


# In[35]:


x = x[:, 1:]


# In[43]:


# Splitting the dataset into training and test set.  
from sklearn.model_selection import train_test_split  
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)


# In[39]:


###Fitting the MLR model to the training set:  
from sklearn.linear_model import LinearRegression  
regressor= LinearRegression()  
regressor.fit(x_train, y_train)


# In[ ]:





# In[47]:


#Predicting the Test set result;  
y_pred= regressor.predict(x_test)


# In[48]:


print('Train Score: ', regressor.score(x_train, y_train))  
print('Test Score: ', regressor.score(x_test, y_test))


# In[ ]:





# In[ ]:




