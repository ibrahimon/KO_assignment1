#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
import sklearn


# In[2]:


data = pd.read_json('articles.json')


# In[3]:


data.head(20)


# In[4]:


data.shape


# In[5]:


data.info()


# In[6]:


data.describe().T


# In[17]:


X = data.iloc[:, [0]].values
y = data.iloc[:, [2]].values


# In[18]:


X.shape


# In[19]:


y.shape


# In[21]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .30, random_state = 0)


# In[43]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:,0] = le.fit_transform(X[:,0])


# In[44]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y[:,0] = le.fit_transform(y[:,0])


# In[45]:


# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB().fit(X_train, y_train)
classifier.fit(X_train, y_train)


# In[ ]:




