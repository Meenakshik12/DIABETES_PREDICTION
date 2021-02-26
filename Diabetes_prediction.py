#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


A= pd.read_csv("/Users/ashish/OneDrive/Desktop/Diabetes/Diab.csv")


# In[3]:


A


# In[4]:


A.head(5)


# In[6]:


A.describe().T


# In[7]:


A.info()


# In[8]:


A.isna().sum()


# In[10]:


A.shape


# In[12]:


import matplotlib.pyplot as plt
import seaborn as sb


# In[13]:


sb.heatmap(A.corr())


# In[14]:


sb.countplot(A["diabetes"])


# In[15]:


sb.pairplot(A,hue="diabetes")


# In[16]:


diabetes_map= {True:1, False:0}


# In[17]:


A["diabetes"]= A["diabetes"].map(diabetes_map)


# In[18]:


A.head()


# In[24]:


X = A.drop(labels=["diabetes"],axis=1)
Y = A[["diabetes"]]


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
model = lr.fit(xtrain,ytrain)
pred = model.predict(xtest)


from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score,accuracy_score
print(confusion_matrix(ytest,pred))
#print(precision_score(ytest,pred))
#print(recall_score(ytest,pred))
#print(f1_score(ytest,pred))
print(accuracy_score(ytest,pred))




# In[ ]:



from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(random_state=20)
model = rfr.fit
pred = model.predict(xtest)

from sklearn.metrics import mean_squared_error
mean_squared_error(ytest,pred)

