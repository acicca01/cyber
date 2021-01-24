#!/usr/bin/env python
# coding: utf-8

# In[132]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split ,cross_val_score , KFold 
from sklearn.preprocessing import MinMaxScaler , Normalizer,StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest , chi2
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import NMF, PCA


# In[3]:


train = pd.read_csv('train_imperson_without4n7_balanced_data.csv')


# In[4]:


test = pd.read_csv('test_imperson_without4n7_balanced_data.csv')


# In[5]:


train.shape


# In[6]:


test.shape


# In[7]:


#splitting features and target variable
x_train = train.drop('155',axis=1)
y_train = train.loc[:,'155']
x_test = test.drop('155',axis=1)
y_test = test.loc[:,'155']


# In[8]:


#defining function to get number of zeros across variables 
def checkzeros(df):
    check = (df == 0).sum(axis=1)
    return check.sum()/(df.shape[0]*df.shape[1])


# In[9]:


checkzeros(x_train)


# In[10]:


checkzeros(x_test)


# In[11]:


x_train = x_train.values
x_test = x_test.values
y_train = y_train.values
y_test = y_test.values


# In[12]:


plt.imshow(x_train, cmap=plt.cm.get_cmap("Reds"), interpolation="nearest", aspect = "auto") 
plt.colorbar()
plt.show()


# In[13]:


plt.imshow(x_test, cmap=plt.cm.get_cmap("Reds"), interpolation="nearest", aspect = "auto") 
plt.colorbar()
plt.show()


# In[14]:


train_x , validation_x,train_y,validation_y = train_test_split(x_train, y_train, random_state = 25, test_size= 0.3)


# In[15]:


svm = SVC(C=100)


# In[16]:


svm.fit(train_x,train_y)


# In[17]:


svm.score(validation_x,validation_y)


# In[18]:


svm.score(x_test,y_test)


# In[19]:


kfold = KFold(n_splits=10 , random_state = 25,shuffle=True)
results = cross_val_score(LogisticRegression(solver='liblinear'),train_x,train_y, cv=kfold)


# In[20]:


results.mean() 


# In[36]:


estimators = [('normalize', Normalizer()), ('knn', KNeighborsClassifier())]
 
model0 = Pipeline(estimators)


# In[37]:


kfold = KFold(n_splits=10, random_state=25,shuffle=True)
results = cross_val_score(model0, x_train, y_train, cv=kfold)


# In[39]:


results.mean()


# In[41]:


results.std()


# In[42]:


model0.fit(x_train,y_train)


# In[43]:


model0.score(x_test,y_test)


# In[83]:


#keeping features only if std != zero 
train = pd.read_csv('train_imperson_without4n7_balanced_data.csv')
x_train = train.drop('155',axis=1)
x_train_clean = x_train.loc[:,x_train.apply(pd.Series.nunique) != 1]


# In[84]:


x_train_clean.shape


# In[85]:


x_train_clean.columns


# In[90]:


test = pd.read_csv('test_imperson_without4n7_balanced_data.csv')
clm = list(x_train_clean.columns) 
x_test_clean = test.loc[:,clm]


# In[92]:


x_test_clean.shape


# In[93]:


x_train_clean = x_train_clean.values
x_test_clean = x_test_clean.values


# In[94]:


plt.imshow(x_train_clean, cmap=plt.cm.get_cmap("Reds"), interpolation="nearest", aspect = "auto") 
plt.colorbar()
plt.show()


# In[154]:


features = [('select_best',SelectKBest(score_func = chi2, k=3)) ,('pca',PCA(n_components=3)),('nmf',NMF(n_components=3,random_state=3))]
feature_union = FeatureUnion(features)


# In[155]:


#estimators = [('standardscaler', StandardScaler()), ('feature_union',feature_union), ('knn', KNeighborsClassifier())]
estimators = [('normalizer', Normalizer()), ('feature_union',feature_union), ('knn', KNeighborsClassifier())] 
model1 = Pipeline(estimators)


# In[156]:


kfold = KFold(n_splits=10, random_state=25,shuffle=True)


# In[157]:


results = cross_val_score(model1, x_train_clean, y_train, cv=kfold)


# In[158]:


results.mean()


# In[159]:


model1.fit(x_train_clean,y_train)


# In[160]:


model1.score(x_test_clean,y_test)


# In[168]:


features = [('select_best',SelectKBest(score_func = chi2, k=3)) ,('pca',PCA(n_components=3))] 
feature_union = FeatureUnion(features)
estimators = [('minmax', MinMaxScaler()), ('feature_union',feature_union), ('svm', SVC(probability=True))] 
model2 = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=25,shuffle=True)
results = cross_val_score(model2, x_train_clean, y_train, cv=kfold, scoring = 'neg_log_loss')


# In[171]:


results.mean() 


# In[172]:


model2.fit(x_train_clean,y_train)


# In[175]:


model2.score(x_test_clean,y_test,scoring = 'neg_log_loss')


# In[174]:


get_ipython().run_line_magic('pinfo', 'cross_val_score')


# In[ ]:




