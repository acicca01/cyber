#!/usr/bin/env python
# coding: utf-8

# In[146]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split ,cross_val_score , KFold 
from sklearn.preprocessing import MinMaxScaler , Normalizer,StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest , chi2
from sklearn.linear_model import LogisticRegression , SGDClassifier
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import NMF, PCA
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV


# In[2]:


train = pd.read_csv('train_imperson_without4n7_balanced_data.csv')


# In[3]:


test = pd.read_csv('test_imperson_without4n7_balanced_data.csv')


# In[4]:


train.shape


# In[5]:


test.shape


# In[6]:


#splitting features and target variable
x_train = train.drop('155',axis=1)
y_train = train.loc[:,'155']
x_test = test.drop('155',axis=1)
y_test = test.loc[:,'155']


# In[7]:


#defining function to get number of zeros across variables 
def checkzeros(df):
    check = (df == 0).sum(axis=1)
    return check.sum()/(df.shape[0]*df.shape[1])


# In[8]:


checkzeros(x_train)


# In[9]:


checkzeros(x_test)


# In[10]:


x_train = x_train.values
x_test = x_test.values
y_train = y_train.values
y_test = y_test.values


# In[11]:


plt.imshow(x_train, cmap=plt.cm.get_cmap("Reds"), interpolation="nearest", aspect = "auto") 
plt.colorbar()
plt.show()


# In[12]:


plt.imshow(x_test, cmap=plt.cm.get_cmap("Reds"), interpolation="nearest", aspect = "auto") 
plt.colorbar()
plt.show()


# In[13]:


train_x , validation_x,train_y,validation_y = train_test_split(x_train, y_train, random_state = 25, test_size= 0.3)


# In[14]:


svm = SVC(C=100)


# In[15]:


svm.fit(train_x,train_y)


# In[16]:


svm.score(validation_x,validation_y)


# In[17]:


svm.score(x_test,y_test)


# In[18]:


kfold = KFold(n_splits=10 , random_state = 25,shuffle=True)
results = cross_val_score(LogisticRegression(solver='liblinear'),train_x,train_y, cv=kfold)


# In[19]:


results.mean() 


# In[20]:


estimators = [('normalize', Normalizer()), ('knn', KNeighborsClassifier())]
 
model0 = Pipeline(estimators)


# In[21]:


kfold = KFold(n_splits=10, random_state=25,shuffle=True)
results = cross_val_score(model0, x_train, y_train, cv=kfold)


# In[22]:


results.mean()


# In[23]:


results.std()


# In[24]:


model0.fit(x_train,y_train)


# In[25]:


model0.score(x_test,y_test)


# In[26]:


#keeping features only if std != zero 
train = pd.read_csv('train_imperson_without4n7_balanced_data.csv')
x_train = train.drop('155',axis=1)
x_train_clean = x_train.loc[:,x_train.apply(pd.Series.nunique) != 1]


# In[27]:


x_train_clean.shape


# In[28]:


x_train_clean.columns


# In[29]:


test = pd.read_csv('test_imperson_without4n7_balanced_data.csv')
clm = list(x_train_clean.columns) 
x_test_clean = test.loc[:,clm]


# In[30]:


x_test_clean.shape


# In[31]:


x_train_clean = x_train_clean.values
x_test_clean = x_test_clean.values


# In[32]:


plt.imshow(x_train_clean, cmap=plt.cm.get_cmap("Reds"), interpolation="nearest", aspect = "auto") 
plt.colorbar()
plt.show()


# In[192]:


features = [('select_best',SelectKBest(score_func = chi2, k=10)) ,('pca',PCA(n_components=10)),('nmf',NMF(n_components=3,random_state=10))]
feature_union = FeatureUnion(features)


# In[197]:


#estimators = [('standardscaler', StandardScaler()), ('feature_union',feature_union), ('knn', KNeighborsClassifier())]
estimators = [('normalizer', Normalizer()), ('feature_union',feature_union), ('std', StandardScaler()) ,('knn', KNeighborsClassifier())] 
model1 = Pipeline(estimators)


# In[198]:


kfold = KFold(n_splits=10, random_state=25,shuffle=True)


# In[199]:


results = cross_val_score(model1, x_train_clean, y_train, cv=kfold)


# In[185]:


results.mean()


# In[200]:


model1.fit(x_train_clean,y_train)


# In[201]:


model1.score(x_test_clean,y_test)


# In[54]:


features = [('select_best',SelectKBest(score_func = chi2, k=30)) ,('pca',PCA(n_components=30))] 
feature_union = FeatureUnion(features)
trans = feature_union.fit_transform(x_train_clean,y=y_train)


# In[61]:


combo = np.concatenate((x_train_clean, trans), axis=1)
combo.shape


# In[171]:


estimators = [('norma', Normalizer()), ('svm',  SGDClassifier(max_iter=1000, tol=1e-3,  random_state = 25))] 
model2 = Pipeline(estimators)
#model2.get_params().keys()
alphas =  [0.1**i for i in range(1,10)]
param_grid = dict(svm__alpha=alphas)
param_grid['svm__penalty'] = ['l2','l1','elasticnet']
grid = GridSearchCV(estimator=model2, param_grid=param_grid)
grid.fit(combo,y_train)
print(grid.best_score_)


# In[176]:


print(grid.best_params_)


# In[164]:


estimators = [('norma', Normalizer()), ('svm',  SGDClassifier(max_iter=1000, tol=1e-3, alpha = 1.0000000000000003e-05, random_state = 25))] 
kfold = KFold(n_splits=10, random_state=25,shuffle=True)
results = cross_val_score(model2, combo, y_train, cv=kfold)
results.mean() 


# In[165]:


model2.fit(combo,y_train)


# In[166]:


#trans_test = feature_union.transform(x_test_clean)
#combo_test = np.concatenate((x_test_clean, trans_test), axis=1)
pred=model2.score(combo_test,y_test)
 


# In[167]:


pred


# In[ ]:





# In[ ]:




