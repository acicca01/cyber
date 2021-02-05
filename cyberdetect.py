#!/usr/bin/env python
# coding: utf-8

# In[448]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split ,cross_val_score , KFold 
from sklearn.preprocessing import MinMaxScaler , Normalizer,StandardScaler,RobustScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest , chi2 , RFE
from sklearn.linear_model import LogisticRegression , SGDClassifier
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import NMF, PCA
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier , VotingClassifier
from keras.layers import Dense, Dropout , ReLU , Input , LeakyReLU ,Flatten,BatchNormalization
from keras.models import Model
from keras.utils import plot_model
from tensorflow.keras import activations
from tensorflow.keras import regularizers
from keras.optimizers import SGD


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
results = cross_val_score(LogisticRegression(solver='liblinear'),x_train,y_train, cv=kfold)


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


# In[33]:


features = [('select_best',SelectKBest(score_func = chi2, k=10)) ,('pca',PCA(n_components=10)),('nmf',NMF(n_components=3,random_state=10))]
feature_union = FeatureUnion(features)


# In[34]:


#estimators = [('standardscaler', StandardScaler()), ('feature_union',feature_union), ('knn', KNeighborsClassifier())]
estimators = [('normalizer', Normalizer()), ('feature_union',feature_union), ('std', StandardScaler()) ,('knn', KNeighborsClassifier())] 
model1 = Pipeline(estimators)


# In[35]:


kfold = KFold(n_splits=10, random_state=25,shuffle=True)


# In[36]:


results = cross_val_score(model1, x_train_clean, y_train, cv=kfold)


# In[37]:


results.mean()


# In[38]:


model1.fit(x_train_clean,y_train)


# In[39]:


model1.score(x_test_clean,y_test)


# In[457]:


scaler = Normalizer()
scaler.fit(x_train_clean)
x_train_scaled = scaler.transform(x_train_clean)
features = [('select_best',SelectKBest(score_func = chi2, k=48)) ,('pca',PCA(n_components=30))] 
feature_union = FeatureUnion(features)
trans_train = feature_union.fit_transform(x_train_scaled,y=y_train)
combo_train = trans_train
#np.concatenate(( trans_train,trans_train), axis=1)


#scaler = StandardScaler()
#scaler.fit(x_train_clean)
#x_train_scaled = scaler.transform(x_train_clean)
#pca = PCA(n_components=30)
#pca.fit(x_train_scaled)
#x_train_pca = pca.transform(x_train_scaled)
#combo_train = np.concatenate((x_train_scaled, x_train_pca), axis=1)


# In[458]:



print(x_train_clean.shape ,trans_train.shape , combo_train.shape )


# In[459]:


#estimators = [('norma', Normalizer()), ('svm',  SGDClassifier(max_iter=10000, tol=1e-3,alpha = 0.001,loss ='hinge', random_state = 25))] 
estimators = [ ('svm',  SGDClassifier(max_iter=10000, tol=1e-3,alpha = 0.001,loss ='hinge', random_state = 25))] 
model2 = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=25,shuffle=True)
results = cross_val_score(model2, combo_train, y_train, cv=kfold)
results.mean() 


# In[460]:


model2.fit(combo_train,y_train)


# In[461]:


x_test_scaled = scaler.transform(x_test_clean)
trans_test = feature_union.transform(x_test_scaled)
combo_test = trans_test 
#np.concatenate((trans_test,trans_test), axis=1)
pred=model2.score(combo_test,y_test) 

#x_test_scaled = scaler.transform(x_test_clean)
#x_test_pca = pca.transform(x_test_scaled)
#combo_test = np.concatenate((x_test_scaled, x_test_pca), axis=1)
#pred=model2.score(combo_test,y_test)
 


# In[462]:


pred


# In[472]:


model3 = Pipeline([('Standard', StandardScaler()), ('forest',  RandomForestClassifier(n_estimators = 80, max_features=5,random_state = 25  ))] )
 
results = cross_val_score(model3, combo_train, y_train, cv=kfold)
results.mean() 


# In[473]:


model3.fit(combo_train,y_train)
model3.score(combo_test,y_test)


# In[588]:


# define encoder
n_inputs = x_train_clean.shape[1]
visible = Input(shape=(n_inputs,))
# encoder level 1
e = Dense(round(float(n_inputs*0.7)))(visible)
e = ReLU()(e)
# encoder level 2
e = Dense(round(n_inputs*0.5))(e)
e = ReLU()(e)
# bottleneck
n_bottleneck = round(n_inputs/3)
bottleneck = Dense(n_bottleneck)(e)


# In[589]:


# define decoder, level 1
d = Dense(round(n_inputs*0.5))(bottleneck)
d = ReLU()(d) 
# decoder level 2
d = Dense(round(n_inputs*0.7))(d)
d = ReLU()(d)
# output layer
output = Dense(n_inputs, activation='linear')(d)
# define autoencoder model
model = Model(inputs=visible, outputs=output)


# In[590]:


model.compile(optimizer='adam', loss='mse')


# In[591]:


plot_model(model , show_shapes=True)


# In[592]:


history = model.fit(x_train_clean, x_train_clean, epochs=35, batch_size=64, verbose=0, validation_data=(x_test_clean,x_test_clean))


# In[593]:


plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()


# In[571]:


# define encoder
n_inputs = x_train_clean.shape[1] 
start = Input(shape=(n_inputs,))
# encoder level 1
e1 = Dense(round(n_inputs*0.7) )(start)
e1 = BatchNormalization()(e1)
e1 = LeakyReLU()(e1)
# encoder level 2
e2 = Dense(round(n_inputs*0.5))(e1)
e2 = BatchNormalization()(e2)
e2 = LeakyReLU()(e2)
# bottleneck
latentdim = round(float(n_inputs)/3)
bottleneck = Dense(latentdim,activity_regularizer=regularizers.l2(1e-5))(e2)


# In[572]:


# define decoder, level 1
d1 = Dense(round(n_inputs*0.5) )(bottleneck)
d1 = BatchNormalization()(d1)
d1 = LeakyReLU()(d1)
# decoder level 2
d2 = Dense(round(n_inputs*0.7) )(d1)
d2 = BatchNormalization()(d2)
d2 = LeakyReLU()(d2)
# output layer
end = Dense(n_inputs, activation='linear')(d2)
# define autoencoder model
model = Model(inputs=start, outputs=end)


# In[572]:


plot_model(model , show_shapes=True)


# In[574]:


model.compile(optimizer='adam', loss='mse',metrics=['accuracy'])
history = model.fit(x_train_clean, x_train_clean, epochs=35, batch_size=64, verbose=0, validation_data=(x_test_clean,x_test_clean))
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()


# In[575]:


encoder = Model(inputs=start, outputs=bottleneck)
x_train_encode = encoder.predict(x_train_clean)
x_test_encode = encoder.predict(x_test_clean)


# In[576]:


#combo train used in model 2 is augmented with the 26 compressed features from autoencode
combo_train_plus = np.concatenate((combo_train, x_train_encode), axis=1)
combo_test_plus = np.concatenate((combo_test, x_test_encode), axis=1)


# In[594]:


x_train_encode.shape 


# In[578]:


rfe_model = SGDClassifier(max_iter=10000, tol=1e-3,alpha = 0.001,loss ='hinge', random_state = 25)
rfe = RFE(rfe_model,  n_features_to_select = 78)
rfe.fit(combo_train_plus,  y_train)


# In[579]:


print("Num Features: %d" % rfe.n_features_)
print("Feature Ranking: %s" % rfe.ranking_)
len(rfe.ranking_)


# In[580]:


combo_auto_train = rfe.transform(combo_train_plus)
combo_auto_test = rfe.transform(combo_test_plus)


# In[581]:


def create_model(inds = combo_auto_train, optimizer='adam', init='glorot_uniform'):
        # define encoder
        n_inputs = inds.shape[1] 
        visible = Input(shape=(n_inputs,))
        # dropout 1
        d1 = Dropout(0.8)(visible )
        # layer 1
        l1 = Dense(round(5),kernel_initializer=init , activation ='tanh')(d1) 
        l1 = BatchNormalization()(l1)
        # layer 2
        l2 = Dense(round(4),kernel_initializer=init , activation ='tanh')(l1)
        d2 = Dropout(0.4)(l2 )
        l3 = Dense(round(3),kernel_initializer=init , activation ='tanh')(d2)
        # output layer
        # output = Dense(1, activation='sigmoid',kernel_initializer=init ,activity_regularizer=regularizers.l2(1e-5))(l3)
        output = Dense(1, activation='sigmoid',kernel_initializer=init )(l3)
        # define neural model
        model = Model(inputs=visible, outputs=output)      
        model.compile(optimizer=optimizer, loss='binary_crossentropy',metrics=["accuracy"])
        return model


# In[569]:


neural = create_model(combo_auto_train,init='random_normal')
history = neural.fit(combo_auto_train  ,y_train,epochs=5,batch_size=128,verbose = 0,validation_data=(combo_auto_test,y_test))
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show()


# In[582]:


neural.evaluate(combo_auto_test, y_test, verbose=0)


# In[583]:


neural = create_model(combo_auto_train,init='random_normal')
history = neural.fit(combo_auto_train  ,y_train,epochs=5,batch_size=128,verbose = 0,validation_data=(combo_auto_test,y_test))
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show()


# In[584]:


neural.evaluate(combo_auto_test, y_test, verbose=0)


# In[ ]:




