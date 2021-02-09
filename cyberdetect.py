#!/usr/bin/env python
# coding: utf-8

# In[190]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split ,cross_val_score , KFold 
from sklearn.preprocessing import MinMaxScaler , Normalizer,StandardScaler,RobustScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest , chi2 , RFE
from sklearn.linear_model import LogisticRegression , SGDClassifier
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import NMF, PCA
from sklearn.metrics import log_loss ,  confusion_matrix, precision_score, recall_score, classification_report,roc_curve,roc_auc_score,matthews_corrcoef
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier , VotingClassifier
from keras.layers import Dense, Dropout , ReLU , Input , LeakyReLU ,Flatten,BatchNormalization
from keras.models import Model
from keras.utils import plot_model
from tensorflow.keras import activations
from tensorflow.keras import regularizers
from keras.optimizers import SGD


# In[191]:


train = pd.read_csv('train_imperson_without4n7_balanced_data.csv')


# In[192]:


test = pd.read_csv('test_imperson_without4n7_balanced_data.csv')


# In[193]:


train.shape


# In[194]:


test.shape


# In[195]:


#splitting features and target variable
x_train = train.drop('155',axis=1)
y_train = train.loc[:,'155']
x_test = test.drop('155',axis=1)
y_test = test.loc[:,'155']


# In[196]:


#defining function to get number of zeros across variables 
def checkzeros(df):
    check = (df == 0).sum(axis=1)
    return check.sum()/(df.shape[0]*df.shape[1])


# In[197]:


checkzeros(x_train)


# In[198]:


checkzeros(x_test)


# In[199]:


x_train = x_train.values
x_test = x_test.values
y_train = y_train.values
y_test = y_test.values


# In[200]:


plt.imshow(x_train, cmap=plt.cm.get_cmap("jet"), interpolation="nearest", aspect = "auto") 
plt.colorbar()
plt.show()


# In[201]:


plt.imshow(x_test, cmap=plt.cm.get_cmap("jet"), interpolation="nearest", aspect = "auto") 
plt.colorbar()
plt.show()


# In[202]:


train_x , validation_x,train_y,validation_y = train_test_split(x_train, y_train, random_state = 25, test_size= 0.3)


# In[211]:


svm = SVC(C=1)


# In[212]:


svm.fit(train_x,train_y)


# In[213]:


svm.score(validation_x,validation_y)


# In[214]:


svm.score(x_test,y_test)


# In[215]:


svm.score(train_x,train_y)


# In[18]:


estimators = [('normalize', Normalizer()), ('knn', KNeighborsClassifier())]
 
model0 = Pipeline(estimators)


# In[19]:


kfold = KFold(n_splits=10, random_state=25,shuffle=True)
results = cross_val_score(model0, x_train, y_train, cv=kfold)


# In[20]:


results.mean()


# In[21]:


results.std()


# In[22]:


model0.fit(x_train,y_train)


# In[23]:


model0.score(x_test,y_test)


# In[24]:


#keeping features only if std != zero 
train = pd.read_csv('train_imperson_without4n7_balanced_data.csv')
x_train = train.drop('155',axis=1)
x_train_clean = x_train.loc[:,x_train.apply(pd.Series.nunique) != 1]


# In[25]:


x_train_clean.shape


# In[26]:


x_train_clean.columns


# In[27]:


test = pd.read_csv('test_imperson_without4n7_balanced_data.csv')
clm = list(x_train_clean.columns) 
x_test_clean = test.loc[:,clm]


# In[28]:


x_test_clean.shape


# In[29]:


x_train_clean = x_train_clean.values
x_test_clean = x_test_clean.values


# In[30]:


plt.imshow(x_train_clean, cmap=plt.cm.get_cmap("jet"), interpolation="nearest", aspect = "auto") 
plt.colorbar()
plt.show()


# In[31]:


plt.imshow(x_train_clean, cmap=plt.cm.get_cmap("jet"), interpolation="nearest", aspect = "auto") 
plt.colorbar()
plt.show()


# In[32]:


features = [('select_best',SelectKBest(score_func = chi2, k=10)) ,('pca',PCA(n_components=10)),('nmf',NMF(n_components=3,random_state=10))]
feature_union = FeatureUnion(features)


# In[33]:


#estimators = [('standardscaler', StandardScaler()), ('feature_union',feature_union), ('knn', KNeighborsClassifier())]
estimators = [('normalizer', Normalizer()), ('feature_union',feature_union), ('std', StandardScaler()) ,('knn', KNeighborsClassifier())] 
model1 = Pipeline(estimators)


# In[34]:


kfold = KFold(n_splits=10, random_state=25,shuffle=True)


# In[35]:


results = cross_val_score(model1, x_train_clean, y_train, cv=kfold)


# In[36]:


results.mean()


# In[37]:


model1.fit(x_train_clean,y_train)


# In[38]:


model1.score(x_test_clean,y_test)


# In[39]:


scaler = Normalizer()
scaler.fit(x_train_clean)
x_train_scaled = scaler.transform(x_train_clean)
features = [('select_best',SelectKBest(score_func = chi2, k=48)) ,('pca',PCA(n_components=30))] 
feature_union = FeatureUnion(features)
trans_train = feature_union.fit_transform(x_train_scaled,y=y_train)
combo_train = trans_train


# In[40]:



print(x_train_clean.shape ,trans_train.shape , combo_train.shape )


# In[184]:


#estimators = [('norma', Normalizer()), ('svm',  SGDClassifier(max_iter=10000, tol=1e-3,alpha = 0.001,loss ='hinge', random_state = 25))] 
estimators = [ ('svm',  SGDClassifier(max_iter=10000, tol=1e-3,alpha = 0.001,loss ='hinge', random_state = 25))] 
model2 = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=25,shuffle=True)
results = cross_val_score(model2, combo_train, y_train, cv=kfold)
results.mean() 


# In[185]:


model2.fit(combo_train,y_train)


# In[186]:


x_test_scaled = scaler.transform(x_test_clean)
trans_test = feature_union.transform(x_test_scaled)
combo_test = trans_test 


# In[187]:


pred=model2.score(combo_test,y_test) 
print(f"Accuracy {pred}")


# In[188]:


svm_combo_pred = model2.predict(combo_test)
confusion_matrix(y_test,svm_combo_pred )
 


# In[189]:


#svm report
report = classification_report(y_test, svm_combo_pred)
print(report)


# In[45]:


model3 = Pipeline([('Standard', StandardScaler()), ('forest',  RandomForestClassifier(n_estimators = 80, max_features=5,random_state = 25  ))] )
results = cross_val_score(model3, combo_train, y_train, cv=kfold)
results.mean() 


# In[46]:


model3.fit(combo_train,y_train)
model3.score(combo_test,y_test)


# In[47]:


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


# In[48]:


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


# In[49]:


model.compile(optimizer='adam', loss='mse')


# In[50]:


plot_model(model , show_shapes=True)


# In[51]:


history = model.fit(x_train_clean, x_train_clean, epochs=35, batch_size=64, verbose=0, validation_data=(x_test_clean,x_test_clean))


# In[52]:


plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()


# In[53]:


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


# In[54]:


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


# In[55]:


plot_model(model , show_shapes=True)


# In[56]:


model.compile(optimizer='adam', loss='mse',metrics=['accuracy'])
history = model.fit(x_train_clean, x_train_clean, epochs=150, batch_size=128, verbose=0, validation_data=(x_test_clean,x_test_clean))
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()


# In[57]:


encoder = Model(inputs=start, outputs=bottleneck)
x_train_encode = encoder.predict(x_train_clean)
x_test_encode = encoder.predict(x_test_clean)


# In[58]:


#combo train used in model 2 is augmented with the 26 compressed features from autoencode
combo_train_plus = np.concatenate((combo_train, x_train_encode), axis=1)
combo_test_plus = np.concatenate((combo_test, x_test_encode), axis=1)


# In[59]:


combo_train_plus.shape 


# In[60]:


rfe_model = SGDClassifier(max_iter=10000, tol=1e-3,alpha = 0.001,loss ='hinge', random_state = 25)
rfe = RFE(rfe_model,  n_features_to_select = 10)
rfe.fit(combo_train_plus,  y_train)


# In[61]:


print("Num Features: %d" % rfe.n_features_)
print("Feature Ranking: %s" % rfe.ranking_)
len(rfe.ranking_)


# In[62]:


combo_auto_train = rfe.transform(combo_train_plus)
combo_auto_test = rfe.transform(combo_test_plus)


# In[63]:


def create_model(inds = combo_auto_train, optimizer='adam', init='glorot_uniform'):
        # define encoder
        n_inputs = inds.shape[1] 
        visible = Input(shape=(n_inputs,))
        # dropout 1
        d1 = Dropout(0.7)(visible )
        # layer 1
        l1 = Dense(round(5),kernel_initializer=init , activation ='tanh')(d1) 
        #l1 = BatchNormalization()(l1)
        # layer 2
        l2 = Dense(round(4),kernel_initializer=init , activation ='tanh')(l1)
        
        d2 = Dropout(0.3)(l2 )
        l3 = Dense(round(3),kernel_initializer=init , activation ='tanh')(d2)
        # output layer
        # output = Dense(1, activation='sigmoid',kernel_initializer=init ,activity_regularizer=regularizers.l2(1e-5))(l3)
        output = Dense(1, activation='sigmoid',kernel_initializer=init )(l3)
        # define neural model
        model = Model(inputs=visible, outputs=output)      
        model.compile(optimizer=optimizer, loss='binary_crossentropy',metrics=["accuracy"])
        return model


# In[90]:


neural = create_model(combo_auto_train,init='random_normal')
history = neural.fit(combo_auto_train  ,y_train,epochs=200,batch_size=512,verbose = 0,validation_split = 0.33)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show()


# In[219]:


plot_model(neural , show_shapes=True)


# In[128]:


from keras import callbacks
earlystopping = callbacks.EarlyStopping(monitor ="val_loss",  
                                        mode ="min", patience = 5,  
                                        restore_best_weights = True) 
neural = create_model(combo_auto_train,init='random_normal')
history = neural.fit(combo_auto_train  ,y_train,epochs=100,batch_size=512,verbose = 1,validation_split = 0.33, callbacks = [earlystopping])


# In[129]:


scores = neural.evaluate(combo_auto_test, y_test)
print("\n%s: %.2f%%" % (neural.metrics_names[1], scores[1]*100))


# In[130]:


from keras.wrappers.scikit_learn import KerasClassifier
model = KerasClassifier(build_fn=create_model, verbose=0, epochs = 53, batch_size = 512)
results = cross_val_score(model, combo_auto_train, y_train, cv=kfold)
print(results.mean())


# In[131]:


neural.evaluate(combo_auto_train, y_train, verbose=0)


# In[132]:


neural_test_pred = neural.predict(combo_auto_test)
neural_test_pred[neural_test_pred>0.5] = 1
neural_test_pred[neural_test_pred<=0.5] = 0 


# In[133]:


confusion_matrix(y_test,neural_test_pred)


# In[142]:


estimators = [ ('svm',  SGDClassifier(max_iter=10000, tol=1e-3,alpha = 0.1,loss ='hinge', random_state = 25))] 
model2 = Pipeline(estimators)
results = cross_val_score(model2, combo_auto_train, y_train, cv=kfold)
results.mean()


# In[145]:


model2.fit(combo_auto_train,y_train)
model2.score(combo_auto_train,y_train) 


# In[146]:


#np.concatenate((trans_test,trans_test), axis=1)
pred=model2.score(combo_auto_test,y_test) 
print(pred)


# In[147]:


svm_test_pred = model2.predict(combo_auto_test)
 
confusion_matrix(y_test,svm_test_pred)


# In[148]:


#neural precision recall
report = classification_report(y_test, neural_test_pred)
print(report)


# In[149]:


#svm report
report = classification_report(y_test, svm_test_pred)
print(report)


# In[150]:


recall_score(y_test, svm_test_pred)


# In[151]:


fpr, tpr, thresholds = roc_curve(y_test, neural_test_pred)                                
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

plot_roc_curve(fpr, tpr)
plt.show()


# In[152]:


fpr, tpr, thresholds = roc_curve(y_test, svm_test_pred)                                
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

plot_roc_curve(fpr, tpr)
plt.show()


# In[179]:


N = round(roc_auc_score(y_test, neural_test_pred) ,4)
S = round(roc_auc_score(y_test, svm_test_pred),4) 
print(f"Neural AUC: {N} Svm AUC: {S}" ) 


# In[180]:


N = round( matthews_corrcoef(y_test, neural_test_pred) ,4)
S = round(roc_auc_score(y_test, svm_test_pred),4) 
print(f"Neural Matthew: {N} Svm Matthew: {S}" ) 


# In[216]:


recall_score(y_test, neural_test_pred)


# In[217]:


matthews_corrcoef(y_test, svm_combo_pred)


# In[ ]:




