import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split ,cross_val_score , KFold 
from sklearn.preprocessing import MinMaxScaler , Normalizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest , chi2 
from sklearn.linear_model import LogisticRegression

train = pd.read_csv('train_imperson_without4n7_balanced_data.csv')
test = pd.read_csv('test_imperson_without4n7_balanced_data.csv')
train.shape
test.shape
#splitting features and target variable
x_train = train.drop('155',axis=1)
y_train = train.loc[:,'155']
x_test = test.drop('155',axis=1)
y_test = test.loc[:,'155']
#defining function to get number of zeros across variables 
def checkzeros(df):
    check = (df == 0).sum(axis=1)
    return check.sum()/(df.shape[0]*df.shape[1])
checkzeros(x_train)
checkzeros(x_test)
x_train = x_train.values
x_test = x_test.values
y_train = y_train.values
y_test = y_test.values
plt.imshow(x_train, cmap=plt.cm.get_cmap("Reds"), interpolation="nearest", aspect = "auto") 
plt.colorbar()
plt.show()
plt.imshow(x_test, cmap=plt.cm.get_cmap("Reds"), interpolation="nearest", aspect = "auto") 
plt.colorbar()
plt.show()
train_x , validation_x,train_y,validation_y = train_test_split(x_train, y_train, random_state = 25, test_size= 0.3)
svm = SVC(C=100)
svm.fit(train_x,train_y)
svm.score(validation_x,validation_y)
svm.score(x_test,y_test)
kfold = KFold(n_splits=10 , random_state = 25,shuffle=True)
results = cross_val_score(LogisticRegression(solver='liblinear'),train_x,train_y, cv=kfold)
results.mean() 




