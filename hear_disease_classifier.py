# -*- coding: utf-8 -*-

import pandas as pd

#read dataset
data = pd.read_csv('heart_cleveland_upload.csv')

data.describe()
data.thalach[data['condition']==1].value_counts()

#define features
X = data.drop(['condition'], axis = 1)

print(data.shape)

print(X.shape)

#define target
y = data['condition']

# split dataset into training and testing data
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.20,random_state=30)

#K-nearest neighbors classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train,y_train)

#decision tree classifier
from sklearn.tree import DecisionTreeClassifier
dtf = DecisionTreeClassifier()
dtf.fit(X_train, y_train)

#support vector machine
from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train, y_train)

#random forrest classifier
from sklearn.ensemble import RandomForestRegressor
rdm_frst = RandomForestRegressor(n_estimators=20, random_state=0)
rdm_frst.fit(X_train, y_train)

#predicting the outcome based on training data for various algorithm
y_predict1= knn.predict(X_test)
y_predict2= dtf.predict(X_test)
y_predict3= svc.predict(X_test)
y_predict4= rdm_frst.predict(X_test)

#deifining a dataset with all the predicted outcomes
df1=pd.DataFrame({'Actual':y_test, 'Predicted_knn':y_predict1,'Predicted_dtf':y_predict2,'Predicted_svc':y_predict3,'Predicted_random_forrest':y_predict4 })

# displaying accuracy of various algorithm
print(knn.score(X_test,y_test))
print(dtf.score(X_test,y_test))
print(svc.score(X_test,y_test))
print(rdm_frst.score(X_test,y_test))

#decision tree has the highest accuracy














