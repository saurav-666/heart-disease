# -*- coding: utf-8 -*-

import pandas as pd

#read dataset
data = pd.read_csv('heart_cleveland_upload.csv')

data.describe()
age_positive = data.sex[data['condition']==1].value_counts()
print('No of people having heart disease (1-male, 0-female) : ',age_positive)

#define features
X = data.drop(['condition'], axis = 1)

print(data.shape)

print(X.shape)

#define target
y = data['condition']

#scaling the features
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
X_scaled = scale.fit_transform(X)


# split dataset into training and testing data
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size = 0.20,random_state=30)

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


#K-nearest neighbors has the highest accuracy

prediction = knn.predict(X_test)

#confusion matrix to check accuracy
from sklearn.metrics import confusion_matrix
results=confusion_matrix(y_test,prediction)
print(results)

# prediction of heart disease based on user input data
#Deployement of model
age = input('Age of the person : ')
sex = input('Gender (0: female, 1: male) : ')
cp = input('Chest pain type (0: Typical Angina 1: Atypical Angina 2: Non-anginal Pain 3: Asymptomatic) : ')
trestbps = input('Resting blood pressure : ')
chol = input('Serum cholesterol : ')
fbs = input('Fasting blood sugar > 120 mg/dl (1: true; 0: false) : ')
restecg = input('resting ECG results (0: normal,1:ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), 2: showing probable or definite left ventricular hypertrophy by Estes criteria : ')
thalach = input('Maximum heart rate achieved : ')
exang = input('exercise induced angina (1: yes, 0: no) :')
oldpeak = input('ST depression induced by exercise relative to rest : ')
slope = input('Slope of the peak exercise ST segment (0: upsloping, 1: flat, 2: downsloping) : ')
ca = input('number of major vessels (0-3) colored by flourosopy : ')
thal = input('thal (0: normal; 1: fixed defect; 2: reversable defect) : ')

input_data = [[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]]

input_df = pd.DataFrame(input_data, columns = [i for i in range(0,13)])

y_pred = knn.predict(input_df)


if (y_pred[0]==0):
    print("The patient has no heart disease")
else:
    print("The patient has heart disease")







