# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 14:38:13 2020

@author: hp
"""
#Basic
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow
import pickle
import seaborn as sns

# Other libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB

# Machine Learning
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

heart = pd.read_csv("heartdisease.csv")
#print(heart.describe())
#Preprocessing the data
# we have unknown values '?'
# change unrecognized value '?' into mean value through the column
for c in heart.columns[:]:
    heart[c] = heart[c].apply(lambda x: heart[heart[c]!='?'][c].astype(float).mean() if x == "?" else x)
    heart[c] = heart[c].astype(float)
    
#Feature Engineering
# heart = pd.get_dummies(heart, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
min_max = MinMaxScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
heart[columns_to_scale ] = min_max.fit_transform(heart[columns_to_scale ])

#Spit into 67% training data and 33% testing data
y = heart['target']
X = heart.drop(['target'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)
print(len(X_train))
len(X_test)

training_accuracy = []
test_accuracy = []
#KNN
knn_scores = []
for k in range(1,21):
    knn_classifier = KNeighborsClassifier(n_neighbors = k)
    knn_classifier.fit(X_train, y_train)
    knn_scores.append(knn_classifier.score(X_test, y_test))
    # record training set accuracy
    training_accuracy.append(knn_classifier.score(X_train, y_train))
    # record test set accuracy
    test_accuracy.append(knn_classifier.score(X_test, y_test))
knn_scores
#print("KN NEAREST NEIGHBOURS:")
#print("Training set score: {:.2f}".format(knn_classifier.score(X_train, y_train)*100))
print("KNN Accuracy: {:.2f}".format(knn_classifier.score(X_test, y_test)*100))

#Decision Tree
dt_scores = []
for i in range(1, len(X.columns) + 1):
    dt_classifier = DecisionTreeClassifier(max_features = i, random_state = 0)
    dt_classifier.fit(X_train, y_train)
    dt_scores.append(dt_classifier.score(X_test, y_test))
    # record training set accuracy
    training_accuracy.append(dt_classifier.score(X_train, y_train))
    # record test set accuracy
    test_accuracy.append(dt_classifier.score(X_test, y_test))
#print("DECISION TREE:")
#print("Training set score: {:.2f}".format(dt_classifier.score(X_train, y_train)*100))
print("Decision Tree Accuracy: {:.2f}".format(dt_classifier.score(X_test, y_test)*100))

#Linear Regression
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
score = (linear_reg.score(X_test, y_test)*100)
print("Linear Regression Accuracy:" + str(score))

rf_scores = []
estimators = [10, 100, 200,210, 220, 230, 240, 250, 500, 1000]
for i in estimators:
    rf_classifier = RandomForestClassifier(n_estimators = i, random_state = 0)
    rf_classifier.fit(X_train, y_train)
    rf_scores.append(rf_classifier.score(X_test, y_test))
    # record training set accuracy
    training_accuracy.append(rf_classifier.score(X_train, y_train))
    # record test set accuracy
    test_accuracy.append(rf_classifier.score(X_test, y_test))
print("Random Forest Accuracy: {:.2f}".format(rf_classifier.score(X_test, y_test)*100))
    
# Saving model to disk
pickle.dump(rf_classifier, open('heartmodel.pkl','wb'))

# Loading model to compare the results
heartmodel = pickle.load(open('heartmodel.pkl','rb'))
print(heartmodel.predict([[63,1,3,145,233,1,0,150,0,2.3,0,0,1]]))

#Ghraphs and Charts

'''#Data Outcome
y = heart["target"]
sns.countplot(y)
target_temp = heart.target.value_counts()
print(target_temp)'''

#We notice, that females are more likely to have heart problems than males
#sns.barplot(heart["sex"],heart["target"])

#Analysing the 'Chest Pain Type' feature
#sns.barplot(heart["cp"],y)

#Fasting Blood sugar Data
#sns.barplot(heart["restecg"],y)

#Analysing the 'exang' feature
#sns.barplot(heart["exang"],y)

#Analysing the Slope feature
#sns.barplot(heart["slope"],y)

#Analysing the 'thal' feature
#sns.barplot(heart["thal"],y)

#Histogram
#heart.hist(figsize=(8,10), bins=10)

'''#Shows the features correlation. ps: there's a bug on matplotlib #3.1.1.
plt.figure(figsize=(12,10))
sns.heatmap(abs(heart.corr()), annot=True) 
plt.show()'''

knn=(knn_classifier.score(X_test, y_test)*100)
dt=(dt_classifier.score(X_test, y_test)*100)
rf=(rf_classifier.score(X_test, y_test)*100)
scores = [knn,dt,score,rf]
algorithms = ["K-Nearest Neighbors","Decision Tree","Linear Regression","Random Forest"]    
for i in range(len(algorithms)):
    print("The accuracy score achieved using "+algorithms[i]+" is: "+str(scores[i])+" %")
sns.set(rc={'figure.figsize':(15,8)})
plt.xlabel("Algorithms")
plt.ylabel("Accuracy score")

sns.barplot(algorithms,scores)