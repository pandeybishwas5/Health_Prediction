import numpy as np
import pickle
# for dataframes
import pandas as pd

# for easier visualization
import seaborn as sns
import os
# for visualization and to display plots
from matplotlib import pyplot as plt

# import color maps
from matplotlib.colors import ListedColormap

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

from math import sqrt

# to split train and test set
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

# to perform hyperparameter tuning
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# Machine Learning Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
df=pd.read_csv('indian_liver_patient.csv')
## if score==negative, mark 0 ;else 1 
def partition(x):
    if x == 2:
        return 0
    return 1

df['Dataset'] = df['Dataset'].map(partition)
## if score==negative, mark 0 ;else 1 
def partition(x):
    if x =='Male':
        return 0
    return 1

df['Gender'] = df['Gender'].map(partition)

df = df.drop_duplicates()
#print( df.shape )
df.Aspartate_Aminotransferase.sort_values(ascending=False).head()
df = df[df.Aspartate_Aminotransferase <=3000 ]
#print( df.shape )

df = df[df.Aspartate_Aminotransferase <=2500 ]
#print( df.shape )
df=df.dropna(how='any')
#print(df.head())


# Create separate object for target variable
y = df.Dataset

# Create separate object for input features
X = df.drop('Dataset', axis=1)
# Split X and y into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234, stratify=df.Dataset)
# Print number of observations in X_train, X_test, y_train, and y_test
#print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
train_mean = X_train.mean()
train_std = X_train.std()
## Standardize the train data set
X_train = (X_train - train_mean) / train_std
X_test = (X_test - train_mean) / train_std

tuned_params = {'n_estimators': [100, 200, 300, 400, 500], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}

#KNN
# creating odd list of K for KNN
neighbors = list(range(1,20,2))
# empty list that will hold cv scores
cv_scores = []

#  10-fold cross validation , 9 datapoints will be considered for training and 1 for cross validation (turn by turn) to determine value of k
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())   

# changing to misclassification error
MSE = [1 - x for x in cv_scores]


# determining best k
optimal_k = neighbors[MSE.index(min(MSE))]
#print('\nThe optimal number of neighbors is %d.' % optimal_k)

knn_classifier = KNeighborsClassifier(n_neighbors = optimal_k)
knn_classifier.fit(X_train, y_train)
y_pred = knn_classifier.predict(X_test)
y_train_pred = knn_classifier.predict(X_train)
knn_acc = accuracy_score(y_test, y_pred, normalize=True) * float(100)  ## get the accuracy on testing data
print("K Nearest Neighbours Accuracy: " + str(knn_acc))
cnf=confusion_matrix(y_test,y_pred).T
# Get just the prediction for the positive class (1)
y_pred_proba = knn_classifier.predict_proba(X_test)[:,1]

#Logistic Regression
tuned_params = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000], 'penalty': ['l1', 'l2']}
lr_classifier = GridSearchCV(LogisticRegression(), tuned_params, scoring = 'roc_auc', n_jobs=-1)
lr_classifier.fit(X_train, y_train)
## Predict Train set results
y_train_pred = lr_classifier.predict(X_train)
## Predict Test set results
y_pred = lr_classifier.predict(X_test)
lr_acc = accuracy_score(y_test, y_pred, normalize=True) * float(100)  ## get the accuracy on testing data
print("Logistic Regression Accuracy: " + str(lr_acc))
# Get just the prediction for the positive class (1)
y_pred_proba = lr_classifier.predict_proba(X_test)[:,1]

#Random Forest
tuned_params = {'n_estimators': [100, 200, 300, 400, 500], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
rf_classifier = RandomizedSearchCV(RandomForestClassifier(), tuned_params, n_iter=15, scoring = 'roc_auc', n_jobs=-1)
rf_classifier.fit(X_train, y_train)
y_train_pred = rf_classifier.predict(X_train)
y_pred = rf_classifier.predict(X_test)
rf_acc = accuracy_score(y_test, y_pred, normalize=True) * float(100)  ## get the accuracy on testing data
print("Random Forest Accuracy: " + str(rf_acc))
# Get just the prediction for the positive class (1)
y_pred_proba = rf_classifier.predict_proba(X_test)[:,1]

#Decision Tree
tuned_params = {'min_samples_split': [2, 3, 4, 5, 7], 'min_samples_leaf': [1, 2, 3, 4, 6], 'max_depth': [2, 3, 4, 5, 6, 7]}
dt_classifier = RandomizedSearchCV(DecisionTreeClassifier(), tuned_params, n_iter=15, scoring = 'roc_auc', n_jobs=-1)
dt_classifier.fit(X_train, y_train)
y_train_pred = dt_classifier.predict(X_train)
y_pred = dt_classifier.predict(X_test)
dt = accuracy_score(y_test, y_pred, normalize=True) * float(100)  ## get the accuracy on testing data
print("Decision Tree Accuracy: " + str(dt))
y_pred_proba = dt_classifier.predict_proba(X_test)[:,1]


# Saving model to disk
pickle.dump(knn_classifier, open('livermodel.pkl','wb'))
# Loading model to compare the results
livermodel = pickle.load(open('livermodel.pkl','rb'))
print(livermodel.predict([[65,1,0.7,0.1,187,16,18,6.8,3.3,0.9]]))

#Graphs and Charts
'''
#Visualizing the features of each category of people (healthy/unhealthy)
data1 = df[df['Dataset']==0] # no disease (original dataset had it labelled as 2 and not 0)
data1 = data1.iloc[:,:-1]

data2 = df[df['Dataset']==1] # with disease
data2 = data2.iloc[:,:-1]

fig = plt.figure(figsize=(10,15))

ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212,sharex=ax1)

ax1.grid()
ax2.grid()

ax1.set_title('Features vs mean values',fontsize=13,weight='bold')
ax1.text(200,0.8,'NO DISEASE',fontsize=20,horizontalalignment='center',color='green',weight='bold')


ax2.set_title('Features vs mean values',fontsize=13,weight='bold')
ax2.text(200,0.8,'DISEASE',fontsize=20,horizontalalignment='center',color='red',weight='bold')

# for axis ax1
plt.sca(ax1)
plt.xticks(rotation = 0, 
           weight='bold', 
           family='monospace',
           size='large')
plt.yticks( weight='bold', 
           family='monospace',
           size='large')
# for axis ax2
plt.sca(ax2)
plt.xticks(rotation = 0, 
           weight='bold', 
           family='monospace',
           size='large')
plt.yticks( weight='bold', 
           family='monospace',
           size='large')

# sns.set_style('whitegrid')

sns.barplot(data=data1,ax=ax1,orient='horizontal', palette='bright') # no disease
sns.barplot(data=data2,ax=ax2,orient='horizontal',palette='bright',saturation=0.80) # with disease
'''

'''
#Visualizing the differences in chemicals in Healthy/Unhealthy people
with_disease = df[df['Dataset']==1]

with_disease = with_disease.drop(columns=['Gender','Age','Dataset'])
names1 = with_disease.columns.unique()
mean_of_features1 = with_disease.mean(axis=0,skipna=True)


without_disease = df[df['Dataset']==2]

without_disease = without_disease.drop(columns=['Gender','Age','Dataset'])
names2 = without_disease.columns.unique()
mean_of_features2 = without_disease.mean(axis=0,skipna=True)

people = []

for x,y in zip(names1,mean_of_features1):
    people.append([x,y,'Diseased'])
for x,y in zip(names2,mean_of_features2):
    people.append([x,y,'Healthy'])
    
new_data = pd.DataFrame(people,columns=['Chemicals','Mean_Values','Status'])

#ValueError: If using all scalar values, you must pass an index
#https://stackoverflow.com/questions/17839973/construct-pandas-dataframe-from-values-in-variables

fig = plt.figure(figsize=(20,8))
plt.title('Comparison- Diseased vs Healthy',size=20,loc='center')
plt.xticks(rotation = 30, 
           weight='bold', 
           family='monospace',
           size='large')
plt.yticks( weight='bold', 
           family='monospace',
           size='large')

g1 = sns.barplot(x='Chemicals',y='Mean_Values',hue='Status',data=new_data,palette="RdPu_r")
plt.legend(prop={'size': 20})
plt.xlabel('Chemicals',size=19)
plt.ylabel('Mean_Values',size=19)

new_data'''

'''#Percentage of Chemicals in Unhealthy People
# create data
with_disease = df[df['Dataset']==1]
with_disease = with_disease.drop(columns=['Dataset','Gender','Age'])
names = with_disease.columns.unique()
mean_of_features = with_disease.mean(axis=0,skipna=True)

list_names = ['Total_Bilirubin','Alkaline_Phosphotase','Direct_Bilirubin','Albumin','Alamine_Aminotransferase',
              'Total_Protiens','Aspartate_Aminotransferase','Albumin_and_Globulin_Ratio']
list_means = [4.164423076923075,319.00721153846155,1.923557692307693,3.0605769230769226,
             99.60576923076923,6.459134615384617,137.69951923076923,0.9141787439613527]

l_names = []
l_means = []
mydict = {}
for x,y in zip(names,mean_of_features):
    mydict[x]=y
    l_names.append(x)
    l_means.append(y)


fig = plt.figure()
plt.title('Percentage of Chemicals in Unhealthy People',size=20,color='#016450')
# Create a pieplot
plt.axis('equal')
explode = (0.09,)*(len(list_means))
color_pink=['#7a0177','#ae017e','#dd3497','#f768a1','#fa9fb5','#fcc5c0','#fde0dd','#fff7f3']

wedges, texts, autotexts = plt.pie( list_means,
                                    explode=explode,
                                    labels=list_names, 
                                    labeldistance=1,
                                    textprops=dict(color='k'),
                                    radius=2.5,
                                    autopct="%1.1f%%",
                                    pctdistance=0.7,
                                    wedgeprops = { 'linewidth' : 3, 'edgecolor' : 'white' })

plt.setp(autotexts,size=17)
plt.setp(texts,size=12)
# plt.show() # don't show pie here [leave it commented]
 
# add a circle at the center
my_circle=plt.Circle( (0,0), 1, color='white')
p=plt.gcf() # get current figure reference
p.gca().add_artist(my_circle) # get current axes

plt.show()'''


'''#Data Outcome
count_class_0, count_class_1 = df['Dataset'].value_counts()
# Divide by class
data_class0 = df[df['Dataset'] == 0]
data_class1 = df[df['Dataset'] == 1]
liverdata = pd.concat([data_class0,data_class1.head(250)], axis=0)
liverdata
sns.countplot(data=liverdata, x = 'Dataset', label='Count')
LD,NLD = liverdata['Dataset'].value_counts()
print('Number of patients diagnosed with liver disease: ',LD)
print('Number of patients not diagnosed with liver disease: ',NLD)'''

'''#Histogram
df.hist(figsize=(8,10), bins=10)
plt.show()'''

'''#Heat Map
plt.figure(figsize=(8,6))
sns.heatmap(abs(df.corr()), annot=True) 
plt.show()'''

#Model Accuracy
knn_model=(knn_classifier.score(X_test, y_test)*100)
lr_model=(lr_classifier.score(X_test, y_test)*100)
rf_model=(rf_classifier.score(X_test, y_test)*100)
dt_model=(dt_classifier.score(X_test, y_test)*100)
scores = [knn_model,lr_model,rf_model,dt_model]
algorithms = ["K-Nearest Neighbors","Linear Regression","Random Forest","Decision Tree"]    
for i in range(len(algorithms)):
    print("The accuracy score achieved using "+algorithms[i]+" is: "+str(scores[i])+" %")
sns.set(rc={'figure.figsize':(15,8)})
plt.xlabel("Algorithms")
plt.ylabel("Accuracy score")

sns.barplot(algorithms,scores)

