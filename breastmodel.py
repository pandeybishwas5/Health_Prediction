import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow
from sklearn.datasets import load_breast_cancer
import pickle
plt.style.use('ggplot')

# Breast cancer dataset for classification
data = load_breast_cancer()
#print (data.feature_names)
#print (data.target_names)
df =pd.read_csv('breastdata1.csv', header=None, 
                     names=["Sample code number", "Clump Thickness", 
                            "Uniformity of Cell Size", "Uniformity of Cell Shape", 
                            "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei" ,
                            "Bland Chromatin", "Normal Nucleoli", "Mitoses", "Class"])
'''
print(df.head())
print(df.info())
print ("Total number of diagnosis are ", str(df.shape[0]), ", ", df.Class.value_counts()[2], "Benign and Malignant are",
       df.Class.value_counts()[4])
'''
  # replace the ? with nans
df=df.replace(to_replace="?", value=0)

# convert the nuclei values to numbers since they are
# actually stored as strings
df['Bare Nuclei'] = pd.to_numeric(df['Bare Nuclei'])

'''df.hist(figsize=(10,15), bins=10)'''
# drop the ID's

df.drop('Sample code number',axis=1,inplace=True)



# set benign = 0, malignant = 1
for i in range(0,df.shape[0]):
    if(df.loc[i,'Class'] == 2):
        df.loc[i,'Class'] = 0
    else:
        df.loc[i,'Class'] = 1

print ("Total number of diagnosis are ", str(df.shape[0]), ", ", df.Class.value_counts()[0], "Benign and Malignant are",
       df.Class.value_counts()[1])

featureMeans = list(df.columns[0:9])
print(featureMeans)

sns.heatmap(data.df.corr())
'''

for i in range(len(featureMeans)):
    print(featureMeans[i],"=",type(featureMeans[i]))

import seaborn as sns


bins = 20
plt.figure(figsize=(15,15))
plt.subplot(3, 2, 1)
sns.distplot(df[df['Class']==1]['Clump Thickness'], bins=bins, color='green', label='M')
sns.distplot(df[df['Class']==0]['Clump Thickness'], bins=bins, color='red', label='B')
plt.legend(loc='upper right')

plt.subplot(3, 2, 2)
sns.distplot(df[df['Class']==1]['Uniformity of Cell Size'], bins=bins, color='green', label='M')
sns.distplot(df[df['Class']==0]['Uniformity of Cell Size'], bins=bins, color='red', label='B')
plt.legend(loc='upper right')

plt.subplot(3, 2, 3)
sns.distplot(df[df['Class']==1]['Uniformity of Cell Shape'], bins=bins, color='green', label='M')
sns.distplot(df[df['Class']==0]['Uniformity of Cell Shape'], bins=bins, color='red', label='B')
plt.legend(loc='upper right')

plt.subplot(3, 2, 4)
sns.distplot(df[df['Class']==1]['Marginal Adhesion'], bins=bins, color='green', label='M')
sns.distplot(df[df['Class']==0]['Marginal Adhesion'], bins=bins, color='red', label='B')
plt.legend(loc='upper right')



plt.subplot(3, 2, 5)
sns.distplot(df[df['Class']==1]['Bland Chromatin'], bins=bins, color='green', label='M')
sns.distplot(df[df['Class']==0]['Bland Chromatin'], bins=bins, color='red', label='B')
plt.legend(loc='upper right')

plt.subplot(3, 2, 6)
sns.distplot(df[df['Class']==1]['Normal Nucleoli'], bins=bins, color='green', label='M')
sns.distplot(df[df['Class']==0]['Normal Nucleoli'], bins=bins, color='red', label='B')
plt.legend(loc='upper right')

plt.subplot(3, 2, 8)
sns.distplot(df[df['Class']==1]['Mitoses'], bins=bins, color='green', label='M')
sns.distplot(df[df['Class']==0]['Mitoses'], bins=bins, color='red', label='B')
plt.legend(loc='upper right')

'''


'''
X = df.loc[:,featureMeans]
y = df.loc[:, 'Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

from sklearn.naive_bayes import GaussianNB

nbclf = GaussianNB().fit(X_train, y_train)
predicted = nbclf.predict(X_test)
print('Breast cancer dataset')
print('Accuracy of GaussianNB classifier on training set: {:.2f}'.format(nbclf.score(X_train, y_train)))
print('Accuracy of GaussianNB classifier on test set: {:.2f}'.format(nbclf.score(X_test, y_test)))



from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(rf.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(rf.score(X_test, y_test)))

rf1 = RandomForestClassifier(max_depth=3, n_estimators=100, random_state=0)
rf1.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(rf1.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(rf1.score(X_test, y_test)))
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm 
from sklearn import metrics 
from sklearn.feature_selection import RFE, f_regression
from sklearn.pipeline import make_pipeline
from sklearn.cross_validation import KFold, cross_val_score
from sklearn import preprocessing, cross_validation,neighbors

model = make_pipeline(preprocessing.StandardScaler(), DecisionTreeClassifier())
model.fit(X,y)
predictions = model.predict(X)
cv_scr = np.mean(cross_val_score(model, X, y, cv=5,n_jobs=-1,scoring='recall'))
print("Crossvalidation recall score:",cv_scr )

'''
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

import time
X = df.loc[:,featureMeans]
y = df.loc[:, 'Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

accuracy_all = []
cvs_all = []



#Support Vector Machines
from sklearn.svm import SVC, NuSVC

start = time.time()

clf = SVC()
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf, X, y, cv=5)

end = time.time()

accuracy_all.append(accuracy_score(prediction, y_test))
cvs_all.append(np.mean(scores))

#print("SVC Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))
#print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
#print("Execution time: {0:.5} seconds \n".format(end-start))

start = time.time()

clf = NuSVC()
clf.fit(X_train, y_train)
prediciton = clf.predict(X_test)
scores = cross_val_score(clf, X, y, cv=5)

end = time.time()

accuracy_all.append(accuracy_score(prediction, y_test))
cvs_all.append(np.mean(scores))

print("NuSVC Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
print("Execution time: {0:.5} seconds \n".format(end-start))


# Saving model to disk
pickle.dump(clf, open('breastmodel.pkl','wb'))

# Loading model to compare the results
breastmodel = pickle.load(open('breastmodel.pkl','rb'))
print(breastmodel.predict([[5,1,1,1,2,1,3,1,1]]))


'''

# Nearest Neighbors

from sklearn.neighbors import KNeighborsClassifier

start = time.time()

clf = KNeighborsClassifier()
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf, X, y, cv=5)

end = time.time()

accuracy_all.append(accuracy_score(prediction, y_test))
cvs_all.append(np.mean(scores))

print("Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
print("Execution time: {0:.5} seconds \n".format(end-start))

'''

