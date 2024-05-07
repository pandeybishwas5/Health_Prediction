import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC, NuSVC
import pickle

df = pd.read_csv('dataset/breastdata1.csv', header=None,
                 names=["Sample code number",
                        "Clump Thickness",
                        "Uniformity of Cell Size",
                        "Uniformity of Cell Shape",
                        "Marginal Adhesion",
                        "Single Epithelial Cell Size",
                        "Bare Nuclei",
                        "Bland Chromatin",
                        "Normal Nucleoli",
                        "Mitoses",
                        "Class"])

df.drop('Sample code number', axis=1, inplace=True)
df = df.replace(to_replace="?", value=0)

df['Bare Nuclei'] = pd.to_numeric(df['Bare Nuclei'])
df["Class"] = df["Class"].replace({2: 0, 4: 1})

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

accuracy_all = []
cvs_all = []

clf = SVC()
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf, X, y, cv=5)

accuracy_all.append(accuracy_score(prediction, y_test))
cvs_all.append(np.mean(scores))

print("SVC Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))

clf = NuSVC()
clf.fit(X_train, y_train)
prediciton = clf.predict(X_test)
scores = cross_val_score(clf, X, y, cv=5)

accuracy_all.append(accuracy_score(prediction, y_test))
cvs_all.append(np.mean(scores))

print("NuSVC Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))


pickle.dump(clf, open(os.path.join('models/', 'breastmodel.pkl'), 'wb'))





