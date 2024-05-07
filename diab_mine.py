import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


diabetes = pd.read_csv('dataset/diabetes.csv')

X = diabetes.drop('Outcome', axis=1)
y = diabetes['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf1 = RandomForestClassifier(max_depth=3, n_estimators=100, random_state=0)
rf1.fit(X_train, y_train)

print("RandomForestClassifier Accuracy on training set: {:.3f}".format(rf1.score(X_train, y_train)))
print("RandomForestClassifier Accuracy on test set: {:.3f}".format(rf1.score(X_test, y_test)))


pickle.dump(rf1, open(os.path.join('models/', 'diabetes.pkl'), 'wb'))
