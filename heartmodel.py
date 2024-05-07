import os
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import pickle
from model_diabetes import rfc

heart = pd.read_csv("dataset/heartdisease.csv")


for c in heart.columns[:]:
    heart[c] = heart[c].apply(lambda x: heart[heart[c] != '?'][c].astype(float).mean() if x == "?" else x)
    heart[c] = heart[c].astype(float)
    
min_max = MinMaxScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
heart[columns_to_scale] = min_max.fit_transform(heart[columns_to_scale])

y = heart['target']
X = heart.drop(['target'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'bootstrap': [True, False],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=0), param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_param = grid_search.best_params_

rfc_classifier = rfc(X_train, y_train, best_param)

y_prediction = rfc_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_prediction)
print(accuracy)

pickle.dump(rfc_classifier, open(os.path.join('models/', 'heartdisease.pkl'), 'wb'))
