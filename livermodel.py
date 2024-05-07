import pickle
import pandas as pd
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")
df = pd.read_csv('dataset/indian_liver_patient.csv')

df_map = {2: 0, 1: 1}
df['Dataset'] = df['Dataset'].map(df_map)

df_map = {'Female': 1, 'Male': 0}
df['Gender'] = df['Gender'].map(df_map)

df = df.drop_duplicates()

df.Aspartate_Aminotransferase.sort_values(ascending=False).head()
df = df[df.Aspartate_Aminotransferase <= 3000]


df = df[df.Aspartate_Aminotransferase <= 2500]

df = df.dropna(how='any')

y = df.Dataset
X = df.drop('Dataset', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234, stratify=df.Dataset)

train_mean = X_train.mean()
train_std = X_train.std()

X_train = (X_train - train_mean) / train_std
X_test = (X_test - train_mean) / train_std
# tuned_params = {'n_estimators': [100, 200, 300, 400, 500], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
neighbors = list(range(1, 20, 2))
cv_scores = []

for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())   

MSE = [1 - x for x in cv_scores]

optimal_k = neighbors[MSE.index(min(MSE))]

knn_classifier = KNeighborsClassifier(n_neighbors=optimal_k)
knn_classifier.fit(X_train, y_train)
y_pred = knn_classifier.predict(X_test)
y_train_pred = knn_classifier.predict(X_train)
knn_acc = accuracy_score(y_test, y_pred, normalize=True) * float(100)
print("K Nearest Neighbours Accuracy: " + str(knn_acc))
cnf = confusion_matrix(y_test,y_pred).T
# Get just the prediction for the positive class (1)
y_pred_proba = knn_classifier.predict_proba(X_test)[:, 1]

pickle.dump(knn_classifier, open(os.path.join('models/', 'livermodel.pkl'), 'wb'))

