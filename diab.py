import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
diabetes = pd.read_csv('diabetes.csv')
print(diabetes.columns)
'''
diabetes_map = {True: 1, False: 0}
diabetes['Outcome'] = diabetes['Outcome'].map(diabetes_map)
'''

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(diabetes.loc[:, diabetes.columns != 'Outcome'], diabetes['Outcome'], stratify=diabetes['Outcome'], random_state=66)

from sklearn.neighbors import KNeighborsClassifier

training_accuracy = []
test_accuracy = []
# try n_neighbors from 1 to 10
neighbors_settings = range(1, 11)

for n_neighbors in neighbors_settings:
    # build the model
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    # record training set accuracy
    training_accuracy.append(knn.score(X_train, y_train))
    # record test set accuracy
    test_accuracy.append(knn.score(X_test, y_test))
'''
plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.savefig('knn_compare_model')
'''

knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train, y_train)

print('Accuracy of K-NN classifier on training set: {:.2f}'.format(knn.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'.format(knn.score(X_test, y_test)))



from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression().fit(X_train, y_train)
logreg001 = LogisticRegression(C=0.01).fit(X_train, y_train)
logreg100 = LogisticRegression(C=100).fit(X_train, y_train)
print("LogisticRegression Training set accuracy: {:.3f}".format(logreg100.score(X_train, y_train)))
print("LogisticRegression Test set accuracy: {:.3f}".format(logreg100.score(X_test, y_test)))

'''
diabetes_features = [x for i,x in enumerate(diabetes.columns) if i!=8]

plt.figure(figsize=(8,6))
plt.plot(logreg.coef_.T, 'o', label="C=1")
plt.plot(logreg100.coef_.T, '^', label="C=100")
plt.plot(logreg001.coef_.T, 'v', label="C=0.001")
plt.xticks(range(diabetes.shape[1]), diabetes_features, rotation=90)
plt.hlines(0, 0, diabetes.shape[1])
plt.ylim(-3, 3)
plt.xlabel("Feature")
plt.ylabel("Coefficient magnitude")
plt.legend()
plt.savefig('log_coef')
'''

from sklearn.tree import DecisionTreeClassifier
'''
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))
'''
#pre-pruning
tree = DecisionTreeClassifier(max_depth=3, random_state=0)
tree.fit(X_train, y_train)

print("DecisionTreeClassifier Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
print("DecisionTreeClassifier Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))


print("Feature importances:\n{}".format(tree.feature_importances_))


from sklearn.ensemble import RandomForestClassifier
rf1 = RandomForestClassifier(max_depth=3, n_estimators=100, random_state=0)
rf1.fit(X_train, y_train)
print("RandomForestClassifier Accuracy on training set: {:.3f}".format(rf1.score(X_train, y_train)))
print("RandomForestClassifier Accuracy on test set: {:.3f}".format(rf1.score(X_test, y_test)))

from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(random_state=0)
gb.fit(X_train, y_train)

print("GradientBoostingClassifier Accuracy on training set: {:.3f}".format(gb.score(X_train, y_train)))
print("GradientBoostingClassifier Accuracy on test set: {:.3f}".format(gb.score(X_test, y_test)))


gb2 = GradientBoostingClassifier(random_state=0, learning_rate=0.01)
gb2.fit(X_train, y_train)

print("GradientBoostingClassifier Accuracy on training set: {:.3f}".format(gb2.score(X_train, y_train)))
print("GradientBoostingClassifier Accuracy on test set: {:.3f}".format(gb2.score(X_test, y_test)))




pickle.dump(rf1, open('diab.pkl','wb'))

# Loading model to compare the results
diab = pickle.load(open('diab.pkl','rb'))
print(diab.predict([[6,148,72,35,0,33.6,0.627,50]]))




knn=(knn.score(X_train, y_train)*100)
dt=(tree.score(X_train, y_train)*100)
rf=(rf1.score(X_train, y_train)*100)
lr=(logreg100.score(X_train, y_train)*100)
scores = [knn,dt,lr,rf]
algorithms = ["K-Nearest Neighbors","Decision Tree","Linear Regression","Random Forest"]    
for i in range(len(algorithms)):
    print("The accuracy score achieved using "+algorithms[i]+" is: "+str(scores[i])+" %")
sns.set(rc={'figure.figsize':(12,7)})
plt.xlabel("Algorithms")
plt.ylabel("Accuracy score")

sns.barplot(algorithms,scores)





'''
knn=(knn.score(X_test, y_test)*100)
dt=(tree.score(X_test, y_test)*100)
rf=(rf1.score(X_test, y_test)*100)
lr=(logreg100.score(X_test, y_test)*100)
scores = [knn,dt,lr,rf]
algorithms = ["K-Nearest Neighbors","Decision Tree","Linear Regression","Random Forest"]    
for i in range(len(algorithms)):
    print("The accuracy score achieved using "+algorithms[i]+" is: "+str(scores[i])+" %")
sns.set(rc={'figure.figsize':(12,7)})
plt.xlabel("Algorithms")
plt.ylabel("Accuracy score")

sns.barplot(algorithms,scores)
'''






