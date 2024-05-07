import os
import warnings
import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
df = pd.read_csv('dataset/pimadata.csv')

df_map = {True: 1, False: 0}
df['diabetes'] = df['diabetes'].map(df_map)

for column in df.columns[:-1]:
    df[column] = df[column].astype(float)

df.loc[df['glucose_conc'] == 0, 'glucose_conc'] = df['glucose_conc'].median()
df.loc[df['diastolic_bp'] == 0, 'diastolic_bp'] = df['diastolic_bp'].median()
df.loc[df['thickness'] == 0, 'thickness'] = df['thickness'].median()
df.loc[df['insulin'] == 0, 'insulin'] = df['insulin'].median()

X = df.drop('diabetes', axis=1)
y = df['diabetes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


def gbc(X_train, X_test, y_train, y_test):
    gbc_model = GradientBoostingClassifier(random_state=42,
                                           learning_rate=0.05,
                                           max_depth=3,
                                           min_samples_split=2,
                                           n_estimators=200)
    gbc_model.fit(X_train, y_train)
    return gbc_model


def rfc(X_train, y_train, best_param=None):
    rfc_model = RandomForestClassifier(**best_param)
    rfc_model.fit(X_train, y_train)
    return rfc_model


def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


def normalization(train, test):
    scaler = StandardScaler()
    scaler.fit(train)
    X_train = scaler.transform(train)
    X_test = scaler.transform(test)
    return X_train, X_test


X_train, X_test = normalization(X_train, X_test)
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
# gbc_model = gbc(X_train, X_test, y_train, y_test)
rfc_m = rfc(X_train, y_train, best_param)
y_pred = rfc_m .predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

pickle.dump(rfc_m, open(os.path.join('models/', 'rfc_model_diabetes.pkl'), 'wb'))
