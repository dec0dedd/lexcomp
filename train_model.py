import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from lexcomp import Model

import joblib

X = pd.read_csv('merged_clean.csv', index_col='id')
y = X.pop('target')

xgb = Model()

X = xgb._preprocess(X).drop(columns=['text'])
X.to_csv('preprocessed.csv', index=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

mdl = SVR(C=0.5).fit(X_train, y_train)
print(mdl.score(X_test, y_test))

joblib.dump(mdl, 'lexcomp/model.joblib')
