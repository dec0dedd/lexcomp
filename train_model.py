import pandas as pd

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

import joblib

X = pd.read_csv('vectorized_clean.csv', index_col='id')
y = X.pop('target')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

xgbreg = XGBRegressor(
    n_estimators=4000,
    learning_rate=0.005,
    n_jobs=-1,
    early_stopping_rounds=100,
    random_state=42
).fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)]
    )

joblib.dump(xgbreg, 'lexcomp/model.joblib')