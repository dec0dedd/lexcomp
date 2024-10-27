import pandas as pd

from xgboost import XGBRegressor

import joblib

X = pd.read_csv('merged_clean_train.csv', index_col='id')
y = X.pop('target')

xgbreg = XGBRegressor(
    n_estimators=4000,
    learning_rate=0.005,
    n_jobs=-1,
    random_state=42
).fit(X, y)

joblib.dump(xgbreg, 'model.joblib')
model = joblib.load('model.joblib')