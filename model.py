from sklearn.linear_model import LinearRegression
import pandas as pd

import matplotlib.pyplot as plt

df = pd.read_csv('merged_clean.csv', index_col='id')
y = df.pop('target')
mdl = LinearRegression(n_jobs=-1)

mdl.fit(df, y)
print(mdl.coef_)
print(mdl.score(df, y))
print(mdl.feature_names_in_)

plt.barh(mdl.feature_names_in_, mdl.coef_)
plt.show()
