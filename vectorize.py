from lexcomp import Model

import pandas as pd

df = pd.read_csv('merged_clean.csv', index_col='id').dropna()
df['id'] = pd.RangeIndex(start=0, stop=df.shape[0])
df.set_index('id', inplace=True)
df.index = df.index.astype(str)

mdl = Model()

x = df.iloc[:1000]

df_emb = pd.concat([x, mdl.vectorize(x)], axis=1)
assert df_emb.shape[0] == x.shape[0]


df_emb.drop(columns=['text'], inplace=True)
df_emb['id'] = pd.RangeIndex(start=0, stop=df_emb.shape[0])
df_emb.set_index('id', inplace=True)

df_emb.columns = df_emb.columns.astype(str)

print(df_emb)

df_emb.to_csv('vectorized_clean.csv', index=True)