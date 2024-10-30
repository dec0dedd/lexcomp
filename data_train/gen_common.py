import pandas as pd
import os

from sklearn.preprocessing import MinMaxScaler

"""
Steps to data preprocessing:

1. Get rid of standard error by adding its' half to the target

2. Scale values to [0, 1] using a scaler (explore different scalers/functions). Possible scalers:
    - MinMaxScaler,
    - Sigmoid function,
    - tanh,

3. Remove useless features like 'url_legal' and 'license'
"""

cdir = os.path.join(os.getcwd(), 'data_train/commonlit')

df = pd.read_csv(os.path.join(cdir, 'train.csv')).drop(columns=['id'])
df['target'] = (df['target'] + df['standard_error']/2).astype('float64')
df.drop(columns=['url_legal', 'license', 'standard_error'], inplace=True)

mnx = MinMaxScaler()
df['target'] = mnx.fit_transform(df['target'].to_frame())
df['target'] = 1-df['target']

df.rename(columns={'excerpt': 'text'}, inplace=True)
df['id'] = pd.RangeIndex(start=0, stop=df.shape[0])
df.set_index('id', inplace=True)

df['text'] = df['text'].apply(lambda x: x.replace('\n', ' '))

df.to_csv(os.path.join(cdir, 'clean.csv'))

print(df)
print(df.dtypes)

for col in df.columns:
    print(f"{col} (#NaN): {df[col].isna().sum()}")

print(df.shape)
