import os
import string

import pandas as pd

lvl2scr = {
    "A1": 0.05,
    "A2": 0.15,
    "B1": 0.3,
    "B2": 0.5,
    "C1": 0.7,
    "C2": 0.9,
}

cdir = os.path.join(os.getcwd(), 'data_train/kaggle')


def clean_str(text):
    printable = str(string.printable)
    return ''.join(filter(lambda x: x in printable, text))


df = pd.read_csv('data_train/kaggle/train.csv')
df['text'] = df['text'].apply(lambda x: x.replace('\n', ' '))
df['target'] = df['label'].map(lvl2scr)
df.drop(columns=['label'], inplace=True)
df['id'] = pd.RangeIndex(start=0, stop=df.shape[0])
df['text'] = df['text'].apply(clean_str)
df.set_index('id', inplace=True)

df.to_csv(os.path.join(cdir, 'clean.csv'), index=True)

print(df)
print(df.dtypes)

for col in df.columns:
    print(f"{col} (#NaN): {df[col].isna().sum()}")
