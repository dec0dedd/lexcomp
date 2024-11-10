import os

import pandas as pd
import numpy as np


# Constants used when translating from classification to regression
E_SCORE = 0.1
I_SCORE = 0.5
A_SCORE = 0.7

cdir = os.path.join(os.getcwd(), 'data_train/onestop')

df = pd.DataFrame()

for _, _, files in os.walk(cdir):
    for file in files:
        if file.endswith('.csv') and file != "clean.csv":
            print(file)

            flpth = os.path.join(cdir, file)

            st = ""
            with open(flpth, 'rb+') as fl:
                st = fl.read().decode('utf-8', errors='ignore')
                st.replace('\n', ' ')

            with open(flpth, 'w') as fl:
                fl.write(st)

            dx = pd.read_csv(flpth, header=0)
            dx.rename(columns={'Elementary': 'E', 'Intermediate ': 'I', 'Intermediate': 'I', 'Advanced': 'A'}, inplace=True)

            tmp = pd.concat([dx['E'], dx['I'], dx['A']]).rename('text').to_frame()
            scr = np.concatenate([
                np.full((dx.shape[0], 1), E_SCORE),
                np.full((dx.shape[0], 1), I_SCORE),
                np.full((dx.shape[0], 1), A_SCORE)
            ])

            tmp['target'] = scr.astype('float64')
            df = pd.concat([df, tmp])

df['id'] = pd.RangeIndex(start=0, stop=df.shape[0])
df.set_index('id', inplace=True)
df.dropna(inplace=True)

df['text'] = df['text'].apply(lambda x: x.replace('\n', ' '))

df = df.sample(n=df.shape[0]//20, random_state=42)
df.to_csv(os.path.join(cdir, 'clean.csv'), index=True)

print(df)
print(df.dtypes)

for col in df.columns:
    print(f"{col} (#NaN): {df[col].isna().sum()}")
