import pandas as pd
import os

cdir = os.getcwd()

df = pd.DataFrame()

for subdir, dirs, files in os.walk(os.path.join(cdir, 'data_train')):
    for file in files:
        if file == "clean.csv":
            dx = pd.read_csv(os.path.join(subdir, file), lineterminator='\n').drop(columns=['id'])
            df = pd.concat([df, dx])

df['id'] = pd.RangeIndex(start=0, stop=df.shape[0])
df.set_index('id', inplace=True)
df.drop(columns=['text'], inplace=True)

df.to_csv('merged_clean_train.csv')

print(df)
print(df.dtypes)

for col in df.columns:
    print(f"{col} (#NaN): {df[col].isna().sum()}")