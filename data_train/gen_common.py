import pandas as pd
import os

import textstat as ts

from sklearn.preprocessing import MinMaxScaler

import lexicalrichness as lex

import numpy as np

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

# Syntacit features
df['flesch_reading'] = df['text'].apply(ts.flesch_reading_ease).astype('float64')
df['dale_chall'] = df['text'].apply(ts.dale_chall_readability_score).astype('float64')
df['coleman_liau'] = df['text'].apply(ts.coleman_liau_index).astype('float64')
df['mcalpine_eflaw'] = df['text'].apply(ts.mcalpine_eflaw).astype('float64')
df['wrd_cnt'] = df['text'].apply(ts.lexicon_count).astype('float64')
df['sen_cnt'] = df['text'].apply(ts.sentence_count).astype('float64')
df['syl_cnt'] = df['text'].apply(ts.syllable_count).astype('float64')
df['polysyl_cnt'] = df['text'].apply(ts.polysyllabcount).astype('float64')
df['avg_wrd_per_sen'] = df['wrd_cnt'] / df['sen_cnt']
df['avg_syl_per_wrd'] = df['syl_cnt'] / df['wrd_cnt']


# Lexical features
df['ttr'] = df['text'].apply(lambda x: lex.LexicalRichness(x).ttr)
df['rttr'] = df['text'].apply(lambda x: lex.LexicalRichness(x).rttr)
df['cttr'] = df['text'].apply(lambda x: lex.LexicalRichness(x).cttr)
df['mtld75'] = df['text'].apply(lambda x: lex.LexicalRichness(x).mtld(threshold=0.75))

df.to_csv(os.path.join(cdir, 'clean.csv'))

print(df)
print(df.dtypes)

for col in df.columns:
    print(f"{col} (#NaN): {df[col].isna().sum()}")

print(df.shape)