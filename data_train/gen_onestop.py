import pandas as pd
import os
import numpy as np
import textstat as ts

import lexicalrichness as lex

import matplotlib.pyplot as plt

import numpy as np


"""
Steps for data preprocessing:

1. Assign constant(?) scores for Elementary/Intermediate/Adv sets

"""


# Constants used when translating from classification to regression
E_SCORE = 0.1
I_SCORE = 0.5
A_SCORE = 0.8

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

df = df.sample(n=df.shape[0]//20, random_state=42)
df.to_csv(os.path.join(cdir, 'clean.csv'), index=True)

print(df)
print(df.dtypes)

for col in df.columns:
    print(f"{col} (#NaN): {df[col].isna().sum()}")