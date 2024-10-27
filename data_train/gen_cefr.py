import pandas as pd
import xml.etree.ElementTree as ET
import os

import textstat as ts

import lexicalrichness as lex

lvl2scr = {
    "A1": 0.05,
    "A2": 0.15,
    "B1": 0.3,
    "B2": 0.6,
    "C1": 0.7,
    "C2": 0.9,
}

cdir = os.path.join(os.getcwd(), 'data_train/cefr')

df = pd.DataFrame()

for _, _, files in os.walk(cdir):
    for file in files:
        if file.endswith('.xml'):
            print(file)

            flpth = os.path.join(cdir, file)

            root = ET.parse(flpth)

            txt = ""
            avg_scr = 0

            for child in root.iter():

                if ('type' in child.attrib and child.attrib['type'] == 'answer'):
                    for ch in child.iter():
                        if isinstance(ch.text, str):
                            txt += ch.text + " "

                if ('corresp' in child.attrib):
                    for ch in child.iter():
                        if (ch.tag == 'span'):
                            avg_scr += lvl2scr[ch.text]

            dx = pd.DataFrame(data=[[txt, avg_scr/3]], columns=['text', 'target'])
            df = pd.concat([df, dx])


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

df.to_csv(os.path.join(cdir, 'clean.csv'), index=True)

print(df)
print(df.dtypes)

for col in df.columns:
    print(f"{col} (#NaN): {df[col].isna().sum()}")
