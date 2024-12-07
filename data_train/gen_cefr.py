import xml.etree.ElementTree as ET
import os

import pandas as pd

lvl2scr = {
    "A1": 0.05,
    "A2": 0.15,
    "B1": 0.3,
    "B2": 0.5,
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
            scr_sum = 0

            for child in root.iter():

                if ('type' in child.attrib and child.attrib['type'] == 'answer'):
                    for ch in child.iter():
                        if isinstance(ch.text, str):
                            txt += ch.text + " "

                if ('corresp' in child.attrib):
                    for ch in child.iter():
                        if (ch.tag == 'span'):
                            scr_sum += lvl2scr[ch.text]

            dx = pd.DataFrame(data=[[txt, scr_sum/3]], columns=['text', 'target'])
            df = pd.concat([df, dx])


df['id'] = pd.RangeIndex(start=0, stop=df.shape[0])
df.set_index('id', inplace=True)
df.dropna(inplace=True)

df['text'] = df['text'].apply(lambda x: x.replace('\n', ' '))

df.to_csv(os.path.join(cdir, 'clean.csv'), index=True)

print(df)
print(df.dtypes)

for col in df.columns:
    print(f"{col} (#NaN): {df[col].isna().sum()}")
