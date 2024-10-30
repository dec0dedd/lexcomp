import joblib
import os

from lexcomp._vectorize import tokenizer, model

import textstat as ts
import pandas as pd
import lexicalrichness as lex
import numpy as np
import torch

from tqdm import tqdm

fpath = os.path.dirname(os.path.realpath(__file__))


def logistic(data, k, mid):
    return np.exp(k*(data-mid))/(1+np.exp(k*(data-mid)))


class Model():
    model = None

    def __init__(self):
        try:
            self.model = joblib.load(os.path.join(fpath, 'model.joblib'))
        except FileNotFoundError:
            raise FileNotFoundError("Could not find model.joblib file!")
    
    def _get_syntactic(self, text: str):
        df = pd.DataFrame(data={'text': text}, index=[0])
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

        df.drop(columns=['text'], inplace=True)

        return df

    def _get_lexical(self, text: str):
        df = pd.DataFrame(data={'text': text}, index=[0])

        df['ttr'] = df['text'].apply(lambda x: lex.LexicalRichness(x).ttr)
        df['rttr'] = df['text'].apply(lambda x: lex.LexicalRichness(x).rttr)
        df['cttr'] = df['text'].apply(lambda x: lex.LexicalRichness(x).cttr)
        df['mtld75'] = df['text'].apply(lambda x: lex.LexicalRichness(x).mtld(threshold=0.75))

        df.drop(columns=['text'], inplace=True)

        return df

    def _preprocess_text(self, text: str) -> pd.DataFrame:
        df = pd.DataFrame(data={"text": text}, index=[0])

        df = pd.concat([df, self._get_syntactic(text), self._get_lexical(text)], axis=1)

        dx = self.vectorize_text(text)
        df = pd.concat([df, dx], axis=1)

        df.columns = df.columns.astype(str)

        return df

    def _preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        res = pd.DataFrame()

        sz = X.shape[0]

        for i in tqdm(range(sz)):
            res = pd.concat([res, self._preprocess_text(X['text'].iloc[i])])

        res['id'] = pd.RangeIndex(start=0, stop=res.shape[0]).astype(str)
        res.set_index('id', inplace=True)

        return res

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        res = pd.DataFrame()

        for txt in X['text']:
            res = pd.concat([res, self._preprocess_text(txt).drop(columns=['text'])])
        booster_cols = self.model.get_booster().feature_names
        return logistic(self.model.predict(res[booster_cols]), 10, 0.5)

    def predict_text(self, text: str):
        return self.predict(self._preprocess_text(text).drop(columns=['text']))

    def vectorize(self, X: pd.DataFrame) -> pd.DataFrame:
        emb = pd.DataFrame()

        sz = X.shape[0]
        for i in tqdm(range(sz)):
            emb = pd.concat([emb, self.vectorize_text(X.iloc[i]['text'])])

        emb['id'] = pd.RangeIndex(start=0, stop=emb.shape[0])
        emb['id'] = emb['id'].astype(str)
        emb.set_index('id', inplace=True)

        return emb

    def vectorize_text(self, text: str) -> pd.DataFrame:
        inp = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)

        with torch.no_grad():
            out = model(**inp)

        return pd.DataFrame(out.last_hidden_state[:, 0, :])
