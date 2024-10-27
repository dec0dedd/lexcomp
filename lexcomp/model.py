import joblib
import os

import textstat as ts

import pandas as pd
import lexicalrichness as lex

fpath = os.path.dirname(os.path.realpath(__file__))

class Model():
    model = None

    def __init__(self):
        try:
            self.model = joblib.load(os.path.join(fpath, 'model.joblib'))
        except FileNotFoundError:
            raise FileNotFoundError("Could not find model.joblib file!")
    
    def _preprocess(self, text):
        df = pd.DataFrame(data={"text": text}, index=[0])
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

        df.drop(columns=['text'], inplace=True)

        return df
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_text(self, text):
        return self.predict(self._preprocess(text))