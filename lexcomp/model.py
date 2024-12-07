import joblib
import os
import warnings

from lexcomp._vectorize import tokenizer, model

import textstat as ts
import pandas as pd
import lexicalrichness as lex
import numpy as np
import torch
import nltk
import spacy
import benepar
from tqdm import tqdm

warnings.filterwarnings("ignore")

nltk.download('punkt_tab')
fpath = os.path.dirname(os.path.realpath(__file__))

nlp = spacy.load("en_core_web_sm")
benepar.download('benepar_en3')
nlp.add_pipe("benepar", config={"model": "benepar_en3"})


def calculate_tree_depth(tree):
    if isinstance(tree, str):
        return 0
    return 1 + max(calculate_tree_depth(child) for child in tree.subtrees())


def average_parse_tree_depth(text):
    text = " ".join(text.split())
    doc = nlp(text)
    total_depth = 0
    sentence_count = 0

    for sent in doc.sents:
        parse_tree = sent._.parse_string
        if parse_tree:
            tree = nltk.Tree.fromstring(parse_tree)
            depth = tree.height()
            total_depth += depth
            sentence_count += 1

    # Avoid division by zero
    return total_depth / sentence_count if sentence_count > 0 else 0


def logistic(data, k, mid):
    return np.exp(k*(data-mid))/(1+np.exp(k*(data-mid)))


def get_unique_word_freq(text):
    words = nltk.tokenize.word_tokenize(text)
    fdist = nltk.FreqDist(words)
    return np.array([
        freq**3 for word, freq in fdist.items() if not word.isdigit()
        ]).mean()


def get_pronoun_count(text):
    pronouns = ['I', 'me', 'my', 'we', 'us', 'our', 'he', 'him', 'his', 'she',
                'her', 'hers', 'it', 'its', 'they', 'them', 'their', 'theirs']
    return sum(1 for word in text.split() if word.lower() in pronouns)


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
        df['unique_freq'] = get_unique_word_freq(text)
        df['pronoun_dens'] = get_pronoun_count(text) / df['wrd_cnt']
        df['avg_parse_tree_dpth'] = average_parse_tree_depth(text)

        oxf_df = pd.read_csv(os.path.join(fpath, 'oxford_clean.csv'), index_col='id')

        scr_dict = {}
        for word in text.split():
            scr = 0

            cut_df = oxf_df[oxf_df['word'] == word]
            if cut_df.shape[0] > 0:
                scr = int(cut_df['score'].iloc[0])

            if scr > 0:
                scr = str(scr)
                if scr in scr_dict:
                    scr_dict[scr] += 1
                else:
                    scr_dict[scr] = 1

        for i in range(1, 6):
            if str(i) in scr_dict:
                df[str(i)+'_word_dens'] = scr_dict[str(i)] / df['wrd_cnt']
            else:
                df[str(i)+'_word_dens'] = 0

        df.drop(columns=['text'], inplace=True)
        return df

    def _get_lexical(self, text: str):
        df = pd.DataFrame(data={'text': text}, index=[0])

        df['ttr'] = df['text'].apply(lambda x: lex.LexicalRichness(x).ttr)
        df['rttr'] = df['text'].apply(lambda x: lex.LexicalRichness(x).rttr)
        df['cttr'] = df['text'].apply(lambda x: lex.LexicalRichness(x).cttr)
        df['mtld75'] = df['text'].apply(lambda x: lex.LexicalRichness(x).mtld(threshold=0.75))

        return df

    def _preprocess_text(self, text: str) -> pd.DataFrame:
        assert isinstance(text, str)
        df = pd.DataFrame(data={"text": text}, index=[0])

        df = pd.concat([df, self._get_syntactic(text), self._get_lexical(text)], axis=1)

        dx = self.vectorize_text(text)
        df = pd.concat([df, dx], axis=1)
        df.columns = df.columns.astype(str)

        return df

    def _preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        assert isinstance(X, pd.DataFrame)
        res = pd.DataFrame()

        sz = X.shape[0]

        for i in tqdm(range(sz)):
            res = pd.concat([res, self._preprocess_text(X['text'].iloc[i])])

        res['id'] = pd.RangeIndex(start=0, stop=res.shape[0]).astype(str)
        res.set_index('id', inplace=True)

        return res

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert isinstance(X, pd.DataFrame)
        res = pd.DataFrame()

        for txt in X['text']:
            res = pd.concat([res, self._preprocess_text(txt).drop(columns=['text'])])
        booster_cols = self.model.get_booster().feature_names
        return logistic(self.model.predict(res[booster_cols]), 3, 0.5)

    def predict_text(self, text: str):
        assert isinstance(text, str)
        return self.predict(self._preprocess_text(text))

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
