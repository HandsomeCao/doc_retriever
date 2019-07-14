# -*- encoding: utf-8 -*-
'''
@File    :   tfidf_sim.py
@Time    :   2019/07/14 16:38:00
@Author  :   Cao Shuai
@Version :   1.0
@Contact :   caoshuai@stu.scu.edu.cn
@License :   (C)Copyright 2018-2019, MILAB_SCU
@Desc    :   None
'''

import json
import pkuseg
import logging
import sys
import codecs
from pprint import pprint
from gensim import corpora, models
from gensim.similarities import MatrixSimilarity

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def read_corpus(fp):
    with open(fp, 'r', encoding='utf-8') as fr:
        jingy_data = json.load(fr)
    return jingy_data


class SimModel(object):
    """
    arg datas: [{'title':"a", 'content':"b"},{....}]
    arg model: choose to use tfidf model or lsi mode
    arg lsi_dim: if use lsi model, the dim to represent the vec
    """

    def __init__(self, datas, model='tfidf', lsi_dim=200):
        assert_print = "model param has to be 'tfidf' or 'lsi'"
        assert model in ['tfidf', 'lsi'], assert_print
        self.datas = datas
        self.model = model
        self.lsi_dim = lsi_dim
        self.seg = pkuseg.pkuseg()
        self.corpus_bow, self.dictionary = self._init_dictionary(datas)
        self.tfidf = models.TfidfModel(self.corpus_bow)
        self.lsi = None
        self.corpus = self._init_corpus()
        self.index = self._init_index()

    def _init_dictionary(self, datas):
        titles = [data['title'] for data in datas]
        texts = [self.seg.cut(title) for title in titles]
        dictionary = corpora.Dictionary(texts)
        corpus_bow = [dictionary.doc2bow(text) for text in texts]
        return corpus_bow, dictionary

    def _init_corpus(self):
        corpus_tfidf = self.tfidf[self.corpus_bow]
        if self.model == 'lsi':
            self.lsi = models.LsiModel(corpus_tfidf,
                                       id2word=self.dictionary,
                                       num_topics=self.lsi_dim)
            corpus_lsi = self.lsi[corpus_tfidf]
            return corpus_lsi
        return corpus_tfidf

    def _init_index(self):
        if self.model == 'tfidf':
            index = MatrixSimilarity(self.tfidf[self.corpus])
        else:
            index = MatrixSimilarity(self.lsi[self.corpus])
        return index

    def sim(self, query):
        vec_bow = self.dictionary.doc2bow(self.seg.cut(query))
        model_vec = self.lsi[vec_bow] if self.lsi else self.tfidf[vec_bow]
        sims = sorted(enumerate(self.index[model_vec]), key=lambda x: -x[1])
        best_index = sims[0][0]
        return self.datas[best_index]['content']


if __name__ == "__main__":
    jiany_datas = read_corpus('./jingyan.json')
    simModel = SimModel(jiany_datas, model='lsi')
    pprint(simModel.sim("肚子痛怎么办？"))
