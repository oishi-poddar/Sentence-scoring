import gensim
from gensim.utils import simple_preprocess
from gensim import corpora, models
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
from nltk.tokenize import sent_tokenize
import pickle
import time
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models.callbacks import PerplexityMetric

import ast


class EpochLogger(CallbackAny2Vec):
    '''Callback to log information about training'''

    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        print("Epoch #{} start".format(self.epoch))

    def on_epoch_end(self, model):
        print("Epoch #{} end".format(self.epoch))
        self.epoch += 1


if __name__ == '__main__':
    # tokenised_text=[]
    # with open("preprocessed_tokenised.txt", encoding="utf8") as file:
    #     l = [line.rstrip() for line in file]
    #     # print(l)
    #     for i in l:
    #         tokenised_text.append(ast.literal_eval(i))
    # print("length",len(tokenised_text))
    # dictionary = gensim.corpora.Dictionary(tokenised_text)
    #
    # count = 0
    # for k, v in dictionary.iteritems():
    #     print(k, v)
    #     count += 1
    #     if count > 10:
    #         break
    # dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
    # bow_corpus = [dictionary.doc2bow(doc) for doc in tokenised_text]
    # pickle.dump(corpus, open('corpus.pkl', 'wb'))
    # dictionary.save('dictionary.gensim')
    # epoch_logger = EpochLogger()

    # perplexity_logger = PerplexityMetric(corpus=bow_corpus)
    # lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=50, id2word=dictionary, passes=2, iterations=10, callbacks=[perplexity_logger])
    # print("nana",lda_model.get_document_topics(dictionary.doc2bow(tokenised_text[0])))
    # print("get_topics - term topic matrix", lda_model.get_topics().shape)
    # print("top_topics- topics and coherence score", len(lda_model.top_topics(bow_corpus)))
    # for idx, topic in lda_model.print_topics(-1):
    #     print('Topic: {} \nWords: {}'.format(idx, topic))

    # theta, _ = lda_model.inference(bow_corpus)
    # theta /= theta.sum(axis=1)[:, None]
    # print(theta.shape)
    from gensim.models import CoherenceModel
    #
    # print('\nPerplexity:', lda_model.log_perplexity(bow_corpus))
    # # Compute Coherence Score
    # coherence_model_lda = CoherenceModel(model=lda_model, texts=tokenised_text[0:10000], corpus=bow_corpus, coherence='c_v')
    # cohere_total = coherence_model_lda.get_coherence()
    # coherence_lda = coherence_model_lda.get_coherence_per_topic()
    # print('\nCoherence Score: ', coherence_lda, cohere_total)


    def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
        """
        Compute c_v coherence for various number of topics

        Parameters:
        ----------
        dictionary : Gensim dictionary
        corpus : Gensim corpus
        texts : List of input texts
        limit : Max num of topics

        Returns:
        -------
        model_list : List of LDA topic models
        coherence_values : Coherence values corresponding to the LDA model with respective number of topics
        """
        coherence_values = []
        model_list = []
        perplexity_logger = PerplexityMetric(corpus=corpus, logger='visdom')
        # for num_topics in range(start, limit, step):
        #     model = gensim.models.LdaMulticore(corpus=corpus, num_topics=num_topics, id2word=dictionary, passes=5, iterations=201, per_word_topics=True)
        #
        #     # model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
        #     model_list.append(model)
        #     coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        #     model.save('lda_models/lda_model'+ str(num_topics))
        #     print("no of topic",num_topics)
        #     print("coherence score", coherencemodel.get_coherence())
        #     coherence_values.append(coherencemodel.get_coherence())

        # return model_list, coherence_values

    # start=time.time()
    # model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=bow_corpus, texts=tokenised_text, start=24,
    #                                                         limit=30, step=1)
    # print("time", time.time()-start)
    # Show graph
    import matplotlib.pyplot as plt

    # limit =30;
    # start = 24;
    # step = 1;
    # x = range(start, limit, step)
    # plt.plot(x, coherence_values)
    # plt.xlabel("Num Topics")
    # plt.ylabel("Coherence score")
    # plt.legend(("coherence_values"), loc='best')
    # plt.show()
    # plt.savefig("lda vs coherence 27-33.png")


    import pyLDAvis.gensim
    import pickle
    import pyLDAvis

    # Visualize the topics
    lda_model = gensim.models.LdaMulticore(corpus=bow_corpus, num_topics=25, id2word=dictionary, passes=5,
                                       iterations=201, per_word_topics=True)

    # lda_model = gensim.models.LdaModel.load('lda_models/lda_model25')
    # print(lda_model)

    # LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, bow_corpus, dictionary)
    # # pyLDAvis.show(LDAvis_prepared)
    # rachel_lda = open('lda_visualisation.html', 'w')
    #
    # pyLDAvis.save_html(LDAvis_prepared,rachel_lda)
    #
