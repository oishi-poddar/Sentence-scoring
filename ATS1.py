from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
import math
import networkx as nx
import re
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import scipy
import time
from functools import partial
from time import sleep
import threading
from pycorenlp import StanfordCoreNLP
from heapq import nlargest
from sentence_transformers import SentenceTransformer, util
import seaborn as sns
import matplotlib.mlab as mlab
import pandas as pd
from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool
# from multiprocessing.dummy import Pool as ThreadPool
import multiprocessing
import itertools


import h5py

print("Hello World from %s!" % __name__)
print("hi")
def preprocessing(file_name):
    i=0
    start=time.time()
    document_sentences=[] # total number of docs=25715
    special_processed_sentences = []
    clean_sentences = []
    with open(file_name, encoding="utf8") as file:

        for line in file:
            # i += 1
            # if i == 3:
            #      break
            document = re.sub(r'[\n\r\t]', ' ', line) #removing new line, tabs
            document = re.sub('[^A-Za-z0-9.]+', ' ', document)
            document_sentences.append(document)

        # preprocess by tokenising into sentence an d removing only few speical characters, numerals etc
            tokenised = sent_tokenize(line)  # sentence tokenisation done before as full stop removed in pre processing
            for special_processed_sentence in tokenised:
                special_processed_sentence = re.sub('\w+[\d*]\w+', ' ', special_processed_sentence)  # removes words with digits in them or \S*\d\S*
                special_processed_sentence = re.sub(r'[\n\r\t]', ' ', special_processed_sentence)  # removes new lines and tabs
                special_processed_sentence = re.sub('[^A-Za-z0-9()"]+', ' ', special_processed_sentence) #remove special characters except parathesis and quotes
                special_processed_sentences.append(special_processed_sentence)

            # preprocess by totally cleaning sentences
            for clean_sentence in tokenised:
                clean_sentence = re.sub('\w*\d\w*', ' ', clean_sentence)# removes words with digits in them and all digits or \S*\d\S*
                clean_sentence=re.sub(r'[\n\r\t]',' ',clean_sentence) # removes new lines and tabs
                clean_sentence = re.sub('[^A-Za-z]+', ' ', clean_sentence) # remove punctuation,keeps only alphabet and certain important special characters
                clean_sentences.append(clean_sentence)

        document_sentences = list(filter(None, document_sentences))
        special_processed_sentences = list(filter(None, special_processed_sentences))
        clean_sentences = list(filter(None, clean_sentences))  # removing empty lists
        special_tokenised_text = [word_tokenize(sentence) for sentence in special_processed_sentences]  # word tokenise
        stop_words = set(stopwords.words('english'))
        porter = PorterStemmer()
        tokenised_text = [word_tokenize(sentence) for sentence in clean_sentences]  # word tokenise
        for i in range(len(tokenised_text)):  # remove stop words and stem
            tokenised_text[i] = [porter.stem(word) for word in tokenised_text[i]]
            tokenised_text[i] = [word for word in tokenised_text[i] if word not in stop_words]
        tokenised_text = list(filter(None, tokenised_text))
        print("No of documents", len(document_sentences)) #20972
        print("clean sentences",len(tokenised_text)) #209718
        print("special processed text",len(special_tokenised_text))
    f = open('preprocessed_tokenised.txt', 'w')
    for token in tokenised_text:
        #clean_sentences = map(lambda x: x + '\n', each_sentence)
        f.writelines(str(token)+'\n')
    f.close()
    f = open('preprocessed_whole.txt', 'w')
    for clean_sentence in clean_sentences:
        # clean_sentences = map(lambda x: x + '\n', each_sentence)
        f.writelines(str(clean_sentence) + '\n')
    f.close()
    end = time.time()
    # total time taken

    asciiList = [n.encode("ascii", "ignore") for n in clean_sentences]
    with h5py.File('preprocessed_whole.hdf5', 'w') as f:
        dataset=f.create_dataset('data',(len(clean_sentences),1), data=(asciiList))
    asciiList=[]
    for string in tokenised_text:
        asciiList.append([n.encode("ascii", "ignore") for n in string])
    #
    # with h5py.File('preprocessed_tokenised.hdf5', 'w') as f:
    #     dataset=f.create_dataset('data',(len(tokenised_text),1), data=(asciiList))

    return document_sentences, special_processed_sentences, special_tokenised_text, tokenised_text, clean_sentences

# to find number of sentences in which each word occurs
def check_word_freq_per_sent(sentences):
    start=time.time()
    no_sentences_word_occurs = {}
    unique_words=set(sentences[0])
    for sent in sentences:
        unique_words.update(set(sent))
    for sent in sentences:
        for word in unique_words:
            if(word in sent):
                if(word in no_sentences_word_occurs):
                    no_sentences_word_occurs[word] +=1
                else:
                    no_sentences_word_occurs[word] =1
    print("Returning number of sentences word occurs")
    end=time.time()
    with h5py.File('no_sentences.hdf5', 'w') as f:
        dataset=f.create_dataset('data',(len(no_sentences_word_occurs),), data=no_sentences_word_occurs)

    # print(f"Runtime of the no_sentences is {end - start}")
    return no_sentences_word_occurs

# calculate frequency of each word
def term_freq_calculation(tokenised_text):
    start=time.time()
    term_freq={}
    count_of_words=0
    for sent in tokenised_text:
        for word in sent:
            count_of_words+=1
            if word in term_freq:
                term_freq[word] += 1
            else:
                term_freq[word] = 1
    print("Returning term freq")
    end=time.time()
    # print(f"Runtime of the term freq is {end - start}")
    return term_freq

# to find the alpha term for each word to find sentence cosine similarity
def alpha(tokenised_text):
    no_sentences_word_occurs=check_word_freq_per_sent(tokenised_text)
    with h5py.File('no_sentences.hdf5', 'w') as f:
        dset=f['data'][()]
    print(len(dset))
    no_of_sentences=len(tokenised_text)
    term_freq= term_freq_calculation(tokenised_text)
    start=time.time()
    final_alpha=[]
    for i in range(len((tokenised_text))):
        alpha=[]
        for word in tokenised_text[i]:
            denom=no_sentences_word_occurs[word]
            num=no_of_sentences
            tf=term_freq[word]
            log_term=np.log(num/denom)
            alpha.append(tf*log_term)

        final_alpha.append(alpha)
    # print((sum(map(lambda i: i * i, final_alpha[709]))))
    # print ("alpha",final_alpha)
    print("Returning final_alpha")
    end=time.time()
    # print(f"Runtime of the alpha is {end - start}")
    return final_alpha

def cosine_similarity(tokenised_text):
    final_alpha = alpha(tokenised_text)
    start=time.time()
    no_of_sentences=len(tokenised_text)

    batch_size=100
    no_of_batches=math.floor(no_of_sentences/batch_size)
    modulo=no_of_sentences%batch_size
    remaining=0
    if modulo!=0:
        remaining=no_of_sentences%batch_size
    print("no of batches",no_of_batches)
    sum_rows=[]
    sentence_aggregate_similarity=[]
    start1=time.time()
    k=0
    m=0
    with h5py.File('random.hdf5', 'w') as f:
        dset = f.create_dataset("default", (no_of_sentences, no_of_sentences))
        for b in range(no_of_batches):
            start2 = time.time()
            similarity_matrix = np.zeros((batch_size, no_of_sentences))
            for i in range(batch_size):
                score = 0
                for j in range(no_of_sentences):
                    if m != j:
                        num = sum([a * b for a, b in zip(final_alpha[m], final_alpha[j])])
                        denom = np.sqrt(sum(map(lambda i: i * i, final_alpha[m]))) * np.sqrt(sum(map(lambda i: i * i, final_alpha[j])))
                        similarity_matrix[i][j] = np.round(num / denom,2)
                        score+=similarity_matrix[i][j]
                m+=1
                sum_rows.append(np.round(score,2))
                sentence_aggregate_similarity.append(score/ no_of_sentences)
            dset[k:k+batch_size,:] = similarity_matrix
            k+=batch_size
            print("outer for loop abd batch", time.time()-start2, b)

        similarity_matrix = np.zeros((remaining, no_of_sentences))
        for i in range(remaining):
            score = 0
            for j in range(no_of_sentences):
                if m != j:
                    num = sum([a * b for a, b in zip(final_alpha[m], final_alpha[j])])
                    denom = np.sqrt(sum(map(lambda i: i * i, final_alpha[m]))) * np.sqrt(
                        sum(map(lambda i: i * i, final_alpha[j])))
                    similarity_matrix[i][j] = np.round(num / denom, 2)
                    score += similarity_matrix[i][j]
            m += 1
            sum_rows.append(np.round(score, 2))
            sentence_aggregate_similarity.append(score / no_of_sentences)
        dset[k:k + remaining, :] = similarity_matrix
    print("matrix creation", time.time() - start1)

    # mu, sigma = scipy.stats.norm.fit(sentence_aggregate_similarity)
    # threshold1 = np.round(mu - (2 * sigma), 2)
    # threshold2 = np.round(mu + (2 * sigma), 2)
    # print("thresholds are ", threshold1, threshold2)
    # for i in range(len(tokenised_text)):
    #     no_of_edges=0
    #     for j in range(len(tokenised_text)):
    #         if i != j:
    #             num = sum([a * b for a, b in zip(final_alpha[i], final_alpha[j])])
    #             denom = np.sqrt(sum(map(lambda i: i * i, final_alpha[i]))) * np.sqrt(sum(map(lambda i: i * i, final_alpha[j])))
    #             similarity_matrix_score = num / denom
    #             if (similarity_matrix_score>threshold1 and similarity_matrix_score<threshold2):
    #                 no_of_edges+=1

    print("Returning similarity scores")
    end=time.time()
    # print(f"Runtime of the calculation of aggregate similarity is {end - start}")
    return sum_rows, sentence_aggregate_similarity

def sentence_cosine_similarity(tokenised_text):
    sum_rows,sentence_aggregate_similarity = cosine_similarity(tokenised_text)
    start = time.time()

    plt.clf()
    kwargs = dict(kde_kws={'linewidth': 2},kde=True,
             bins=50, color = 'darkblue',
             hist_kws={'edgecolor':'black'})
    fig=sns.distplot(sentence_aggregate_similarity, hist=True, **kwargs)
    plt.xlabel('Similarity score')
    plt.ylabel('frequency')

    # add a 'best fit' line
    # y = mlab.normpdf(20, mu, sigma)
    # l = plt.plot(20, y, 'r--', linewidth=2)

    plt.savefig("histogramUSINGbatchAll2.png")
    plt.clf()
    mu, sigma = scipy.stats.norm.fit(sentence_aggregate_similarity)
    threshold1=np.round(mu-(2*sigma),2)
    threshold2=np.round(mu+(2*sigma),2)
    print("thresholds are ", threshold1, threshold2)
    start3=time.time()
    with h5py.File('random.hdf5', 'a') as f:
        similarity_matrix = f['default']
        print(similarity_matrix[0:8, 0:8])
        # with h5py.File('random.hdf5', 'w') as f:
        similarity_matrix[similarity_matrix<threshold1]=0
        similarity_matrix[similarity_matrix>threshold2]=0
        print(similarity_matrix[0:8, 0:8])
        print("Generating graph")
        nx_graph = nx.from_numpy_array(similarity_matrix)  # generating graph fom adjacency matrix
        raw_labels= map(str, (np.arange(len(tokenised_text)))) #loads converts a string of int to list of strings
        lab_node = dict(zip(nx_graph.nodes, raw_labels))
        layout = nx.spring_layout(nx_graph)
        nx.draw(nx_graph, layout)
        labels = nx.get_edge_attributes(nx_graph, "weight")
        nx.draw_networkx_edge_labels(nx_graph, pos=layout, edge_labels=labels)
        nx.draw_networkx_labels(nx_graph, layout, labels=lab_node, font_size=10, font_family='sans-serif')
        plt.savefig("graphWithBatches.png")
    print("read", time.time()-start3)

    end=time.time()
    print("runtime of aggregate similarity", end-start)
    # # print(nx.adjacency_spectrum(nx_graph))
    # print("finding centrality")
    centrality = nx.eigenvector_centrality(nx_graph)
    # return sentence_aggregate_similarity, list(centrality.values())

#NER
def named_entity_recognition(special_preprocessed_text):
    ne_per_sentence=[]
    noun_per_sentence=[]
    verb_per_sentence=[]
    proper_noun_per_sentence=[]
    print("inside ner")
    start=time.time()
    nlp = StanfordCoreNLP('http://localhost:9000')
    for i in range(len(special_preprocessed_text)):
        result = nlp.annotate(special_preprocessed_text[i],
                          properties={
                              'annotators': 'ner, pos',
                              'outputFormat': 'json',
                              'timeout': 1000,
                          })

       # checking number of named entities per sentece
        ne=0
        noun=0
        verb=0
        proper_noun=0

        for word in result["sentences"][0]['tokens']:
            if not word['ner'] == 'O':
               ne+=1
        ne_per_sentence.append(ne)
        # checking number of noun  phrases per sentence
        for word in result["sentences"][0]['tokens']:
            if 'NN' in word['pos']:
                noun+=1
        noun_per_sentence.append(noun)
        # checking number of verb phrases per sentence
        for word in result["sentences"][0]['tokens']:
            if 'VB' in word['pos']:
                verb+=1
        verb_per_sentence.append(verb)

        # checking number of proper nouns per sentence
        for word in result["sentences"][0]['tokens']:
            if 'NNP' in word['pos']:
                proper_noun+=1
        proper_noun_per_sentence.append(proper_noun)
        if i in ne_per_sentence:
            ne_per_sentence[i] /= len(special_preprocessed_text[i])
        if i in noun_per_sentence:
            noun_per_sentence[i] /= len(special_preprocessed_text[i])
        if i in verb_per_sentence:
            verb_per_sentence[i] /= len(special_preprocessed_text[i])
        if i in proper_noun_per_sentence:
            proper_noun_per_sentence[i] /= len(special_preprocessed_text[i])

    plt.hist(ne_per_sentence,bins=10)
    plt.savefig("NER.png")
    end=time.time()
    # print(f"Runtime of the NER is {end - start}")
    return ne_per_sentence, noun_per_sentence, verb_per_sentence, proper_noun_per_sentence

# cue phrases
def sentence_cue_phrases(tokenised_text): # TO DO phrase freq
    cue_phrase=["embodi"]
    cue_phrases_per_sentence={}
    for i in range(len(tokenised_text)):
        for word in tokenised_text[i]:
            if word in cue_phrase:
                if i in cue_phrases_per_sentence:
                    cue_phrases_per_sentence[i] +=1
                else:
                    cue_phrases_per_sentence[i] =1
        if i in cue_phrases_per_sentence:
            cue_phrases_per_sentence[i] /= len(tokenised_text[i])
    return cue_phrases_per_sentence

# scorer bases on sentence length
def sentence_length_scorer(tokenised_text):
    start=time.time()
    sentence_length=[]
    max_length_of_sentence = max(map(len, tokenised_text)) #finding max number of words in any sentence
    for i in range(len(tokenised_text)):
        sentence_length.append(len(tokenised_text[i])/max_length_of_sentence)
    end=time.time()
    # print(f"Runtime of the no_sentences is {end - start}")
    return sentence_length

#sentence significance scorer
def sentence_significance_scorer(tokenised_text):
    start=time.time()
    sentence_significance=[]
    final_alpha = alpha(tokenised_text)
    for i in range(len(tokenised_text)):
        sum_alpha=sum(final_alpha[i])
        sentence_significance.append(sum_alpha/len(tokenised_text[i]))
    end=time.time()
    # print(f"Runtime of the significance scorer is {end - start}")
    return sentence_significance

# frequency scorer
def freq_scorer(tokenised_text):
    start=time.time()
    sentence_freq_score=[]
    term_freq = term_freq_calculation(tokenised_text)
    top_30_perc=round(0.3*len(term_freq))
    top_30_perc_words = nlargest(top_30_perc, term_freq, key=term_freq.get) #list of top 30 percent high freq words
    for i in range(len(tokenised_text)):
        count = 0
        for word in tokenised_text[i]:
            if word in top_30_perc_words:
                count+=1
        sentence_freq_score.append(count)
    end=time.time()
    # print(f"Runtime of the freq_scorer is {end - start}")
    return sentence_freq_score

def numeric_sentence_score(special_tokenised_text):
    start=time.time()
    sentence_number_numeral=[]
    for i in range(len(special_tokenised_text)):
        number_count=0
        for word in special_tokenised_text[i]:
            if word.isnumeric():
                number_count+=1
        sentence_number_numeral.append(number_count)
    end=time.time()
    # print(f"Runtime of the numeral is {end - start}")
    return sentence_number_numeral

def punctuation_sentence_scorer(special_preprocessed_text):
    start=time.time()

    sentence_special_character=[]

    for i in range(len(special_preprocessed_text)):
        sentence=special_preprocessed_text[i]
        number_of_paranthesis=sentence.count("(")+sentence.count(")")
        number_of_quotes=sentence.count("\"")
        sentence_special_character.append(number_of_paranthesis+number_of_quotes)
    end=time.time()
    # print(f"Runtime of the punctaution scorer is {end - start}")
    return sentence_special_character

def upper_case_sentence_scorer(special_tokenised_text):
    start=time.time()
    sentence_upper_case=[]

    for i in range(len(special_tokenised_text)):
        upper_case=0
        for word in special_tokenised_text[i]:
            if word.isupper():
                upper_case+=1
        sentence_upper_case.append(upper_case)
    end=time.time()
    # print(f"Runtime of the upper case is {end - start}")
    return sentence_upper_case

def sentence_position_scorer(document_sentences):
    start=time.time()
    sentence_position=[]
    for doc in document_sentences:
        k=0
        sentences=sent_tokenize(doc)
        document_length=len(sentences)
        for i in range(len(sentences)):
            score=abs(document_length-i)/document_length
            sentence_position.append(np.round(score,2))
            k+=1
    end=time.time()
    # print(f"Runtime of the sentence_position is {end - start}")
    return sentence_position


def sentence_transformer_scorer(clean_sentences,embeddings2):
    # print("fucntion1")
    start=time.time()
    no_of_sentences = len(clean_sentences)
    with h5py.File('random.hdf5', 'w') as f:
        dset = f.create_dataset("default", (no_of_sentences, len(embeddings2)))
        k=0
        batch_size=1
        no_of_batches = math.floor(no_of_sentences / batch_size)
        modulo = no_of_sentences % batch_size
        remaining = 0
        if modulo != 0:
            remaining = no_of_sentences % batch_size
        # print("no of batches", no_of_batches)
        first = 0
        second =0
        # print(len(clean_sentences))
        # star=time.time()
        # embeddings2 = model.encode(clean_sentences)
        # print("timee fr embedding",time.time()-star)
        # sta=time.time()
        for i in range(no_of_batches):
            start1=time.time()
            embeddings1 = embeddings2[i]
            cos_sim = (util.pytorch_cos_sim(embeddings1, embeddings2))
            dset[first:first+batch_size, :] = cos_sim
            first += batch_size
            # print("for loop ", time.time()-start1,i)


    # print("fucntion1 over")
    # print("time for function 1", time.time()-start)


def sentence_transformer_scorer2(embeddings2,batch_size,a):
    # print("fucntion2")
    start=time.time()
    with h5py.File('random1.hdf5', 'a') as f:
        # dataset_name = f.create_dataset("default"+str(index), shape=(no_of_sentences, len(embeddings2)), dtype='f')
        dataset=f['default']
        # no_of_batches = math.floor(no_of_sentences / batch_size)
        # modulo = no_of_sentences % batch_size
        # remaining = 0
        # if modulo != 0:
        #     remaining = no_of_sentences % batch_size
        # print("no of batches", no_of_batches)
        start1=time.time()
        embeddings1 = embeddings2[a:a+batch_size]
        # print(embeddings)
        # print(embeddings.shape)
        cos_sim = (util.pytorch_cos_sim(embeddings1, embeddings2))
        dataset[a:a+batch_size, :] = cos_sim
        # print("for loop ", time.time()-start1,i)
    return cos_sim
# Encode all sentence
    end = time.time()
    # print(f"Runtime of the sentence_transformer2 is {end - start}")
    # Compute cosine similarity between all pairs
    # print("computing cosine similarity between all pairs")
    # cos_sim = util.pytorch_cos_sim(embeddings, embeddings)
    # with h5py.File('name-of-file.h5', 'w') as hf:
    #     hf.create_dataset("h5py dataset", data=cos_sim)
    # with h5py.File('name-of-file.h5', 'r') as hf:
    #     data = hf['h5py dataset'][:]
    # print(data)
    # print("matrix made")
    # sum_rows=np.round(cos_sim.sum(axis=1)/len(clean_sentences),2)
    # plt.figure(1)
    # kwargs = dict(kde_kws={'linewidth': 2}, kde=True,
    #               bins=20, color='darkblue',
    #               hist_kws={'edgecolor': 'black'})
    # fig = sns.distplot(sum_rows, hist=True, **kwargs)
    # plt.xlabel('Similarity score')
    # plt.ylabel('frequency')
    #
    # # add a 'best fit' line
    # # y = mlab.normpdf(20, mu, sigma)
    # # l = plt.plot(20, y, 'r--', linewidth=2)
    #
    # plt.savefig("histogramUSINGbert1.png")
    # plt.clf()
    # mu, sigma = scipy.stats.norm.fit(sum_rows)
    # threshold1 = np.round(mu - (2 * sigma), 2)
    # threshold2 = np.round(mu + (2 * sigma), 2)
    # print("thresholds are ", threshold1, threshold2)
    # cos_sim[cos_sim <= threshold1] = 0.0
    # cos_sim[cos_sim >= threshold2] = 0.0
    # print(cos_sim)
    #
    # print("Generating graph")
    # plt.figure(2)
    # nx_graph = nx.from_numpy_array(cos_sim.numpy())  # generating graph fom adjacency matrix
    # raw_labels = map(str, (np.arange(len(tokenised_text))))  # loads converts a string of int to list of strings
    # lab_node = dict(zip(nx_graph.nodes, raw_labels))
    # layout = nx.spring_layout(nx_graph)
    # nx.draw(nx_graph, layout)
    # labels = nx.get_edge_attributes(nx_graph, "weight")
    # nx.draw_networkx_edge_labels(nx_graph, pos=layout, edge_labels=labels)
    # nx.draw_networkx_labels(nx_graph, layout, labels=lab_node, font_size=10, font_family='sans-serif')
    # plt.savefig("graph.png")
    # end = time.time()
    # print("runtime of aggregate similarity", end - start)
    # print("finding centrality using Bert method")
    # centrality = nx.eigenvector_centrality(nx_graph)
    # print(centrality)

    # return sum_rows, list(centrality.values())

# document_sentences, special_preprocessed_text, special_tokenised_text, tokenised_text, clean_sentences = preprocessing("Final_text.txt")
# with open("preprocessed_tokenised.txt", encoding="utf8") as file:
#     tokenised_text = [line.rstrip() for line in file]
# print(tokenised_text)

def embeddings_calculator(model,clean_sentences,a,b):
    print(multiprocessing.current_process())
    sentence_embeddings = model.encode(clean_sentences[a:b])
    with h5py.File('embeddings.hdf5', 'a') as f:
        f['embed'][a:b]=sentence_embeddings

def test():
    print("test")

def main():
    test()
    print("meow")
if __name__ == '__main__':
    print("main")
    main()
    with open("preprocessed_whole.txt", encoding="utf8") as file:
        clean_sentences = [line.rstrip() for line in file]
    print(
        "done............................................................................................................................")
    # model = SentenceTransformer('roberta-large-nli-stsb-mean-tokens')
    # s = time.time()
    # # e = model.encode(clean_sentences[0:10000])
    # # print(e)
    # print("serial embedding", time.time() - s)
    # embeddings_batch_size = 5000
    # # embeddings=np.empty((len(clean_sentences[0:8]),1024))
    # first = 0
    # last = embeddings_batch_size
    no_of_sentences = len(clean_sentences[0:10000])
    print("No of sentences", no_of_sentences)
    # no_of_batches = math.floor(no_of_sentences / embeddings_batch_size)
    # modulo = no_of_sentences % embeddings_batch_size
    # remaining = 0
    # if modulo != 0:
    #     remaining = no_of_sentences % embeddings_batch_size
    # print("no of batches", no_of_batches)
    # firstList=[]
    # lastList=[]
    # for i in range(no_of_batches):
    #     firstList.append(first)
    #     first+=embeddings_batch_size
    #     lastList.append(last)
    #     last+=embeddings_batch_size
    #
    # #
    # with h5py.File('embeddings.hdf5', 'w') as f:
    #     dset = f.create_dataset("embed", (no_of_sentences, 1024))
    # #         dset = embeddings
    # s = time.time()
    # print("starting pool 1")
    # with multiprocessing.Pool(2) as pool:
    #     print(multiprocessing.current_process())
    #     print(multiprocessing.active_children())
    #     pool.starmap(partial(embeddings_calculator,model,clean_sentences),list(zip(firstList,lastList)))
    #
    # print(" thread time",time.time()-s)
    # print(results[0:4])
    with h5py.File('embeddings.hdf5', 'r') as f:
        embeddings = f['embed'][()]
    sta=time.time()
    with h5py.File('random1.hdf5', 'w') as f:
        dataset_name = f.create_dataset("default", shape=(no_of_sentences, no_of_sentences), dtype='f')
    a=[]
    a1=0
    batch_size=5000
    no_of_batches = math.floor(no_of_sentences / batch_size)
    modulo = no_of_sentences % batch_size
    remaining = 0
    if modulo != 0:
        remaining = no_of_sentences % batch_size
    print("no of batches", no_of_batches)
    for i in range(no_of_batches):
        a.append(a1)
        a1+=batch_size
    with multiprocessing.Pool(2) as pool:
        pool.starmap(partial(sentence_transformer_scorer2, embeddings,batch_size), list(zip(a)))
    pool.join()
    print("starting pool2")

    with h5py.File('random1.hdf5', 'a') as f:
        cos_sim = f['default'][()]
        print(cos_sim)
        sentence_aggregate_similarity = cos_sim.sum(axis=1) / len(clean_sentences)
        print(type(sentence_aggregate_similarity))
        print(sentence_aggregate_similarity)
        plt.clf()
        kwargs = dict(kde_kws={'linewidth': 2}, kde=True,
                      bins=50, color='darkblue',
                      hist_kws={'edgecolor': 'black'})
        fig = sns.distplot(sentence_aggregate_similarity, hist=True, **kwargs)
        plt.xlabel('Similarity score')
        plt.ylabel('frequency')
        plt.savefig("histogram10k.png")
        plt.clf()
        mu, sigma = scipy.stats.norm.fit(sentence_aggregate_similarity)
        threshold1 = mu - (2 * sigma)
        threshold2 = mu + (2 * sigma)
        print("thresholds are ", threshold1, threshold2)

        print("yaaay tie",time.time()-sta)

        plt.clf()
        cos_sim[cos_sim < threshold1] = 0
        cos_sim[cos_sim > threshold2] = 0
        s = time.time()
        print("Generating graph")
        nx_graph = nx.from_numpy_array(cos_sim)  # generating graph fom adjacency matrix
        # raw_labels = map(str, (np.arange(len(clean_sentences))))  # loads converts a string of int to list of strings
        # lab_node = dict(zip(nx_graph.nodes, raw_labels))
        # layout = nx.spring_layout(nx_graph)
        # nx.draw(nx_graph, layout)
        # labels = nx.get_edge_attributes(nx_graph, "weight")
        # nx.draw_networkx_edge_labels(nx_graph, pos=layout, edge_labels=labels)
        # nx.draw_networkx_labels(nx_graph, layout, labels=lab_node, font_size=10, font_family='sans-serif')
        # plt.savefig("graph10k.png")
        print("time for graph", time.time()-s)
    # # print(nx.adjacency_spectrum(nx_graph))
    # print("finding centrality")
        s=time.time()
        centrality = nx.eigenvector_centrality(nx_graph, max_iter=1000)
        print("centrality",centrality)
        print("centrality time",time.time()-s)
    # s=time.time()
    # sentence_transformer_scorer(clean_sentences[0:8],embeddings)
    # print("serial",time.time()-s)
    # with h5py.File('random.hdf5', 'r') as f:
    #     dset=f['default'][()]
    #     print(dset)
    #     sentence_aggregate_similarity = dset.sum(axis=1) / len(clean_sentences)
    #     print(sentence_aggregate_similarity)
    # jobs = [None] * 2# list of jobs
    # jobs_num = 2 # number of workers
    # for i in range(jobs_num):
    #     # Declare a new process and pass arguments to it
    #     print(multiprocessing.current_process())
    #     jobs[i] = multiprocessing.Process(target=embeddings_calculator, args=(clean_sentences,first,last,embeddings,model))
    #     jobs[i].start()
    # Make the Pool of workers
    # workThread1 = threading.Thread(target=sentence_transformer_scorer2, args=(clean_sentences[0:4],embeddings,0))
    # workThread2 = threading.Thread(target=sentence_transformer_scorer2, args=(clean_sentences[4:8],embeddings,4))
    # workThread1.start()
    # workThread2.start()
    # workThread1.join()
    # workThread2.join()
# st=time.time()
# first =0
# last=first+embeddings_batch_size
# # embeddings_calculator(clean_sentences,0,3,np.empty((3,)),model)
# threads = [None] * 2
# embeddings=np.empty((len(clean_sentences[0:1000]),1024))
# for i in range(len(threads)):
#     print("thread active count",threading.active_count())
#     threads[i] = threading.Thread(target=embeddings_calculator, args=(clean_sentences,first,last,embeddings,model))
#     threads[i].start()
#     first+=embeddings_batch_size
#     last+=embeddings_batch_size
#
#     # workThread2 = threading.Thread(target=embeddings, args=(clean_sentences, first2, last2,embeddings))
# for i in range(len(threads)):
#     threads[i].join()
# # print(embeddings.shape)
# # print("thread one",embeddings)
# print("threading",time.time()-st)
# #
# star=time.time()
# model = SentenceTransformer('roberta-large-nli-stsb-mean-tokens')
# embeddings2 = model.encode(clean_sentences[0:8])
# print(type(embeddings2))
# print("timee fr embedding",time.time()-star)


# workThread1.run()

# sentence_cosine_similarity(tokenised_text)
# sentence_position = sentence_position_scorer(document_sentences)
# sentence_special_character = punctuation_sentence_scorer(special_preprocessed_text)
# sentence_upper_case = upper_case_sentence_scorer(special_tokenised_text)
# sentence_number_numeral = numeric_sentence_score(special_tokenised_text)

# ne_per_sentence, noun_per_sentence, verb_per_sentence, proper_noun_per_sentence = named_entity_recognition(special_preprocessed_text)
# cue_phrases_per_sentence = sentence_cue_phrases(tokenised_text)
# sentence_length = sentence_length_scorer(tokenised_text)
# sentence_significance = sentence_significance_scorer(tokenised_text)
# sentence_freq_score = freq_scorer(tokenised_text)
# # dictionary of lists
# dict = {'position': sentence_position,
#         'special_char': sentence_special_character,
#         'upper_case': sentence_upper_case,
#         'numeral': sentence_number_numeral,
#         'aggregate': aggregate_similarty,
#         'centrality': centrality,
#         'length': sentence_length,
#         'significance': sentence_significance,
#         'freq_score': sentence_freq_score
#         }
#
# df1 = pd.DataFrame(dict)
# df1.to_csv('dataframe.csv',index=False)
# df = pd.read_csv("dataframe.csv")
# print(df)
# x = StandardScaler().fit_transform(df) # normalizing the features
# print(x.shape)
# print(np.mean(x),np.std(x))
# feat_cols = ['feature'+str(i) for i in range(x.shape[1])]
# normalised_breast = pd.DataFrame(x,columns=feat_cols)
# normalised_breast.head()
# pca_breast = PCA(n_components=3)
# principalComponents_breast = pca_breast.fit_transform(x)
# principal_breast_Df = pd.DataFrame(data = principalComponents_breast
#              , columns = ['principal component 1', 'principal component 2','3'])
# print(principal_breast_Df.head())
# print('Explained variation per principal component: {}'.format(pca_breast.explained_variance_ratio_))