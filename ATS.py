from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
import math
import ast
import networkx as nx
from memory_profiler import profile
import re
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import scipy
import csv
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

def preprocessing(file_name):
    i=0
    start=time.time()
    document_sentences=[] # total number of docs=25715
    special_processed_sentences = []
    clean_sentences = []
    with open(file_name, encoding="utf8") as file:

        for line in file:
            i += 1
            if( i==20):
               break
            #     break
            document = re.sub(r'[\n\r\t]', ' ', line) #removing new line, tabs
            document = re.sub('[^A-Za-z0-9.]+', ' ', document)
            document_sentences.append(document)

        # preprocess by tokenising into sentence and removing only few speical characters, numerals etc
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
        special_processed_sentences = list(filter(str.strip, special_processed_sentences))
        clean_sentences = list(filter(str.strip, clean_sentences))  # removing empty spaced strings
        document_sentences = list(filter(str.strip, document_sentences))
        special_tokenised_text = [word_tokenize(sentence) for sentence in special_processed_sentences]  # word tokenise
        special_tokenised_text = list(filter(None, special_tokenised_text))
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
        print("toknised", len(tokenised_text))
        print("special tokenised", len(special_tokenised_text))
    # with open ('preprocessed_words.txt', 'w') as tokenised_words_file:
    #     for clean_sentence in tokenised_text:
    #             #clean_sentences = map(lambda x: x + '\n', each_sentence)
    #         tokenised_words_file.writelines('%s\n' % clean_sentence)
    # with open ('special_preprocessed_sentences.txt', 'w') as special_preprocessed:
    #     for clean_sentence in special_processed_sentences:
    #     #clean_sentences = map(lambda x: x + '\n', each_sentence)
    #         special_preprocessed.writelines(str(clean_sentence)+'\n')
    # with open('special_preprocessed_text.txt', 'w') as special_preprocessed_text:
    #     for clean_sentence in special_tokenised_text:
    #         # clean_sentences = map(lambda x: x + '\n', each_sentence)
    #         special_preprocessed_text.write('%s\n' % clean_sentence)
    # with open('document_sentences_text.txt', 'w') as document_sentences_file:
    #     for clean_sentence in document_sentences:
    #         # clean_sentences = map(lambda x: x + '\n', each_sentence)
    #         document_sentences_file.write(str(clean_sentence)+'\n')


    end = time.time()
    # total time taken
    print(f"Runtime of the preprcessing is {end - start}")
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
    print(f"Runtime of the no_sentences is {end - start}")
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
    print(f"Runtime of the term freq is {end - start}")
    return term_freq

# to find the alpha term for each word to find sentence cosine similarity
def alpha(tokenised_text):
    no_sentences_word_occurs=check_word_freq_per_sent(tokenised_text)
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
    print(f"Runtime of the alpha is {end - start}")
    return final_alpha

def cosine_similarity(tokenised_text):

    final_alpha = alpha(tokenised_text)
    start=time.time()
    similarity_matrix = np.zeros((len(tokenised_text), len(tokenised_text)))
    sum_row=[]
    for i in range(len(tokenised_text)):
        for j in range(len(tokenised_text)):
            if i != j:
                num = sum([a * b for a, b in zip(final_alpha[i], final_alpha[j])])
                denom=np.sqrt(sum(map(lambda i : i * i, final_alpha[i]))) * np.sqrt(sum(map(lambda i : i * i, final_alpha[j])))
                similarity_matrix[i][j] = np.round(num/denom,2)
                print(similarity_matrix[i][j])
    print("Returning similarity matrix")
    print(similarity_matrix)
    end=time.time()
    print(f"Runtime of the creation of matrix is {end - start}")
    return np.round(similarity_matrix,2)

def sentence_cosine_similarity(tokenised_text):

    sentence_aggregate_similarity=[]
    similarity_matrix = cosine_similarity(tokenised_text)
    start = time.time()
    sum_rows = np.sum(similarity_matrix, axis=1)
    print(sum_rows)
    for i in range(len(tokenised_text)):
        sentence_aggregate_similarity.append(sum_rows[i]/len(tokenised_text))

    print("Aggregate matrix made")
    plt.clf()
    kwargs = dict(kde_kws={'linewidth': 2},kde=True,
             bins=20, color = 'darkblue',
             hist_kws={'edgecolor':'black'})
    fig=sns.distplot(sentence_aggregate_similarity, hist=True, **kwargs)
    plt.xlabel('Similarity score')
    plt.ylabel('frequency')

    # add a 'best fit' line
    # y = mlab.normpdf(20, mu, sigma)
    # l = plt.plot(20, y, 'r--', linewidth=2)

    plt.savefig("histogram1.png")
    plt.clf()
    mu, sigma = scipy.stats.norm.fit(sentence_aggregate_similarity)
    threshold1=np.round(mu-(2*sigma),2)
    threshold2=np.round(mu+(2*sigma),2)
    print("thresholds are ", threshold1, threshold2)
    similarity_matrix[similarity_matrix<threshold1]=0.0
    similarity_matrix[similarity_matrix>threshold2]=0.0
    print(similarity_matrix)
    print("Generating graph")
    nx_graph = nx.from_numpy_array(similarity_matrix)  # generating graph fom adjacency matrix
    raw_labels= map(str, (np.arange(len(tokenised_text)))) #loads converts a string of int to list of strings
    lab_node = dict(zip(nx_graph.nodes, raw_labels))
    layout = nx.spring_layout(nx_graph)
    nx.draw(nx_graph, layout)
    labels = nx.get_edge_attributes(nx_graph, "weight")
    nx.draw_networkx_edge_labels(nx_graph, pos=layout, edge_labels=labels)
    nx.draw_networkx_labels(nx_graph, layout, labels=lab_node, font_size=10, font_family='sans-serif')
    plt.savefig("graphNoBERT.png")
    end=time.time()
    print("runtime of aggregate similarity", end-start)
    # similarity_matrix[similarity_matrix > 0] = 1.0
    # print(nx.adjacency_spectrum(nx_graph))
    print("finding centrality")
    centrality = nx.eigenvector_centrality(nx_graph)
    return sentence_aggregate_similarity, list(centrality.values())

#NER
def named_entity_recognition(special_preprocessed_text):
    ne_per_sentence=[]
    noun_per_sentence=[]
    verb_per_sentence=[]
    proper_noun_per_sentence=[]
    start=time.time()
    nlp = StanfordCoreNLP('http://localhost:9000')
    for i in range(len(special_preprocessed_text)):
        result = nlp.annotate(special_preprocessed_text[i],
                          properties={
                              'annotators': 'ner, pos',
                              'outputFormat': 'json',
                              'timeout': 900000,
                          })

       # checking number of named entities per sentence
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
    print("Time taken for NER is", end-start)
    return ne_per_sentence, noun_per_sentence, verb_per_sentence, proper_noun_per_sentence
# cue phrases
def sentence_cue_phrases(clean_sentences, special_tokenised_text): # TO DO phrase freq
    start=time.time()
    cue_phrases_per_sentence = []
    le=[]
    with open('AutoPhrase_multi-words.txt', encoding="utf8") as f:
        list=f.readlines()
            # print(line.split('\t')[1])
         # for x in lines:
    print(list)
    for l in list[1]:

        # print(l.split('\t'))
        # print(l.split('\t')[1])
        # print(re.split('\t([^]]+)\n',l))
        print(l)
        print(r'\t([^\t\n]*)\n')



    #     phrases.append(x.split('')[1]) #phrases.append( [x.split(' ')[1] for x in open(file).readlines()) or  np.loadtxt("myfile.txt")[:, 1]
    phrases=['The self']
    print(l)
    print(clean_sentences)
    for i in range(len(clean_sentences)):
       count=0
       for phrase in phrases:
           v= clean_sentences[i].find(phrase)
           print(v)

       cue_phrases_per_sentence.append(count/ len(special_tokenised_text[i]))
    end=time.time()
    print(f"Runtime of cue phrase is {end - start}")
    return cue_phrases_per_sentence

# scorer bases on sentence length
def sentence_length_scorer(tokenised_text):
    start=time.time()
    sentence_length=[]
    max_length_of_sentence = max(map(len, tokenised_text)) #finding max number of words in any sentence
    for i in range(len(tokenised_text)):
        sentence_length.append(len(tokenised_text[i])/max_length_of_sentence)
    end=time.time()
    print(f"Runtime of the sentence length is {end - start}")
    return sentence_length

#sentence significance scorer
def sentence_significance_scorer(tokenised_text):
    start=time.time()
    sentence_significance=[]
    final_alpha = alpha(tokenised_text)
    for i in range(len(tokenised_text)):
        sum_alpha=sum(final_alpha[i])
        sentence_significance.append(np.round(sum_alpha/len(tokenised_text[i]),2))
    end=time.time()
    print(f"Runtime of the significance scorer is {end - start}")
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
    print(f"Runtime of the freq_scorer is {end - start}")
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
    print(f"Runtime of the numeral is {end - start}")
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
    print(f"Runtime of the punctaution scorer is {end - start}")
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
    print(f"Runtime of the upper case is {end - start}")
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
    print(f"Runtime of the sentence_position is {end - start}")
    return sentence_position

def sentence_transformer_scorer(clean_sentences):
    start=time.time()
    model = SentenceTransformer('roberta-large-nli-stsb-mean-tokens')
    # Encode all sentences

    embeddings1 = model.encode(clean_sentences)
    embeddings2=model.encode(clean_sentences[3:4])
    # Compute cosine similarity between all pairs
    print("computing cosine similarity between all pairs")
    cos_sim = (util.pytorch_cos_sim(embeddings2, embeddings1))

    # with h5py.File('name-of-file.h5', 'w') as hf:
    #     hf.create_dataset("h5py dataset", data=cos_sim)
    # with h5py.File('name-of-file.h5', 'r') as hf:
    #     data = hf['h5py dataset'][:]
    # print(data)
    sum_rows=np.round(cos_sim.sum(axis=1)/len(clean_sentences),2)
    print("sum",sum_rows)
    kwargs = dict(kde_kws={'linewidth': 2}, kde=True,
                  bins=20, color='darkblue',
                  hist_kws={'edgecolor': 'black'})
    fig = sns.distplot(sum_rows, hist=True, **kwargs)
    plt.xlabel('Similarity score')
    plt.ylabel('frequency')

    # add a 'best fit' line
    # y = mlab.normpdf(20, mu, sigma)
    # l = plt.plot(20, y, 'r--', linewidth=2)

    plt.savefig("histogramUSINGbert.png")
    plt.clf()
    mu, sigma = scipy.stats.norm.fit(sum_rows)
    threshold1 = np.round(mu - (2 * sigma), 2)
    threshold2 = np.round(mu + (2 * sigma), 2)
    print("thresholds are ", threshold1, threshold2)
    cos_sim[cos_sim <= threshold1] = 0.0
    cos_sim[cos_sim >= threshold2] = 0.0
    print(cos_sim)

    print("Generating graph")
    plt.figure(2)
    nx_graph = nx.from_numpy_array(cos_sim.numpy())  # generating graph fom adjacency matrix
    raw_labels = map(str, (np.arange(len(tokenised_text))))  # loads converts a string of int to list of strings
    lab_node = dict(zip(nx_graph.nodes, raw_labels))
    layout = nx.spring_layout(nx_graph)
    nx.draw(nx_graph, layout)
    labels = nx.get_edge_attributes(nx_graph, "weight")
    nx.draw_networkx_edge_labels(nx_graph, pos=layout, edge_labels=labels)
    nx.draw_networkx_labels(nx_graph, layout, labels=lab_node, font_size=10, font_family='sans-serif')
    plt.savefig("graph.png")
    end = time.time()
    print("runtime of aggregate similarity", end - start)
    print("finding centrality using Bert method")
    centrality = nx.eigenvector_centrality(nx_graph)
    end=time.time()
    print(f"Runtime of the sentence_transformer is {end - start}")
    return sum_rows, list(centrality.values())

def embeddings_calculator(model,clean_sentences,a,b):
    sentence_embeddings = model.encode(clean_sentences[a:b])
    with h5py.File('embeddings_test.hdf5', 'a') as f:
        f['embed_test'][a:b]=sentence_embeddings


def cosine_similarity_matrix(embeddings2, a,b):
    # print("fucntion2")
    start = time.time()
    with h5py.File('random1.hdf5', 'a') as f:
        # dataset_name = f.create_dataset("default"+str(index), shape=(no_of_sentences, len(embeddings2)), dtype='f')
        dataset = f['default']
        sentence_aggregate_similarity=f['sum_rows']
        # no_of_batches = math.floor(no_of_sentences / batch_size)
        # modulo = no_of_sentences % batch_size
        # remaining = 0
        # if modulo != 0:
        #     remaining = no_of_sentences % batch_size
        # print("no of batches", no_of_batches)
        start1 = time.time()
        embeddings1 = embeddings2[a:b]
        # print(embeddings)
        # print(embeddings.shape)
        cos_sim = (util.pytorch_cos_sim(embeddings1, embeddings2))
        sum_row=(cos_sim.sum(axis=1))/len(embeddings2)
        dataset[a:b, :] = cos_sim
        sentence_aggregate_similarity[a:b]=np.round(sum_row,2)

def sentence_aggregate_similarity_calculator(clean_sentences):
    # if __name__ == '__main__':
    model = SentenceTransformer('roberta-large-nli-stsb-mean-tokens')
    embeddings_batch_size = 5000
    first = 0
    last = embeddings_batch_size

    no_of_sentences = len(clean_sentences)
    print("No of sentences", no_of_sentences)
    no_of_batches = math.floor(no_of_sentences / embeddings_batch_size)
    modulo = no_of_sentences % embeddings_batch_size
    print("no of batches", no_of_batches)
    firstList=[]
    lastList=[]
    for i in range(no_of_batches):
        firstList.append(first)
        first+=embeddings_batch_size
        lastList.append(last)
        last+=embeddings_batch_size
    if modulo != 0:
        remaining = no_of_sentences % embeddings_batch_size
        firstList.append(first)
        lastList.append(first+remaining)
    # with h5py.File('embeddings_test.hdf5', 'w') as f:
    #     dset = f.create_dataset("embed", (no_of_sentences, 1024))
    # #         dset = embeddings
    s = time.time()
    print("starting pool 1")
    # with multiprocessing.Pool(2) as pool:
    #     pool.starmap(partial(embeddings_calculator,model,clean_sentences),list(zip(firstList,lastList)))
    # pool.join()

    print(" thread time",time.time()-s)

    with h5py.File('embeddings_test.hdf5', 'r') as f:
        embeddings = f['embed_test'][()]
        print("embedding length",len(embeddings))
    sta = time.time()
    sentence_aggregate_similarity=[]
    with h5py.File('random1.hdf5', 'w') as f:
        dataset_name = f.create_dataset("default", shape=(no_of_sentences, no_of_sentences), dtype='f')
        sum_rows = f.create_dataset("sum_rows", dtype='f', shape=(no_of_sentences,))
    batch_size = 50
    no_of_batches = math.floor(no_of_sentences / batch_size)
    modulo = no_of_sentences % batch_size
    print("no of batches", no_of_batches)
    first=0
    last=first+batch_size
    firstList = []
    lastList = []
    for i in range(no_of_batches):
        firstList.append(first)
        first += batch_size
        lastList.append(last)
        last += batch_size
    if modulo != 0:
        remaining = no_of_sentences % batch_size
        firstList.append(first)
        lastList.append(first + remaining)
    with multiprocessing.Pool(2) as pool:
        pool.starmap(partial(cosine_similarity_matrix, embeddings[0:5000]), list(zip(firstList,lastList)))
    pool.join()
    print("starting pool2")

    with h5py.File('random1.hdf5', 'a') as f:
        cos_sim = f['default']
        sentence_aggregate_similarity=f['sum_rows'][()]

        # print(sum_row)
        # sentence_aggregate_similarity = cos_sim.sum(axis=1) / len(clean_sentences)
        # print(sentence_aggregate_similarity)
        plt.clf()
        kwargs = dict(kde_kws={'linewidth': 2}, kde=True,
                      bins=50, color='darkblue',
                      hist_kws={'edgecolor': 'black'})
        fig = sns.distplot(sentence_aggregate_similarity, hist=True, **kwargs)
        plt.xlabel('Similarity score')
        plt.ylabel('frequency')
        plt.savefig("histogram100k.png")
        plt.clf()
        mu, sigma = scipy.stats.norm.fit(sentence_aggregate_similarity)
        threshold1 = mu - (2 * sigma)
        threshold2 = mu + (2 * sigma)
        print("thresholds are ", threshold1, threshold2)

        print("yaaay tie", time.time() - sta)

        plt.clf()
        cos_sim[cos_sim < threshold1] = 0
        cos_sim[cos_sim > threshold2] = 0
        s = time.time()
        print("cos sim  size", len(cos_sim))
        # print("Generating graph")
        # nx_graph = nx.from_numpy_array(cos_sim)  # generating graph fom adjacency matrix
        # print("time for graph", time.time() - s)
        # # # print(nx.adjacency_spectrum(nx_graph))
        # print("finding centrality")
        # s = time.time()
        # centrality = nx.eigenvector_centrality(nx_graph, max_iter=1000)
        # print("centrality time", time.time() - s)
        # centrality = list(np.round(centrality.values(),2))
        return sentence_aggregate_similarity

@profile
def main():
    print("in main")
    start=time.time()
    # document_sentences, special_preprocessed_sentences, special_tokenised_text, tokenised_text, clean_sentences = preprocessing("Final_text.txt")
    # with open("preprocessed_whole.txt", encoding="utf8") as file:
    #     clean_sentences = [line.rstrip() for line in file]
    # tokenised_text=[]
    # with open("preprocessed_words.txt", encoding="utf8") as file:
    #     l = [line.rstrip() for line in file]
    #     for i in l:
    #         tokenised_text.append(ast.literal_eval(i))
    with open("special_preprocessed_sentences.txt", encoding="utf8") as file:
        special_preprocessed_sentences = [line.rstrip() for line in file]
    sentences = special_preprocessed_sentences[0:5000]
    print(sentences[4990:5000])
    # field names
    fields=['Sentences']
    filename = "sentences_data.csv"
    rows = sentences
    # writing to csv file
    with open(filename, 'w') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)

        # writing the fields
        csvwriter.writerow(fields)
        for line in sentences:
            csvfile.write(line)
            csvfile.write('\n')


        # special_tokenised_text=[]
    # with open("special_preprocessed_text.txt", encoding="utf8") as file:
    #     l = [line.rstrip() for line in file.readlines()]
    #     for i in l:
    #         special_tokenised_text.append(ast.literal_eval(i))
    # with open("document_sentences_text.txt", encoding="utf8") as file:
    #     document_sentences = [line.rstrip('\n') for line in file.readlines()]
    # print(len(document_sentences))
    #
    # document_sentences=document_sentences[0:20]
    # clean_sentences=clean_sentences[0:5000]
    # special_preprocessed_sentences = special_preprocessed_sentences[0:5000]
    # tokenised_text=tokenised_text[0:5000]
    # special_tokenised_text=special_tokenised_text[0:5000]
    print("stoooop")
    print(len(clean_sentences))
    print(len(special_preprocessed_sentences))
    print(len(tokenised_text))
    print(len(special_tokenised_text))

    # print(len(document_sentences))
    print(
        "done............................................................................................................................")
    sentence_position = sentence_position_scorer(document_sentences)
    sentence_position=sentence_position[0:5000]
    sentence_special_character = punctuation_sentence_scorer(special_preprocessed_sentences)
    sentence_upper_case = upper_case_sentence_scorer(special_tokenised_text)
    sentence_number_numeral = numeric_sentence_score(special_tokenised_text)
    sentence_aggregate_similarity = sentence_aggregate_similarity_calculator(special_preprocessed_sentences)
    ne_per_sentence, noun_per_sentence, verb_per_sentence, proper_noun_per_sentence = named_entity_recognition(special_preprocessed_sentences)
    # cue_phrases_per_sentence = sentence_cue_phrases(special_preprocessed_sentences, special_tokenised_text)
    sentence_length = sentence_length_scorer(tokenised_text)
    sentence_significance = sentence_significance_scorer(tokenised_text)
    sentence_freq_score = freq_scorer(tokenised_text)



    # # dictionary of lists
    dict = {
             'position': sentence_position,
            'special_char': sentence_special_character,
            'upper_case': sentence_upper_case,
            'numeral': sentence_number_numeral,
            'aggregate': sentence_aggregate_similarity,
            # 'centrality': centrality,
            'length': sentence_length,
            'significance': sentence_significance,
            'freq_score': sentence_freq_score,
            'proper_noun': proper_noun_per_sentence,
            'noun': noun_per_sentence,
            'verb': verb_per_sentence,
            'named_entity': ne_per_sentence
            }

    df1 = pd.DataFrame(dict)
    df1.to_csv('dataframe_1.csv',index=False)
    print("Total time", time.time()-start)


if __name__ == '__main__':

    import cProfile, pstats, io


    #
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    profiler.dump_stats('data-profiling')
    with open('data-profiling', 'w') as f:
        ps = pstats.Stats(profiler, stream=f)
        ps.strip_dirs().sort_stats('cumulative').print_stats()


    # profiler.dump_stats('data-profiling')
    # .strip_dirs().sort_stats('cumulative').print_stats()
    # stats.dump_stats('export-data-profiling')
    # print(stats.print_stats())

    # h = hpy()
    # print(h.heap())



    #
    #     s = io.StringIO()
    #     pr = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    #     # # stats.strip_dirs()
    #     # stats.print_stats()
    #     ps = pstats.Stats(pr, stream=f)
    #     if strip_dirs:
    #         ps.strip_dirs()
    #     if isinstance(sort_by, (tuple, list)):
    #         ps.sort_stats(*sort_by)
    #     else:
    #         ps.sort_stats(sort_by)
    #     ps.print_stats(lines_to_print)
