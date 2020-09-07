from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
import networkx as nx
import re
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from pycorenlp import StanfordCoreNLP
from heapq import nlargest

def preprocessing(file_name):

    file = open(file_name,encoding="utf8")
    filedata = file.readlines()
    sentences=[]
    tokenised=sent_tokenize("".join(filedata)) # sentence tokeisation done before as full stop removed in pre processing
    for sentence in tokenised:
        sentence = re.sub('\w*\d\w*', ' ', sentence)# removes words with digits in them or \S*\d\S*
        sentence=re.sub(r'[\n\r\t]',' ',sentence) # removes new lines and tabs
        sentence = re.sub('[^A-Za-z]+', ' ', sentence) # remove punctuation,keeps only alphabet and full stop
        sentences.append(sentence)
    # sentences = list(filter(str.strip, sentences)) # remove empty spaces and strings
    # print("new",sentences)
    # cleaned_text="".join(sentences)
    # cleaned_text= re.sub("\s\s+",' ', cleaned_text)  # remove multiple spaces one after another
    stop_words = set(stopwords.words('english'))
    # tokenised_text = sent_tokenize(cleaned_text) #sentence tokenise
    porter = PorterStemmer()
    tokenised_text = [word_tokenize(sentence) for sentence in sentences] #word tokenise
    for i in range(len(tokenised_text)): # remove stop words and stem
        tokenised_text[i]= [porter.stem(word) for word in tokenised_text[i]]
        tokenised_text[i] = [word for word in tokenised_text[i] if word not in stop_words]
    print("Final",tokenised_text)
    print("Number of sentences",len(tokenised_text)) #209312
    f = open('preprocessed.txt', 'w')
    for clean_sentence in tokenised_text:
        #clean_sentences = map(lambda x: x + '\n', each_sentence)
        f.writelines(str(clean_sentence)+'\n')
    f.close()
    return tokenised_text

# to find number of sentences in which each word occurs
def check_word_freq_per_sent(sentences):
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
    return no_sentences_word_occurs

# calculate frequency of each word
def term_freq_calculation(tokenised_text):
    term_freq={}
    for sent in tokenised_text:
        for word in sent:
            if word in term_freq:
                term_freq[word] += 1
            else:
                term_freq[word] = 1
    return term_freq

# to find the alpha term for each word to find sentence cosine similarity
def alpha(tokenised_text):
    no_sentences_word_occurs=check_word_freq_per_sent(tokenised_text)
    no_of_sentences=len(tokenised_text)
    term_freq= term_freq_calculation(tokenised_text)
    final_alpha=[]
    for sent in tokenised_text:
        alpha=[]
        for word in sent:
            denom=no_sentences_word_occurs[word]
            num=no_of_sentences
            tf=term_freq[word]
            log_term=np.log(num/denom)
            alpha.append(tf*log_term)
        final_alpha.append(alpha)
    return final_alpha

def cosine_similarity(tokenised_text):
    final_alpha = alpha(tokenised_text)
    similarity_matrix = np.zeros((len(tokenised_text), len(tokenised_text)))

    for i in range(len(tokenised_text)):
        for j in range(len(tokenised_text)):
            if i != j:
                num = sum([a * b for a, b in zip(final_alpha[i], final_alpha[j])])
                denom=np.sqrt(sum(map(lambda i : i * i, final_alpha[i]))) * np.sqrt(sum(map(lambda i : i * i, final_alpha[j])))
                similarity_matrix[i][j] = num/denom
    # nx_graph = nx.from_numpy_array(similarity_matrix) # generating graph fom adjacency matrix
    # draw_graph=nx.draw(nx_graph)
    # plt.savefig("filename.png")
    # #plt.show()
    return similarity_matrix

def sentence_cosine_similarity(tokenised_text):
    sentence_aggregate_similarity=[]
    similarity_matrix = cosine_similarity(tokenised_text)
    sum_rows = np.sum(similarity_matrix, axis=1)
    for i in range(len(tokenised_text)):
        sentence_aggregate_similarity.append(sum_rows[i]/len(tokenised_text[i]))
    return sentence_aggregate_similarity

#NER
def named_entity_recognition(tokenised_text): # to do NO STOP WORDS OR STEM, BREAK UP INTO NOUN AND VERB
    ne_per_sentence={}
    pos_per_sentence={}
    proper_noun_per_sentence={}
    nlp = StanfordCoreNLP('http://localhost:9000')
    for i in range(len(tokenised_text)):
        text=(" ".join(tokenised_text[i]))
        result = nlp.annotate(text,
                          properties={
                              'annotators': 'ner, pos',
                              'outputFormat': 'json',
                              'timeout': 1000,
                          })
       # checking number of named entities per sentece
        for word in result["sentences"][0]['tokens']:
            if not word['ner'] == 'O':
                if (i in ne_per_sentence):
                    ne_per_sentence[i] += 1
                else:
                    ne_per_sentence[i] = 1

        # checking number of noun and verb phrases per sentence
        for word in result["sentences"][0]['tokens']:
            if 'NN' in word['pos'] or 'VB' in word['pos']:
                if(i in pos_per_sentence):
                    pos_per_sentence[i] +=1
                else:
                    pos_per_sentence[i] = 1

        # checking number of proper nouns per sentence
        for word in result["sentences"][0]['tokens']:
            if 'NNP' in word['pos']:
                if (i in proper_noun_per_sentence):
                    proper_noun_per_sentence[i] += 1
                else:
                    proper_noun_per_sentence[i] = 1
        text = ""
        if i in ne_per_sentence:
            ne_per_sentence[i] /= len(tokenised_text[i])
        if i in pos_per_sentence:
            pos_per_sentence[i] /= len(tokenised_text[i])
        if i in proper_noun_per_sentence:
            proper_noun_per_sentence[i] /= len(tokenised_text[i])
    return ne_per_sentence, pos_per_sentence, proper_noun_per_sentence

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
    sentence_length={}
    max_length_of_sentence = max(map(len, tokenised_text)) #finding max number of words in any sentence
    for i in range(len(tokenised_text)):
        sentence_length[i]=len(tokenised_text[i])/max_length_of_sentence

    return sentence_length

#sentence significance scorer
def sentence_significance_scorer(tokenised_text):
    sentence_significance={}
    final_alpha = alpha(tokenised_text)
    for i in range(len(tokenised_text)):
        sum_alpha=sum(final_alpha[i])
        sentence_significance[i]=sum_alpha/len(tokenised_text[i])

    return sentence_significance

# frequency scorer
def freq_scorer(tokenised_text):
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
    return sentence_freq_score


tokenised_text = preprocessing("input1.txt")
aggregate_similarty = sentence_cosine_similarity(tokenised_text)
ne_per_sentence, pos_per_sentence, proper_noun_per_sentence = named_entity_recognition(tokenised_text)
cue_phrases_per_sentence = sentence_cue_phrases(tokenised_text)
sentence_length = sentence_length_scorer(tokenised_text)
sentence_significance = sentence_significance_scorer(tokenised_text)
sentence_freq_score = freq_scorer(tokenised_text)



