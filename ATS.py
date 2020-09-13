from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
import networkx as nx
import re
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import scipy
from pycorenlp import StanfordCoreNLP
from heapq import nlargest
from sentence_transformers import SentenceTransformer, util
from torch_geometric.utils import to_networkx

def preprocessing(file_name):
    file = open(file_name, encoding="utf8")
    filedata = file.read()

    # preprocess by splitting according to docs
    split_documents= re.split('[A-Z]{1,3}\d{1,12}[A-Z0-9]{1,3}',filedata)# splitting the extracted string into documents based on serialID
    document_sentences=[] # total number of docs=25715
    split_documents = list(filter(None, split_documents))
    for document in split_documents:
        document = re.sub(r'[\n\r\t]', ' ', document) #removing new line, tabs
        document = re.sub('[^A-Za-z0-9.]+', ' ', document)
        document_sentences.append(document)
    document_sentences = list(filter(None, document_sentences))
    print("No of documents",(len(document_sentences)))

    # preprocess by tokenising into sentence and removing only few speical characters, numerals etc
    special_processed_sentences = []
    tokenised = sent_tokenize(filedata)  # sentence tokenisation done before as full stop removed in pre processing
    for special_processed_sentence in tokenised:
        special_processed_sentence = re.sub('\w+[\d*]\w+', ' ', special_processed_sentence)  # removes words with digits in them or \S*\d\S*
        special_processed_sentence = re.sub(r'[\n\r\t]', ' ', special_processed_sentence)  # removes new lines and tabs
        special_processed_sentence = re.sub('[^A-Za-z0-9()"]+', ' ', special_processed_sentence) #remove special characters except parathesis and quotes
        special_processed_sentences.append(special_processed_sentence)
    special_tokenised_text = [word_tokenize(sentence) for sentence in special_processed_sentences]  # word tokenise
    special_tokenised_text = list(filter(None, special_tokenised_text))
    # preprocess by totally cleaning sentences
    clean_sentences=[]
    for clean_sentence in tokenised:
        clean_sentence = re.sub('\w*\d\w*', ' ', clean_sentence)# removes words with digits in them and all digits or \S*\d\S*
        clean_sentence=re.sub(r'[\n\r\t]',' ',clean_sentence) # removes new lines and tabs
        clean_sentence = re.sub('[^A-Za-z]+', ' ', clean_sentence) # remove punctuation,keeps only alphabet and certain important special characters
        clean_sentences.append(clean_sentence)
    clean_sentences = list(filter(None, clean_sentences))  # removing empty lists
    print("clean sentences",len(clean_sentences))
    stop_words = set(stopwords.words('english'))
    porter = PorterStemmer()
    tokenised_text = [word_tokenize(sentence) for sentence in clean_sentences] #word tokenise
    for i in range(len(tokenised_text)): # remove stop words and stem
        tokenised_text[i]= [porter.stem(word) for word in tokenised_text[i]]
        tokenised_text[i] = [word for word in tokenised_text[i] if word not in stop_words]
    tokenised_text = list(filter(None, tokenised_text)) # removing empty lists
    print("Final",tokenised_text)
    print("Number of sentences",len(tokenised_text)) #209312
    f = open('preprocessed.txt', 'w')
    for clean_sentence in tokenised_text:
        #clean_sentences = map(lambda x: x + '\n', each_sentence)
        f.writelines(str(clean_sentence)+'\n')
    f.close()
    return document_sentences, special_processed_sentences, special_tokenised_text, tokenised_text, clean_sentences

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
    count_of_words=0
    for sent in tokenised_text:
        for word in sent:
            count_of_words+=1
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
    l=0
    res = list(filter(None, tokenised_text))
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
    plt.hist(sentence_aggregate_similarity, bins=20)
    plt.savefig("histogram15.png")
    return sentence_aggregate_similarity

#NER
def named_entity_recognition(special_preprocessed_text):
    ne_per_sentence={}
    noun_per_sentence={}
    verb_per_sentence={}
    proper_noun_per_sentence={}
    nlp = StanfordCoreNLP('http://localhost:9000')
    for i in range(len(special_preprocessed_text)):
        text=(" ".join(special_preprocessed_text[i]))
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

        # checking number of noun  phrases per sentence
        for word in result["sentences"][0]['tokens']:
            if 'NN' in word['pos']:
                if(i in noun_per_sentence):
                    noun_per_sentence[i] +=1
                else:
                    noun_per_sentence[i] = 1

        # checking number of verb phrases per sentence
        for word in result["sentences"][0]['tokens']:
            if 'VB' in word['pos']:
                if(i in verb_per_sentence):
                    verb_per_sentence[i] +=1
                else:
                    verb_per_sentence[i] = 1

        # checking number of proper nouns per sentence
        for word in result["sentences"][0]['tokens']:
            if 'NNP' in word['pos']:
                if (i in proper_noun_per_sentence):
                    proper_noun_per_sentence[i] += 1
                else:
                    proper_noun_per_sentence[i] = 1
        text = ""
        if i in ne_per_sentence:
            ne_per_sentence[i] /= len(special_preprocessed_text[i])
        if i in noun_per_sentence:
            noun_per_sentence[i] /= len(special_preprocessed_text[i])
        if i in verb_per_sentence:
            verb_per_sentence[i] /= len(special_preprocessed_text[i])
        if i in proper_noun_per_sentence:
            proper_noun_per_sentence[i] /= len(special_preprocessed_text[i])
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

def numeric_sentence_score(special_tokenised_text):
    sentence_number_numeral={}
    for i in range(len(special_tokenised_text)):
        for word in special_tokenised_text[i]:
            if word.isnumeric():
                if i in sentence_number_numeral:
                    sentence_number_numeral[i]+=1
                else:
                    sentence_number_numeral[i]=1
    return sentence_number_numeral

def punctuation_sentence_scorer(special_preprocessed_text):
    sentence_special_character={}
    for i in range(len(special_preprocessed_text)):
        sentence=special_preprocessed_text[i]
        number_of_paranthesis=sentence.count("(")+sentence.count(")")
        number_of_quotes=sentence.count("\"")
        sentence_special_character[i]=number_of_paranthesis+number_of_quotes

    return sentence_special_character

def upper_case_sentence_scorer(special_tokenised_text):
    sentence_upper_case={}
    for i in range(len(special_tokenised_text)):
        for word in special_tokenised_text[i]:
            if word.isupper():
                print(word)
                if i in sentence_upper_case:
                    sentence_upper_case[i] +=1
                else:
                    sentence_upper_case[i] = 1
    return sentence_upper_case

def sentence_position_scorer(document_sentences):
    sentence_position={}
    k=0
    for doc in document_sentences:
        sentences=sent_tokenize(doc)
        halved_document_length=len(sentences)/2
        for i in range(len(sentences)):
            score=abs(halved_document_length-i)/halved_document_length
            sentence_position[k]= round(score,2)
            k+=1
    return sentence_position

def sentence_transformer_scorer(clean_sentences):
    model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
    # Encode all sentences
    embeddings = model.encode(clean_sentences)

    # Compute cosine similarity between all pairs
    cos_sim = util.pytorch_cos_sim(embeddings, embeddings)
    print(cos_sim.size())
    sum_rows=cos_sim.sum(axis=1)/len(clean_sentences)
    plt.hist(sum_rows, bins=10)
    mu, sigma = scipy.stats.norm.fit(sum_rows)
    best_fit_line = scipy.stats.norm.pdf(10, mu, sigma)
    plt.plot(10, best_fit_line)
    plt.savefig("histogram15.png")
    nx_graph=to_networkx(cos_sim)
    # nx_graph = nx.from_numpy_array(cos_sim.numpy()) # generating graph fom adjacency matrix
    draw_graph=nx.draw(nx_graph)
    plt.savefig("filename.png")
    # #plt.show()


document_sentences, special_preprocessed_text, special_tokenised_text, tokenised_text, clean_sentences = preprocessing("input2.txt")
sentence_transformer_scorer(clean_sentences)
#sentence_position = sentence_position_scorer(document_sentences)
# sentence_special_character = punctuation_sentence_scorer(special_preprocessed_text)
# sentence_upper_case = upper_case_sentence_scorer(special_tokenised_text)
# sentence_number_numeral = numeric_sentence_score(special_tokenised_text)
#aggregate_similarty = sentence_cosine_similarity(tokenised_text)
# ne_per_sentence, noun_per_sentence, verb_per_sentence, proper_noun_per_sentence = named_entity_recognition(special_preprocessed_text)
# cue_phrases_per_sentence = sentence_cue_phrases(tokenised_text)
# sentence_length = sentence_length_scorer(tokenised_text)
# sentence_significance = sentence_significance_scorer(tokenised_text)
# sentence_freq_score = freq_scorer(tokenised_text)



