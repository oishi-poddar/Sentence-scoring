import pandas as pd
import phrasemachine
from pycorenlp import StanfordCoreNLP
from nltk.tokenize import sent_tokenize
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scipy
import time
start=time.time()
phrases=[]
Autophrase_score=[]
phrases.append("smart card")
Autophrase_score.append(0.9508378559)
# with open('AutoPhrase_multi-words.txt', encoding="utf8") as f:
#     cnt = 0
#     for line in f:
#         line = line.rstrip()
#         contents = line.split("\t")
#         if len(contents) != 2:
#             cnt += 1
#         else:
#             Autophrase_score.append(contents[0])
#             phrases.append(contents[1])
#     print(len(phrases))
i=0
# nlp = StanfordCoreNLP('http://localhost:9000')
# sentence_phrase_count=[]
# idx=0
# all_sentences=[]
with_titles=0
number_sentences=0
with open('new-patentdb-unmanned.csv', encoding="utf8", ) as f:
    for line in f:
        if(i==50):
            break
        i+=1
        title=line.split(",")[1].strip()
        # if (title != ""):
        #     with_titles+=1
#             result = nlp.annotate(title,
#                                   properties={
#                                       'annotators': 'ner, pos',
#                                       'outputFormat': 'json',
#                                       'timeout': 900000,
#                                   })
#
#             title_phrases = []
#             for word in result["sentences"][0]['tokens']:
#                 if 'NN' in word['pos']:
#                     title_phrases.append(word['word'])
#                 if 'VB' in word['pos']:
#                     title_phrases.append(word['word'])
        sentences = sent_tokenize(line)
#         for s in sentences:
#             all_sentences.append(s)
#         print("no of sentencws",len(sentences))
        for sentence in sentences:
            number_sentences+=1
            if (title == ""):
                with_titles += 1
#             count = 0
#             for g in range(len(phrases)):
#                 count += (sentence.count(phrases[g])* ((float(Autophrase_score[g]))))
#                 # print(Autophrase_score[g])
#                 # print((sentence.count(phrases[g]) * int(float(Autophrase_score[g]))))
#                 # if(sentence.count(phrases[g])!=0):
#                 #     print(phrases[g],"contained in sentence",idx)
#                 #     print ((float(Autophrase_score[g])), "is the phrase score")
#                 #     print(count , "count")
#             # sentence_phrase_count.append(count*Autophrase_score[g])
#             sentence_phrase_count.append(count)
#
#             if(title!=""):
#                 for phrase in title_phrases:
#                     sentence_phrase_count[idx]+=sentence.count(phrase)
#                     # if (sentence.count(phrase) != 0):
#                     #     print(phrase, " title phrases contained in sentence", idx)
#             idx+=1
#     print(sentence_phrase_count)
# feature = {
#          'phrases_score': sentence_phrase_count,
#         }
#
# df1 = pd.DataFrame(feature)
# df1.to_csv('phrase_score_feature.csv',index=False)
print(number_sentences)
print(with_titles)
sentence_phrase_count=[]
# with open('phrase_score_feature.csv', encoding="utf8" ) as f:
#     for line in f:
#         sentence_phrase_count.append(int(line.rstrip()))
df=pd.read_csv('phrase_score_feature.csv')
sentence_phrase_count = df["phrases_score"].tolist()
# sentence_phrase_count=sentence_phrase_count[0:10]
# print(sorted(sentence_phrase_count, reverse=True))
kwargs = dict(kde_kws={'linewidth': 2},kde=False,
         bins=50,
         hist_kws={'edgecolor':'black'})
# plt.hist(sentence_phrase_count, bins=10)
# plt.show()
# fig=sns.distplot(sentence_phrase_count, hist=True, **kwargs)
# plt.xlabel('phrase count  score')
# plt.ylabel('frequency')
# # plt.show()
# plt.savefig("phrase count vs sentence.png")
indices = list(range(len(sentence_phrase_count)))
print(indices)
print(sentence_phrase_count)
indices.sort(key=lambda x: sentence_phrase_count[x])
print(indices)
list1 = [x for x in sentence_phrase_count if x == 0.]
print(len(list1))
print(indices[:len(list1)])
csv_input = pd.read_csv('special_preprocessed_sentences.txt')
csv_input['Predicted'] = pd.Series(sentence_phrase_count)
csv_input.to_csv('output.csv', index=False)
# print("time", time.time()-start)
# add a 'best fit' line
# y = mlab.normpdf(20, mu, sigma)
# l = plt.plot(20, y, 'r--', linewidth=2)

# plt.savefig("histogram1.png")
# mu, sigma = scipy.stats.norm.fit(sentence_phrase_count)
# threshold1=np.round(mu-(2*sigma),2)
# threshold2=np.round(mu+(2*sigma),2)
# print("thresholds are ", threshold1, threshold2)
# sorted_list = sorted(sentence_phrase_count, reverse=True)
# print(np.median(sentence_phrase_count))
#
# print(sentence_phrase_count[sentence_phrase_count<4.9] )

# similarity_matrix[similarity_matrix<threshold1]=0.0
# similarity_matrix[similarity_matrix>threshold2]=0.0
# print(similarity_matrix)
