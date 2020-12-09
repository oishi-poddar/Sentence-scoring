from re import search
import csv
import pandas as pd
import numpy as np
phrases = []
with open('AutoPhrase_new.txt', encoding="utf8") as f:

    cnt=0
    for line in f:
        line = line.rstrip()
        contents = line.split("\t")
        if len(contents)!=2:
            cnt+=1
        else:
            phrases.append(contents[1])
    print(len(phrases))
    print(phrases[0:41])
    count = 0

with open("preprocessed_whole.txt", encoding="utf8") as file:
    clean_sentences = [line.rstrip() for line in file]
    print(len(clean_sentences))
    clean=[]
    # stopwords=["figure","figures","diagram","example","examples"]
    # for i in range(len(clean_sentences[0:273])):
    #     querywords = clean_sentences[i].split()
    #     resultwords = [word for word in querywords if word.lower() not in stopwords]
    #     result = ' '.join(resultwords)
    #     clean.append(result)
    # print(len(clean))
    with open('Autophrase_updated.txt', 'w') as special_preprocessed:
        for clean_sentence in clean:
            special_preprocessed.writelines(str(clean_sentence)+'\n')
    # # for i in range(len(clean_sentences[0:273])):
    #     print(clean_sentences[i])
    result=[]
    for i in range(len(clean_sentences[0:273])):
        v=0
        strSentence = clean_sentences[i].replace(" ", "")
        for phrase in phrases[0:32]:
            wordList = phrase.split()
            str=''.join(wordList)
            if str in strSentence:
                v+=1
        result.append(v)
    print(len(result))
    # sortedL=result
    # sortedL.sort(reverse=True)
    # print(sortedL[0:151])
    # print(result)
    # sortedL=result.sort(reverse=True)
    # print(sortedL)
    # result.sort()
    print(result)
    list1=[]
    for x in result:
        if x>=2:
            list1.append(1)
        else:
            list1.append(0)
    print(list1)
    print(len(list1[0:5]))
    indices = list(range(len(result)))
    indices.sort(key=lambda x: result[x], reverse=True)
    # list1 = [x for x in indices if x >= 2]
    # print(indices[0:151])
    # print(len(indices))
    # fields = ['Autophrase_labeling']
    filename = "sentences_data.csv"

    # writing to csv file
    # with open(filename, 'w') as csvfile:
    print(len(list1))
    csv_input = pd.read_csv('sentences_data.csv')
    csv_input['Predicted'] = pd.Series(list1)
    csv_input.to_csv('output.csv', index=False)
    df = pd.read_csv('output.csv')
    TP = (df.Predicted == df['Oracle label']).sum()
    print(TP)
    recallDenom = (df['Oracle label'] == 1.0).sum()
    precisionDen=(df.Predicted==1.0).sum()
    print(recallDenom)
    print("precision", TP/precisionDen)
    print("denom", TP/recallDenom)
            # for i in indices:
    #     print(clean_sentences[i])
    #     #precision 31.1 TP=52  Recall 52/97 53.6%
    # 54/165=32.7    54/97=55.67
    # 53/151 =35      53/97=54.6

            # wCount = len(wordList)
            # print("cou",wCount)
            # for w in wordList:
            #     if w in clean_sentences[i]:
            #         print("mk",w)
            #         wCount -= 1
            # if wCount == 0 and :
            #     print("i",i)
        # for phrase in phrases:
        #
        #     if search(phrase, clean_sentences[i]):
        #         print(phrase)
            # v = phrase in clean_sentences[i]
        # print(v)


# print("count",cnt)
print(phrases[0:34])