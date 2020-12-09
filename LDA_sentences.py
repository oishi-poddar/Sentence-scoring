from nltk.tokenize import TweetTokenizer, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import LatentDirichletAllocation
from sentence_transformers import SentenceTransformer

tokenised_text = []
with open("preprocessed_whole.txt", encoding="utf8") as file:
    l = [line.rstrip() for line in file]
print(l[0:10])
#TFIDF vectorizer
# model = SentenceTransformer('distilbert-base-nli-mean-tokens')
# sentence_embeddings = model.encode(l[0:10])

#
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(l[0:9000])
tfidf_feature_names = vectorizer.get_feature_names()
print(tfidf_feature_names)
lda = LatentDirichletAllocation(n_components=3, random_state=0)
print(lda.get_document_topics)
X_lda = lda.fit_transform(X)
for idx, topic in enumerate(lda.components_):
        print ("Topic ", idx, " ".join(tfidf_feature_names[i] for i in topic.argsort()[:-10 - 1:-1]))