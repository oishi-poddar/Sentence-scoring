# Sentence-scoring

To run locally, the StanfordCoreNLP server needs to be started. Otherwise you can comment out the function call to named_entity_recognition function.
These are the steps:
1)To download Stanford CoreNLP, go to https://stanfordnlp.github.io/CoreNLP/index.html#download and click on “Download CoreNLP.
2)Install Java 8 (if not installed)
3)Running Stanford CoreNLP Server
4) Go to folder with extracted files and run the following command.
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -annotators "tokenize,ssplit,pos,lemma,parse,sentiment" -port 9000 -timeout 30000
