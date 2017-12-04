from __future__ import print_function
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

# Initializing stemmer
tokenizer = RegexpTokenizer(r'\w+')
en_stop = get_stop_words('en')
p_stemmer = PorterStemmer()

''' Function that takes an input file and performs stemming to generate the output file '''
def getStemmedDocument(inputFileName, outputFileName):
	with open(inputFileName) as f:
	    docs = f.readlines()

	out = open(outputFileName, 'w')

	for doc in docs:
		raw = doc.lower()
		split = raw.split('\t', 1)
		docClass = split[0]
		tokens = tokenizer.tokenize(split[1])
		stopped_tokens = [token for token in tokens if token not in en_stop]
		stemmed_tokens = [p_stemmer.stem(token) for token in stopped_tokens]
		documentWords = ' '.join(stemmed_tokens)
		print((docClass + "\t" + documentWords), file=out)

	out.close();

# Creates the new stemmed documents with the suffix 'new' for both train and test files
getStemmedDocument('r8-train-all-terms.txt', 'r8-train-all-terms-new.txt')
getStemmedDocument('r8-test-all-terms.txt', 'r8-test-all-terms-new.txt')


