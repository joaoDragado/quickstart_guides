'''
we define a tokenizer function that cleans the unprocessed text data from the
csv file and separate it into word tokens while removing stop words:
'''

import numpy as np
import re
from nltk.corpus import stopwords

stop = stopwords.words('english')

def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
                           text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

'''
we define a generator function stream_docs that reads in and returns one document at a time
'''

def stream_docs(path):
    with open(path, 'r', encoding='utf-8') as csv:
        next(csv) # skip header
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label

# let's read in the first document from the file
next(stream_docs(path='movie_data.csv'))

'''
We define function get_minibatch , that takes a document stream from the stream_docs function and return a set no of documents, specified by the size parameter
'''

def get_minibatch(doc_stream, size):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y

'''
HashingVectorizer is data-independent.
we initialized HashingVectorizer with our tokenizer function and set the number of features to 2**21. we reinitialized a logistic regression classifier by setting the loss parameter of the SGDClassifier to 'log'.
'''

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
vect = HashingVectorizer(decode_error='ignore',
                         n_features=2**21,
                         preprocessor=None,
                         tokenizer=tokenizer)

clf = SGDClassifier(loss='log', random_state=1, n_iter=1)
doc_stream = stream_docs(path='movie_data.csv')

'''
we use the pyprind package to estimate the progress of our learning algorithm.
We initialized the progress bar object with 45 iterations and,
in the for loop, we iterated over 45 mini-batches of documents,
with each mini-batch consisting of 1,000 documents.
'''

import pyprind

pbar = pyprind.ProgBar(45)
classes = np.array([0, 1])

for _ in range(45):
    X_train, y_train = get_minibatch(doc_stream, size=1000)
    if not X_train:
        break
    X_train = vect.transform(X_train)
    clf.partial_fit(X_train, y_train, classes=classes)
    pbar.update()

'''
we use the last 5,000 documents to evaluate the performance of our model:
'''

X_test, y_test = get_minibatch(doc_stream, size=5000)
X_test = vect.transform(X_test)
print('Accuracy: %.3f' % clf.score(X_test, y_test))

# we can use the last 5,000 documents to update our model:
clf = clf.partial_fit(X_test, y_test)
