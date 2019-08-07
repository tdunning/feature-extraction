### We will use the hotel reviews from https://kavita-ganesan.com/entity-ranking-data
### to build word representations using cooccurrence
import collections
import random
import re
import time
from math import floor
from typing import Optional, Set, Callable, List

import numpy as np
import scipy.sparse as sparse
from sklearn import preprocessing as pre

# We just use a simple regex here to define words
# this is a bit lossy compared to fancier tokenizers
# but it is also about 50x faster
wordPattern = re.compile(r'''(?x)
    ([A-Z]\.)+
    |\d+:(\.\d)+
    |(https?://)?(\w+\.)(\w{2,})+([\w/]+)?
    |[@#]?\w+(?:[-']\w+)*
    |\$\d+(\.\d+)?%?
    |\\[Uu]\w+
    |\\[Uu]\w+'t
    |\.\.\.
    |[!?]+
    ''')


def docs(max_docs=-1, ignore=None):
    """Returns a generator of generators. The inner generators return the tokens of
    each document in our corpus."""
    if ignore is None:
        ignore = set()
    with open("/Users/tdunning/tmp/OpinRank/hotels.txt", "r", encoding="latin_1") as f:
        doc = 0
        step = 1
        scale = 10
        t0 = time.time_ns() / 1e9
        i0 = 0
        for line in f.read().split("\n"):
            doc = doc + 1
            if max_docs != -1 and doc > max_docs:
                break
            if doc % (step * scale) == 0:
                t1 = time.time_ns() / 1e9
                print("Doc %d (%.0f doc/s)" % (doc, (doc - i0) / (t1 - t0)))
                i0 = doc
                t0 = t1
                step = floor(step * 2.55)
                if step >= 10:
                    step = 1
                    scale = scale * 10
            pieces = line.split('\t', maxsplit=2)
            if len(pieces) == 3:
                yield (m.group(0) for m in wordPattern.finditer(pieces[2].lower()) if m and m.group(0) not in ignore)


def H(k):
    """Computes unnormalized entropy of a stack of vectors"""
    if k.ndim == 2:
        k = k[:, :, np.newaxis]
    p = (k + 0.0) / k.sum(axis=1).sum(axis=1)[:, np.newaxis, np.newaxis]
    raw = -(k * np.log(p + (p == 0)))
    while raw.ndim > 1:
        raw = raw.sum(axis=1)
    return raw


def llr(k):
    """Computes the log-likelihood ratio test for binomials in a vector-wise fashion.
    K is assumed to contain an n x 2 x 2 array of counts presumed to be a 2x2 table for
    each of n cases. We return an n-long vector of scores."""
    s_row = H(k.sum(axis=1))
    s_col = H(k.sum(axis=2))
    s = H(k)
    return 2 * (s_row + s_col - s)


def encode(docs: collections.abc.Iterable, lexicon: List[str], ignore=Optional[Set],
           matrix: Optional[Callable] = sparse.csr_matrix) -> sparse.spmatrix:
    if ignore is None:
        ignore = {}
    lookup = dict(zip(sorted(lexicon), range(len(lexicon))))
    rows = []
    cols = []
    data = []
    k = 0
    for d in docs:
        cols.extend({lookup[w] for w in d if w not in ignore})
        n = len(cols) - len(rows)
        rows.extend(itertools.repeat(k, n))
        data.extend(itertools.repeat(1, n))
        k += 1
    zz = matrix((data, (rows, cols)))
    return (zz)


from collections import Counter
from nltk.corpus import brown
import itertools


def count():
    k = Counter()
    for w in wordPattern.split(brown.raw()):
        k[w] += 1
    return k


def test():
    return Counter((w for d in docs() for w in d))


Ndocs = 50000
minScore = 15
maxAssociates = 30

# count all the words that appear
lexicon = Counter(itertools.chain.from_iterable(docs(Ndocs)))

# kill words too rare to have interesting collocation
kill = {w for w in lexicon if lexicon[w] < 3}
for w in kill:
    del lexicon[w]

allWords = sorted(lexicon)

# build the doc x word matrix using the lexicon we have slightly tuned
# note column friendly result
z = encode(docs(Ndocs), allWords, ignore=kill, matrix=sparse.csc_matrix)

# downsample frequent words (don't kill them entirely)
targetMaxFrequency = max(200.0, Ndocs / 30.0)
downSampleRate = [min(1, targetMaxFrequency / lexicon[w]) for w in allWords]
print("downsample %.0f words out of %d" % (sum(1 if p < 1 else 0 for p in downSampleRate), len(lexicon)))
for w in range(len(allWords)):
    p = downSampleRate[w]
    if p < 1:
        # only a few words will get whacked
        nz = z[:, w].nonzero()
        v = [1 if random.random() < p else 0 for i in nz[0]]
        z[nz[0], w] = v

# so here are final counts
wordCounts = z.sum(axis=0)
total = sum(wordCounts)
print("doc x word matrix ready")

# compute raw cooccurrence
cooc = z.T @ z
# but avoid self-cooccurrence
cooc[(range(cooc.shape[0]), range(cooc.shape[1]))] = 0
print('cooccurrence computation done %.3f sparsity' % ((cooc > 0).sum() / (lambda s: s[0] * s[1])(cooc.shape)))

# now find interesting cooccurrence
# we build a 3D array with one 2x2 contingency tables for each non-zero in the cooccurrence table
# the four elements count how often two particular words cooccur or not
nz = cooc.nonzero()
# A and B together
k11 = cooc[nz]
# A anywhere
k1_ = wordCounts[0, nz[0]]
# A without B
k12 = k1_ - k11

# B anywhere
k_1 = wordCounts[0, nz[1]]
# B without A
k21 = k_1 - k11
# neither A nor B
k22 = Ndocs - k12 - k21 - k11

# final shape should be n x 2 x 2
k = np.array([k11, k12, k21, k22]).reshape((2, 2, k11.shape[1])).transpose()
print("%d x %d x %d counts ready" % k.shape)

# constructs scores whereever cooc was non-zero. Note cooc is symmetric, extra work here
scores = sparse.csr_matrix((llr(k), nz))
print("scoring done")

# now review each word and limit the number of associates
rows = []
cols = []
for row in range(scores.shape[0]):
    # find nth highest score
    index = (scores[row, :] >= minScore).nonzero()[1]
    if len(index) > 0:
        s = sorted((scores[row, index].toarray().flat), reverse=True)
        cutoff = s[min(len(s), maxAssociates) - 1]
        cols.extend(i for i in index if scores[row, i] >= cutoff)
        rows.extend(itertools.repeat(row, len(cols) - len(rows)))
# final result has row per word consisting of unweighted associates
# might should consider idf weighting here
associates = sparse.csr_matrix((list(itertools.repeat(1, len(rows))), (rows, cols)), shape=scores.shape)
print("associates ready")
synonyms = associates * associates.T

lookup = dict(zip(sorted(lexicon), range(len(lexicon))))
unlook = list(sorted(lexicon))
print([unlook[i] for i in (synonyms[lookup['railway'], :] > 3).nonzero()[1]])
print([unlook[i] for i in (synonyms[lookup['hot'], :] > 3).nonzero()[1]])
print([unlook[i] for i in (synonyms[lookup['cold'], :] > 3).nonzero()[1]])
print([unlook[i] for i in (synonyms[lookup['food'], :] > 3).nonzero()[1]])
print([unlook[i] for i in (synonyms[lookup['room'], :] > 3).nonzero()[1]])

# lookup = dict(zip(sorted(lexicon), range(len(lexicon))))
# unlook = list(sorted(lexicon))
# m = lookup['railway']
# index = scores[m,:].nonzero()[1]
# print(sorted(((scores[m,i], i, unlook[i]) for i in index), key=lambda x: -x[0])[1:15])

limit = 1000

n1 = (wordCounts ** 2).sum()
n2 = (wordCounts[wordCounts < limit] ** 2).sum() + ((wordCounts >= limit) * limit * limit).sum()
print(n1 / n2)

all_docs = [bag for bag in docs()]
wordEncoder = pre.OneHotEncoder()
words = [[w] for bag in all_docs for w in bag]
wordEncoder.fit(words)
vectors = [v.sum(axis=0) for bag in all_docs]
wordEncoder.transform([[x] for x in all_docs[0]])
