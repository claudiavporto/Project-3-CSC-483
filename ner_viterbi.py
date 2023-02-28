"""Named Entity Recognition as a sequence tagging task.

Author: Kristina Striegnitz and Claudia Porto

I affirm that I have carried out my academic endeavors with full
academic honesty. Claudia Porto

Complete this file for part 2 of the project.
"""
from nltk.corpus import conll2002
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support

import math
import numpy as np
from memm import MEMM

#################################
#
# Word classifier
#
#################################

def getfeats(word, o):
    """Take a word its offset with respect to the word we are trying to
    classify. Return a list of tuples of the form (feature_name,
    feature_value).
    """
    o = str(o)
    features = [
        (o + 'word', word),
        (o + 'isCapitalized', word[0].isupper()),
        (o + 'isDigit', word.isdigit()),
        (o + 'isAllCaps', word.isupper()),
        (o + 'isAllLower', word.islower()),
        (o + 'isPunct', len(word) == 1 and not word.isalnum()),
        (o + 'hasHyphen', '-' in word),
        (o + 'isIdentifier', word.isidentifier()),
        (o + 'hasApostrophe', "'" in word)
    ]
    return features
    

def word2features(sent, i):
    """Generate all features for the word at position i in the
    sentence. The features are based on the word itself as well as
    neighboring words.
    """
    features = []

    for o in [-1,0,1]:
        if i+o >= 0 and i+o < len(sent):
            word = sent[i+o][0]
            featlist = getfeats(word, o)
            features.extend(featlist)
    return features


#################################
#
# Viterbi decoding
#
#################################

def viterbi(obs, memm, pretty_print=False):
    """The Viterbi algorithm. Calculates what sequence of states is most
        likely to produce the given sequence of observations.

        V is the main data structure that represents the trellis/grid of
        Viterbi probabilities. It is a list of np.arrays. Each np.array
        represents one column in the grid.

        The variable 'path' maintains the currently most likely paths. For
        example, once we have finished processing the third observation,
        then 'path' contains for each possible state, the currently most
        likely path that leads to that state given these three
        observations.
        """
    V = [{}]
    path = {}

    init_obs_feats = dict(word2features(obs, 0))
    init_obs_feats['-1label'] = "<S>"
    vector_feats = memm.vectorize(init_obs_feats)
    init_state_probs = memm.classifier.predict_log_proba(vector_feats)

    for i, state in enumerate(memm.states):
        V[0][state] = init_state_probs[0][i]
        path[state] = [state]

    for i in range(1, len(obs)):
        V.append({})
        new_path = {}
        for j, state in enumerate(memm.states):
            max_v = float('-inf')
            max_prev_state = None
            for prev_state in memm.states:
                obs_feats = dict(word2features(obs, i))
                obs_feats['-1label'] = prev_state
                vector_feats = memm.vectorize(obs_feats)
                state_probs = memm.classifier.predict_log_proba(vector_feats)
                v = V[i - 1][prev_state] + state_probs[0][j]
                if v > max_v:
                    max_v = v
                    max_prev_state = prev_state
            V[i][state] = max_v
            new_path[state] = path[max_prev_state] + [state]
        path = new_path

    if pretty_print:
        pretty_print_trellis(V)

    (prob, state) = max([(V[len(obs) - 1][state], state) for state in memm.states])

    return path[state]

def pretty_print_trellis(V):
    """Prints out the Viterbi trellis formatted as a grid."""
    print("    ", end=" ")
    for i in range(len(V)):
        print("%7s" % ("%d" % i), end=" ")
    print()

    for y in V[0].keys():
        print("%.5s: " % y, end=" ")
        for t in range(len(V)):
            print("%.7s" % ("%f" % V[t][y]), end=" ")
        print()

if __name__ == "__main__":
    print("\nLoading the data ...")
    train_sents = list(conll2002.iob_sents('esp.train'))
    dev_sents = list(conll2002.iob_sents('esp.testa'))
    test_sents = list(conll2002.iob_sents('esp.testb'))

    print("\nTraining ...")
    train_feats = []
    train_labels = []

    for sent in train_sents:
        for i in range(len(sent)):
            feats = dict(word2features(sent, i))
            train_labels.append(sent[i][-1])
            if i == 0:
                feats['-1label'] = "<S>"
            else:
                feats['-1label'] = train_labels[-2]
            train_feats.append(feats)

    vectorizer = DictVectorizer()
    X_train = vectorizer.fit_transform(train_feats)

    model = LogisticRegression(max_iter=800)
    model.fit(X_train, train_labels)

    print("\nTesting ...")

    y_pred = []
    states = model.classes_
    vocabulary = 0
    memm = MEMM(states, vocabulary, vectorizer, model)

    print("Writing to results_memm.txt")

    j = 0
    with open("results_memm.txt", "w") as out:
        for sent in test_sents[:100]:
            y_pred.append(viterbi(sent, memm, False))
            for i in range(len(sent)):
                word = sent[i][0]
                gold = sent[i][-1]
                pred = y_pred[j][i]
                out.write("{}\t{}\t{}\n".format(word, gold, pred))
            j += 1
        out.write("\n")

print("Now run: python3 conlleval.py results_memm.txt")








