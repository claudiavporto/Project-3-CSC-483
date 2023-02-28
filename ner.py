"""Named Entity Recognition as a classification task.

Author: Kristina Striegnitz and Claudia Porto

I affirm that I have carried out my academic endeavors with full
academic honesty. Claudia Porto

Complete this file for part 1 of the project.
"""
from nltk.corpus import conll2002
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support

def getfeats(word, o):
    """Take a word and its offset with respect to the word we are trying
    to classify. Return a list of tuples of the form (feature_name,
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
            word = sent[i+o][0] # get word
            featlist = getfeats(word, o)
            features.extend(featlist)
    return features

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
            feats = dict(word2features(sent,i))
            train_feats.append(feats)
            train_labels.append(sent[i][-1])

    vectorizer = DictVectorizer()
    X_train = vectorizer.fit_transform(train_feats)
    
    model = LogisticRegression(max_iter=800)
    model.fit(X_train, train_labels)

    print("\nTesting ...")
    test_feats = []
    test_labels = []

    for sent in test_sents:
        for i in range(len(sent)):
            feats = dict(word2features(sent,i))
            test_feats.append(feats)
            test_labels.append(sent[i][-1])

    X_test = vectorizer.transform(test_feats)
    y_pred = model.predict(X_test)

    print("Writing to results_classifier.txt")
    # format is: word gold pred
    j = 0
    with open("results_classifier.txt", "w", encoding="utf8") as out:
        for sent in test_sents[:100]:
            for i in range(len(sent)):
                word = sent[i][0]
                gold = sent[i][-1]
                pred = y_pred[j]
                j += 1
                out.write("{}\t{}\t{}\n".format(word,gold,pred))
        out.write("\n")

    print("Now run: python3 conlleval.py results_classifier.txt")






