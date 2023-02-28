
class MEMM:
    def __init__(self, states, vocabulary, vectorizer, classifier):
        """Save the components that define a Maximum Entropy Markov Model: set of
        states, vocabulary, and the classifier information.
        """
        self.states = states
        self.vocabulary = vocabulary
        self.vectorizer = vectorizer
        self.classifier = classifier

    def vectorize(self, features):
        """Take a list of feature tuples and returns a vector representation
         of those features"""
        return self.vectorizer.transform(features)


