import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import random
import pickle
from collections import Counter
from nltk.stem import WordNetLemmatizer
import os
BASE_FOLDER = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))

pos_path = os.path.join(BASE_FOLDER, "resources/pos.txt")
neg_path = os.path.join(BASE_FOLDER, "resources/neg.txt")


lemmatizer = WordNetLemmatizer()
hm_lines = 100000

def create_lexicon(pos, neg):
    lexicon = []
    for fi in [pos, neg]:
        with open(fi, 'r') as f:
            contents = f.readlines()
            for l in contents[:hm_lines]:
                all_words = word_tokenize(l)
                lexicon += list(all_words)

    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    w_counts = Counter(lexicon)
    l2 = []
    # remove words like and, the, a ... or very rare words
    for w in w_counts:
        if 1000 > w_counts[w] > 50:
          l2.append(w)

    return l2

def sample_handling(sample, lexicon, classification):
    featureset = []

    with open(sample, 'r') as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            current_words = word_tokenize(l.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon))
            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    features[index_value] += 1
            features = list(features)
            featureset.append([features, classification])

    return featureset

def create_feature_sets_and_labels(pos, neg, test_size=0.1):
    lexicon = create_lexicon(pos,neg)
    features = []
    features += sample_handling(pos, lexicon, [1,0])
    features += sample_handling(neg, lexicon, [0,1])
    random.shuffle(features)

    features = np.array(features)
    testing_size = int(test_size * len(features))
    train_x = list(features[:,0][:-testing_size])
    train_y = list(features[:,1][:-testing_size])
    test_x = list(features[:, 0][-testing_size:])
    test_y = list(features[:, 1][-testing_size:])

    return train_x, train_y, test_x, test_y

if __name__ == '__main__':
    train_x, train_y, test_x, test_y = create_feature_sets_and_labels(pos_path, neg_path)
    sentiment_path = os.path.join(BASE_FOLDER, "resources/sentiment_set.pickle")
    with open(sentiment_path, 'wb') as f:
        pickle.dump([train_x, train_y, test_x, test_y], f)


