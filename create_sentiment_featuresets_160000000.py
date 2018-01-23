import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import pickle
import pandas as pd
from nltk.stem import WordNetLemmatizer
import os

### need run only with GPU. CPU very slow
BASE_FOLDER = os.path.abspath(os.path.dirname( __file__ ))

path_16 = os.path.join(BASE_FOLDER, "resources/training.1600000.processed.noemoticon.csv")
path_2009_14 = os.path.join(BASE_FOLDER, "resources/testdata.manual.2009.06.14.csv")
train_path = os.path.join(BASE_FOLDER, "train_set.csv")
test_path = os.path.join(BASE_FOLDER, "test_set.csv")
processed_path = os.path.join(BASE_FOLDER, "processed-test-set.csv")
lexicon_path = os.path.join(BASE_FOLDER, "lexicon-2500-2638.pickle")
train_shuffled = os.path.join(BASE_FOLDER, "train_set_shuffled.csv")


lemmatizer = WordNetLemmatizer()


def init_process(fin, fout):
    outfile = open(fout, 'a')
    with open(fin, buffering=200000, encoding='latin-1') as f:
        try:
            for line in f:
                line = line.replace('"', '')
                initial_polarity = line.split(',')[0]
                if initial_polarity == '0':
                    initial_polarity = [1, 0]
                elif initial_polarity == '4':
                    initial_polarity = [0, 1]

                tweet = line.split(',')[-1]
                outline = str(initial_polarity) + ':::' + tweet
                outfile.write(outline)
        except Exception as e:
            print(str(e))
    outfile.close()


init_process(path_16, train_path)
init_process(path_2009_14, test_path)


def create_lexicon(fin):
    lexicon = []
    with open(fin, 'r', buffering=100000, encoding='latin-1') as f:
        try:
            counter = 1
            content = ''
            for line in f:
                counter += 1
                if (counter / 2500.0).is_integer():
                    tweet = line.split(':::')[1]
                    content += ' ' + tweet
                    words = word_tokenize(content)
                    words = [lemmatizer.lemmatize(i) for i in words]
                    lexicon = list(set(lexicon + words))
                    print(counter, len(lexicon))

        except Exception as e:
            print(str(e))

    with open(lexicon_path, 'wb') as f:
        pickle.dump(lexicon, f)


create_lexicon(train_path)


def convert_to_vec(fin, fout, lexicon_pickle):
    with open(lexicon_pickle, 'rb') as f:
        lexicon = pickle.load(f)
    outfile = open(fout, 'a')
    with open(fin, buffering=20000, encoding='latin-1') as f:
        counter = 0
        for line in f:
            counter += 1
            label = line.split(':::')[0]
            tweet = line.split(':::')[1]
            current_words = word_tokenize(tweet.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]

            features = np.zeros(len(lexicon))

            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    # OR DO +=1, test both
                    features[index_value] += 1

            features = list(features)
            outline = str(features) + '::' + str(label) + '\n'
            outfile.write(outline)

        print(counter)


convert_to_vec(test_path, processed_path, lexicon_path)


def shuffle_data(fin):
    df = pd.read_csv(fin, error_bad_lines=False)
    df = df.iloc[np.random.permutation(len(df))]
    print(df.head())
    df.to_csv(train_shuffled, index=False)


shuffle_data(train_path)


def create_test_data_pickle(fin):
    feature_sets = []
    labels = []
    counter = 0
    with open(fin, buffering=20000) as f:
        for line in f:
            try:
                features = list(eval(line.split('::')[0]))
                label = list(eval(line.split('::')[1]))

                feature_sets.append(features)
                labels.append(label)
                counter += 1
            except:
                pass
    print(counter)
    feature_sets = np.array(feature_sets)
    labels = np.array(labels)


create_test_data_pickle(processed_path)