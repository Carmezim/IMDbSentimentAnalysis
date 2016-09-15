import numpy as np
import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')

labels = {'pos': 1, 'neg': 0}
df = pd.DataFrame()

# Moving movie reviews into a single data frame. This may take a few minutes.
print("Moving reviews into a single data frame. It may take a few minutes.")
for s in ('test', 'train'):
    for l in ('pos', 'neg'):
        path = os.path.join('./aclImdb/', s, l)
        for file in os.listdir(path):
            with open(os.path.join(path, file), 'r', encoding="utf-8") as infile:
                txt = infile.read()
            df = df.append([[txt, labels[l]]], ignore_index=True)


print("Data frame successfully created.")
df.columns = ['review', 'sentiment']

# Shuffling the data frame.
np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))

# Saving the assembled data as CSV file.
df.to_csv('./movie_data.csv', index=False)

df = pd.read_csv('./movie_data.csv', encoding="ISO-8859-1")

# First three samples of the data frame.
print("First three samples:\n ", df.head(3))

# Cleaning Text Data.
print("Cleaning Text Data. \n")


# Removing punctuation marks keeping emoticons.
def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + \
           ' '.join(emoticons).replace('-', '')
    return text


# preprocessor("</a>This :) is :( a test :-)!")

# Applying preprocessor to movie reviews in the data frame.


stop = stopwords.words('english')

# Documents into tokens
porter = PorterStemmer()

def tokenizer(text):
    return text.split()


def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


# Training Logistic Regression Model

# Finding optimal set of parameters using 5-fold stratified cross-validation.


if __name__ == '__main__':
    print("Removing punctuation marks. \n")
    df['review'] = df['review'].apply(preprocessor)

    print("Splitting training and test data. \n")
    X_train = df.loc[:25000, 'review'].values
    Y_train = df.loc[:25000, 'sentiment'].values
    X_test = df.loc[25000:, 'review'].values
    Y_test = df.loc[25000:, 'sentiment'].values

    print("\nSetting optimal parameters. \n")

    tfidf = TfidfVectorizer(strip_accents=None,
                            lowercase=False,
                            preprocessor=None)

    param_grid = [{'vect__ngram_range': [(1, 1)],
                   'vect__stop_words': [stop, None],
                   'vect__tokenizer': [tokenizer,
                                       tokenizer_porter],
                   'clf__penalty': ['l1', 'l2'],
                   'clf__C': [1.0, 10.0, 100.0]},
                  {'vect__ngram_range': [(1, 1)],
                   'vect__stop_words': [stop, None],
                   'vect__tokenizer': [tokenizer,
                                       tokenizer_porter],
                   'vect__use_idf': [False],
                   'vect__norm': [None],
                   'clf__penalty': ['l1', 'l2'],
                   'clf__C': [1.0, 10.0, 100.0]},
                  ]

    lr_tfidf = Pipeline([('vect', tfidf),
                         ('clf',
                          LogisticRegression(random_state=0))])

    gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid,
                               scoring='accuracy',
                               cv=5, verbose=1,
                               n_jobs=-1)

    print("Fitting 5 folds for each of 48 candidates, totalling 240 fits")
    gs_lr_tfidf.fit(X_train, Y_train)

    print('Best parameter set: %s ' % gs_lr_tfidf.best_params_)
    print('CV Accuracy: %.3f' % gs_lr_tfidf.best_score_)

    clf = gs_lr_tfidf.best_estimator_
    print('Test Accuracy: %.3f' % clf.score(X_test, Y_test))
