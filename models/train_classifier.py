import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import numpy as np
import pandas as pd
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin

from sqlalchemy import create_engine

def load_data(database_filepath):
    '''
    INPUTs
        database_filepath: the DB file path and name including the extension
    OUTPUTs
        X: the messages columns
        Y: the categories columns
        category_names: the categories columns names
    '''
    # read in file
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql("SELECT * FROM msg_cat", engine)
    X = df.message.values
    Y = df.iloc[:, 4:]
    
    category_names = Y.columns
    
    return X, Y, category_names


def tokenize(text):
    '''
    Input: the text to be tokenized
    Output: array of tokens after replacing URLs with place holder, lemmatization, stripping and converting to lower case 
    '''
    
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def display_results(model, Y_test, y_pred, category_names):
    '''
    INPUTs
        model: the prediction model
        Y_test: the test labels
        y_pred: the predicted lables
        category_names: the categories columns names list
    OUTPUTs
        print the calssification report and the model best parameters
    '''
    print("Classification_report:", classification_report(Y_test, y_pred, target_names=category_names))
    print("\nBest Parameters:", model.best_params_)


class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def build_model():
    '''
    Initiate the model and define the grid search parameters
    OUTPUTs
        return the model pipeline
    '''
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', RandomForestClassifier())
    ])

    parameters = {
        #'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        #'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
        #'features__text_pipeline__vect__max_features': (None, 5000, 10000),
        #'features__text_pipeline__tfidf__use_idf': (True, False),
        'clf__n_estimators': [50, 100, 200],
        'clf__min_samples_split': [2, 3, 4]#,
        #'features__transformer_weights': (
        #    {'text_pipeline': 1, 'starting_verb': 0.5},
        #    {'text_pipeline': 0.5, 'starting_verb': 1},
        #    {'text_pipeline': 0.8, 'starting_verb': 1},
        #)
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    INPUTs
        model: the prediction model
        X_test: the test dataset
        Y_test: the test lables
        category_names: the categories columns names list
    OUTPUTs
        predict the model
        and call display_results to print the calssification report and the model best parameters
    '''
    y_pred = model.predict(X_test)
    display_results(model, Y_test, y_pred, category_names)


def save_model(model, model_filepath):
    '''
    INPUTs
        model: the prediction model
        model_filepath: the file path and file name including the extension '.pk'
    OUTPUTs
        Export the model as a pickle file 
    '''
    filename = model_filepath
    f = open(filename, 'wb')
    pickle.dump(model, f)
    f.close()


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()