'''
Import all the packages needed 
'''
import sys
import pandas as pd
import sqlite3
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])

import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords 
import string
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report,  accuracy_score
from sklearn.metrics import f1_score, make_scorer

import sys
import pickle
import os

def load_data(database_filepath):
    #Load dataset from database with read_sql_table
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('df_clean', engine)
    df = df.drop(['original'], 1)
    df.dropna(inplace=True)
    
    # Define feature and target variables X and Y
    X = df['message']
    Y = df.drop(['id', 'message', 'genre'], axis=1)
    #to ensure no value is > 1
    for i in Y.columns.tolist():
        Y[i]=Y[i].map(lambda x: 1 if x >= 2 else x)
    
    #List of the category names for classification
    global category_names
    category_names = Y.columns.tolist()
    
    return X,Y,category_names


def tokenize(text, stopW = True):
    
     #I want a unique word if there are an url
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
        
    #to erase punctuation simbols
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    

    clean_tokens = []
    for tok in tokens:
                
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    #we can choose if we want to erase the stopwords
    if stopW == True:
        # Remove stop words
        clean_tokens = [w for w in clean_tokens if not w in stopwords.words("english")]
    else:
        pass

    return clean_tokens


def build_model():
    # Returns the GridSearchCV object to be used as the model
    
    pipeline = Pipeline([('text_pipeline', Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer())
    ])),
                         ('clf', MultiOutputClassifier(AdaBoostClassifier()))
                        ])
    
    
    
    parameters = [
        {
            'text_pipeline__vect__max_df': (.75, 1.0),
            'text_pipeline__tfidf__use_idf': (True, False),
            'clf': (MultiOutputClassifier(AdaBoostClassifier()),),
            'clf__estimator__learning_rate':[.1,1,2]
        }, {
            'text_pipeline__vect__max_df': (.75, 1.0),
            'text_pipeline__tfidf__use_idf': (True, False),
            'clf': (MultiOutputClassifier(KNeighborsClassifier()),),
            'clf__estimator__n_neighbors': (3, 5, 9)
        }
    ]
    
    cv = GridSearchCV(pipeline, param_grid=parameters, scoring=make_scorer(f1_score , average='weighted'),cv=3,n_jobs=-1)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)        
    print(classification_report(Y_test.values, Y_pred, target_names=category_names))
    
    # Calculate the test for each category.
    for i in range(len(category_names)):
        print('Category: {} '.format(category_names[i]))
        print(classification_report(y_test.iloc[:, i].values, y_pred[:, i]))
        print('Accuracy {}\n\n'.format(accuracy_score(Y_test.iloc[:, i].values, Y_pred[:, i])))


def save_model(model, model_filepath):
    #Export your model as a pickle file
    pickle.dump(model, open(model_filepath, 'wb'))


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
        print("\nBest Parameters:", model.best_params_)
        
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
