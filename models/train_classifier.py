import sys
sys.setrecursionlimit(10000)
#Data Manipulation
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
#Database and Model
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
 
#Model Evaluation and Saving
from sklearn.metrics import classification_report
import pickle

def load_data(database_filepath):
    '''
    Load data from database

    Args:
    database_filepath: path to database

    Returns:
    X: features dataframe
    Y: labels dataframe
    category_names: category names
    '''
    #loads data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    '''
    Tokenize and lemmatize text

    Args:
    text: text to be tokenized

    Returns:
    tokens: list of tokens
    '''
    #Remove stop words and punctuation, and lemmatize words
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word).lower().strip() for word in tokens]
    return tokens    
    

def build_model():
    '''
    Builds the model

    Args:
    None

    Returns:
    pipeline: model
    '''
    #Using custom transformer to build model
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
])
    return pipeline


def train_model(model, X_train, Y_train):
    '''
    Train the model using grid search

    Args:
    model: model to be trained
    X_train: training features
    Y_train: training labels

    Returns:
    model: trained model
    '''
    #grid search
    parameters = {
    'clf__estimator__n_estimators': [50, 100],
    'clf__estimator__min_samples_split': [2, 4]
}
    cv = GridSearchCV(model, param_grid=parameters)
    cv.fit(X_train, Y_train)
    return cv.best_estimator_


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate model performance

    Args:
    model: trained model
    X_test: test features
    Y_test: test labels
    category_names: category names

    Returns:
    None
    '''
    #evaluates model
    Y_pred = model.predict(X_test)
    for i, col in enumerate(category_names):
        print(col)
        print(classification_report(Y_test[col], Y_pred[:, i]))


def save_model(model, model_filepath):
    '''
    Save model to pickle file

    Args:
    model: trained model
    model_filepath: path to save model

    Returns:
    None
    '''
    #saves model to pickle file
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model = train_model(model, X_train, Y_train)
        
        
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