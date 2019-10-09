import sys
from sqlalchemy import create_engine
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
# NLP
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
# For ML
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
# For Model Evaluation
from sklearn.metrics import classification_report, accuracy_score
# For Saving
import pickle

def load_data(database_filepath):
    """
    Given the database filepath, load the engine, and divide the data into X and Y

    :param database_filepath: database filepath
    :return: X, Y, and categories
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('cleanData', con=engine)
    # Read messages as X
    X = df['message']
    # Read categories as Y
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)

    return X, Y, Y.columns


def tokenize(text):
    """
    Given a text, tokenize, lemmatize, normalize, and remove leading and trailing white space

    :param text:
    :return:
    """
    # tokenize text
    tokens = word_tokenize(text)
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        # lemmatize, normailize, and remove leading and trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Setup the pipeline and model

    :return: instance of GridSearchCV model
    """
    # Creating the pipeline
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultiOutputClassifier(
                             OneVsRestClassifier(LinearSVC())))])

    # Parameter grid
    parameters = {'vect__ngram_range': ((1, 1), (1, 2)),
                  'vect__max_df': (0.75, 1.0)
                  }

    # Create the model
    model = GridSearchCV(estimator=pipeline,
                         param_grid=parameters,
                         verbose=3,
                         cv=3)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """

    :param model:
    :param X_test:
    :param Y_test:
    :param category_names:
    :return:
    """
    y_preds = model.predict(X_test)
    print(classification_report(y_preds, Y_test.values, target_names=category_names))
    print("**** Accuracy scores for each category *****\n")
    for i in range(36):
        print("Accuracy score for " + Y_test.columns[i], accuracy_score(Y_test.values[:, i], y_preds[:, i]))


def save_model(model, model_filepath):
    """

    :param model:
    :param model_filepath:
    :return:
    """
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