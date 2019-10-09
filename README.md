# Disaster Response Pipeline Project

## Udacity Data Scientist Nanodegree

### Project Overview

In this project, I apply my skills to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages. The project includes a web app where an emergency worker can input a new message and get classification results in several categories.   

### Project Components

#### 1. ETL Pipeline
* Loads teh messages and categories datasets
* Merges the two datasets
* Cleans the data
* Stores the results in a SQLite databse
#### 2. ML Pipeline
* Loads data from the SQLite database
* Splits the dataset into training and test sets
* Builds a text processing and machine learning pipeline
* Trains and tunes a model using GridSearchCV
* Outputs results on the test set
* Exports the final model as a pickle file
#### 3. Flask Web App
* A flask web application where an emergency worker can input a new message and get classification results in several categories. 
* Provides data visualizations using Plotly

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
