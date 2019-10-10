# Disaster Response Pipeline Project

## Udacity Data Scientist Nanodegree

### Project Overview

In this project, I apply my skills to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages. The project includes a web app where an emergency worker can input a new message and get classification results in several categories.   

### Project Components

#### 1. ETL Pipeline
* Loads the messages and categories datasets
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

### Files:
* /app/templates/go.html - web page that handles user query and displays model results
* /app/templates/master.html - index webpage displays cool visuals and receives user input text for model
* /app/run.py - flask application
* /data/disaster_categories.csv - categories dataset
* /data/disaster_messages.csv - messages dataset
* /data/DisasterResponse.db - SQLite database container the merged dataset
* /data/process_data.csv - data cleaning ETL pipeline that merges the two dataset above
* /models/classifier.pkl - classifier model as a pickle file
* /models/train_classifier.py - machine learning pipeline that builds the classifier model above
* /dsnd.yml - anaconda environment used to run this code
* /README.md - github readme file

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
