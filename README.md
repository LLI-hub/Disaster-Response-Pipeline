# Disaster Response Pipeline
Python file and files for a working web app.

### by Iván Lucas López

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

The code should run with no issues using Python versions 3.*.

1. pandas
2. numpy
3. sqlalchemy
4. matplotlib
5. plotly
6. NLTK
7. NLTK [punkt, wordnet, stopwords]
8. sklearn
9. joblib
10. flask
11. re
12. sqlite3
13. pickle
14. sys
15.string
16.sqlalchemy

## Project Motivation<a name="motivation"></a>

**Study on traffic accidents in Madrid**

For this project, I was interestested in using El Ayuntamiento de Madrid data from 2019 about traffic accidents to better understand:

1. How many traffic accident occurred in Madrid during 2019?, 
  How many people are involved? and 
  What is the average number of people involved in a traffic accident?
  
2. What day of the week are there the most accidents?

3. What time do most accidents occur?

4. At what age are more traffic accidents suffered?

## File Descriptions <a name="files"></a>

There are 3 files available here to showcase work related to the above questions.  

1. data

    In a Python script, process_data.py, there is a data cleaner pipeline that:
  - Loads the messages and categories datasets
  - Merges the two datasets
  - Cleans the data
  - Stores it in a SQLite database

2. model

    In a Python script, train_classifier.py, there is a machine learning pipeline that:
  - Loads data from the SQLite database
  - Splits the dataset into training and test sets
  - Builds a text processing and machine learning pipeline
  - Trains and tunes a model using GridSearchCV
  - Outputs results on the test set
  - Exports the final model as a pickle file

3. app

    Flask web app.

## Results<a name="results"></a>

Here is how to see the Flask app.

Open a new terminal window. You should be in the app folder, if not, then use terminal commands to navigate inside the folder with the run.py file.

Type in the command line: **python run.py**

Your web app should now be running if there were no errors.

Now, open another Terminal Window.

Type: **env|grep WORK**

You'll see output that looks something like this:

SPACEDOMAIN: udacity-student-workspaces.com/

SPACEID: viewa7a4999b

In a new web browser window, type in the following:

https://SPACEID-3001.SPACEDOMAIN

In this example, that would be: **"https://viewa7a4999b-3001.udacity-student-workspaces.com/"** (Don't follow this link now, this is just an example.)

Your SPACEID might be different.

You should be able to see the web app. The number 3001 represents the port where your web app will show up. 
Make sure that the 3001 is part of the web address you type in.


## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Must give credit to El Ayuntamiento de Madrid for the data.  
You can find the Licensing for the data and other descriptive information at the link available [here](https://datos.madrid.es/portal/site/egob/menuitem.c05c1f754a33a9fbe4b2e4b284f1a5a0/?vgnextoid=7c2843010d9c3610VgnVCM2000001f4a900aRCRD&vgnextchannel=374512b9ace9f310VgnVCM100000171f5a0aRCRD&vgnextfmt=default).
