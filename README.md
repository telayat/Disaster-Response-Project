# Disaster-Response-Project
Disaster Response Project (ETL and Modeling Pipelines)

### Table of Contents

1. [Installation](#installation)
2. [Project Overview](#overview)
3. [File Descriptions](#files)
4. [How to Use](#how_to_use)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

plotly library should be installed which can be done by running "pip install plotly==4.14.3" command on terminal.
Other than this there should be no necessary libraries to run the code here beyond the Anaconda distribution of Python.  The code should run with no issues using Python versions 3.*.

## Project Overview<a name="overview"></a>

In this project, I am demonestrating how an end-to-end machine learning project can be build through ETL pipeline and Machine Learning pipeline including GridSearch and Feature Union features till having the proper deployment and visualization required.

## File Descriptions<a name="files"></a>

data folder:

    -disaster_messages.csv: this is the disaster messages.
    -disaster_categories.csv: this is the disatser categories file.
    -process_data.py: this is the python file for the ETL pipeline

models folder:

    -train_classifier.py: this is python file for the ML modeling pipeline.
    
app folder:

    -run.py: this is python file to run the web app.

notebooks folder:

    -ETL Pipeline Preparation.ipynb: ETL pipeline code in notebook format.
    -ML Pipeline Preparation.ipynb: Machine Learning pipeline code in notebook format.


## How to Use<a name="how_to_use"></a>

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database; below an example:
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves it; below an example:
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app (you need to adjust DB name, table name and the model pkl file according to the parameters used in the previous two steps).
    `python run.py`

3. Go to http://0.0.0.0:3001/


## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Here I am using disaster data from "Figure Eight", credit must be given to them for making this data available.





