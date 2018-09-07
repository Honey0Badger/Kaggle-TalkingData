# Kaggle TalkingData AdTracking Fraud Detection Challenge
-----
Predict whether or not an Ad click collected by TalkingData is a click fraud or not based on
the relevant information about the click (device, OS, IP, timestamp, and etc). Link to the
competition page [Kaggle TalkingData](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection). The jupyter notebook ```talkingData_EDA.ipynb``` contains
preliminary analysis and visualization of the dataset.

Authors: Yunkun Xie (honeyo0obadger@gmail.com), Jianhua Ma (jm9yq@virginia.edu)

# Installation
-----

## Download the data
* Clone this repo to your computer (it comes with small sample inputs for test/debug purposes).
* Download the complete dataset from Kaggle [here](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/data) and save them into the ```inputs``` folder (be cautious that the datasets are big).
* switch into the model folder ```cd model```.

## Install the requirements
* Install the requirements using ```pip install -r requirements.txt```.
 * Make sure to use the Python 3 version (we recommend using [Anaconda](https://anaconda.org/anaconda/python) python distribution).
 * It is recommended to use virtual environment to have a clean package setup.

# Usage
-----
We use [click](http://click.pocoo.org/5/) package to create simple CLI for the modeling tasks.
Each task is created as an indepndent command and all tasks are grouped in the ```main.py``` file
for simplicity.

* Run ```mkdir data_dir``` to create a directory for processed and engineered data.
* Run ```python main.py``` to see what commands (task) are available.
* Run ```python main.py COMMAND [ARGS]``` to run each task.

# Example
-----
Here we show a simple workflow as an example of how to run the script. Assume the raw data files
are put under the ```*/inputs``` directory (here we use sample inputs for fast calculations) and
the python scripts are under ```*/model``` folder. Two separate folders are also created ```*/data```
and ```*/outputs``` for processed/engineered dataset and predictions respectively. Notice all the folders
are on the same level.

* Run ```cd model``` to switch into ```model``` folder.
* Run ```python main.py preprocess ../inputs/train_sample.csv ../inputs/test_sample.csv ../data/``` to preprocess
  the raw dataset and save processed dataset in ```../data/``` folder.
* Run ```python main.py feature_engineer ../data/``` to generate engineered features and save under the same ```../data/``` folder. Due to the large size and long processing time of each feature, they are saved in individual files so that they can be assembled later for testing different combinations of features without regenerating them.
* Run ```python main.py model_train_predict ../data/ ../outputs/``` to train the model and make predictions on the test set. The predictions are saved under ```../outputs/``` folder.

# Extending this
-----
Building a good model requires a lot of testing and adjusting. Current scripts offer the basic (but complete) workflow for the modeling process. It can be easily extended due to its modular nature. For example, each engineered feature is created as a separate function put in the ```feature_engineer.py``` file. Different statistical model can be created and adjusted (in terms of parameters) in the ```models.py``` file. Different tasks (train, predict, CV, and etc.) can be added in the ```pipelines.py``` and called from the ```main.py``` file by creating a command function in the ```main.py``` file. ```settings.py``` file contains runtime global parameters such as data filename, features/models to use.

