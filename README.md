# DSLR
 
Subject created by the 42AI association. Discover Data Science in the projects where you re-constitute Poudlard’s Sorting Hat. Warning: this is not a subject on cameras.

## Project status: [![pde-bakk's 42 dslr Score](https://badge42.vercel.app/api/v2/cl1kxvlgu002109lfx5bumh9s/project/2295758)](https://github.com/JaeSeoKim/badge42)

## Setup

`git clone https://github.com/pde-bakk/DSLR.git && cd DSLR`

`pip3 install -r requirements.txt`

## V.1 Data Analysis
This program displays information about all numerical features of the provided dataset.

`python3 srcs/describe.py datasets/dataset_train.csv`

## V.2 Data Visualization
I created a set of scripts, each using a particular visualization method to answer a question.
These scripts require `datasets/dataset_train.csv` as a parameter to be able to answer the questions.
### V.2.1 Histogram
Which Hogwarts course has a homogenous score distribution between all four houses?
`python3 srcs/histogram.py datasets/dataset_train.csv`

### V.2.2 Scatter plot
What are the two features that are similar ?
`python3 srcs/scatter_plot.py datasets/dataset_train.csv`

### V.2.3 Pair plot
From this visualization, what features are you going to use for your logistic regression?
`python3 srcs/pair_plot.py datasets/dataset_train.csv`

## V.3 Logistic Regression
First off, train the models by running `python3 srcs/logreg_train.py datasets/dataset_train.csv`.
This will generate a `datasets/weights` file which can then be used for the predictions.

Then, run the predictions with `python3 srcs/logreg_predict.py datasets/dataset_test.csv datasets/weights`.
This will generate a file with all predictions in `datasets/houses.csv`.
