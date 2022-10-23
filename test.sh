#!/usr/bin/env bash

pip3 install -r requirements.txt > /dev/null

python3 srcs/logreg_train.py datasets/dataset_train.csv
python3 srcs/logreg_predict.py datasets/dataset_test.csv models.pickle

python3 secret/evaluate.py
