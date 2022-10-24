#!/usr/bin/env bash

if [[ "${1}" == "ALL" ]]; then
  python3 srcs/describe.py datasets/dataset_train.csv
  python3 srcs/histogram.py datasets/dataset_train.csv
  python3 srcs/scatter_plot.py datasets/dataset_train.csv
  python3 srcs/pair_plot.py datasets/dataset_train.csv
fi

python3 srcs/logreg_train.py datasets/dataset_train.csv
python3 srcs/logreg_predict.py datasets/dataset_test.csv models.pickle

python3 secret/evaluate.py
