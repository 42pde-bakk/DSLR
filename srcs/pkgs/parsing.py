import os
import sys


def check_input(argv, argv_len: int = 2):
    if len(argv) != argv_len:
        print(f'Please provide one parameter with the csv file.', file=sys.stderr)
        quit()

    filename, extension = os.path.splitext(argv[1])
    if extension != '.csv' or not os.path.exists(argv[1]):
        print(f'Please provide a valid .csv file.', file=sys.stderr)
        quit()

    if argv_len == 3 and not os.path.exists(argv[2]):
        print(f'Please provide a valid file for the models', file=sys.stderr)
        quit()


def predict_check_input(argv):
    if len(argv) != 3 or 'dataset_test.csv' not in argv[1]:
        print('It takes as a parameter dataset_test.csv and a file containing the weights trained by previous program.', file=sys.stderr)
        quit()

    filename, extension = os.path.splitext(argv[1])
    if extension != '.csv' or not os.path.exists(argv[1]):
        print(f'Please provide a valid .csv file.', file=sys.stderr)
        quit()
