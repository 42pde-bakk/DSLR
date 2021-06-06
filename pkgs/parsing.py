import os


def check_input(argv):
    if len(argv) != 2:
        print(f'Please provide one parameter with the csv file.')
        quit()

    filename, extension = os.path.splitext(argv[1])
    if extension != '.csv' or not os.path.exists(argv[1]):
        print(f'Please provide a valid .csv file.')
        quit()
