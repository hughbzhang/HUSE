import csv
import numpy as np
import collections
import csv

from utils import plot, calculate_HUSE

def get_data():
    HJ, model_probs, labels, length_list = [], [], [], []

    with open('../data/summary_clean.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        first_line = True
        for row in csv_reader:
            if first_line:
                first_line = False
                print('Column names are {", ".join(row)}')
            else:
                HJ.append(float(row[0]))
                model_probs.append(float(row[1]))
                labels.append(row[2] == "True")
                length_list.append(int(row[3]))

    return HJ, model_probs, labels, length_list

def main():

    # For custom use, rewrite this section with your own data
    ###################################################################
    taskname = "SUMMARIZATION"
    HJ, model_probs, labels, length_list = get_data()
    ###################################################################

    HUSE, HUSEQ, HUSED = calculate_HUSE(HJ, model_probs, labels, length_list)

    # OUTPUT
    print("For the task of {}".format(taskname))
    print("Overall HUSE score is: {}".format(HUSE))
    print("HUSE-Q (just human) score is: {}".format(HUSEQ))
    print("HUSE-D score is: {}".format(HUSED))

    # Plot saved to {taskname}.pdf
    plot(taskname, HJ, model_probs, labels, length_list)

if __name__ == "__main__":
    main()
