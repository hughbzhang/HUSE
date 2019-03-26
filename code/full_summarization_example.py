# Full summarization example processing code directly from Mechanical Turk
from utils import plot, calculate_HUSE
from full_example_utils import *

# This functionality is the same as summarization_example.py
def basics(filename):
    logps, bucket, labels, lens = extract(filename)
    buck_list, p_list, l_list, len_list = get_data(logps, bucket, labels, lens, types, nrep_max=40)
    taskname = "SUMMARIZATION"
    HUSE, HUSEQ, HUSED = calculate_HUSE(buck_list, p_list, l_list, len_list)
    
    print("For the task of {}".format(taskname))
    print("Overall HUSE score is: {}".format(HUSE))
    print("HUSE-Q (just human) score is: {}".format(HUSEQ))
    print("HUSE-D score is: {}".format(HUSED))
    plot(taskname, buck_list, p_list, l_list, len_list)

def main():
    filename = "../data/summarization_raw_outputs.csv"
    basics(filename)
    detailed_analysis(filename)

def detailed_analysis(filename):
    classifier = KNeighborsClassifier(n_neighbors=15, algorithm='brute')

    # Load data from MTURK output csv
    logps, bucket, labels, lens, ctxs, outs = extract_ctx(filename)
    buck_list, p_list, l_list, len_list, ctx_list, ex_list = get_data_ctx(logps, bucket, labels, lens, types, ctxs, outs)
    
    # Find examples of model failures
    perr_list, ev_list, examples_list = classify_examples(buck_list, p_list, l_list, len_list, ctx_list, ex_list, classifier)
    locate_model_failures(perr_list, ev_list, examples_list)

    # Run experiments on replicates needed
    all_accs_list, n_samp_test = replicate_exps(logps, bucket, labels, lens, classifier)
    plot_replicates(all_accs_list, n_samp_test)

if __name__ == "__main__":
    main()
