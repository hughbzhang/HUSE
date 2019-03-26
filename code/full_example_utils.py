import os
import matplotlib.pyplot as plt
import csv
import numpy as np
from collections import defaultdict
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
import matplotlib as mpl
mpl.use('Agg')

types = ["SUM", "HUMAN_SUM"]
bucket = ["Invalid", "Rare", "Specific", "Average", "Typical", "Very Typical"]
unbucket = {}
for index, buck in enumerate(bucket):
    unbucket[buck] = index + 1

def classify_examples(buck_list, p_list, l_list, len_list, ctx_list, ex_list, classifier):
    np.array(l_list)
    np_list = np.array(p_list) / np.array(len_list)
    x_mat = np.vstack([np_list/np.std(np_list), buck_list/np.std(buck_list) ]).transpose()
    y_vec = np.array(l_list)
    pred_vec = []
    for train_index, test_index in LeaveOneOut().split(x_mat):
        X_train, X_test = x_mat[train_index], x_mat[test_index]
        y_train, y_test = y_vec[train_index], y_vec[test_index]
        _ = classifier.fit(X_train, y_train)
        pred_vec.append(2.0*(classifier.predict_proba(X_test)[0,1]-0.5)*(2.0*y_test - 1.0))

    examples_list = []
    perr_list = []
    ev_list = []
    for key, value in ctx_list.items():
        if l_list[value[0]] == 0:
            id_mod = 1
            id_hum = 0
        else:
            id_mod = 0
            id_hum = 1

        exl = (ex_list[value[id_mod]][0], ex_list[value[id_mod]][1], ex_list[value[id_hum]][1])
        pvl = (pred_vec[value[id_mod]][0], pred_vec[value[id_hum]][0])
        evl = (np_list[value[id_mod]], np_list[value[id_hum]], buck_list[value[id_mod]], buck_list[value[id_hum]])
        examples_list.append(exl)
        ev_list.append(evl)
        perr_list.append(pvl)

    return perr_list, ev_list, examples_list

def replicate_exps(logps, bucket, labels, lens, classifier):
    np.random.seed(0)
    reps = 20
    all_accs_list = []
    acc_v_list = []
    nrep = 15
    for j in range(reps):
        sub_acc = []
        for k in range(nrep):
            buck_list, p_list, l_list, len_list = get_data_rand(logps, bucket, labels, lens, types, nrep_max=j+1)
            np_list = np.array(p_list)/np.array(len_list)
            x_mat = np.vstack([buck_list/np.std(buck_list), np_list/np.std(np_list)]).transpose()
            y_vec = np.array(l_list)
            sub_acc.append(np.mean(get_accs(classifier, x_mat, y_vec)))
        acc_v_list.append(np.mean(sub_acc))
    acc_n_list = []
    nrep = 30
    n_samp_test = np.linspace(50, 200, 11)
    for nsamp in n_samp_test:
        sub_acc = []
        buck_list, p_list, l_list, len_list = get_data(logps, bucket, labels, lens, types, nrep_max=40)
        for k in range(nrep):
            idx_set_1 = np.random.choice(np.nonzero(l_list)[0], int(nsamp/2), replace=False)
            idx_set_2 = np.random.choice(np.nonzero(np.array(l_list)==False)[0], int(nsamp/2), replace=False)
            idx_set = np.concatenate([idx_set_1, idx_set_2])
            bsub = np.array(buck_list)[idx_set]
            psub = np.array(p_list)[idx_set]
            npsub = psub / (np.array(len_list)[idx_set])
            x_mat = np.vstack([bsub/np.std(bsub), npsub/np.std(npsub)]).transpose()
            y_vec = np.array(l_list)[idx_set]
            sub_acc.append(np.mean(get_accs(classifier, x_mat, y_vec)))
        acc_n_list.append(np.mean(sub_acc))
    all_accs_list.append((acc_v_list, acc_n_list))
    return all_accs_list, n_samp_test

def plot_replicates(all_accs_list, n_samp_test):
    fig, (ax1, ax2) = plt.subplots(1,2 , sharey=True, figsize=(5,2.5))

    ax1.plot(np.linspace(50, 200, 11), all_accs_list[0][1],linewidth=2, color='red',label='summarization')
    ax1.set_xlabel('Number of examples in test set')
    ax1.set_ylabel('Average classification accuracy')
    ax1.set_xticks(np.floor(n_samp_test[np.linspace(0,len(n_samp_test)-1,5).astype(int)]))

    ax2.plot(np.arange(20,dtype=int)+1, all_accs_list[0][0],linewidth=2, color='red',label='summarization')
    ax2.set_xlabel('Number of replicates')
    ax2.set_xticks(np.linspace(0,20,5))
    ax2.legend(loc='lower right')

    fig.suptitle('Effect of experiment design on accuracy')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust(hspace=0.01)
    plt.savefig('../figures/EXPERIMENT_DESIGN.pdf')
    plt.close()

def extract(filename):
    reverse = {}
    with open(filename) as fopen:
        read = csv.reader(fopen)
        header = next(read)
        for j, x in enumerate(header):
            reverse[x] = j
        logps = defaultdict(lambda: [])
        lens = defaultdict(lambda: [])
        bucket = defaultdict(lambda: [])
        labels = dict()
        for arr in read:
            hitid = arr[reverse["HITId"]]
            who = arr[reverse["Input.TYPE0"]]
            for i in range(25):
                slen = len(arr[reverse["Input.OUT{}".format(i)]].split(' '))
                logp = float(arr[reverse["Input.LOGP{}".format(i)]])#/float(slen)
                bucket_guess = unbucket[arr[reverse["Answer.{}".format(i)]]]
                labels[(hitid, i, who)] = who
                lens[(hitid, i, who)] = slen
                logps[(hitid, i, who)].append(logp)
                bucket[(hitid, i, who)].append(bucket_guess)
        return logps, bucket, labels, lens

def get_data(logps, bucket, labels, lens, types, nrep_max = 20):
    l_list = []
    len_list = []
    p_list = []
    buck_list = []
    for key in sorted(logps.keys()):
        if (labels[key] == types[0]) or (labels[key] == types[1]):
            p = np.mean(logps[key][0:nrep_max])
            buck = np.mean(bucket[key][0:nrep_max])
            len_list.append(lens[key])
            l_list.append(labels[key] == types[0])
            if np.isinf(p):
                p_list.append(-1000)
            else:
                p_list.append(p)
            buck_list.append(buck)
    return buck_list, p_list, l_list, len_list

def get_accs(clf, x_mat, y_vec):
    loo = LeaveOneOut()
    acc_vec = []
    for train_index, test_index in loo.split(x_mat):
        X_train, X_test = x_mat[train_index], x_mat[test_index]
        y_train, y_test = y_vec[train_index], y_vec[test_index]
        _ = clf.fit(X_train, y_train)
        acc_vec.append(clf.predict(X_test)==y_test[0])
    return acc_vec

def get_stats(acc_all, acc_hum, acc_log):
    sub_vec = [np.mean(acc_all), np.mean(acc_hum), np.mean(acc_log)]
    acc_est = np.mean(acc_all)
    half_tv_est = 2.0*acc_est - 1.0
    quality_term = 2.0*np.mean(acc_hum) - 1.0
    diversity_term = half_tv_est - quality_term
    return half_tv_est, diversity_term, quality_term, sub_vec

def opt_thresh(x, y):
    np.argsort(x)
    acmax = 0.0
    for xi in np.sort(x):
        acc_1 = (np.sum(y[x >= xi]) + np.sum((1-y)[x < xi]))/float(len(x))
        acc_2 = (np.sum((1-y)[x >= xi]) + np.sum(y[x < xi]))/float(len(x))
        if acc_1 > acmax:
            acmax = acc_1
        if acc_2 > acmax:
            acmax = acc_2
    return acmax

def get_data_ctx(logps, bucket, labels, lens, types, ctx, outs):
    l_list = []
    len_list = []
    p_list = []
    buck_list = []
    ctx_ind = {}
    i = 0
    j = 0
    ctx_list = defaultdict(list)
    ex_list = []
    for key in sorted(logps.keys()):
        if (labels[key] == types[0]) or (labels[key] == types[1]):
            p = np.mean(logps[key])
            buck = np.mean(bucket[key])
            if ctx[key] in ctx_ind:
                ctx_list[ctx_ind[ctx[key]]].append(j)
            else:
                ctx_list[i].append(j)
                ctx_ind[ctx[key]] = i
                i = i+1
            len_list.append(lens[key])
            l_list.append(labels[key] == types[0])
            if np.isinf(p):
                p_list.append(-1000)
            else:
                p_list.append(p)
            buck_list.append(buck)
            ex_list.append((ctx[key], outs[key]))
            j = j + 1
    return buck_list, p_list, l_list, len_list, ctx_list, ex_list

def get_data_rand(logps, bucket, labels, lens, types, nrep_max = 20):
    l_list = []
    p_list = []
    len_list = []
    buck_list = []
    for key in logps.keys():
        p = np.mean(np.random.choice(logps[key],nrep_max, replace=False)) 
        buck = np.mean(np.random.choice(bucket[key],nrep_max, replace=False))
        len_list = lens[key]
        l_list.append(labels[key] == types[0])
        if np.isinf(p):
            p_list.append(-1000)
        else:
            p_list.append(p)
        buck_list.append(buck)
    return buck_list, p_list, l_list, len_list

def print_matrix(ex_list, ev_list, idx):
    np.array(ex_list)[idx]
    np.array(ev_list)[idx]
    str_list = []
    for i in idx:
        examples = (ex_list[i][0], ex_list[i][1], ev_list[i][0], ev_list[i][2], ex_list[i][2], ev_list[i][1], ev_list[i][3])
        str_list.append('\t'.join([str(substr) for substr in examples]))
    return str_list

def dump_file(ex_list, ev_list, idx, filename):
    with open(os.path.join("../records/", filename),'w') as fopen:
        fopen.write('\n'.join(print_matrix(ex_list, ev_list, idx)))

def locate_model_failures(perr_list, ev_list, examples_list):
    perr_arr = np.array(perr_list)
    number_examples = 15
    output_incorrect = np.argsort(np.sum(perr_arr,axis = 1))[0:number_examples]
    output_correct = np.argsort(-1*np.sum(perr_arr,axis = 1))[0:number_examples]
    div_idx = np.nonzero(np.abs(np.array(ev_list)[:,3]-np.array(ev_list)[:,2]) < 0.5)[0]
    output_correct_div = div_idx[np.argsort(-1*np.sum(perr_arr,axis = 1)[div_idx])[0:number_examples]]

    output_indisting = np.argsort(np.sum(np.abs(perr_arr),axis = 1))[0:number_examples]

    dump_file(examples_list, ev_list, output_correct, 'correct_examples.txt')
    dump_file(examples_list, ev_list, output_incorrect, 'incorrect_examples.txt')
    dump_file(examples_list, ev_list, output_correct_div, 'correct_examples_diversity.txt')
    dump_file(examples_list, ev_list, output_indisting, 'indistinguishable_examples.txt')

def extract_ctx(filename):
    reverse = {}
    with open(filename) as fopen:
        read = csv.reader(fopen)
        header = next(read)
        for j, x in enumerate(header):
            reverse[x] = j
        logps = defaultdict(lambda: [])
        lens = defaultdict(lambda: [])
        bucket = defaultdict(lambda: [])
        labels = dict()
        ctxs = dict()
        outs = dict()
        for arr in read:
            hitid = arr[reverse["HITId"]]
            who = arr[reverse["Input.TYPE0"]]
            for i in range(25):
                slen = len(arr[reverse["Input.OUT{}".format(i)]].split(' '))
                logp = float(arr[reverse["Input.LOGP{}".format(i)]])
                bucket_guess = unbucket[arr[reverse["Answer.{}".format(i)]]]
                ctxs[(hitid, i, who)] = arr[reverse["Input.CTX{}".format(i)]]
                outs[(hitid, i, who)] = arr[reverse["Input.OUT{}".format(i)]]
                labels[(hitid, i, who)] = who
                lens[(hitid, i, who)] = slen
                logps[(hitid, i, who)].append(logp)
                bucket[(hitid, i, who)].append(bucket_guess)
                
        return logps, bucket, labels, lens, ctxs, outs
