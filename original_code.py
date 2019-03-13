#from IPython import embed
from matplotlib.pyplot import cm
from scipy.stats import spearmanr, kendalltau, pearsonr
import math
import matplotlib.pyplot as plt
import matplotlib
import csv
import numpy as np
import collections
from collections import defaultdict

bucket = ["Invalid", "Rare", "Specific", "Average", "Typical", "Very Typical"]
unbucket = {}
for index, buck in enumerate(bucket):
    unbucket[buck] = index + 1


def extract(fname):
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


from scipy.stats import trim_mean

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


from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import quantile_transform
from sklearn.linear_model import LogisticRegression
from scipy.stats import rankdata

def get_accs(clf, x_mat, y_vec):
    loo = LeaveOneOut()
    acc_vec = []
    for train_index, test_index in loo.split(x_mat):
        X_train, X_test = x_mat[train_index], x_mat[test_index]
        y_train, y_test = y_vec[train_index], y_vec[test_index]
        _ = clf.fit(X_train, y_train)
        acc_vec.append(clf.predict(X_test)==y_test[0])
    return acc_vec

def splice_files(input_file, file_partial, file_full):
    """ makes a spliced human eval set, where the logp is with respect to input_file, the human eval is taken from file_full, and the evaluation on the model generation is taken from file_partial"""
    output_file = file_partial[:-4]+'_spliced.csv'
    # load and store logps from input_file
    logp_dict = {}
    reverse = {}
    fopen = open(input_file)
    read = csv.reader(fopen)
    code = next(read)
    for j, x in enumerate(code):
        reverse[x] = j
    for line in read:
        for i in range(25):
            cin = line[reverse['CTX'+str(i)]].replace('<unk>','UNKNOWN')
            cin = cin.replace("\"","")
            cin = cin.replace("  "," ")
            lpout = line[reverse['LOGP'+str(i)]]
            typ = line[reverse['TYPE'+str(i)]].split('_')[0]
            logp_dict[(cin, typ)] = lpout
    fopen.close()
    # go through file_full and splice the logps into the human eval
    reverse = {}
    fopen = open(file_full)
    read = csv.reader(fopen)
    code = next(read)
    for j, x in enumerate(code):
        reverse[x] = j
    nlines = [','.join(code)]
    for line in read:
        if 'HUMAN' in line[reverse['Input.TYPE0']]:
            nline = line
            for i in range(25):
                cin = line[reverse['Input.CTX'+str(i)]]
                cin = cin.replace("\"","")
                cin = cin.replace("  "," ")
                typ = line[reverse['Input.TYPE'+str(i)]].split('_')[0]
                try:
                    nline[reverse['Input.LOGP'+str(i)]] = logp_dict[(cin, typ)]
                except:
                    print(logp_dict)
            nline = ["\""+item+"\"" for item in nline]
            nlines.append(','.join(nline))
    # read contents of file_partial (human eval of model generated text) - just combine this directly
    fopen = open(file_partial)
    code = next(fopen)
    for line in fopen:
        nlines.append(line.strip())
    fopen.close()
    fopen = open(output_file,'wt')
    fopen.write('\n'.join(nlines))
    fopen.close()

splice_list = [['../MTURK_inputs/pretrained-turk07.csv', '../MTURK_results/summarization_round2_temp_patched.csv', '../MTURK_results/summarization_round2_results_patched.csv'],
               ['../MTURK_inputs/roc-turk.csv', '../MTURK_results/roc_round2_notemp_patched.csv',               '../MTURK_results/roc_round2_results_patched.csv'],
               ['../MTURK_inputs/pretrained-turk09.csv', '../MTURK_results/summarization_round2_sum09.csv', '../MTURK_results/summarization_round2_results_patched.csv'],
               ['../MTURK_inputs/giga-ret-turk-top1.txt', '../MTURK_results/summarization_round2_retrieval.csv', '../MTURK_results/summarization_round2_results_patched.csv'],
               ['../MTURK_inputs/roc-ret-turk-top1.txt', '../MTURK_results/roc_round2_retrieval.csv',                 '../MTURK_results/roc_round2_results_patched.csv']
               ]

for splice_target in splice_list:
    print(splice_target[0])
    #splice_files(splice_target[0], splice_target[1], splice_target[2])    

run_list = [["summarization", "../MTURK_results/summarization_round2_results_patched.csv", ["SUM", "HUMAN_SUM"]],
            ["summarization t=0.7", "../MTURK_results/summarization_round2_temp_patched_spliced.csv", ["SUM07", "HUMAN_SUM"]],
            ["summarization t=0.9", "../MTURK_results/summarization_round2_sum09_spliced.csv", ["SUM09", "HUMAN_SUM"]],
            ["summarization (ret)", "../MTURK_results/summarization_round2_retrieval_spliced.csv", ["SUMRET", "HUMAN_SUM"]],
            ["ROC stories (overfit)","../MTURK_results/roc_round2_results_patched.csv", ["ROCLONG", "HUMAN_ROCLONG"]],
            ["ROC stories","../MTURK_results/roc_round2_notemp_patched_spliced.csv", ["ROC","HUMAN_ROCLONG"]],
            ["ROC stories (retrieval)","../MTURK_results/roc_round2_retrieval_spliced.csv", ["ROCRET","HUMAN_ROCLONG"]],
            ["Language modeling","../MTURK_results/lm_results.csv", ["LM","HUMAN_LM"]],
            ["Reddit dialogue t=1.0","../MTURK_results/reddit_results_reformatted.csv", ["LOGP_1.0.txt","HUMAN_REDDIT1.0"]],
            ["Reddit dialogue t=0.7","../MTURK_results/reddit_results_reformatted.csv", ["LOGP_0.7.txt","HUMAN_REDDIT0.7"]],
            ["Reddit dialogue t=0.5","../MTURK_results/reddit_results_reformatted.csv", ["LOGP_0.5.txt","HUMAN_REDDIT0.5"]],
            ]
#            ["roc stories t=0.5","../MTURK_results/ROC_results.csv", ["MODEL05", "HUMAN"]]]


def get_stats(acc_all, acc_hum, acc_log):
    sub_vec = [np.mean(acc_all), np.mean(acc_hum), np.mean(acc_log)]
    #acc_est = np.max(sub_vec)
    acc_est = np.mean(acc_all)
    half_tv_est = 2.0*acc_est - 1.0
    quality_term = 2.0*np.mean(acc_hum) - 1.0
    diversity_term = half_tv_est - quality_term
    return half_tv_est, diversity_term, quality_term, sub_vec

def opt_thresh(x, y):
    x_srt = np.argsort(x)
    acmax = 0.0
    for xi in np.sort(x):
        acc_1 = (np.sum(y[x >= xi]) + np.sum((1-y)[x < xi]))/float(len(x))
        acc_2 = (np.sum((1-y)[x >= xi]) + np.sum(y[x < xi]))/float(len(x))
        if acc_1 > acmax:
            acmax = acc_1
        if acc_2 > acmax:
            acmax = acc_2
    return acmax

class MyKNNClassifier:
    def __init__(self, n_neighbors, algorithm):
        self.n_nb = n_neighbors

    def fit(self, x, y):
        self.x = x
        self.y = y

    def predict(self, x):
        dists = np.sum((self.x - x)**2.0,axis=1)
        dcut = np.sort(dists)[self.n_nb]
        return np.bincount(self.y[dists <= dcut]).argmax()

    def predict_proba(self, x):
        dists = np.sum((self.x - x)**2.0,axis=1)
        dcut = np.sort(dists)[self.n_nb]
        return np.bincount(self.y[dists <= dcut])/np.sum(self.y[dists <= dcut])



#clf = KNeighborsClassifier(n_neighbors=15,algorithm='brute')
clf = MyKNNClassifier(n_neighbors=15,algorithm='brute')
stat_list = []
#run_list = [run_list[-2]]
for run in run_list:
    print(run[0])
    filename = run[1]
    logps, bucket, labels, lens = extract(filename)
    types = run[2]
    buck_list, p_list, l_list, len_list = get_data(logps, bucket, labels, lens, types, nrep_max=40)
    y_vec = np.array(l_list)
    np_list = np.array(p_list)/np.array(len_list)
    #x_mat = np.vstack([buck_list/np.std(buck_list), p_list/np.std(p_list)]).transpose()
    #acc_vec_all = get_accs(clf, x_mat, y_vec)
    nx_mat = np.vstack([buck_list/np.std(buck_list), np_list/np.std(np_list)]).transpose()
    acc_vec_nall = get_accs(clf, nx_mat, y_vec)
    #print(np.mean(acc_vec_nall))
    #print(np.mean(acc_vec_all))
    #if True or np.mean(acc_vec_nall) > np.mean(acc_vec_all):
    #print('switch')
    acc_vec_all = acc_vec_nall
    x_mat = nx_mat
    acc_vec_hum = get_accs(clf, x_mat[:,0][:,np.newaxis], y_vec)
    acc_vec_logp = get_accs(clf, x_mat[:,1][:,np.newaxis], y_vec)
    statsout = get_stats(acc_vec_all, acc_vec_hum, acc_vec_logp)
    stat_list.append(statsout)
    print(statsout)

with open('summary.csv','wt') as fopen:
    hdr = ['name','tv-hp','tv-div','tv-h','acc-hp','acc-h','acc-p']
    fopen.write(','.join(hdr)+'\n')
    for i, statline in enumerate(stat_list):
        num_list = list(statline[0:3])+statline[3]
        num_str = [str(sub) for sub in num_list]
        fopen.write(','.join([run_list[i][0]]+num_str)+'\n')

clf = KNeighborsClassifier(n_neighbors=15,algorithm='brute')
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


for i in range(len(run_list)):
    filename = run_list[i][1]
    logps, bucket, labels, lens = extract(filename)
    buck_list, p_list, l_list, len_list = get_data(logps, bucket, labels, lens, run_list[i][2], nrep_max=40)
    larray =np.array(l_list)
    np_list = np.array(p_list)/np.array(len_list)
    pin = np_list
    #x_mat = np.vstack([p_list/np.std(p_list), buck_list/np.std(buck_list) ]).transpose()
    x_mat = np.vstack([pin/np.std(pin), buck_list/np.std(buck_list) ]).transpose()
    y_vec = np.array(l_list)
    clf.fit(x_mat,y_vec)
    x_min, x_max = x_mat[:,0].min() - .5, x_mat[:, 0].max() + .5
    y_min, y_max = x_mat[:, 1].min() - .5, x_mat[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2),
                         np.arange(y_min, y_max, 0.2))
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    xx_1, yy_1 = np.meshgrid(np.arange(x_min, x_max, 0.2)*np.std(pin),
                             np.arange(y_min, y_max, 0.2)*np.std(buck_list))
    cm = plt.cm.RdBu
    fig, ax = plt.subplots(figsize=(4.5,3.5))
    ax.contourf(xx_1, yy_1, Z, cmap=cm, alpha=.6)
    ax.scatter(np.array(pin)[np.nonzero(larray)[0]], np.array(buck_list)[np.nonzero(larray)[0]],color='blue',edgecolor='black',label='Model',marker='s')
    ax.scatter(np.array(pin)[np.nonzero(larray==False)[0]],np.array(buck_list)[np.nonzero(larray==False)[0]], color='red',edgecolor='black',label='Human')
    divider = make_axes_locatable(ax)
    axHistx = divider.append_axes("top", 0.7, pad=0.0, sharex=ax)
    axHisty = divider.append_axes("right", 0.7, pad=0.0, sharey=ax)
    # make some labels invisible
    axHistx.xaxis.set_tick_params(labelbottom=False,bottom='off')
    axHistx.yaxis.set_tick_params(labelleft=False,left='off')
    axHisty.xaxis.set_tick_params(labelbottom=False,bottom='off')
    axHisty.yaxis.set_tick_params(labelleft=False,left='off')
    # now determine nice limits by hand:
    b_subset = np.nonzero(larray)[0]
    r_subset = np.nonzero(larray==False)[0]
    x = np.array(pin)
    y = np.array(buck_list)
    _ = axHistx.hist(x[b_subset], color='blue', bins=np.arange(x_min, x_max, 0.3)*np.std(pin), alpha=0.7)
    _ = axHistx.hist(x[r_subset], color='red', bins=np.arange(x_min, x_max, 0.3)*np.std(pin), alpha=0.7)
    _ = axHisty.hist(y[b_subset], color='blue', bins=np.arange(y_min, y_max, 0.3)*np.std(buck_list), orientation='horizontal', alpha=0.7)
    _ = axHisty.hist(y[r_subset], color='red', bins=np.arange(y_min, y_max, 0.3)*np.std(buck_list), orientation='horizontal', alpha=0.7)
    #
    ax.legend(loc='lower left')
    ax.set_xlim(np.min(xx_1), np.max(xx_1))
    ax.set_ylim(np.min(yy_1), np.max(yy_1))
    ax.set_xlabel('Model log likelihood')
    ax.set_ylabel('Human evaluation')
    axHistx.set_title(run_list[i][0])
    plt.tight_layout()
    plt.savefig('../figs/task-scatter'+str(i)+'.pdf')
    #plt.savefig('../figs/task-scatter'+str(i)+'.png')



###
# Detailed analysis

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

def extract_ctx(fname):
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
                logp = float(arr[reverse["Input.LOGP{}".format(i)]])#/float(slen)
                bucket_guess = unbucket[arr[reverse["Answer.{}".format(i)]]]
                ctxs[(hitid, i, who)] = arr[reverse["Input.CTX{}".format(i)]]
                outs[(hitid, i, who)] = arr[reverse["Input.OUT{}".format(i)]]
                labels[(hitid, i, who)] = who
                lens[(hitid, i, who)] = slen
                logps[(hitid, i, who)].append(logp)
                bucket[(hitid, i, who)].append(bucket_guess)
        return logps, bucket, labels, lens, ctxs, outs


i=0
filename = run_list[i][1]
logps, bucket, labels, lens, ctxs, outs = extract_ctx(filename)
buck_list, p_list, l_list, len_list, ctx_list, ex_list = get_data_ctx(logps, bucket, labels, lens, run_list[i][2], ctxs, outs)
larray =np.array(l_list)
np_list = np.array(p_list)/np.array(len_list)
pin = np_list
x_mat = np.vstack([pin/np.std(pin), buck_list/np.std(buck_list) ]).transpose()
y_vec = np.array(l_list)
loo = LeaveOneOut()
pred_vec = []
for train_index, test_index in loo.split(x_mat):
    X_train, X_test = x_mat[train_index], x_mat[test_index]
    y_train, y_test = y_vec[train_index], y_vec[test_index]
    _ = clf.fit(X_train, y_train)
    pred_vec.append(2.0*(clf.predict_proba(X_test)[0,1]-0.5)*(2.0*y_test - 1.0))

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

perr_arr = np.array(perr_list)

numex = 15
o_flip = np.argsort(np.sum(perr_arr,axis = 1))[0:numex]
o_corr = np.argsort(-1*np.sum(perr_arr,axis = 1))[0:numex]
div_idx = np.nonzero(np.abs(np.array(ev_list)[:,3]-np.array(ev_list)[:,2]) < 0.5)[0]
o_corr_div = div_idx[np.argsort(-1*np.sum(perr_arr,axis = 1)[div_idx])[0:numex]]

o_ind = np.argsort(np.sum(np.abs(perr_arr),axis = 1))[0:numex]

def print_matrix(ex_list, ev_list, idx):
    m1 = np.array(ex_list)[idx]
    m2 = np.array(ev_list)[idx]
    str_list = []
    for i in idx:
        examples = (ex_list[i][0], ex_list[i][1], ev_list[i][0], ev_list[i][2], ex_list[i][2], ev_list[i][1], ev_list[i][3])
        str_list.append('\t'.join([str(substr) for substr in examples]))
    return str_list

def dump_file(ex_list, ev_list, idx, filename):
    with open(filename,'w') as fopen:
        fopen.write('\n'.join(print_matrix(ex_list, ev_list, idx)))

dump_file(examples_list, ev_list, o_corr, 'correct_examples.txt')
dump_file(examples_list, ev_list, o_flip, 'incorrect_examples.txt')
dump_file(examples_list, ev_list, o_corr_div, 'correct_examples_diversity.txt')
dump_file(examples_list, ev_list, o_ind, 'indistinguishable_examples.txt')

###
# triangle plot

from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon

def glyphplot(x,y, s, c, m, ax):
    um = np.unique(m)
    for mtype in um:
        xs = np.array(x)[np.array(m) == mtype]
        ys = np.array(y)[np.array(m) == mtype]
        ax.scatter(xs, ys, s=s, color=c, marker=mtype)


plt.figure(figsize=(6.5,3.5))

ax = plt.subplot(111)
ax.plot((1,0),(0,1), color='gray')
for sp in np.linspace(0,1, 11):
    ax.plot((sp,0),(0,sp), color='lightgray', zorder=0, alpha=0.5)

idx_list = {'HUMAN_SUM': [0, 1, 2, 3], 'HUMAN_ROCLONG': [4, 5, 6], 'HUMAN_LM': [7], 'HUMAN_REDDIT': [8,9,10]}

cdict = {'HUMAN_SUM':'red', 'HUMAN_ROCLONG':'blue','HUMAN_REDDIT':'green'}
shorthands = ['1.0','0.7','0.9','ret','*','1.0','ret','lm','1.0','0.7','0.5']
glist = ['o','o','o','s','*','o','s','.','o','o','o']
mcenter = {'HUMAN_SUM':(0.75,0.175), 'HUMAN_ROCLONG':(0.14,0.8), 'HUMAN_REDDIT':(0.2,0.5)}
madjust = {'HUMAN_SUM':(0.8,0.3), 'HUMAN_ROCLONG':(0.2,0.9), 'HUMAN_REDDIT':(0.55,0.55)}
m_name = {'HUMAN_SUM':'summarization', 'HUMAN_ROCLONG':'story generation', 'HUMAN_REDDIT':'chitchat dialogue'}
#dict([(rsub[2][1],i) for i, rsub in enumerate(run_list)])
xtot, xin, yin, slist = zip(*stat_list)
for mtype in idx_list.keys():
    sub_idx_list = idx_list[mtype]
    if len(sub_idx_list) > 1:
        coords = np.vstack([np.array(xin)[sub_idx_list],np.array(yin)[sub_idx_list]])
        c_aug = coords/np.sum(coords,axis=0)
        cfull = np.vstack([coords.transpose(), c_aug.transpose()])
        chull = ConvexHull(cfull)
        ax.add_patch(Polygon(cfull[chull.vertices,:],color=cdict[mtype],alpha=0.2))
        #ax.scatter(coords[0,:], coords[1,:], s=100,color=cdict[mtype])
        glyphplot(coords[0,:], coords[1,:], 100, cdict[mtype], np.array(glist)[sub_idx_list], ax)
        for i in sub_idx_list:
            if glist[i] == 'o':
                ax.annotate(shorthands[i], xy=(xin[i]+0.01,yin[i]+0.01),size=10)
        adj = madjust[mtype]
        ax.annotate(m_name[mtype], xy=mcenter[mtype], xytext = madjust[mtype],arrowprops=dict(facecolor='black', frac=0.2, width=0.02,headwidth=8.0, alpha=0.7),size=15,color=cdict[mtype])


ax.scatter(xin[idx_list['HUMAN_LM'][0]], yin[idx_list['HUMAN_LM'][0]], s=100, color='purple', marker='o')
ax.annotate('language modeling', xy=(xin[idx_list['HUMAN_LM'][0]], yin[idx_list['HUMAN_LM'][0]]), xytext = (1.0,0.1),arrowprops=dict(facecolor='black', frac=0.2, width=0.02,headwidth=8.0, alpha=0.7),size=15,color='purple')

from matplotlib.lines import Line2D

legend_elements = [Line2D([0], [0], marker='o', color='w', label='NLM (w/ temperature)',
                          markerfacecolor='black', markersize=10),
                   Line2D([0], [0], marker='*', color='w', label='NLM (overfitted)',
                          markerfacecolor='black', markersize=10),
                   Line2D([0], [0], marker='s', color='w', label='Retrieval',
                          markerfacecolor='black', markersize=10)]

# Create the figure
ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.5,1.0))
#plt.legend(loc='top right')
ax.set_xlim((-0.0,1))
ax.set_ylim((-0.0,1))
ax.set_xlabel('Diversity error')
ax.set_ylabel('Quality error')

# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.tight_layout()
plt.savefig('../figs/div-qual.pdf')
#plt.savefig('../figs/div-qual.png')




###
# number of replicates / samples needed

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

np.random.seed(0)
run_sub = [0, 5, 7, 9]
rep_list = [20, 20 , 20, 20]
nsmax = [200, 200, 100, 200]
all_accs_list = []
for i, idx in enumerate(run_sub):
    print(idx)
    run = run_list[idx]
    filename = run[1]
    logps, bucket, labels, lens = extract(filename)
    types = run[2]
    acc_v_list = []
    nrep = 15
    for j in range(rep_list[i]):
        sub_acc = []
        for k in range(nrep):
            buck_list, p_list, l_list, len_list = get_data_rand(logps, bucket, labels, lens, types, nrep_max=j+1)
            np_list = np.array(p_list)/np.array(len_list)
            x_mat = np.vstack([buck_list/np.std(buck_list), np_list/np.std(np_list)]).transpose()
            y_vec = np.array(l_list)
            sub_acc.append(np.mean(get_accs(clf, x_mat, y_vec)))
        acc_v_list.append(np.mean(sub_acc))
    acc_n_list = []
    nrep = 30
    n_samp_test = np.linspace(50, nsmax[i], 11)
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
            sub_acc.append(np.mean(get_accs(clf, x_mat, y_vec)))
        acc_n_list.append(np.mean(sub_acc))
    all_accs_list.append((acc_v_list, acc_n_list))


fig, (ax1, ax2) = plt.subplots(1,2 , sharey=True, figsize=(5,2.5))

ax1.plot(np.linspace(50, nsmax[0], 11), all_accs_list[0][1],linewidth=2, color='red',label='summarization')
ax1.plot(np.linspace(50, nsmax[1], 11), all_accs_list[1][1],linewidth=2, color='blue',label='story generation')
ax1.plot(np.linspace(50, nsmax[2], 11), all_accs_list[2][1],linewidth=2, color='purple',label='language modeling')
ax1.set_xlabel('Number of examples in test set')
ax1.set_ylabel('Average classification accuracy')
ax1.set_xticks(np.floor(n_samp_test[np.linspace(0,len(n_samp_test)-1,5).astype(int)]))
#plt.title('Effect of replicate human judgements on ')

ax2.plot(np.arange(20,dtype=int)+1, all_accs_list[0][0],linewidth=2, color='red',label='summarization')
ax2.plot(np.arange(20,dtype=int)+1, all_accs_list[1][0],linewidth=2, color='blue',label='story generation')
ax2.plot(np.arange(20,dtype=int)+1, all_accs_list[2][0],linewidth=2, color='purple',label='language modeling')
ax2.set_xlabel('Number of replicates')
ax2.set_xticks(np.linspace(0,20,5))
ax2.legend(loc='bottom right')

fig.suptitle('Effect of experiment design on accuracy')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.subplots_adjust(hspace=0.01)
plt.savefig('../figs/exptdesign.pdf')
plt.close()



