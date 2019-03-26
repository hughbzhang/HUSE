import matplotlib as mpl
from IPython import embed
mpl.use('Agg')
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier

# All arrays are 1x2N where N is the number of samples in test set.
# HJ is human judgements
# model_probs contains model probabilities for each utterance
# labels denotes whether an utterance is generated or real
# length_list records the number of tokens in each utterance
def calculate_HUSE(HJ, model_probs, labels, length_list):
    classifier = KNNClassifier(15)

    y_vec = np.array(labels)

    # Normalize p_model by number of tokens
    nmodel_probs = np.array(model_probs)/np.array(length_list)
    features = np.vstack([HJ/np.std(HJ), nmodel_probs/np.std(nmodel_probs)]).transpose()

    # Accuracy with both pmodel and HJ
    acc_vec_all = get_accuracy(classifier, features, y_vec)

    # Accuracy with just HJ
    acc_vec_hum = get_accuracy(classifier, features[:,0][:,np.newaxis], y_vec)

    # Accuracy with just pmodel (unused for HUSE)
    acc_vec_logp = get_accuracy(classifier, features[:,1][:,np.newaxis], y_vec)
    statsout = get_stats(acc_vec_all, acc_vec_hum, acc_vec_logp)

    HUSE = 1.-statsout[0]
    HUSEQ = 1.-statsout[2]
    HUSED = 1-statsout[0]+statsout[2]

    return HUSE, HUSEQ, HUSED

def get_accuracy(classifier, features, labels):
    acc_vec = []
    for train_index, test_index in LeaveOneOut().split(features):
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        _ = classifier.fit(X_train, y_train)
        acc_vec.append(classifier.predict(X_test) == y_test[0])
    return acc_vec

def get_stats(acc_all, acc_hum, acc_log):
    sub_vec = [np.mean(acc_all), np.mean(acc_hum), np.mean(acc_log)]
    acc_est = np.mean(acc_all)
    half_tv_est = 2.0*acc_est - 1.0
    quality_term = 2.0*np.mean(acc_hum) - 1.0
    diversity_term = half_tv_est - quality_term
    return half_tv_est, diversity_term, quality_term, sub_vec

# We implement our own KNN to deal with ties.
class KNNClassifier:
    def __init__(self, num_neighbors):
        self.num_neighbors = num_neighbors

    def fit(self, features, labels):
        self.features = features
        self.labels = labels

    def predict(self, datapoint):
        dists = np.sum((self.features - datapoint)**2.0,axis=1)
        dcut = np.sort(dists)[self.num_neighbors]
        return np.bincount(self.labels[dists <= dcut]).argmax()

    def predict_proba(self, datapoint):
        dists = np.sum((self.features - datapoint)**2.0,axis=1)
        dcut = np.sort(dists)[self.num_neighbors]
        return np.bincount(self.labels[dists <= dcut])/np.sum(self.y[dists <= dcut])

# Here be dragons.
def plot(taskname, HJ, model_probs, labels, length_list):
    classifier = KNeighborsClassifier(n_neighbors=15, algorithm='brute')

    larray = np.array(labels)
    nmodel_probs = np.array(model_probs)/np.array(length_list)
    pin = nmodel_probs
    x_mat = np.vstack([pin/np.std(pin), HJ/np.std(HJ) ]).transpose()
    y_vec = np.array(labels)
    
    classifier.fit(x_mat,y_vec)
    x_min, x_max = x_mat[:,0].min() - .5, x_mat[:, 0].max() + .5
    y_min, y_max = x_mat[:, 1].min() - .5, x_mat[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2),
                         np.arange(y_min, y_max, 0.2))
    Z = classifier.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    xx_1, yy_1 = np.meshgrid(np.arange(x_min, x_max, 0.2)*np.std(pin),
                             np.arange(y_min, y_max, 0.2)*np.std(HJ))
    cm = plt.cm.RdBu
    fig, ax = plt.subplots(figsize=(4.5,3.5))
    ax.contourf(xx_1, yy_1, Z, cmap=cm, alpha=.6)
    ax.scatter(np.array(pin)[np.nonzero(larray)[0]], np.array(HJ)[np.nonzero(larray)[0]],color='blue',edgecolor='black',label='Model',marker='s')
    ax.scatter(np.array(pin)[np.nonzero(larray==False)[0]],np.array(HJ)[np.nonzero(larray==False)[0]], color='red',edgecolor='black',label='Human')
    divider = make_axes_locatable(ax)
    axHistx = divider.append_axes("top", 0.7, pad=0.0, sharex=ax)
    axHisty = divider.append_axes("right", 0.7, pad=0.0, sharey=ax)
    axHistx.xaxis.set_tick_params(labelbottom=False,bottom=False)
    axHistx.yaxis.set_tick_params(labelleft=False,left=False)
    axHisty.xaxis.set_tick_params(labelbottom=False,bottom=False)
    axHisty.yaxis.set_tick_params(labelleft=False,left=False)
    b_subset = np.nonzero(larray)[0]
    r_subset = np.nonzero(larray==False)[0]
    x = np.array(pin)
    y = np.array(HJ)
    _ = axHistx.hist(x[b_subset], color='blue', bins=np.arange(x_min, x_max, 0.3)*np.std(pin), alpha=0.7)
    _ = axHistx.hist(x[r_subset], color='red', bins=np.arange(x_min, x_max, 0.3)*np.std(pin), alpha=0.7)
    _ = axHisty.hist(y[b_subset], color='blue', bins=np.arange(y_min, y_max, 0.3)*np.std(HJ), orientation='horizontal', alpha=0.7)
    _ = axHisty.hist(y[r_subset], color='red', bins=np.arange(y_min, y_max, 0.3)*np.std(HJ), orientation='horizontal', alpha=0.7)
    ax.legend(loc='lower left')
    ax.set_xlim(np.min(xx_1), np.max(xx_1))
    ax.set_ylim(np.min(yy_1), np.max(yy_1))
    ax.set_xlabel('Model log likelihood')
    ax.set_ylabel('Human Judgement')
    axHistx.set_title(taskname)
    plt.tight_layout()
    plt.savefig("../figures/" + taskname +'.pdf')
