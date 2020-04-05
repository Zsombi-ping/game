from sklearn.svm import OneClassSVM
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
from random import *
import os
import sys
import glob

def compute_fpr_tpr(userid, positive_scores, negative_scores, plot = True):
    zeros = np.zeros(len(negative_scores))
    ones = np.ones(len(positive_scores))
    y = np.concatenate((zeros, ones))
    scores = np.concatenate((negative_scores, positive_scores))
    fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    if( plot == True ):
        plot_ROC( userid, fpr, tpr, roc_auc )
    return roc_auc

def plot_ROC(userid, fpr, tpr, roc_auc):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
    lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC - user ' + userid)
    plt.legend(loc="lower right")
    plt.show()


def split_list(a_list):
    half = len(a_list)//2
    return a_list[:half], a_list[half:]

DATASETS = []

PATH = "C:/Users/Zsombi/Desktop/Mesterseges/101_ObjectCategories"
for folder in os.listdir(PATH):
    folder_str = ""
    ll = []
    ll.append(folder)
    folder_str = PATH +"/"+folder+"/*.jpg"
    for file in glob.iglob(folder_str):
        ll.append(file)
    DATASETS.append(ll)

SIZE = len(DATASETS)
n = 2

Classes = []

Train = []
Test = []

for i in range (0,n):
    id = randrange(SIZE)
    Classes.append(DATASETS[id])
    train , test = split_list(DATASETS[i])
    Train.append(train)
    Test.append(test)

for i in range (n):
    user_train = Train
    user_test =Test

    clf = OneClassSVM(gamma='scale').fit(user_train)
    clf.fit(user_train)
    positive_scores = clf.score_samples(user_test)
    negative_scores = clf.score_samples(Classes[i])

    print(str(Classes[i][0]) + " : " +str('%.2f' % compute_fpr_tpr(Classes[i][0],positive_scores,negative_scores)))







