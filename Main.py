
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import roc_curve, auc
#rom sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle

from AdaBoost import AdaBoost
from K_Nearest_Neighbor import K_Nearest_Neighbor
from Naive_Bayes import Naive_Bayes
from TreeDecision import TreeDecision


def load_magic_data():


    Ynames = ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym',
              'fM3Long', 'fM3Trans', 'fAlpha', 'fDist', 'identity']

    featureNames = ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym',
                    'fM3Long', 'fM3Trans', 'fAlpha', 'fDist']

    filepath = 'data/dataset.data'

    data = pd.read_csv(filepath, names=Ynames, header=None)

    data['identity'][data['identity'] == 'g'] = 1
    data['identity'][data['identity'] == 'h'] = 0

    X = data[featureNames].values
    Y = data['identity'].values.astype('int64')

    return (X, Y, Ynames)


def main():
    (X, Y, Ynames) = load_magic_data()
    X = StandardScaler().fit_transform(X)
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.3, random_state=None)
    print("-----------------------------------------------------------------------------")
    print("DECISION TREE")
    print("++++++++++++++++")
    treedecison= TreeDecision(Xtrain, Xtest, Ytrain, Ytest)
    treedecison.treedecision()
    print("-----------------------------------------------------------------------------")
    print("K NEAREST NEIGHBOT")
    print("+++++++++++++++++++")
    k_nearest_neighbor=K_Nearest_Neighbor(Xtrain, Xtest, Ytrain, Ytest)
    k_nearest_neighbor.knearest()
    print("-----------------------------------------------------------------------------")
    print("NAIVE BAYES")
    print("+++++++++++++")
    naivebayes = Naive_Bayes(Xtrain, Xtest, Ytrain, Ytest)
    naivebayes.naivebayes()
    print("-----------------------------------------------------------------------------")
    print("ADA BOOST")
    print("+++++++++++++")
    adaboost = AdaBoost(Xtrain, Xtest, Ytrain, Ytest)
    adaboost.adaboost()

if __name__ == "__main__":

    main()