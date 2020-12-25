# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 15:35:14 2019

@author: mmrra
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
train = pd.read_csv("F:\MachineLearning\Datasets\Titanic/trainpp.csv")
##test = pd.read_csv("F:\MachineLearning\Datasets\Titanic/testpp.csv")
train.head()

target=train["Survived"]
train_data=train.drop(["Survived"], axis=1)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits = 10, shuffle=True, random_state=0)

clf = DecisionTreeClassifier()
scoring ='accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)

print(round(np.mean(score)*100,2))