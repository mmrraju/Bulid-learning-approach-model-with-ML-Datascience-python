# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 16:11:57 2019

@author: mmrra
"""

import math
import operator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv("F:\MachineLearning\Datasets\Titanic/trainpp.csv")


target=train["Survived"]
train_data=train.drop(["Survived"], axis=1)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits = 10, shuffle=True, random_state=0)

# Create KNN classifier
knn = KNeighborsClassifier(n_neighbors = 3)

scoring = 'accuracy'
score = cross_val_score(knn, train_data, target, cv=5, n_jobs=1, scoring=scoring)
print(score)

print(round(np.mean(score)*100, 2))
