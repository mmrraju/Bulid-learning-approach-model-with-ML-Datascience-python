# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 09:51:47 2019

@author: mmrra
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

### Loading our train and testing dataset.
train = pd.read_csv("F:\MachineLearning\Datasets\Titanic/Train.csv")
test = pd.read_csv("F:\MachineLearning\Datasets\Titanic/test.csv")
train.head()

###Now we will findout how much data is absent in our training dataset
train.info()
train.isnull().sum()

test.info()
test.isnull().sum()

### Now we will define a function to findout relationship among variable and Target variable 
def bar_chart(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived, dead])
    df.index=['Survived', 'Dead']
    df.plot(kind = 'bar', stacked=True, figsize=(10, 5))
    
bar_chart('Pclass')
bar_chart('Sex')
bar_chart('Embarked')

### Now we will combine our train and test dataset
train_test_data=[train, test]
for dataset in  train_test_data:
    dataset['Title']=dataset['Name'].str.extract('([A-Za-z]+)\.', expand=False)

train['Title'].value_counts() 

title_mapping = {"Mr":0, "Miss":1, "Mrs":2, "Master":3, "Dr":3, "Rev":3, "Col":3, "Mlle":3, "Major":3, "Lady":3, "Jonkheer":3, "Countess":3, "Sir":3, "Capt":3, "Don":3, "Ms":3, "Mme":3}

for dataset in train_test_data:
    dataset['Title']=dataset['Title'].map(title_mapping)
    
test['Title'].value_counts()

bar_chart('Title')

### Delete unnecessary feature from dataset
train.drop('Name', axis=1, inplace=True)
test.drop('Name', axis=1, inplace=True)

### Sex mapping
sex_mapping ={"male":0, "female":1}
for dataset in train_test_data:
    dataset['Sex']=dataset['Sex'].map(sex_mapping)

bar_chart('Sex')

###Now we will fill the missing value in 'age' with average value
train["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace=True)
test["Age"].fillna(test.groupby("Title")["Age"].transform("median"), inplace=True)

train.groupby("Title")["Age"].transform("median")
train.head()

train.info()

### Now we will see which age people more die. 
facet=sns.FacetGrid(train, hue="Survived", aspect=4)
facet.map(sns.kdeplot, 'Age', shade= True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()
plt.show()

### Now we will mapping our age child=0, young=1, adult=2, mid_ag=3, senior=4
for dataset in train_test_data:
    dataset.loc[dataset['Age']<=16, 'Age']= 0,
    dataset.loc[(dataset['Age']>16) & (dataset['Age']<=26) , 'Age']=1,
    dataset.loc[(dataset['Age']>26) & (dataset['Age']<=36), 'Age']=2,
    dataset.loc[(dataset['Age']>36) & (dataset['Age']<=62), 'Age']=3,
    dataset.loc[dataset['Age']>62, 'Age']=4
    
bar_chart('Age')

### Now we will count from which different class people
Pclass1 = train[train['Pclass']==1]['Embarked'].value_counts()
Pclass2 = train[train['Pclass']==2]['Embarked'].value_counts()
Pclass3 = train[train['Pclass']==3]['Embarked'].value_counts()
df= pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index=['1st class', '2nd class', '3rd class']
df.plot(kind = 'bar', stacked=True, figsize=(10, 5))

for dataset in train_test_data:
    dataset['Embarked']=dataset['Embarked'].fillna('S')
    
### Now we will mapping embarked
embarked_mapping={"S":0, "C":1, "Q":2}
for dataset in train_test_data:
    dataset['Embarked']=dataset['Embarked'].map(embarked_mapping)

train["Fare"].fillna(train.groupby("Pclass")["Fare"].transform("median"), inplace=True)
test["Fare"].fillna(test.groupby("Pclass")["Fare"].transform("median"), inplace=True)
train.head()

### Now we findout the which fare people die much
facet=sns.FacetGrid(train, hue="Survived", aspect=4)
facet.map(sns.kdeplot, 'Fare', shade=True)
facet.set(xlim=(0, train['Fare'].max()))
facet.add_legend()
plt.show()

for dataset in train_test_data:
    dataset.loc[dataset['Fare']<=17, 'Fare']=0,
    dataset.loc[(dataset['Fare']>17) & (dataset['Fare']<=30), 'Fare']=1,
    dataset.loc[(dataset['Fare']>30) & (dataset['Fare']<=100), 'Fare']=2,
    dataset.loc[dataset['Fare']>100, 'Fare']=3

train.Cabin.value_counts()

for dataset in train_test_data:
    dataset['Cabin']=dataset['Cabin'].str[:1]

Pclass1 = train[train['Pclass']==1]['Cabin'].value_counts()
Pclass2 = train[train['Pclass']==2]['Cabin'].value_counts()
Pclass3 = train[train['Pclass']==3]['Cabin'].value_counts()
df=pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index=['1st class', '2nd class', '3rd class']
df.plot(kind ='bar', stacked= True, figsize=(10,5))

cabin_mapping={"A":0, "B":0.4, "C":0.8, "D":1.2, "E":1.6, "F":2.0, "G":2.4, "T":2.8}
for dataset in train_test_data:
    dataset['Cabin']=dataset['Cabin'].map(cabin_mapping)
    
train["Cabin"].fillna(train.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
test["Cabin"].fillna(test.groupby("Pclass")["Cabin"].transform("median"), inplace=True)

train["FamilySize"]=train["SibSp"]+train["Parch"]+1
test["FamilySize"]=test["SibSp"]+test["Parch"]+1

facet= sns.FacetGrid(train, hue="Survived", aspect=4)
facet.map(sns.kdeplot, 'FamilySize', shade=True)
facet.set(xlim=(0, train['FamilySize'].max()))
facet.add_legend()
plt.xlim(0)

family_mapping={1: 0, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.6, 6: 2.0, 7: 2.4, 8: 2.8, 9: 3.2, 10: 3.6, 11: 4.0}
for dataset in train_test_data:
    dataset['FamilySize']=dataset['FamilySize'].map(family_mapping)

feature_drop =['Ticket', 'SibSp', 'Parch']
train =train.drop(feature_drop, axis=1)
test = test.drop(feature_drop, axis=1)
#train.to_csv("TrainPP.csv") test.to_csv("TestPP.csv")
train = train.drop(['PassengerId'], axis=1)

train_data=train.drop('Survived', axis=1)

#train.to_csv("trainAfter.csv")
target= train['Survived']
train_data.shape, target.shape
#target.to_csv("targetpp.csv")

###################Now we will take machinelearning model#####################
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
###Random forest
clf=RandomForestClassifier(n_estimators = 13)
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)

print(round(np.mean(score)*100, 2))
### Support vector machine
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')
k_fold = KFold(n_splits = 10, shuffle=True, random_state=0)

support = svm.LinearSVC(random_state=20)
scoring='accuracy'
score = cross_val_score(support, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
print(round(np.mean(score)*100, 2))
###KNN###
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits = 10, shuffle=True, random_state=0)

# Create KNN classifier
knn = KNeighborsClassifier(n_neighbors = 3)

scoring = 'accuracy'
score = cross_val_score(knn, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)

print(round(np.mean(score)*100, 2))
