# -*- coding: utf-8 -*-
"""
Created on Thursday Jul 25 10:44:07 2019

@author: lajkonik
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

# Read neuron feature/role dataset
#df = pd.read_csv("output_loghasum_080419.csv")
df = pd.read_csv("output_harmmean_062120.csv")
X = df.iloc[:,1:-1].values
y = df.iloc[:,-1].values.astype('S')

# Make predictions
# Build classifier on all data (full connectome)
#clf = RidgeClassifier(max_iter=100)
#clf = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1,max_depth=2, random_state=0)
#clf = RandomForestClassifier(n_estimators=2000, max_features=3, n_jobs=-1, random_state=0)

# Grid search for optimizing parameters
"""
parameters = {'max_features':[2,4,6,8,10], 'min_samples_split':[2,4,8,16], 'max_depth':[None,2,4,8], 'criterion':['gini','entropy']}
clf = RandomForestClassifier(n_estimators=100, max_features=3, n_jobs=-1, random_state=0)
clfgs = GridSearchCV(clf, parameters, cv=3, verbose=3)
clfgs.fit(X, y)
print(clfgs.best_estimator_)
print(clfgs.best_score_)
"""
clf_best = RandomForestClassifier(n_estimators=1000, n_jobs=-1, random_state=0)

# Evaluate the best classifier
clf_best.fit(X, y)
preds = clf_best.predict(X)

# Get neurons that changed roles
df_changed_roles = df[y!=preds]

role_dist = 100 * np.array([preds[y=='I'].size, preds[y=='M'].size, preds[y=='S'].size]) / float(preds.size)

print('num changes = ',df_changed_roles.shape[0])

from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf_best, X, y, cv=93, verbose=100, n_jobs=-1)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

df_features = pd.DataFrame({'feature':df.columns[1:-1], 'importance':clf_best.feature_importances_})
print(df_features.sort_values(by='importance', ascending=False))

"""
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
skbest = SelectKBest(mutual_info_classif, k=4).fit(X, y)
f_selected = skbest.get_support()
df2 = pd.DataFrame(df.iloc[:,2:-1])
print(df2.T[f_selected].index)


X_new = skbest.fit_transform(X, y)
scores = cross_val_score(clf_best, X_new, y, cv=10, verbose=100, n_jobs=-1)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
"""


