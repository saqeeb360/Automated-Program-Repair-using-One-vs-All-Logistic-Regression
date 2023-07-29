import pandas as pd
import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn import tree
from sklearn.metrics import accuracy_score   
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import scipy.sparse
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE

import sys
np.set_printoptions(threshold=sys.maxsize)

dictSize = 225
X, y = load_svmlight_file( "train", multilabel = False, n_features = dictSize, offset = 1 )
X = X.toarray()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=15, stratify = y)

z = Counter(y_train)
z = dict(sorted(z.items(), key=lambda item: item[1]))
z

lst = list(z.keys())
print(lst)
lst = lst[:-4]
len(lst)

X_train1=[]
X_train2=[]
y_train1 = []
y_train2 = []
for i in range(X_train.shape[0]):
  if(y_train[i] in lst):
    X_train1.append(X_train[i])
    y_train1.append(y_train[i])
  else:
    X_train2.append(X_train[i])
    y_train2.append(y_train[i])

sampling = RandomOverSampler(random_state=42)
X_train1, y_train1 = sampling.fit_resample(X_train1, y_train1)

X_train = X_train1 + X_train2
y_train = y_train1 + y_train2

############## Logistic Regression ################
model = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True,
                   intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs',
                   max_iter=1000, multi_class='multinomial', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
score = accuracy_score(y_test, y_pred)  
print(score)

filename = 'model.sav'
pickle.dump(model, open(filename, 'wb'))