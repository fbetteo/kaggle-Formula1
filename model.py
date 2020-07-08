import pandas as pd
import numpy as np
from sklearn.metrics.classification import fbeta_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, precision_recall_fscore_support

val_size = 0.3
seed = 8

# data
results = pd.read_csv("data/results.csv")

# cleaning
df = results[['grid', 'position']]
df = df[(df.position != "\\N") & (df.grid != 0)]
df['position'] = df['position'].astype('int32')

# classify first place
df['position'] = np.where(df['position'] > 1, 2, 1)

X = df.drop(columns="position")
y = df['position']

# isolating test
test_index = X.shape[0]-1000  # ultimos 1000
X_test = X.iloc[test_index:]
y_test = y.iloc[test_index:]
X = X[:test_index]
y = y[:test_index]

# train validation
X_train, X_val, y_train,  y_val = train_test_split(X, y, test_size=val_size,
                                                   random_state=seed)

# model
clf = DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

# validation
y_predict = clf.predict(X_val)

precision_val, recall_val, fbeta_val, support_val = precision_recall_fscore_support(y_val, y_predict)

precision_val

accuracy_score(y_val, y_predict)
roc_auc_score(y_val, y_predict)

pd.DataFrame(
    confusion_matrix(y_val, y_predict), columns=["Pred First", "Pred Not First"],
    index=["True First", "True Not First"]
)

# test 
y_predict_test = clf.predict(X_test)
precision_test, recall_test, fbeta_test, support_test = precision_recall_fscore_support(y_test, y_predict_test)
precision_test

pd.DataFrame(
    confusion_matrix(y_test, y_predict_test), columns=["Pred First", "Pred Not First"],
    index=["True First", "True Not First"]
)
