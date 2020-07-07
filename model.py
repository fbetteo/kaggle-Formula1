import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

# data
results = pd.read_csv("data/results.csv")

# cleaning
df = results[['grid', 'position']]
df = df[(df.position != "\\N") & (df.grid != 0)]
df['position'] = df['position'].astype('int32')

# classify first place
df['position'] = np.where(df['position'] > 1, 2,1)

X = df.drop(columns = "position")
y = df['position']

# isolating test
test_index = X.shape[0]-1000 # ultimos 1000
X_test = X.iloc[test_index:]
y_test = y.iloc[test_index:]
X = X[:test_index]
y = y[:test_index]


clf = DecisionTreeClassifier()
clf = clf.fit(X,y)

y_predict = clf.predict(X )

accuracy_score(y, y_predict)
roc_auc_score(y, y_predict)

pd.DataFrame(
    confusion_matrix(y, y_predict), columns = ["Pred First", "Pred Not First"],
    index = ["True First", "True Not First"]
)

df.describe()
df.shape

y.shape

