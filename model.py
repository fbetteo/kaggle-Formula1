import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

laptimes = pd.read_csv("data/lap_times.csv")
races = pd.read_csv("data/races.csv")
results = pd.read_csv("data/results.csv")

df.head(20)
df.describe()
df = results[['grid', 'position']]
df = df[(df.position != "\\N") & (df.grid != 0)]
df['position'] = df['position'].astype('int32')

# classify first place
df['position'] = np.where(df['position'] > 1, 2,1)

X = df.drop(columns = "position")
y = df['position']


clf = DecisionTreeClassifier()
clf = clf.fit(X,y)

y_predict = clf.predict(X )

accuracy_score(y, y_predict)

from sklearn.metrics import confusion_matrix

pd.DataFrame(
    confusion_matrix(y, y_predict), columns = ["Pred First", "Pred Not First"],
    index = ["True First", "True Not First"]
)

df.describe()
df.shape

y.shape

