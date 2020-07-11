import pandas as pd
import numpy as np
from sklearn.metrics.classification import fbeta_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from time import time

val_size = 0.3
seed = 8

# data
results_data = "data/results.csv"
qualifying_data = 'working/qualifying.p'
driver_points_data = 'working/driver_points.p'
constructor_points_data = 'working/constructor_points.p'

results = pd.read_csv(results_data)
qualifying = pd.read_pickle(qualifying_data)
driver_points = pd.read_pickle(driver_points_data)
constructor_points = pd.read_pickle(constructor_points_data)

# add  data
qualifying_merge = qualifying[['raceId', 'driverId', 'dif_to_min_perc']]
driver_points_merge = driver_points.iloc[:, lambda df:df.columns.str.contains('_points')] # select column based on name
constructor_points_merge = constructor_points.iloc[:, lambda df:df.columns.str.contains('_points')] # select column based on name

results = results.join(driver_points_merge) # join by index (no need to merge with column)
results = results.join(constructor_points_merge)
results.describe()
results.head()
results = results.merge(qualifying_merge, how="left",
                        on=['raceId', 'driverId'])

# cleaning

target = ['position']
qualy_vars = ['grid', 'dif_to_min_perc']
point_vars = list(results.columns[results.columns.str.contains('_points')])

vars_keep = target + qualy_vars + point_vars

df = results[vars_keep]
df = df[(df.position != "\\N") & (df.grid != 0)]
df['position'] = df['position'].astype('int32')
df = df.dropna()  # drop because tree don't handle NA


# Target being first place
df['position'] = np.where(df['position'] > 1, 0, 1)
X = df.drop(columns="position")
y = df['position']

# isolating test
test_index = X.shape[0]-1000  # ultimos 1000
X_test = X.iloc[test_index:]
y_test = y.iloc[test_index:]
X = X[:test_index]
y = y[:test_index]

# train validation in case of need
X_train, X_val, y_train,  y_val = train_test_split(X, y, test_size=val_size,
                                                   random_state=seed)


# benchmark
# predicting the pole position wins
# metrics in whole training
y_pred_benchmark = np.where(X['grid'] == 1, 1, 0)

precision_val, recall_val, fbeta_val, support_val = precision_recall_fscore_support(
    y, y_pred_benchmark)

precision_val
recall_val

pd.DataFrame(
    confusion_matrix(y, y_pred_benchmark), columns=["Pred Not First", "Pred  First"],
    index=["True Not First", "True  First"]
)

# models
# TREE
pipeline_tree = Pipeline(
    steps=[
        ("tree", DecisionTreeClassifier())
    ]
)

criterion = ['gini', 'entropy']
max_depth = [3, 5, 10]
min_samples_split = [0.01, 0.05, 0.1, 0.2]
min_samples_leaf = [0.01, 0.05, 0.1, 0.2]

param_grid = dict(
    tree__criterion=criterion,
    tree__max_depth=max_depth,
    tree__min_samples_split=min_samples_split,
    tree__min_samples_leaf=min_samples_leaf
)

clf = GridSearchCV(pipeline_tree, param_grid=param_grid, verbose=8, scoring=['precision', 'recall', 'f1'],
                   refit='precision',
                   cv=5)
clf = clf.fit(X, y)


scores_df = pd.DataFrame(clf.cv_results_)
scores_df = scores_df.sort_values(
    by=['rank_test_recall']).reset_index(drop='index')
scores_df.head()

scores_df.to_csv('tree_score.csv')

# test
y_predict_test = clf.predict(X_test) # uses the best of the grid search
precision_test, recall_test, fbeta_test, support_test = precision_recall_fscore_support(
    y_test, y_predict_test)


pd.DataFrame(
    confusion_matrix(y_test, y_predict_test),  columns=["Pred Not First", "Pred  First"],
    index=["True Not First", "True  First"]
)
