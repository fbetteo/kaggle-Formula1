from numpy.lib.function_base import append
import pandas as pd
import numpy as np
from pandas import read_stata
from sklearn.metrics.classification import fbeta_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import ClusterCentroids
from imblearn.pipeline import Pipeline as imbPipeline # needed because RandomOverSamples does not have transform method
from time import time
import helpers
import pickle
from xgboost import XGBClassifier


val_size = 0.3
seed = 8
# I chose precision since I want to minimize the amount of False Positives 
# Betting, I would rather fail as less as possible. It would be better to optimize profit but I don't have the odds.
objective_metric = 'precision' #recall, precison, f1. Choose one so CV refits the classifier with the best model according to this metric

# data
results_data = "data/results.csv"
qualifying_data = 'working/qualifying.p'
driver_points_data = 'working/driver_points.p'
constructor_points_data = 'working/constructor_points.p'
q_stops_data = 'working/driver_q_stops.p'
races_data   = 'data/races.csv'
weather_data = 'working/weather_info.p'


results = pd.read_csv(results_data)
qualifying = pd.read_pickle(qualifying_data)
driver_points = pd.read_pickle(driver_points_data)
constructor_points = pd.read_pickle(constructor_points_data)
q_stops = pd.read_pickle(q_stops_data)
races = pd.read_csv(races_data)
weather = pd.read_pickle(weather_data)

# add  data
qualifying_merge = qualifying[['raceId', 'driverId', 'dif_to_min_perc']]
driver_points_merge = driver_points.iloc[:, lambda df:df.columns.str.contains(
    '_points')]  # select column based on name
constructor_points_merge = constructor_points.iloc[:, lambda df:df.columns.str.contains(
    '_points')]  # select column based on name
q_stops_merge = q_stops.iloc[:, lambda df:df.columns.str.contains(
    '_qstops|driverId|raceId')]
races_merge = races[['raceId', 'year', 'round', 'circuitId']]
weather_merge = weather.iloc[:, lambda df:df.columns.str.contains(
    'weather_|year|round|circuitId')]

# join by index (no need to merge with column)
results = results.join(driver_points_merge)
results = results.join(constructor_points_merge)
results.describe()
results.head()
results = results.merge(qualifying_merge, how="left",
                        on=['raceId', 'driverId'])
# q_stops has much less observations, we lose a lot of obs  because we later remove NA
# results = results.merge(q_stops_merge, how="left",
#                         on=['raceId', 'driverId'])
results = results.merge(races_merge, how = 'left',
on = ['raceId'])
results = results.merge(weather_merge, how = 'left',
on = ['year', 'round', 'circuitId'])                   

# cleaning

target = ['position']
qualy_vars = ['grid', 'dif_to_min_perc']
point_vars = list(results.columns[results.columns.str.contains('_points')])
stop_vars = list(results.columns[results.columns.str.contains('_qstops')])
weather_vars = list(results.columns[results.columns.str.contains('weather_')])

vars_keep = target + qualy_vars + point_vars +  weather_vars # + stop_vars  

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

bench_precision_val, bench_recall_val, bench_fbeta_val, bench_support_val = precision_recall_fscore_support(
    y, y_pred_benchmark)

bench_precision_val
bench_recall_val
bench_fbeta_val


pd.DataFrame(
    confusion_matrix(y, y_pred_benchmark), columns=["Pred Not First", "Pred  First"],
    index=["True Not First", "True  First"]
)

# models

model_results = pd.DataFrame()  # we will store each model iteration here

# ClusterCentroids takes a long time in TREE
samplers_to_test = [None, RandomOverSampler(), SMOTE(), ADASYN()] #, ClusterCentroids()]

# TREE

pipeline_tree = imbPipeline(
    steps=[
        ('samp',None),
        ("tree", DecisionTreeClassifier())
    ]
)

# # TUNING
# samplers_to_test = [None, RandomOverSampler(), SMOTE(), ADASYN()]
# criterion = ['gini', 'entropy']
# max_depth = [3, 5, 10]
# min_samples_split =  [0.01, 0.05, 0.1, 0.2]
# min_samples_leaf =  [0.01, 0.05, 0.1, 0.2]

# FINAL
samplers_to_test = [None] #, RandomOverSampler(), SMOTE(), ADASYN()]
criterion = ['entropy'] # ['gini', 'entropy']
max_depth = [5] # [3, 5, 10]
min_samples_split = [0.05] #  [0.01, 0.05, 0.1, 0.2]
min_samples_leaf = [0.01] # [0.01, 0.05, 0.1, 0.2]

param_grid = dict(
    samp = samplers_to_test,
    tree__criterion=criterion,
    tree__max_depth=max_depth,
    tree__min_samples_split=min_samples_split,
    tree__min_samples_leaf=min_samples_leaf
)

# Need to do stratification due to unbalanced?
# Nop. GridSearch used StratifiedKFold for classifiers if using cv = integer
clf_tree = GridSearchCV(pipeline_tree, param_grid=param_grid, verbose=8, scoring=['precision', 'recall', 'f1'],
                        refit=objective_metric,
                        cv=3)
clf_tree = clf_tree.fit(X, y)

model_results = model_results.append(helpers.extract_scores_kfold(
    clf_tree, None))  # appends CV results

pickle.dump(model_results, open('working/model_results.p', 'wb'))

# Logistic
pipeline_lr =imbPipeline(
    steps=[
        ('samp', None),
        ("LR", LogisticRegression())
    ]
)

# # TUNING
# samplers_to_test =  [None, RandomOverSampler(), SMOTE(), ADASYN()]
# penalty =  ['l1', 'l2']
# class_weight = [None, "balanced"]

# FINAL
samplers_to_test = [None] # [None, RandomOverSampler(), SMOTE(), ADASYN()]
penalty = ['l2'] # ['l1', 'l2']
class_weight = [None] # [None, "balanced"]

param_grid = dict(
    samp = samplers_to_test,
    LR__penalty=penalty,
    LR__class_weight=class_weight
)

# Need to do stratification due to unbalanced?
# Nop. GridSearch used StratifiedKFold for classifiers if using cv = integer
clf_lr = GridSearchCV(pipeline_lr, param_grid=param_grid, verbose=8, scoring=['precision', 'recall', 'f1'],
                      refit=objective_metric,
                      cv=3)
clf_lr = clf_lr.fit(X, y)

model_results = model_results.append(helpers.extract_scores_kfold(clf_lr, None))
pickle.dump(model_results, open('working/model_results.p', 'wb'))

# Random Forest
# Takes a couple of minutes

pipeline_rf = Pipeline(
    steps=[
        ('samp', None),  
        ("rf", RandomForestClassifier())
    ]
)


# # TUNING
# samplers_to_test = [None, RandomOverSampler(), SMOTE(), ADASYN()]
# n_estimators = [10, 50, 100]
# criterion =  ['gini', 'entropy']
# max_depth = [3, 5, 10]
# min_samples_split =  [0.01, 0.05, 0.1, 0.2]
# min_samples_leaf = [0.01, 0.05, 0.1, 0.2]
# max_features = [None, "sqrt"]
# class_weight = [None, "balanced"]

# FINAL
samplers_to_test = [None] # [None, RandomOverSampler(), SMOTE(), ADASYN()]
n_estimators = [100] # [10, 50, 100]
criterion = ['entropy'] # ['gini', 'entropy']
max_depth = [5] # [3, 5, 10]
min_samples_split = [0.01] # [0.01, 0.05, 0.1, 0.2]
min_samples_leaf = [0.01] # [0.01, 0.05, 0.1, 0.2]
max_features = ['sqrt'] # [None, "sqrt"]
class_weight = [None] # [None, "balanced"]


param_grid = dict(
    samp = samplers_to_test,  
    rf__n_estimators=n_estimators,
    rf__criterion=criterion,
    rf__max_depth=max_depth,
    rf__min_samples_split=min_samples_split,
    rf__min_samples_leaf=min_samples_leaf,
    rf__max_features=max_features,
    rf__class_weight=class_weight
)

# Need to do stratification due to unbalanced?
# Nop. GridSearch used StratifiedKFold for classifiers if using cv = integer
clf_rf = GridSearchCV(pipeline_rf, param_grid=param_grid, verbose=8, scoring=['precision', 'recall', 'f1'],
                      refit=objective_metric,
                      cv=5)
clf_rf = clf_rf.fit(X, y)


model_results = model_results.append(helpers.extract_scores_kfold(clf_rf, None))

pickle.dump(model_results, open('working/model_results.p', 'wb'))

# Neural Net

pipeline_nn = imbPipeline(
    steps=[
        ('samp', None),
        ('standard', StandardScaler()),
         ("NN", MLPClassifier())
    ]
)

# # TUNING
# samplers_to_test =  [None, RandomOverSampler(), SMOTE(), ADASYN()]
# hidden_layer_sizes = [(4,1), (4), (7),(5,2) , (10,5), (5,4,3)]
# activation = ['identity', 'relu', 'logistic', 'tanh']

# FINAL
samplers_to_test = [None] # [None, RandomOverSampler(), SMOTE(), ADASYN()]
hidden_layer_sizes = [(5,4,3)] # [(4,1), (4), (7),(5,2) , (10,5), (5,4,3)]
activation = ['tanh'] # ['identity', 'relu', 'logistic', 'tanh']

param_grid = dict(
    samp = samplers_to_test,
    NN__hidden_layer_sizes=hidden_layer_sizes,
    NN__activation=activation
)

clf_nn = GridSearchCV(pipeline_nn, param_grid=param_grid, verbose=8, scoring=['precision', 'recall', 'f1'],
                      refit=objective_metric,
                      cv=5)
clf_nn = clf_nn.fit(X, y)



model_results = model_results.append(helpers.extract_scores_kfold(clf_nn, None))
pickle.dump(model_results, open('working/model_results.p', 'wb'))

## XGBOOST
# model_results = pd.read_pickle('working/model_results.p')
pipeline_xgb = imbPipeline(
    steps=[
        ('samp', None),
      #  ('standard', StandardScaler()),
         ("xgb", XGBClassifier())
    ]
)

# # TUNING
# samplers_to_test = [None] # [None, RandomOverSampler(), SMOTE(), ADASYN()]
# eval_metric = ['aucpr', "map"]
# num_boost_round = [999]
# early_stopping_rounds= [10]
# max_depth = [1,3,5,10]
# min_child_weight =  [1,3,5,10]
# subsample =  [0.5, 0.8, 1]
# colsample_bytree =  [0.5,0.8,1]
# base_score =  [0.5, 0.7, 0.9]

# FINAL
samplers_to_test = [None] # [None, RandomOverSampler(), SMOTE(), ADASYN()]
eval_metric = ["aucpr"] # ['aucpr', "map"]
num_boost_round = [999]
early_stopping_rounds= [10]
max_depth = [5] # [1,3,5,10]
min_child_weight = [1] # [1,3,5,10]
subsample = [0.8] # [0.5, 0.8, 1]
colsample_bytree = [1] # [0.5,0.8,1]
base_score = [0.5] # [0.3, 0.5, 0.7, 0.9]

param_grid = dict(
    samp = samplers_to_test,
    xgb__eval_metric = eval_metric,
    xgb__max_depth = max_depth,
    xgb__min_child_weight = min_child_weight,
    xgb__subsample = subsample,
    xgb__colsample_bytree = colsample_bytree,
    xgb__base_score = base_score
    # xgb__num_boost_round=num_boost_round,
    # xgb__early_stopping_rounds = early_stopping_rounds
    )

clf_xgb = GridSearchCV(pipeline_xgb, param_grid=param_grid, verbose=8, scoring=['precision', 'recall', 'f1'],
                      refit=objective_metric,
                      cv=3)
clf_xgb = clf_xgb.fit(X, y)


model_results = model_results.append(helpers.extract_scores_kfold(clf_xgb, None))
pickle.dump(model_results, open('working/model_results.p', 'wb'))

# -------------- #
# All models ran #
# -------------- #


# important metrics and comparison to benchmark
# keeping best models for each classifier
results = model_results[(model_results.rank_test_precision == 1) | (
    model_results.rank_test_recall == 1) | (model_results.rank_test_f1 == 1)].loc[:, ['mean_test_precision', 'mean_test_recall', 'mean_test_f1', 'rank_test_precision', 'rank_test_recall', 'rank_test_f1', 'params']]
results['benchmark_test_precision'] = bench_precision_val[1]
results['benchmark_test_recall'] = bench_recall_val[1]
results['benchmark_test_fbeta'] = bench_fbeta_val[1]

results.to_csv('gridsearch_results.csv')

# generate metrics in test set for best model (refit)
# also generates the confusion matrix
models_test = {'tree': clf_tree,
'logistic': clf_lr,
'rf': clf_rf,
'neural_net': clf_nn,
'xgb': clf_xgb}

# NESTED CV. TA BIEN ESTO?
# EXPANDIR PARA LOS OTROS MODELOS
# https://machinelearningmastery.com/nested-cross-validation-for-machine-learning-with-python/
# from sklearn.model_selection import cross_val_score, KFold
# outer_cv = KFold(n_splits=3, shuffle=True, random_state=5)

# nested_score = cross_val_score(clf_xgb, X=X, y=y, scoring = 'precision', cv=outer_cv)


ensemble_models = list(models_test.values())

test_results= pd.DataFrame()
confusion_test = {}
for name, model in models_test.items():
    test_results = test_results.append(helpers.metrics_in_test(name, model, X_test, y_test)[0])
    confusion_test[name] = helpers.metrics_in_test(name, model, X_test, y_test)[1]

# include ensemble
test_results = test_results.append(helpers.ensemble_test(list(models_test.values()), X_test, y_test)[0])
confusion_test['ensemble'] = confusion_test.get('ensemble', helpers.ensemble_test(list(models_test.values()), X_test, y_test)[1])

# include benchmark
test_results = test_results.append(helpers.benchmark_test(X_test, y_test)[0])
confusion_test['benchmark'] = confusion_test.get('benchmark', helpers.benchmark_test(X_test, y_test)[1])


pickle.dump(test_results, open('working/test_results.p','wb'))
pickle.dump(confusion_test, open('working/confusion_test.p','wb'))