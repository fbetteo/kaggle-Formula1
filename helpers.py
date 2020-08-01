import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def extract_scores_kfold(model, _metric):
    scores = pd.DataFrame(model.cv_results_)
    # scores = scores.sort_values(
    #     by = ['rank_test'+metric]
    # ).reset_index(drop  = 'index')
    return(scores)


def sort_results(results, metric):
    results = results.sort_values(
        by=['mean_test_' + metric],
        ascending=False
    ).reset_index(drop='index')
    return results


def metrics_in_test(model_name, model, X_test, y_test):
    y_predict_test = model.predict(X_test)  # uses the best of the grid search
    precision_test, recall_test, fbeta_test, support_test = precision_recall_fscore_support(
        y_test, y_predict_test)

    metrics = pd.DataFrame([[model_name, precision_test[1], recall_test[1], fbeta_test[1], support_test[1]]],
                           columns=["Model", "precision", "recall", "fbeta", "support"])

    confusion= pd.DataFrame(confusion_matrix(y_test, y_predict_test),  columns=["Pred Not First", "Pred  First"],
                                    index=["True Not First", "True  First"]
                                    )
    return [metrics, confusion]

def benchmark_test(X_test, y_test):
    y_predict_test = np.where(X_test['grid'] == 1, 1, 0)

    precision_test,recall_test, fbeta_test, support_test = precision_recall_fscore_support(
        y_test, y_predict_test)

    metrics = pd.DataFrame([['benchmark', precision_test[1], recall_test[1], fbeta_test[1], support_test[1]]],
                           columns=["Model", "precision", "recall", "fbeta", "support"])

    confusion= pd.DataFrame(confusion_matrix(y_test, y_predict_test),  columns=["Pred Not First", "Pred  First"],
                                    index=["True Not First", "True  First"]
                                    )
    return [metrics, confusion]

  


def ensemble_test(models, X_test, y_test):
    pred = np.zeros((y_test.shape[0]))
    for m in models:
        pred = pred +  m.predict(X_test)
    ensemble_pred = np.where(pred > len(models)/2,1,0)
    # return ensemble_pred
    precision_test, recall_test, fbeta_test, support_test = precision_recall_fscore_support(
        y_test, ensemble_pred)

    metrics = pd.DataFrame([['ensemble', precision_test[1], recall_test[1], fbeta_test[1], support_test[1]]],
                           columns=["Model", "precision", "recall", "fbeta", "support"])

    confusion= pd.DataFrame(confusion_matrix(y_test, ensemble_pred),  columns=["Pred Not First", "Pred  First"],
                                    index=["True Not First", "True  First"]
                                    )
    return [metrics, confusion]
    

