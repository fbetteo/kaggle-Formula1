import pandas as pd

def extract_scores_kfold(model, _metric):
    scores = pd.DataFrame(model.cv_results_)
    # scores = scores.sort_values(
    #     by = ['rank_test'+metric]
    # ).reset_index(drop  = 'index')
    return(scores)


def sort_results(results, metric):
    results = results.sort_values(
        by = ['mean_test_' + metric],
        ascending = False
    ).reset_index(drop = 'index')
    return results
