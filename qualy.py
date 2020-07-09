import pandas as pd
import numpy as np

qualifying = pd.read_csv("data/qualifying.csv")
qualifying = qualifying.replace('\\N', np.nan)  # replace \N by NaN

# coalesce (q3,q2,q1). That's the order that counts for the starting grid
qualifying['best_qualy'] = qualifying.q3.combine_first(
    qualifying.q2).combine_first(qualifying.q1)


def time_in_miliseconds(x):
    """ 
    Takes a string with mm:ss.ms
    and returns minutes, seconds, ms and value in ms.
    """
    try:
        minute = int(x.split(":")[0])
        seconds = int(x.split(":")[1].split(".")[0])
        miliseconds = int(x.split(":")[1].split(".")[1])
        best_in_mili = minute * 60000 + seconds * 1000 + miliseconds
        return [minute, seconds, miliseconds, best_in_mili]
    except:
        return [np.NaN, np.NaN, np.NaN, np.NaN]

# applies function to all rows of best_qualy and returns one column per list object (expand)
# but we keep just the last 
qualifying['best_qualy_ms'] = qualifying.apply(
    lambda x: time_in_miliseconds(x.best_qualy), axis=1, result_type='expand')[3]


qualifying.head()

