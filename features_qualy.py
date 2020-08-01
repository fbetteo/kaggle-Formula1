import pandas as pd
import numpy as np
import pickle

qualifying = pd.read_csv("data/qualifying.csv")
qualifying = qualifying.replace('\\N', np.nan)  # replace \N by NaN


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


# translate times to ms
# applies function to all rows of best_qualy and returns one column per list object (expand)
# but we keep just the last
for q in ['q1', 'q2', 'q3']:
    qualifying[q + ("_ms")] = qualifying.apply(
        lambda x: time_in_miliseconds(x[q]), axis=1, result_type='expand')[3]

# min valid qualy time
qualifying['best_qualy_ms'] = qualifying.apply(
    lambda x: min(x.q1_ms, x.q2_ms, x.q3_ms), axis=1)

# min per group.
# returns a series with the group as index
min_qualy_by_race = qualifying.groupby('raceId')['best_qualy_ms'].min()

# percentage  difference to the min of the race qualy
# filling with 0.1 if don't have qualy time. Totally arbitrary for now
qualifying['dif_to_min_perc'] = qualifying.apply(lambda x: (
    x.best_qualy_ms - min_qualy_by_race[x.raceId])/min_qualy_by_race[x.raceId], axis=1).fillna(0.1)

# export
pickle.dump(qualifying, open("working/qualifying.p", "wb"))
