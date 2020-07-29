import pandas as pd
import pickle

from pandas import read_csv

# data
results_data = "data/results.csv"
races_data = 'data/races.csv'

results = pd.read_csv(results_data)
races = pd.read_csv(races_data)

results_2 = results.merge(races, how = "left", on = "raceId")

# Generate cumulated points over last 1 to 5 races per driver
driver_points = results_2.sort_values(['driverId','year', 'raceId'])[['raceId', 'driverId', 'points']] # by year because raceid is not chronological between years (but yes in each one)
for i in range(1,6):
    driver_points["driver_last_" + str(i) + "_points"] = driver_points.groupby('driverId')['points'].rolling(i, min_periods = i).sum().groupby(level = 0).shift().fillna(0).reset_index(level = 0)['points']


# Generate cumulated points over last 1 to 5 races per constructor
constructor_points = results_2.sort_values(['constructorId','year', 'raceId'])[['raceId', 'constructorId', 'points']]
for i in range(1,6):
    constructor_points["constructor_last_" + str(i) + "_points"] = constructor_points.groupby('constructorId')['points'].rolling(i, min_periods = i).sum().groupby(level = 0).shift().fillna(0).reset_index(level = 0)['points']


# export
pickle.dump(driver_points, open("working/driver_points.p", "wb"))
pickle.dump(constructor_points, open("working/constructor_points.p", "wb"))