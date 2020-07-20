import pandas as pd
import pickle
import numpy as np

# data
pits_data = "data/pit_stops.csv"
races_data = 'data/races.csv'

results = pd.read_csv(pits_data)
races = pd.read_csv(races_data)

results_2 = results.merge(races, how = "left", on = "raceId")
results_2
# Generate cumulated points over last 1 to 5 races per driver
driver_q_stops = results_2.sort_values(['driverId','year', 'raceId'])[['raceId', 'driverId', 'stop', 'milliseconds']] # by year because raceid is not chronological between years (but yes in each one)
# table to get the amount of stops per race and driver
driver_q_stops = driver_q_stops.groupby(['driverId', 'raceId'])['stop'].max().reset_index()
# how many stops had on average in the last i races
for i in range(1,6):
    driver_q_stops["driver_last_" + str(i) + "_qstops"] = driver_q_stops.groupby('driverId')['stop'].rolling(i, min_periods = i).mean().groupby(level = 0).shift().fillna(0).reset_index(level = 0)['stop']
# Should I change the 0 with some value? Average of the driver in the circuit?


# how much time in average the stop lasted (last i stops)
# no funciono. tengo que calcular para cada driver id pero solo si stop = 1 porque si calculo para stop = 2, voy a incluir data de la misma carrera.
# revisarrrr
driver_time_stops = results_2.sort_values(['driverId','year', 'raceId'])[['raceId', 'driverId', 'stop', 'milliseconds']] # by year because raceid is not chronological between years (but yes in each one)

driver_time_stops["driver_last_timestops"] = driver_time_stops.groupby('driverId').apply(lambda x: pits_time_prev_race(x))#.groupby(level = 0).shift().fillna(0).reset_index(level = 0)['milliseconds']
driver_time_stops
# for i in range(1,6):
#     driver_time_stops["driver_last_" + str(i) + "_timestops"] = driver_time_stops.groupby('driverId')['milliseconds'].rolling(i, min_periods = i).mean().groupby(level = 0).shift().fillna(0).reset_index(level = 0)['milliseconds']



# Generate cumulated points over last 1 to 5 races per constructor
constructor_points = results_2.sort_values(['constructorId','year', 'raceId'])[['raceId', 'constructorId', 'points']]
for i in range(1,6):
    constructor_points["constructor_last_" + str(i) + "_points"] = constructor_points.groupby('constructorId')['points'].rolling(i, min_periods = i).sum().groupby(level = 0).shift().fillna(0).reset_index(level = 0)['points']


# export
pickle.dump(driver_q_stops, open("working/driver_q_stops.p", "wb"))
# pickle.dump(constructor_points, open("working/constructor_points.p", "wb"))