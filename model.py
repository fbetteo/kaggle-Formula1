import pandas as pd
import numpy as np

laptimes = pd.read_csv("data/lap_times.csv")
races = pd.read_csv("data/races.csv")
results = pd.read_csv("data/results.csv")
laptimes_2 = pd.merge(laptimes, races, on = "raceId")