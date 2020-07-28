# Formula1 - Who will win?

Based on [this dataset](https://www.kaggle.com/rohanrao/formula-1-world-championship-1950-2020) that has information about Formula 1 races since 1950 I tried to create a model to predict who will end in the first position in each race.  
The dataset not only has the ending position in each race for each driver (our dependent variable) but also has a lot of useful features such as lap times during classification stages, starting postion, team results, car failures, etc. This enables us to create a variety of predictors using information of previous races for each driver.  
I have addded and cleaned some weather information scraped from Wikipedia for each race based on [this repo](https://github.com/veronicanigro/Formula_1.


Need to Download the odds from somewhere.
Betfair has historical odds for F1 but will need to understand the API.
