# Formula 1 - Who will win?

Based on [this dataset](https://www.kaggle.com/rohanrao/formula-1-world-championship-1950-2020) that has information about Formula 1 races since 1950 I tried to create a model to predict who will end in the first position in each race.  

The dataset not only has the ending position in each race for each driver (our dependent variable) but also has a lot of useful features such as lap times during classification stages, starting postion, team results, car failures, etc. This enables us to create a variety of predictors using information previous to the races for each driver.  
I have also addded and cleaned some weather information scraped from Wikipedia for each race based on [this repo](https://github.com/veronicanigro/Formula_1.

## Goals

1) Create a classifier ML project using Python and in particular Pandas and Sklearn with Pipelines that will serve for own future reference. 
2) Predict the first position of races which is a problem with unbalanced classes.
3) Compare different models tuned with CV (GridSearch Cross validation in this case), generate an ensemble and check results againts a benchmark.

## Steps and Results

I have cleaned the data and generated a set of features using different rolling windows since the optimal number is not something I could know beforehand. I have merged all the datasets and removed incomplete rows. That reduces drastically the amount of observations but we are still in a couple thousands obs so let's say it's ok.

Then I have designated a benchmark model: predict that the winner is the driver that starts in the first position of the grid, called pole position. It's a decent rule of thumb and has a score of 0.587 in precision using all the training data. Not beating that would be a failure.  

I tried the following four model architectures and searched for the best parameters using grid search within a 5 fold cross validation setup. After that I generated a simple voting ensemble model. Between parenthesis I state the mean precision score 





Need to Download the odds from somewhere.
Betfair has historical odds for F1 but will need to understand the API.
