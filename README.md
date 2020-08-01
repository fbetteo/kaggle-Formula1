# Formula 1 - Who will win?

Based on [this dataset](https://www.kaggle.com/rohanrao/formula-1-world-championship-1950-2020) that has information about Formula 1 races since 1950 I tried to create a model to predict who will end in the first position in each race.  

The dataset not only has the ending position in each race for each driver (our dependent variable) but also has a lot of useful features such as lap times during classification stages, starting position, team results, car failures, etc. This enables us to create a variety of predictors using information previous to the races for each driver.  
I have also addded and cleaned some weather information scraped from Wikipedia for each race based on [this repo](https://github.com/veronicanigro/Formula_1).

## Goals

1) Create a classifier ML project using Python and in particular Pandas and Sklearn with Pipelines that will serve for own future reference. 
2) Predict the first position of races which is a problem with unbalanced classes.
3) Compare different models tuned with CV (GridSearch Cross validation in this case), generate an ensemble and check results againts a benchmark.

## Steps and Results

I have cleaned the data and generated a set of features using different rolling windows since the optimal number is not something I could know beforehand. I have merged all the datasets and removed incomplete rows. That reduces drastically the amount of observations but we are still in a couple thousands obs so let's say it's ok.
At this point I have separated the last 1000 observations and kept them apart from the training setup. These will be our test set. This corresponds roughly to the last 3 seasons where we have data.

Using only the training set I have designated a benchmark model: predict that the winner is the driver that starts in the first position of the grid, called pole position. It's a decent rule of thumb and has a score of 0.587 in precision using all the training data. Not beating that would be a failure.  

Then I tried the following five model architectures and searched for the best parameters using grid search within a 3 fold cross validation setup (the reason of 3 is we don't have that many positive observations). After that I generated a simple voting ensemble model. The Cross validation mean precision scores are:  

### CV results
* Decision Tree: 0.643
* Logistic Regression: 0.748
* Random Forest:0.823
* Neural Network: 0.722
* XGBoost: 0.732
* Benchmark: 0.587

*Side note: The right way should be using Nested CV to avoid over optimistic results. A quick test using Nested was quite close though.*

Overall we get better results than the benchmark in the CV approach which is good. Random Forest being the better one.  
To validate these results we must try the models in the previously isolated test set, which was never used to train anything.

### Test results
* Decision Tree: 0.533
* Logistic Regression: 0.562
* Random Forest: 0.55
* Neural Network: 0.547
* XGBoost: 0.55
* Ensemble: 0.592
* Benchmark: 0.509

We see a surprisringly high drop in precision score for all the models. This could lead to think we committed overfitting during training. However, the benchmark also suffers a drop and is still the worst model. This might be  because of having not enough observations to have consistent scores or because the test set has a different pattern in the positive labels. Since this problem has a temporary dimension, it is possible that in the last 3 seasons the relation between predictors and label mutated or some races had unexpected results compared to the historical pattern.

Interestingly, the ensemble model was able to classify the races with a bit higher precision. Inspectig the confusion matrix it seems this is due to a more conservative approach leading to less False Positives (and keeping True Positives in the ball park of the individual models)

#### DRAFT

oversampling y undersampling para imbalanced class. Por el momento aumnetaron el recall pero redujeron precision. Mas falsos positivos.



Need to Download the odds from somewhere.
Betfair has historical odds for F1 but will need to understand the API.
