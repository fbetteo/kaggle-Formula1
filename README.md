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

Using only the training set I have designated a benchmark model: predict that the winner is the driver that starts in the first position of the grid, called pole position. It's a decent rule of thumb and has a score of 0.596 in precision using all the training data. Not beating that would be a failure.  

Then I tried the following five model architectures and searched for the best parameters using grid search within a 3 fold cross validation setup (the reason of 3 is we don't have that many positive observations). After that I generated a simple voting ensemble model. The Cross validation mean precision scores are:  

### CV results
* Decision Tree: 0.618
* Logistic Regression: 0.622
* Random Forest:0.745
* Neural Network: 0.619
* XGBoost: 0.606
* Benchmark: 0.596

*Side note: The right way should be using Nested CV to avoid over optimistic results. A quick test using Nested was quite close though.*

Overall we get better results than the benchmark in the CV approach which is good. Random Forest being the better one.  
To validate these results we must try the models in the previously isolated test set, which was never used to train anything.

### Test results
* Decision Tree: 0.58
* Logistic Regression: 0.538
* Random Forest: 0.50
* Neural Network: 0.565
* XGBoost: 0.558
* Ensemble: 0.585
* Benchmark: 0.509

We see a notorious drop in precision score for all the models. This could lead to think we committed overfitting during training. However, the benchmark also suffers a drop and is the second worst model. This might be  because of having not enough observations to have consistent scores or because the test set has a different pattern in the positive labels. Since this problem has a temporary dimension, it is possible that in the last 3 seasons the relation between predictors and label mutated or some races had unexpected results compared to the historical pattern.

Interestingly, the ensemble model was able to classify the races with a bit higher precision. Inspecting the confusion matrix it seems this is due to an -expected- more conservative approach keeping the False Positives in the low end and True positives in the high end but without getting to the maximum values of any individual model.

All in all the models seem to learn something from the data and predict with higher accuracy the winner than the benchmark. 

## Comments

* I tried a few Sampling approachs (RandomOverSampling, Smote, Adasyn -oversampling-, ClusterCentroids -undersampling-) and precision score was consistently better in CV without using them.  
* I have removed the predictors related to amount of pit stops in previous races because it reduced the sample by an extra 50% and CV vs test metrics had a huge gap. Without them the score is much more stable so I think it is convenient to keep more observations but less predictors in this case.  
* Nested CV should be used to reduce over optimistic results in CV metrics.

## Next steps

More features could be included into the model. For example, Age of the driver or predictors related to failures of the cars in previous races.  
Get the odds on the raceday to see if the model is able to generate some profit. For the test set they might be possible to get (betfair?) I doubt it for all the training data. Having it would be great to optimize the parameters based on profit and not precision.

