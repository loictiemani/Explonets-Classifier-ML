# machine-learning-challenge

<img src= "https://github.com/loictiems/machine-learning-challenge/blob/main/image/exoplanets.jpg">

Over a period of nine years in deep space, the NASA Kepler space telescope has been out on a planet-hunting mission to discover hidden planets outside of our solar system.
To help process this data, this project will create machine learning models capable of classifying candidate exoplanets from the raw dataset.

## Files Index

Following files are attached:

1. <a href="https://github.com/loictiems/machine-learning-challenge/blob/main/model_1.ipynb">model_1.ipynb</a>: Model 1 with **KNN** classifier

2. <a href="https://github.com/loictiems/machine-learning-challenge/blob/main/model_2.ipynb">model_2.ipynb</a>: Model 2 with **Logistic Regression**

3. <a href="https://github.com/loictiems/machine-learning-challenge/blob/main/model_3.ipynb">model_2.ipynb</a>: Model 3 with **Random Forest**

4. <a href="https://github.com/loictiems/machine-learning-challenge/blob/main/model_rf.sav">model_rf.sav</a>: Dumped trained model file


## GridSearch for Optimization of Model Parameters

* For KNN model:

```python
param_grid = {
    "n_neighbors": range(1, 20, 2),
    "weights": ['uniform', 'distance'],
    "metric": ["euclidean", "manhattan"]
}

# Output of Best Estimator: 
KNeighborsClassifier(metric='manhattan', n_neighbors=17, weights='distance')

```
* For Logistic Regression model:

```python
param_grid = {
    "C": np.logspace(-3,3,7),
    "penalty": ['l1', 'l2'],
    "solver": ['newton-cg', 'lbfgs', 'liblinear', 'sag']
}

# Output of Best Estimator: 
LogisticRegression(C=100.0, penalty='l1', solver='liblinear')

```

* For Random Forest model:

```python
param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}

# Output of Best Estimator: 
RandomForestClassifier(criterion='entropy', max_depth=8, n_estimators=200)

```

## Tesing Score Comparison

| Model | Testing Data Score | Testing Data Score after Feature Selection |
|---|---|---|
| KNN | 84.6% | NA |
| Logistic Regression | 89.3% | 88.7% (CFE) |
| Random Forest | 88.9% | 87.2% (.feature_importances_) |

By using the GridSearchCV function, it takes a dict of all possible parameters in a for loop for getting the best parameters. If `n_jobs=-1` is given, the search can be done in parallel calculation which saves a lot of time in fitting hundreds of candidates/fits.

So far, *Logistic Regression and Random Forest models* have the best score among these three models. Results of other models are to be determined. The current model has more than 85% of accuracy to predict new exoplanets.

## Feature Selection

### Recursive Feature Elimination (RFE)

After using Recursive Feature Elimination (RFE) for Logistic Regression, **20 features are considered less important** to the model and got removed from the training data. 

Results show that the **RFE testing data score has 88.7%** which is very similar to the **original score (89.3%)**. It is obvious that RFE can *reduce the complexity* of this model, and accelerate the calculation while keeping *a similar accuracy result*.

### .feature_importances_ (FI)

Similar to RFE, `.feature_importances_` can provide the ranking of the important features. After sorting the list of feature importances, **all 7 features with importance lower than 0.01** have been removed from the training data.

Results show that new training data with **selected features gives 87.2% of accuracy** while original data has **88.9% accuracy**. They are pretty close, but not as close as RFE method.

* Feature Importances Ranking of Random Forest Model:

<img src="https://github.com/loictiems/machine-learning-challenge/blob/main/image/model3_randomforest_featuresimportances.png">


* Random Forest Model Classification Report:

<img src="https://github.com/loictiems/machine-learning-challenge/blob/main/image/model3_randomforest_report.png">
