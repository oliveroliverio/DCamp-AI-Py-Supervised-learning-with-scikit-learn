# [Supervised learning with Scikit learn](https://www.datacamp.com/courses/supervised-learning-with-scikit-learn)

# 1 Classification



## Supervised learning


## Which of these is a classification problem?

## Exploratory data analysis
### Load Iris dataset
```python
from sklearn import datasets
import pandas as pd

import numpy as np

import matpLotlib.pyplot as plt

plt.style.use('ggplot')
iris = datasets.load_iris()

type(iris)
```

### View data types and dimensions in dataset

```python
type(iris.data)
iris.data.shape
iris.target_names
```

### Load as pandas dataframe
```python
X = iris.data
y = iris.target
df = pd.DataFrame(X, columns=iris.feature_names)
print(df.head())
```

## Numerical EDA

## Visual EDA

```python
_ = pd.plotting.scatter_matrix(df, c = y, figsize = [8, 8],
s=150, marker = 'D')

```

## The classification challenge

### Using scikit-learn to fit a classifier

```python
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=6)
iris = datasets.load_iris()
knn.fit(iris['data'], iris['target'])
```

## k-Nearest Neighbors: Fit

## k-Nearest Neighbors: Predict

## Measuring model performance

### Train/test split
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21, stratify=y)
knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(\"Test set predictions:\\n {}\".format(y_pred))

```


## The digits recognition dataset

## Train/Test Split + Fit/Predict/Accuracy

Now that you have learned about the importance of splitting your data into training and test sets, it's time to practice doing this on the digits dataset! After creating arrays for the features and target variable, you will split them into training and test sets, fit a k-NN classifier to the training data, and then compute its accuracy using the .score() method.

- Import KNeighborsClassifier from sklearn.neighbors and
- train_test_split from sklearn.model_selection
- Create an array for the features using digits.data and an array for the
- target using digits.target .
- Create stratified training and test sets using 0.2 for the size of the test
- set. Use a random state of 42 . Stratify the split according to the labels so
- that they are distributed in the training and test sets as they are in the
- original dataset.
- Create a k-NN classifier with 7 neighbors and fit it to the training data.
- Compute and print the accuracy of the classifier's predictions using the
.score() method.

```python
# Import necessary modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Create feature and target arrays
X = digits.data
y = digits.target

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)

# Create a k-NN classifier with 7 neighbors: knn
knn = KNeighborsClassifier(n_neighbors=7)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Print the accuracy
print(knn.score(X_test, y_test))

```

## Overfitting and underfitting

Remember the model complexity curve that Hugo showed in the video? You will now construct such a curve for the digits dataset! In this exercise, you will compute and plot the training and testing accuracy scores for a variety of different neighbor values. By observing how the accuracy scores differ for the training and testing sets with different values of k, you will develop your intuition for overfitting and underfitting.

The training and testing sets are available to you in the workspace as X_train, X_test, y_train, y_test. In addition, KNeighborsClassifier has been imported from sklearn.neighbors.

Inside the for loop:

- Setup a k-NN classifier with the number of neighbors equal to k .
- Fit the classifier with k neighbors to the training data.
- Compute accuracy scores the training set and test set separately using the .score() method and assign the results to the train_accuracy and test_accuracy arrays respectively.



---










---

# 2 Regression



## Introduction to regression

## Which of the following is a regression problem?

## Importing data for supervised learning

## Exploring the Gapminder data

## The basics of linear regression

## Fit & predict for regression

## Train/test split for regression

## Cross-validation

## 5-fold cross-validation

## K-Fold CV comparison

## Regularized regression

## Regularization I: Lasso

## Regularization II: Ridge


# 3 Fine-tuning your model



## How good is your model?

## Metrics for classification

## Logistic regression and the ROC curve

## Building a logistic regression model

## Plotting an ROC curve

## Precision-recall Curve

## Area under the ROC curve

## AUC computation

## Hyperparameter tuning

## Hyperparameter tuning with GridSearchCV

## Hyperparameter tuning with RandomizedSearchCV

## Hold-out set for final evaluation

## Hold-out set reasoning

## Hold-out set in practice I: Classification

## Hold-out set in practice II: Regression


# 4 Preprocessing and pipelines



## Preprocessing data

## Exploring categorical features

## Creating dummy variables

## Regression with categorical features

## Handling missing data

## Dropping missing data

## Imputing missing data in a ML Pipeline I

## Imputing missing data in a ML Pipeline II

## Centering and scaling

## Centering and scaling your data

## Centering and scaling in a pipeline

## Bringing it all together I: Pipeline for classification

## Bringing it all together II: Pipeline for regression

## Final thoughts

