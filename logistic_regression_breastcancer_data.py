# -*- coding: utf-8 -*-
"""
Created on Wed May 19 17:58:34 2021

@author: RaZz oN
"""
#Importing the breast cancer dataset
from sklearn.datasets import load_breast_cancer

# Loading the dataset into the dataset varaible
dataset = load_breast_cancer()


# =============================================================================
# Splitting the datset into data and target
# =============================================================================

# Data
x = dataset.data

# Target
y = dataset.target 



from sklearn.model_selection import train_test_split


# =============================================================================
# Training and testing of the dataset
# =============================================================================

X_train , X_test , y_train , y_test = train_test_split(x, y, random_state=40, test_size=0.2)



# =============================================================================
# Model Creation and Evaluation
# =============================================================================


from sklearn.linear_model import LogisticRegression

# Use Logistic Regressioon Model 
lR = LogisticRegression(solver='lbfgs', max_iter=10000)

# Fitting the train data into the model
lR.fit(X_train, y_train)

# Using X_test to predict the model
y_pred = lR.predict(X_test)

from sklearn.metrics import accuracy_score

# Checking the accuracy of the model
accuracy_score(y_test, y_pred)

from sklearn.model_selection import cross_val_score

# Using cross validation score for more optimized accuracy measurement
cross_val_score(lR, x, y , cv =10)


"""

Model Evaluation metrics


"""


# =============================================================================
# Confusion matrix
# =============================================================================

from sklearn.metrics import confusion_matrix , classification_report

conf_matrix = confusion_matrix(y_test,y_pred)

class_rep = classification_report(y_test,y_pred)


# =============================================================================
# ROC CURVE
# =============================================================================

from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

# Here, instead of predicting the X_test, we use prediction probability

y_pred = lR.predict_proba(X_test)

# We want only 1 column i.e cancer and we reject not cancer column i.e column 0
y_pred = y_pred[:,1]


FPR, TPR, Thresholds = roc_curve(y_test, y_pred)

plt.plot(FPR,TPR)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()

# =============================================================================
# ROC_AUC SCORE ACCURACY
# =============================================================================
from sklearn.metrics import roc_auc_score

# Using auc and roc score for accuracy
roc_auc_score(y_test, y_pred)
