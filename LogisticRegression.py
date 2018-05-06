
# coding: utf-8
# Using Breast Cancer DataSet from sklearn
# In[5]:


## import all the necessary packages##
import numpy as np 
import pandas as pd 
import sklearn

from sklearn.datasets import load_breast_cancer 

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

## loading dataset called "load_breast_cancer" into cancer##

cancer = load_breast_cancer()

#print(cancer)

x_train,X_test,y_train,y_test=train_test_split(cancer.data, cancer.target,stratify=cancer.target,random_state=42)

#print(X_test)
#print(y_test)

#print(x_train)
#print(y_train)

log_reg = LogisticRegression()
## C is set to default 1 when the LogisticRegression() is called

p= log_reg.fit(x_train,y_train)

##print(p) - to see the characteristics of the LogisticRegression() function

## lets print the accuracy of the model on both train and test datasets

print('Acurracy on train dataset: {:.3f}'.format(log_reg.score(x_train,y_train)))

print('Accuracy on test dataset: {:.3f}'.format(log_reg.score(X_test,y_test)))


# LogisticRegression depends on : REgularization and C value
#L1: assumes only few features as important
#L2: assumes all features as important


# Lower C can result in underfitting"
## improve the accuracy of the model by setting C to a higher value, as compared to the default C=1 

log_reg100 = LogisticRegression(C=100)

log_reg100.fit(x_train,y_train)

print('Acurracy on train dataset: {:.3f}'.format(log_reg100.score(x_train,y_train)))

print('Accuracy on test dataset: {:.3f}'.format(log_reg100.score(X_test,y_test)))




plt.plot(log_reg.coef_.T,'o',label= 'C=1')
plt.plot(log_reg100.coef_.T,'^',label='C=100')
plt.xticks(range(cancer.data.shape[1]),cancer.feature_names,rotation=90)
plt.hlines(0,0,cancer.data.shape[1])
plt.ylim(-5,5)
plt.xlabel('Coeffcient index')
plt.ylabel('Coeffcient magnitude')
plt.legend()




