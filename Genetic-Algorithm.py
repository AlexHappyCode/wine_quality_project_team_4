#!/usr/bin/env python
# coding: utf-8

# # Genetic Data Machine Learning Model
# 
# ### By Alexander Pena
# 
# #### Assisted by: Ted

# # Library imports

# In[1]:


import os
import warnings
from IPython import display
import yfinance as yf
import datetime as dt
import pandas as pd
import numpy as np
from numpy import arange
import matplotlib.pyplot as plt
from pandas import read_csv
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import f1_score 
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance


# # Data Import
# 
# First will look at the e1_positive.csv dataset

# In[2]:


df = pd.read_csv("e1_positive.csv")
df


# # Dataset Characteristics
# 
# All the samples in the dataset are numeric

# ### Number of samples 

# In[3]:


print('Number of samples:', df.shape[0])


# ### Classification Split

# In[4]:


print('Number of 0 labels: ', len(df[df.Label==0]))
print('Number of 1 labels: ', len(df[df.Label==1]))


# ### Datatype

# ### Range of Data

# In[5]:


print('Biggest value in the dataset:', df.min().min())
print('Biggest value in the dataset:', df.max().max())
print('Mean of all the means:', df.mean().mean())
print('Standard Deviation of all the standard deviations:', df.std().std())


# ### How many values are just zeros?

# In[6]:


vals = []
for index, row in df.iterrows():
    vals.append(row[row == 0].value_counts()[0])
num_zeroes = pd.Series(vals)
print('Mean number of zeroes between features:', num_zeroes.mean())
print('Standard Deviation of the number of zeroes between features:', num_zeroes.std())


# # splitting data into features and labels
# ### Dropping the labels for our feature matrix

# In[7]:


y = df.iloc[:, df.shape[1] - 1].values
x = df.drop(['Label'], axis=1)


# # Split the data into trainning set and testing set
# ### Supervised Learning

# In[8]:


x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.25,
    #random_state=0
    shuffle=True
)


# # Creating the Random Forest Classifier

# In[9]:


model = RandomForestClassifier (
    n_estimators=500, # ntree
    max_features=6, # mtry
    bootstrap=True, # yes
    random_state=42, # for testing
    oob_score=True # need oob score
)


# # Hyperparameter tuning
# 
# The hyperparameters in the random forest model are either used to increase the predictive power of the model or to make the model faster. 

# # Cross Validation

# | n_estimators      | number of trees in the forest 
# |-------------------|:------------------------------
# | max_depth         | maximum depth in a tree
# | min_samples_split | minimum number of samples to allow a split in an internal node
# | min_samples_leaf  | specifies the minimum number of samples required to be at a leaf node
# | bootstrap         |Bootstrap=True (default): samples are drawn with replacement Bootstrap=False : samples are drawn without replacement
# | random_state  | generated random numbers for the random forest.

# In[10]:


grid_ranges = {
    'n_estimators': [500, 1000],
    'max_features': np.arange(1, 5, 1),
    'bootstrap': [True],
    #'random_state': [1, 2, 30, 42],
}
gscv = GridSearchCV(
    estimator=model,
    param_grid=grid_ranges,
    cv=3,
    n_jobs=-1,
    verbose=1,
)
gscv_fit = gscv.fit(x_train, y_train)
best_parameters = gscv_fit.best_params_
print(best_parameters)


# # Recreate the Random Forest Model with the best found hyperparameters

# In[11]:


model = RandomForestClassifier(
    n_estimators=best_parameters['n_estimators'],
    max_features=best_parameters['max_features'],
    bootstrap=best_parameters['bootstrap'],
    oob_score=True
)
model.fit(x_train, y_train)


# # Now running the model on the test data

# ### Running model on entire testing data

# In[74]:


predict = model.predict(x_test)


# ### Taking a single positive and negative sample

# #### Negative Sample

# In[81]:


index = 0 # index of sample to predict
single_sample = [x_test.iloc[index].to_numpy()]


# In[82]:


get_ipython().run_cell_magic('time', '', 'with warnings.catch_warnings(record=True):\n    single_predict = model.predict(single_sample)\n')


# In[85]:


print('predicted:', single_predict[0], ' actual', y_test[index])


# #### Positive Sample

# In[84]:


index = 1 # index of sample to predict
single_sample = [x_test.iloc[index].to_numpy()]


# In[83]:


get_ipython().run_cell_magic('time', '', 'with warnings.catch_warnings(record=True):\n    single_predict = model.predict(single_sample)\n')


# In[ ]:


print('predicted:', single_predict[0], ' actual', y_test[index])


# In[46]:


#predict = model.predict(x_test.iloc[0])
x_test.shape


# # Cut Off
# - Change the cutoff from anyhwere to 0 and 1
# - The model should be already trained and is the variable 'model'
# - The default value is 0.5

# In[32]:


cutoff = 0.5
y_pred_threshold = (model.predict_proba(x_test)[:, 1] > cutoff).astype('float')

# Confusion Matrix of the results with change in cutoff
confusion_matrix(y_test, y_pred_threshold)


# In[15]:


model.predict_proba(x_test)[0:10] # proportion of votes for each sample


# # Metrics

# ### Confusion Matrix

# In[ ]:


cm = confusion_matrix(y_test, predict, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()


# ### Accuracy

# In[19]:


ac = accuracy_score(predict, y_test)
print('Accuracy is:', ac)


# ### Classification
# 
# Report, precision, recall, f1-score, support, accuracy

# In[38]:


from sklearn.metrics import classification_report
labels = ['class 0', 'class 1']
print(classification_report(y_test, predict, target_names=labels))


# ### The out of bag error, OOB

# In[39]:


print('The out of bag error is', model.oob_score_)


# ### Feature Ranking

# In[16]:


''' First extracting feature rankings and standardizing '''
# Get importances
importances = model.feature_importances_
# Standardize Importances
std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
feature_names = list(x_train.columns)
features_dict = {key: val for key, val in zip(feature_names, std)}
feature_rankings = {k: v for k, v in sorted(features_dict.items(), key=lambda item: item[1], reverse=True)}
feature_rankings_list = list(feature_rankings.items())


# In[17]:


import matplotlib.pyplot as plt

# number of features to compare

n = 10
target_features = feature_rankings_list[:n]
feature_names = [tup[0] for tup in target_features]
feature_values = [tup[1] for tup in target_features]

fig = plt.figure(figsize = (10, 5))

plt.xticks(rotation=45, ha='right')

# creating the bar plot
plt.bar(feature_names, feature_values, color ='maroon',
        width = 0.4)

plt.xlabel('Feature')
plt.ylabel('Rank Normalized [0, 1]')
plt.title(f'Feature Ranking first {n}')

plt.show()
           


# In[18]:


import matplotlib.pyplot as plt

# number of features to compare

n = 50
target_features = feature_rankings_list[:n]
feature_names = [tup[0] for tup in target_features]
feature_values = [tup[1] for tup in target_features]

fig = plt.figure(figsize = (10, 5))

plt.xticks(rotation=45, ha='right')

# creating the bar plot
plt.bar(feature_names, feature_values, color ='maroon',
        width = 0.4)

plt.xlabel('Feature')
plt.ylabel('Rank Normalized [0, 1]')
plt.title(f'Feature Ranking first {n}')

plt.show()
           

