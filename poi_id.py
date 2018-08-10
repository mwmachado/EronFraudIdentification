#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

import pickle
import pandas as pd
import numpy as np

from time import time

from sklearn.preprocessing import Imputer, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier, dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# features_list = ['poi','salary'] # You will need to use more features

### Load the dictionary containing the dataset
with open('final_project_dataset.pkl', 'rb') as f:
    dic = pickle.load(f)

data = pd.DataFrame.from_dict(dic, orient='index')
data.replace('NaN', np.nan, inplace=True)

label = 'poi'

del data['email_address']

s = data[data[label] == 1].isnull().sum()
limit = data[label].value_counts()[1]/3
few_poi_values = s[s > limit].index.tolist()

payments = ['salary',
            'deferral_payments',
            'loan_advances',
            'bonus',
            'deferred_income',
            'expenses',
            'long_term_incentive',
            'other',
            'director_fees',
            'total_payments']

data[payments] = data[payments].fillna(0)

correct = ['deferred_income','deferral_payments', 'expenses', 'director_fees', 'total_payments']
data.loc['BELFER ROBERT',correct] = np.array([-102500, 0, 3285, 102500, 3285])

correct = ['other', 'expenses', 'director_fees', 'total_payments']
data.loc['BHATNAGAR SANJAY',correct] = np.array([0, 137864, 0, 137864])

stock = ['restricted_stock_deferred',
         'restricted_stock',
         'exercised_stock_options',
         'total_stock_value']

data[stock] = data[stock].fillna(0)

correct = ['restricted_stock_deferred','restricted_stock', 'exercised_stock_options', 'total_stock_value']
data.loc['BELFER ROBERT',correct] = np.array([-44093, 44093, 0, 0])

correct = ['restricted_stock_deferred','restricted_stock', 'exercised_stock_options', 'total_stock_value']
data.loc['BHATNAGAR SANJAY',correct] = np.array([-2604490, 2604490, 15456290, 15456290])

email = ['to_messages',
         'from_poi_to_this_person',
         'from_messages',
         'from_this_person_to_poi',
         'shared_receipt_with_poi']

imp = Imputer(np.nan)
data.loc[data[label] == 1, email] = imp.fit_transform(data[email][data[label]==1])
data.loc[data[label] == 0, email] = imp.fit_transform(data[email][data[label]==0])

payments = list(set(payments)-set(few_poi_values))
stock    = list(set(stock)-set(few_poi_values))
email    = list(set(email)-set(few_poi_values))

data = data[[label]+payments+stock+email]





### Task 2: Remove outliers

data.drop('LOCKHART EUGENE E', inplace=True)
data.drop(['TOTAL', 'THE TRAVEL AGENCY IN THE PARK'], inplace=True)


### Task 3: Create new feature(s)

data['ratio_from_poi'] = data.from_this_person_to_poi/data.from_messages
data['ratio_to_poi']   = data.from_poi_to_this_person/data.to_messages
data['ratio_with_poi'] = data.shared_receipt_with_poi/data.to_messages
new = ['ratio_with_poi', 'ratio_to_poi', 'ratio_from_poi']
email = new


features = payments+stock+email
features_list = [label]+features


### Store to my_dataset for easy export below.
# my_dataset = data_dict
my_dataset = data.to_dict(orient='index')

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes  import GaussianNB
from sklearn.tree         import DecisionTreeClassifier
from sklearn.ensemble     import RandomForestClassifier

# clf = GaussianNB()
clf = DecisionTreeClassifier()
# clf = RandomForestClassifier()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
cv = StratifiedShuffleSplit(n_splits=1000, random_state=42)
scoring = 'f1_macro'
skb = {'SKB__k': ['all'] + list(range(2,len(features[0]),2))}

params = {}
params.update(skb)

t0 = time()

dt = {'clf__criterion'        : ['gini', 'entropy'],
      'clf__max_depth'        : [2, 4, 6],
      'clf__min_samples_leaf' : [2, 4, 6],
      'clf__random_state'     : [42]}

dt.update(params)

pipe = Pipeline(steps=[('SKB', SelectKBest()), ('clf', DecisionTreeClassifier())])
clf = GridSearchCV(pipe, param_grid = dt, cv=cv, scoring=scoring).fit(features, labels)
dt = clf.best_estimator_


print('Params Tunning:', round(time() - t0, 3), 'segundos')
print('Best Params: ', clf.best_params_)

t0 = time()
test_classifier(dt, my_dataset, features_list)
print ('Validation Time:', round(time() - t0, 3), 'segundos')

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)