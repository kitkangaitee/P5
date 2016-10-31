#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn import preprocessing
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import tree
from tester import test_classifier

### Select features
### The first feature must be "poi".

features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 
'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 
'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 
'long_term_incentive', 'restricted_stock', 'director_fees', 'to_messages', 
'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 
'shared_receipt_with_poi'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Remove outliers
data_dict.pop('TOTAL',0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK',0)

### Create new feature(s)
for name in data_dict:
    data_point = data_dict[name]
    if data_point['total_payments'] == 'NaN' or data_point['total_stock_value'] == 'NaN':
        data_point['stocks_payment_fraction'] = 0.0
    else:
        data_point['stocks_payment_fraction'] = \
        float(data_point['total_stock_value'])/float(data_point['total_payments'])

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

sss = StratifiedShuffleSplit(labels, 100, random_state = 42)

scaler = preprocessing.StandardScaler()
pca = PCA(n_components=2)
selector = SelectKBest(k=2)
combined_features = FeatureUnion([("pca", pca), ("univ_select", selector)])

### Test different classifers
'''
### Gaussian Naive Bayes
gnb = GaussianNB()
pipe_gnb = Pipeline([("scaler", scaler), ("features", combined_features), ("gnb", gnb)])

param_grid_gnb = dict(features__pca__n_components=[1, 2, 3, None],
                  features__univ_select__k=[1, 2, 3])

gs_gnb = GridSearchCV(pipe_gnb, param_grid=param_grid_gnb, cv=sss, scoring='f1')
gs_gnb.fit(features, labels)
clf_gnb = gs_gnb.best_estimator_

print 'Best parameters for Gaussian Naive Bayes'
print gs_gnb.best_params_
print ' '
print "Tester Classification report:" 
test_classifier(clf_gnb, my_dataset, features_list)

### RandomForest
rf = RandomForestClassifier()
pipe_rf = Pipeline([("scaler", scaler), ("features", combined_features), ("rf", rf)])

param_grid_rf = dict(features__pca__n_components=[1, 2, 3, None],
                  features__univ_select__k=[1, 2, 3],
                rf__n_estimators=[10, 30, 50],
                rf__min_samples_split=[5,20,50])

gs_rf = GridSearchCV(pipe_rf, param_grid=param_grid_rf, cv=sss, scoring='f1')
gs_rf.fit(features, labels)
clf_rf = gs_rf.best_estimator_

print 'Best parameters for RandomForest'
print gs_rf.best_params_
print ' '
print "Tester Classification report:" 
test_classifier(clf_rf, my_dataset, features_list)

### Decision Tree
dt = tree.DecisionTreeClassifier(random_state = 42)
pipe_dt = Pipeline([("scaler", scaler), ("features", combined_features), ("dt", dt)])

param_grid_dt = dict(features__pca__n_components=[1, 2, 3, None],
                  features__univ_select__k=[1, 2, 3],
                dt__min_samples_split=[6, 10, 30])

gs_dt = GridSearchCV(pipe_dt, param_grid=param_grid_dt, cv=sss, scoring='f1')
gs_dt.fit(features, labels)

clf_dt = gs_dt.best_estimator_

print 'Best parameters for Decision Tree'
print gs_dt.best_params_
print ' '
print "Tester Classification report:" 
test_classifier(clf_dt, my_dataset, features_list)
'''

### AdaBoost
ada = AdaBoostClassifier(random_state = 42)
pipe_ada = Pipeline([("scaler", scaler), ("features", combined_features), ("ada", ada)])

param_grid_ada = dict(features__pca__n_components=[1, 2, 3, None],
                  features__univ_select__k=[1, 2, 3, 4],
                ada__n_estimators=[5, 10, 30, 50])

gs_ada = GridSearchCV(pipe_ada, param_grid=param_grid_ada, cv=sss, scoring='f1')
gs_ada.fit(features, labels)
clf_ada = gs_ada.best_estimator_

print 'Best parameters for AdaBoost'
print gs_ada.best_params_
print ' '
print "Tester Classification report:" 
test_classifier(clf_ada, my_dataset, features_list)

# Getting feature score for SelectKBest
features_k= gs_ada.best_params_['features__univ_select__k']
SKB_k=SelectKBest(k=features_k)
SKB_k.fit_transform(features, labels)   
feature_scores = ['%.2f' % elem for elem in SKB_k.scores_]
features_selected=[(features_list[i+1], feature_scores[i]) for i in SKB_k.get_support(indices=True)]
features_selected=sorted(features_selected, key=lambda feature: float(feature[1]) , reverse=True)

print features_selected

dump_classifier_and_data(clf_ada, my_dataset, features_list)

