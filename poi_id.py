#!/usr/bin/python

import sys
import pickle
import numpy as np
import pandas as pd
sys.path.append("C:/Users/Dominik/Documents/GitHub/ud120-projects/tools")
from time import time

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from tester import test_classifier
import pprint

### Task 1: Select what features you'll use.
# features_list is a list of strings, each of which is a feature name.
# The first feature must be "poi".

# I follow two different strategies to select features. First, I select features
# which show the highest correlations with 'poi'. These should be good predictors.
# Second, I use a maximum amount of data (all features with more than 50
# percent of non-missing observations) and PCA to predict 'poi'. 
# The performance of both strategies are compared.

features_list = ['poi','to_messages', 'from_poi_to_this_person', 'from_this_person_to_poi',
                 'shared_receipt_with_poi'] 

### Load the dictionary containing the dataset
data_dict = pickle.load(open("C:/Users/Dominik/Documents/GitHub/ud120-projects/final_project/final_project_dataset.pkl", "r"))

### Impute missing values
enron_data = data_dict
enron_df = pd.DataFrame.from_dict(enron_data, dtype = float)
enron_df
d = {True:1, False:0}
enron_df = enron_df.applymap(lambda x: d.get(x,x))
enron_df = enron_df.transpose()
enron_df = enron_df[['poi', 'salary', 'from_poi_to_this_person', 'from_this_person_to_poi', 
'shared_receipt_with_poi', 'bonus', 'deferral_payments', 'deferred_income', 'exercised_stock_options',
'expenses', 'from_messages', 'loan_advances', 'long_term_incentive', 'restricted_stock',
'restricted_stock_deferred', 'to_messages', 'total_payments', 'total_stock_value']] 
enron_df.dtypes
enron_df = enron_df.astype(float)
enron_df.dtypes

enron_df.count()
# Features with more than 50 percent of missing observations:
# deferral_payments, deferred_income, loan_advances, long_term_incentive,
# restricted_stock_deferred
# I remove these features as the large number of missings does not allow us to
# estimate reliable means
enron_df = enron_df[['poi', 'salary', 'from_poi_to_this_person', 'from_this_person_to_poi', 
'shared_receipt_with_poi', 'bonus', 'exercised_stock_options',
'expenses', 'from_messages', 'restricted_stock',
'to_messages', 'total_payments', 'total_stock_value']] 
enron_df.count()

# I use the median as it is more robust to outliers
enron_df = enron_df.fillna(enron_df.median())

# Check that I have to features with highest correlation to poi
corrmat = enron_df.corr()
corrmat

### Remove outliers

# Even though I do not use salary as feature, it could help to
# identify outlier
pd.DataFrame.hist(enron_df, column= 'salary')
x = max(enron_df["salary"])
enron_df[(enron_df.salary==x)]
enron_df = enron_df.drop(['TOTAL'])

pd.DataFrame.hist(enron_df, column= 'from_poi_to_this_person')
pd.DataFrame.hist(enron_df, column= 'from_this_person_to_poi')
pd.DataFrame.hist(enron_df, column= 'shared_receipt_with_poi')
pd.DataFrame.hist(enron_df, column= 'to_messages')

enron_df_final = enron_df[features_list]

### Task 3: Create new feature(s)
# It might be that from_poi_to_this_person and from_this_person_to_poi are
# strongly correlated as email traffic usually goes both ways.
# If that would be true we could aggregate both variables, e.g. factor analysis 
email_vars = enron_df_final[['from_poi_to_this_person', 'from_this_person_to_poi']]
corrmat = email_vars.corr()
corrmat
# Yet, correlation is quite low. The two features probabliy do not belong to 
# a common latent factor.

# The histograms also showed that the scales of the feature vary substantially.
# I therefore apply the MinMax Scaler below. This is required for the 
# Support Vector Machines Classifier I will use later on. 
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()

for feature in features_list:
    enron_df_final[feature] = min_max_scaler.fit_transform(np.array(enron_df[feature]))


# Store to my_dataset for easy export below.
enron_df_final = enron_df_final.transpose()
final_dict = enron_df_final.to_dict()
my_dataset = final_dict

# Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# Note: Additional features are created below when using the PCA strategy!

### Task 4: Try a varity of classifiers
### I first try a number of classifiers using the four selected features
### After selecting the algorithm, I compare the performance of the four features
### to the PCA strategy 
 
### Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve

clf_nbays = GaussianNB()

# fit classifier
t0 = time()    
clf_nbays.fit(features, labels)
print "fitting time:", round(time()-t0, 3), "s"
    
# use classifier to predict labels
t1 = time()     
pred = clf_nbays.predict(features)
print "prediction time:", round(time()-t1, 3), "s"

accuracy_nbays = accuracy_score(labels, pred)
print accuracy_nbays

precision, recall, thresholds = precision_recall_curve(labels, pred)
print(precision)
print(recall)
print(thresholds)


### Decision Tree
from sklearn import tree

# create classifier
clf_dc = tree.DecisionTreeClassifier()
    
# fit the classifier 
t0 = time()    
clf_dc.fit(features, labels)
print "fitting time:", round(time()-t0, 3), "s"
    
# use classifier to predict labels
t1 = time()     
pred = clf_dc.predict(features)
print "prediction time:", round(time()-t1, 3), "s"

accuracy_dc = accuracy_score(labels, pred)
print(accuracy_dc)

precision, recall, thresholds = precision_recall_curve(labels, pred)
print(precision)
print(recall)
print(thresholds)

### Support Vector Machines
from sklearn.svm import SVC
svc_clf = SVC()

t0 = time()    
svc_clf.fit(features, labels) 
print "fitting time:", round(time()-t0, 3), "s"

t1 = time()     
pred = svc_clf.predict(features)
print "prediction time:", round(time()-t1, 3), "s"

accuracy_svc = accuracy_score(labels, pred)
print(accuracy_svc)

precision, recall, thresholds = precision_recall_curve(labels, pred)
print(precision)
print(recall)
print(thresholds)

# In sum, the Decision Tree seems quite fast and provides highest accuracy

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

print(features_list)
print(labels)
### Decision Tree with four features and cross-validation
# Trying different specifications, I was able to boost precision and recall
# by increasing min_samples_split with gini criterion and balanced class weight

clf = tree.DecisionTreeClassifier(min_samples_split=15, criterion="gini",
                                  class_weight="balanced")
    
t0 = time()    
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
    
t1 = time()     
pred = clf.predict(features_test)
print "prediction time:", round(time()-t1, 3), "s"

accuracy = accuracy_score(labels_test, pred)
print(accuracy)

precision, recall, thresholds = precision_recall_curve(labels_test, pred)
print(precision)
print(recall)
print(thresholds)

# Feature importance
clf.feature_importances_

# The most important feature is 'from_this_person_to_poi'. The second most important
# feature is 'shared_receipt_with_poi'. Compoared to these to features, the
# contributions of 'to_messages' and 'from_poi_to_this_person' are marginal.

# The previous cross-validation stategy is of limited use as N is quite low
# I rather rely on stratified shuffle split cross validation provided in
# the test_classifier function
test_classifier(clf, my_dataset, features_list, folds = 1000)
# Feature importance
clf.feature_importances_

# I now will compare this performance to the PCA strategy

### Decision Tree with PCA and cross-validation
from sklearn.decomposition import PCA
pca = PCA(n_components=4)
enron_pca = enron_df.iloc[0:145,1:13]
pca.fit(enron_pca)
comps = pd.DataFrame(pca.components_, columns=list(enron_pca.columns.values))
print "Explained variance by component: %s" % pca.explained_variance_ratio_
# The first principal component explains already 0.82 percent of the variance
# in the data. 

enron_pca = pd.DataFrame(pca.transform(enron_pca))
enron_poi = enron_df.iloc[0:145,0:1]
enron_poi = enron_poi.reset_index()
enron_poi = enron_poi.iloc[0:145,1:2]
frames = [enron_poi, enron_pca]
enron_pca_df = pd.concat(frames, axis = 1, ignore_index=True)
enron_pca_df.columns = ['poi', 'pc1', 'pc2', 'pc3', 'pc4']

enron_pca_df = enron_pca_df.transpose()
final_dict_2 = enron_pca_df.to_dict()
my_dataset_2 = final_dict_2
pprint.pprint(my_dataset_2)

features_list_pca = ['poi', 'pc1', 'pc2', 'pc3', 'pc4']

test_classifier(clf, my_dataset_2, features_list_pca, folds = 1000)
clf.feature_importances_

# Comparing the performance of the PCA strategy with the correlation-based
# feature selection, we see that correlation-based selection performes a bit better.
# Accuracy and precision are higher with the correlaton-based selection.
# Yet, recall is larger using the PCA strategy. Hence, the larger amount of 
# information under the PCA strategy boosts the ability to recover the positive
# samples. Yet, it is likely that the larger amount of noise that goes into
# the PCA strategy (all 13 features, some rather irrelevant) depresses 
# accuracy and precision. In sum, both strategies deliver acceptable results. 



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
