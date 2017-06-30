# Antonio Minondo
# dengai.py
# this file performs explanatory analysis on dengai data as well as produces
# a DNN regressor on the data to predict the total cases of outbreaks in San
# Juan PR and Iquitos Peru
# 06/30/2017
# this script was generated from a jupyter notebook

get_ipython().magic('matplotlib inline')

import pandas as  pd
import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf
import seaborn as sns

from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection

# just for the sake of this blog post!
from warnings import filterwarnings
filterwarnings('ignore')



'''
AFTER EXPLANATORY ANALYSIS SHOWN FURTHER DOWN IT WAS DETERMINED THAT
THE VARIABLES THAT HAVE THE BIGGEST EFFECT ARE TEMP AND HUMIDITY
this following function preprocesses that data, by only selecting these
features, fills their na values and separates them into separate cities

'''


# make function to preprocess data
def preprocess_data(data_path, labels_path=None):
    # load data and set index to city, year, weekofyear
    df = pd.read_csv(data_path, index_col=[0, 1, 2])

    # select features we want
    features = ['reanalysis_specific_humidity_g_per_kg',
                 'reanalysis_dew_point_temp_k',
                 'station_avg_temp_c',
                 'station_min_temp_c']
    df = df[features]

    # fill missing values
    df.fillna(method='ffill', inplace=True)

    # add labels to dataframe
    if labels_path:
        labels = pd.read_csv(labels_path, index_col=[0, 1, 2])
        #df = df.join(labels)

    # separate san juan and iquitos
    sj_features = df.loc['sj']
    iq_features = df.loc['iq']
    sj_labels = labels.loc['sj']
    iq_labels = labels.loc['iq']
    return sj_features, iq_features, sj_labels, iq_labels



# In[100]:


sj_features, iq_features, sj_labels, iq_labels = preprocess_data(
                                                 'data/dengue_features_train.csv',
                                                 labels_path="data/dengue_labels_train.csv")




#since data is linear it makes sense to separate data linearly
#split train and test data
sj_feat_train = sj_features.head(800)
sj_labels_train = sj_labels.head(800)
sj_feat_test = sj_features.tail(sj_features.shape[0] - 800)
sj_labels_test = sj_labels.tail(sj_labels.shape[0] - 800)



iq_feat_train = iq_features.head(400)
iq_labels_train = iq_labels.head(400)
iq_feat_test = iq_features.tail(iq_features.shape[0] - 400)
iq_labels_test = iq_labels.tail(iq_labels.shape[0] - 400)



iq_feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(iq_features)
sj_feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(sj_features)



# Build 3 layer DNN with 10, 20, 10 units respectively
iq_regressor = tf.contrib.learn.DNNRegressor(feature_columns=iq_feature_columns,
                                            hidden_units=[10, 20, 10])
sj_regressor = tf.contrib.learn.DNNRegressor(feature_columns=sj_feature_columns,
                                            hidden_units=[10, 20, 10])



#fitting regressor iq
iq_regressor.fit(iq_feat_train, iq_labels_train, steps=1000)


#fitting regressor sj
sj_regressor.fit(sj_feat_train, sj_labels_train, steps=1000)



iq_predictions = list(iq_regressor.predict(iq_feat_test, as_iterable=True))
score = metrics.mean_absolute_error(iq_labels_test, iq_predictions)
print("Mean Error: {0:f}".format(score))
#iq_labels_test.total_cases



sj_predictions = list(sj_regressor.predict(sj_feat_test, as_iterable=True))
score = metrics.mean_absolute_error(sj_labels_test, sj_predictions)
print("Mean Error: {0:f}".format(score))



#_------------------------------------------------------------------
#EXPLANATORY ANALYSIS FOLLOWS
#-------------------------------------------------------------------

# Seperate data for San Juan
sj_train_features = train_features.loc['sj']
sj_train_labels = train_labels.loc['sj']

# Separate data for Iquitos
iq_train_features = train_features.loc['iq']
iq_train_labels = train_labels.loc['iq']



print('San Juan')
print('features: ', sj_train_features.shape)
print('labels  : ', sj_train_labels.shape)

print('\nIquitos')
print('features: ', iq_train_features.shape)
print('labels  : ', iq_train_labels.shape)


# Remove `week_start_date` string.
sj_train_features.drop('week_start_date', axis=1, inplace=True)
iq_train_features.drop('week_start_date', axis=1, inplace=True)



# Null check
#pd.isnull(sj_train_features).any()

(sj_train_features
     .ndvi_ne
     .plot
     .line(lw=0.8))

plt.title('Vegetation Index over Time')
plt.xlabel('Time')


sj_train_features.fillna(method='ffill', inplace=True)
iq_train_features.fillna(method='ffill', inplace=True)



#distribution of labels
print('San Juan')
print('mean: ', sj_train_labels.mean()[0])
print('var :', sj_train_labels.var()[0])

print('\nIquitos')
print('mean: ', iq_train_labels.mean()[0])
print('var :', iq_train_labels.var()[0])



sj_train_labels.hist()
iq_train_labels.hist()




#variance  >>  mean suggests total_cases can be described by a negative
# binomial distribution, so we'll use a negative binomial regression below.



#add total cases column in features train data
sj_train_features['total_cases'] = sj_train_labels.total_cases
iq_train_features['total_cases'] = iq_train_labels.total_cases



# compute correlations matrix
sj_correlations = sj_train_features.corr()
iq_correlations = iq_train_features.corr()



# plot san juan
sj_corr_heat = sns.heatmap(sj_correlations)
plt.title('San Juan Variable Correlations')




# plot iquitos
iq_corr_heat = sns.heatmap(iq_correlations)
plt.title('Iquitos Variable Correlations')


#the total_cases variable doesn't have many obvious strong correlations.


#sorted look at correlations
# San Juan
(sj_correlations
     .total_cases
     .drop('total_cases') # don't compare with myself
     .sort_values(ascending=False)
     .plot
     .barh())

# Iquitos
(iq_correlations
     .total_cases
     .drop('total_cases') # don't compare with myself
     .sort_values(ascending=False)
     .plot
     .barh())


#END OF BASIC EXPLORATORY ANALYSIS --------------------------------------------------------
