
# coding: utf-8

# **Jake McRoberts  
# July 9th, 2017  
# Case Study: Retail Sales Data**

# # Improve Demand Forecasting using Retail Sales Data

# Python version used for analyses: 3.6.X
# 
# Data files: 
# - sales.txt
# - article_master.txt
# 
# Tasks:
# 1. Prepare and analyze the data to assure sufficient quality and suitability for the business case study.
# 2. Identify what is driving sales and which type of promotion has a stronger impact on sales.
# 3. Forecast sales (sold units) for the following month. 
# 4. Perform appropriate diagnostics to check whether your methods and results from above are reliable.
# 5. Conclusions and what I would have improved on
# 6. Appendix

# ## 1. Import, clean, and wrangle sales data to ensure data quality and suitability for analysis

# ### Import Data

# In[506]:

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# import machine learning packages
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, SelectPercentile
from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit

from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score

# import necessary seasonality decomposition and ARIMA modeling packages
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA


# In[507]:

# import transactional sales data of articles into a pandas dataframe
data_sales = pd.read_csv("sales.txt", sep = ";", parse_dates=['retailweek'])

# import attribute data of sold articles
data_attribute = pd.read_csv("article_master.txt", sep = ";")


# ### Inspect Data

# Verify data looks clean and suitable from a high level view of it.

# In[508]:

data_sales.head()


# In[509]:

data_sales.shape


# Check for any NaN values in sales transactional data. False means there are no NaN's in the dataset.

# In[510]:

data_sales.isnull().values.any()


# In[511]:

data_sales.describe()


# In[512]:

data_attribute.head()


# In[513]:

data_attribute.shape


# Check for any NaN values in sales attribute data. False means there are no NaN's in the dataset.

# In[514]:

data_attribute.isnull().values.any()


# In[515]:

data_attribute.describe()


# From a high level perspective, the data looks okay. There aren't any NaN's in either dataset and the size of the dataframes makes sense based on the size of the raw data in the text files. The descriptive statistics on the data looks great as well.
# 
# Use pandas `merge` function to combine the transactional sales data with the sales attribute data like a SQL join. Use an inner join since I want to add cost data (if there is any) for the given transactional data. If there isn't any cost data, then I will not use the transactional sales data. I'll try a left join first to see if there are any cases where there is transactional data but not cost data for a given article. If there aren't many cases like this, then I will go ahead with the inner join.

# In[516]:

data_leftJoined = pd.merge(data_sales, data_attribute, how = 'left', on = 'article')
data_leftJoined.head()


# In[517]:

data_leftJoined.describe()


# Investigate why there are less cost data points (count) than the other columns of data - likely due to there not being cost data for an article that had transactional sales data.

# In[518]:

data_leftJoined[data_leftJoined.isnull().values].shape


# As expected, there are 369 NaN's for missing cost data for an article that had cost data. I can fix this by replacing the NaN's with 0's as I show below.

# In[519]:

data_leftJoined.fillna(0, inplace = True)
data_leftJoined[data_leftJoined.isnull().values]


# In[520]:

data_leftJoined.describe()


# Even though I can fix the NaN values in the left join, I am going to go forward with the inner join so for all of the data I have both transactional sales and cost data. I want to avoid filling in NaN's with artificial 0's (or some other arbitrary value). Perform inner join of sales and attribute data.

# In[521]:

data_joined = pd.merge(data_sales, data_attribute, how = 'inner', on = 'article')
data_joined.head()


# In[522]:

data_joined.describe()


# Verify there are no NaN values. False means there are no NaN's in the dataset.

# In[523]:

data_joined.isnull().values.any()


# There are no NaN values. Joined data looks good to move on to exploratory data analysis through plotting. It will be very useful to visualize the different categories of sales data.

# ### Plot Data

# In[524]:

plt.figure()
data_joined.plot(x = 'retailweek', y = 'sales')
plt.show()


# In[525]:

fig, axes = plt.subplots(nrows=1, ncols=2, figsize = (12,6))
data_joined[0:100].plot(ax=axes[0], x = 'retailweek', y = 'sales')
axes[0].set_title('First 100 sales')
data_joined[-100:-1].plot(ax=axes[1], x = 'retailweek', y = 'sales')
axes[1].set_title('Last 100 sales')
plt.show()


# In[526]:

fig, axes = plt.subplots(nrows=1, ncols=2, figsize = (12,6))
data_joined.plot(ax = axes[0], y = 'sales', kind = 'hist')
data_joined.plot(ax = axes[1], y = 'sales', kind = 'box')
plt.show()


# Sales data looks as expected; there are a lot of outliers likely at high sale times (like Christmas / winter holiday and summer before school starts). The sales data is pretty cyclical in addition to being heavily skewed with the vast majority of weeks with sales of less than 100 items for each item type.
# 
# Examine price data now.

# In[527]:

fig, axes = plt.subplots(nrows=1, ncols=2, figsize = (12,6))
data_joined.plot(ax = axes[0], x = 'retailweek', y = ['regular_price', 'current_price'])
axes[0].set_title('All data for regular and current price')
data_joined[0:100].plot(ax = axes[1], x = 'retailweek', y = ['regular_price', 'current_price'])
axes[1].set_title('First 100 data points for regular and current price')
plt.show()


# In[528]:

plt.figure()
data_joined.plot(y = ['regular_price', 'current_price'], kind = 'hist', subplots = True)
plt.show()


# In[529]:

plt.figure()
data_joined.plot(y = ['regular_price', 'current_price'], kind = 'box')
plt.show()


# Pricing data (regular price and current price) both look as expected; the regular price is higher than the current price (indicating that discounts, like promo1 and/or promo2, reduce the price). Both price datasets are skewed but the regular price looks like it could be closer to a normal distribution. The majority of items that are sold are priced under 75 euro.
# 
# Look into the ratio data (`ratio = current_price / regular_price`).

# In[530]:

fig, axes = plt.subplots(nrows=1, ncols=2, figsize = (12,6))
data_joined.plot(ax = axes[0], x = 'retailweek', y = 'ratio')
axes[0].set_title('All ratio data')
data_joined[0:200].plot(ax = axes[1], x = 'retailweek', y = 'ratio')
axes[1].set_title('First 200 ratio data points')
plt.show()


# In[531]:

fig, axes = plt.subplots(nrows=1, ncols=2, figsize = (12,6))
data_joined.plot(ax = axes[0], y = 'ratio', kind = 'hist')
data_joined.plot(ax = axes[1], y = 'ratio', kind = 'box')
plt.show()


# Ratio data is similarly skewed like the sales and pricing data - the cyclical nature of pricing items is very apparent in the data when zooming in on a smaller sample of the ratio data.
# 
# Examine the promo data (1 and 2) next.

# In[532]:

fig, axes = plt.subplots(nrows=1, ncols=2, figsize = (12,6))
data_joined.plot(ax = axes[0], x = 'retailweek', y = ['promo1', 'promo2'])
axes[0].set_title('All promo data (1 and 2)')
data_joined[0:1000].plot(ax = axes[1], x = 'retailweek', y = ['promo1', 'promo2'])
axes[1].set_title('First 1000 data points for promo data')
plt.show()


# In[533]:

plt.figure()
data_joined.plot(y = ['promo1', 'promo2'], kind = 'hist', subplots = True)
plt.show()


# In[534]:

plt.figure()
data_joined.plot(y = ['promo1', 'promo2'], kind = 'box')
plt.show()


# Plotting the promotion data doesn't yield much when looking for a distribution since the values are either '0' or '1' at all times. It is good to confirm that the data is clean.
# 
# Inspect the actual cost data from the second text data file holding attribute data for each article of clothing.

# In[535]:

fig, axes = plt.subplots(nrows=1, ncols=2, figsize = (12,6))
data_joined.plot(ax = axes[0], x = 'retailweek', y = 'cost')
axes[0].set_title('All cost data')
data_joined[0:500].plot(ax = axes[1], x = 'retailweek', y = 'cost')
axes[1].set_title('First 500 cost data points')
plt.show()


# In[536]:

fig, axes = plt.subplots(nrows=1, ncols=2, figsize = (12,6))
data_joined.plot(ax = axes[0], y = 'cost', kind = 'hist')
data_joined.plot(ax = axes[1], y = 'cost', kind = 'box')
plt.show()


# Cost data looks reasonable; there aren't any large outliers. There is one item that costs roughly 20 euro. The data is skewed similarly to the other numerical data, although the shape is closest to the price data. I'll examine the potential outlier to see if it seems like a real data point.

# In[537]:

data_joined[data_joined['cost'] > 17].shape


# The data point above is a real data point. There are 123 transactional sales data points for the Porsche shoes. I know that adidas has a relationship with Porsche for branding of clothing and it would make sense for it to be more expensive that a non-Porsche branded piece of clothing considering the Porsche brand.
# 
# Overall, the data looks great. I'll move on to digging into the data to find what drives sales and if the promotions provide value add to sales numbers while keeping profits high.

# ## 2. - 4. Identify what is driving sales and forecast future sales, and which type of promotion has a stronger impact on sales while ensuring evaluation techniques and modeling are robust and validated

# ### Transactional Sales Data Correlation Matrix

# Create a correlation matrix to see how each of the promotions and ratio play into the number of sales.

# In[538]:

# calculate correlation matrix
corr_sales = data_joined.corr()
corr_sales


# In[539]:

# plot the heatmap
sns.heatmap(corr_sales)
plt.show()


# The two price variables (regular and current price) are heavily correlated which makes a lot of sense. Regular price is heavily correlated with cost which makes perfect sense and current price is also positively correlated with cost. Ratio looks to be moderately correlated with the current_price which again makes sense since it is the results of regular_price/current_price. Ratio looks to be moderately negatively correlated with sales which would imply that as teh ratio goes down, the sales go up; this intuitively makes sense since reduced prices hypothetically make for more purchases.
# 
# Lastly, it looks like promo1 is loosely correlated with sales; more so than promo2's correlation with sales.
# 
# At this point, it isn't very clear which promotion drives sales more. I would hazard a guess that promo1 does influence sales more based on the more positive correlation. As far as pricing goes, it looks like the like there is a moderate correlation between reducing the ratio (decreasing the current_price) and increasing sales.
# 
# I think the best approach would be to prepare the joined sales data for applying machine learning for a numerical output (not a classifer). It looks like some form of ordinary least squares regression, logistic regression, lasso regression, ARIMA, or random forest regression will be the best way to approach this problem.

# ### Machine Learning: Regression

# Steps required for adequately performing machine learning on sales data using a Pipeline for evaluating many different combinations of features/algorithms and using proper validation technique(s) like cross-validation:
# 
# A. Feature creation (Ideas listed below for creating new variables to help with machine learning, re-run correlation matrix with the new features):
#     - month (1 through 12)
#     - season (1 through 4: winter, spring, summer, fall)
#     - type of clothing (convert to a unique number for each article of clothing)
#     - country number (convert country variable to a unique number)
# 
# B. Split data into features (data, ~ X) and target (sales ~ Y) for machine learning regressions. Take the features and targets and split into training and testing data as well so I can properly evaluate my results against 'new' data so I can better judge the performance.
# 
# C. Use a SKLearn Pipeline for steps 1 through 5
#     1. Transform data (feature scaling, etc.)
#     2. Dimensionality reduction (PCA)
#     3. Feature selection (k-best, k-means, etc.)
#     4. Machine learning algorithm selection (evaluate different combinations and machine learning algorithms while using cross-validation methods to ensure valid results)
#     5. Evaluate results using the following metrics: accuracy, precision, recall

# ##### A. Feature Creation

# Create week, month, and year variables in the joined dataframe using pandas datetime.
# 
# Create season variable based on the month: Winter (1), Spring (2), Summer (3), Fall (4).

# In[540]:

# create date columns
data_joined['week'] = data_joined['retailweek'].dt.week
data_joined['month'] = data_joined['retailweek'].dt.month
data_joined['year'] = data_joined['retailweek'].dt.year


# In[541]:

# create season column
data_joined['season'] = data_joined['retailweek'].apply(lambda dt: (dt.month%12 + 3)//3)


# In[542]:

# verify columns were created correctly
data_joined.head()


# Create unique identifier (numerical) for country, category of clothing, and product group of clothing.

# In[543]:

# country
data_joined['country'] = pd.Categorical(data_joined['country'])
data_joined['country_code'] = data_joined.country.cat.codes


# In[544]:

# clothing category
data_joined['category'] = pd.Categorical(data_joined['category'])
data_joined['category_code'] = data_joined.category.cat.codes


# In[545]:

# product group of clothing
data_joined['productgroup'] = pd.Categorical(data_joined['productgroup'])
data_joined['productgroup_code'] = data_joined.productgroup.cat.codes


# In[546]:

# verify columns were created correctly
data_joined.head()


# Verify that all of the feature creations have worked - get the summary data from the dataframe.

# In[547]:

data_joined.describe()


# Created features look good. I'll re-run the correlation matrix and heatmap to see if the new features turn up any noticeable correlations.

# In[548]:

# calculate correlation matrix
corr_sales2 = data_joined.corr()
corr_sales2


# In[549]:

# plot the heatmap
sns.heatmap(corr_sales2)
plt.show()


# Of the new features, the majority don't have any correlations that would be useful (or not intuitive). One that sticks out as possibly important is the category_code feature is negatively correlated with cost, regular_price, and current_price. A set of new correlations of note are the correlations between date (like week and month) and seasonality with ratio and promo1. These correlations could be useful in creating a robust machine learning algorithm for predicting future sales.
# 
# Create the features and target data along with splitting into training and testing data.

# ##### B. Create Feature and Target Data and Training/Testing Data Subsets

# SKLearn machine learning functions receive lists and generally can handle numpy arrays so I'll split out the data into two lists: 1 for the target (Y) which is the sales column and 1 for the features (X) which are all of the other columns that are numerical.

# In[550]:

# define features I want to use and what the target data is
features_list = ['regular_price', 'current_price', 'ratio', 'promo1', 'promo2', 'cost', 'week', 'month', 'year', 'season', 'country_code', 'category_code', 'productgroup_code']
target_list = ['sales']


# In[551]:

# subset joined dataframe into df_features and target
df_features = data_joined[features_list]
df_features.head()


# In[552]:

df_features.shape


# In[553]:

target = data_joined[target_list]
target.head()


# In[554]:

target.shape


# Subsetted dataframes look good, now split the data into train and testing splits. Generally, a 80%/20% or 70%/30% training/testing data split works so I'm going to start with 70/30 to maximize my testing data so I have the best possible idea of performance for new data since that is the goal of this case study.

# In[555]:

features_train, features_test, target_train, target_test = train_test_split(df_features, target,                                                                             test_size=0.3, random_state=42)


# Check sizing of created feature and target train/test splits

# In[556]:

print(features_train.shape, features_test.shape, target_train.shape, target_test.shape)


# In[557]:

print(type(features_train), type(features_test), type(target_train), type(target_train))


# Training and testing data looks good. Next, I'll set up the SKLearn Pipeline for performing regressions.

# ##### C. Pipeline: Use a pipeline for finding and evaluating the best feature scaler, dimensionality reduction, feature selection, and machine learning algorithm

# Create regressors, scalers, etc. for use in Pipeline

# In[173]:

# regressions
LinReg = LinearRegression()
Las = Lasso() # optimize with GridSearchCV: alpha=[0.1, 0.3, 0.5, 0.7, 0.9, 1] (default=1)
DTree = DecisionTreeRegressor(random_state = 42) # min_samples_leaf (default=1), min_samples_split (default=2)
RF = RandomForestRegressor(random_state = 42)
# n_estimators (default=10), min_samples_leaf (default=1), min_samples_split (default=2)

# feature scaling
minMaxScaler = MinMaxScaler()
stdScaler = StandardScaler() #(could) optimize with: with_std=False

# dimensionality reduction and feature selection
PCAreducer = PCA(svd_solver='auto', random_state=42)
KBestSelector = SelectKBest(f_regression) # optimize: k = [3, 5, 7, 10, 13, 15] (default = 10)
KPercentSelector = SelectPercentile(f_regression) # optimize: percentile = [5, 10, 15, 20, 25] (default=10)


# Create Pipeline including the machine learning tools I want to evaluate and the parameters I want to optimize for each of the tools used.
# 
# Use StratifiedShuffleSplit for cross-validation and GridSearchCV for parameter optimization within pipeline. I chose to optimize around r^2 since it encompasses a lot of quality of fit within 1 performance metric.

# In[168]:

# convert features and target to numpy arrays
target_np = target.as_matrix(columns = target_list).ravel()
df_features_np = df_features.as_matrix(columns = features_list)
target_train_np = target_train.as_matrix(columns = target_list).ravel()
features_train_np = features_train.as_matrix(columns = target_list)


# In[213]:

# create pipeline
pipe = Pipeline([
    ('scaler', stdScaler),
    ('PCA', PCAreducer),
    ('KBest', KBestSelector),
    ('reg', RF) # LinReg, Las, DTree, RF
])

# create parameters for GridSearchCV using pipeline
params = {
#     'PCA__n_components': [None, 13, 11, 9],
    'KBest__k': [9, 7, 5, 3],
    'reg__n_estimators': [3, 5, 8, 10, 15],
    'reg__min_samples_split': [2, 4, 6, 8, 10, 12],
    'reg__min_samples_leaf': [1, 2, 4, 6, 8]
#     'reg__alpha': [0.1, 0.3, 0.5, 0.7, 0.9, 1]
}
# scaler doesn't have any parameters to be used in this case
# linear regression algorithm doesn't have any params that need tuning

# create StratifiedShuffleSplit for cross-validation
# folds = 1000
# cvSetup = StratifiedShuffleSplit(target_np, folds, random_state=42)

# set up GridSearchCV
# gridSearch = GridSearchCV(pipe, param_grid = params, scoring = 'r2', \
#                          cv = cvSetup, verbose = 10)
gridSearch = GridSearchCV(pipe, param_grid = params, scoring = 'r2', verbose = 10) # scoring = 'r2',

# fit pipeline and grid search to training features and target dataset
gridSearch.fit(features_train, target_train_np)
# gridSearch.fit(df_features_np, target_np)

# store the best estimator (best classifier with parameters)
regr = gridSearch.best_estimator_


# In[214]:

# create handle
KBest_handle = gridSearch.best_estimator_.named_steps['KBest']

# Get SelectKBest scores rounded to 2 decimal places
feature_scores = ['%.2f' % elem for elem in KBest_handle.scores_]
# Get SelectKBest pvalues, rounded to 2 decimal places
feature_scores_pvalues = ['%.2f' % elem for elem in  KBest_handle.pvalues_]

# Create a tuple of SelectKBest feature names, scores and pvalues
features_selected_tuple=[(features_list[i], feature_scores[i],                           feature_scores_pvalues[i]) for i in KBest_handle.get_support(indices=True)]
# Sort by reverse score order
features_selected_tuple = sorted(features_selected_tuple, key=lambda feature: float(feature[0]) , reverse=True)
# features_selected_tuple = sorted(features_selected_tuple, key=lambda feature: float(feature[1]) , reverse=True)

# Print selected feature names and scores
print(' ')
print('Selected Features, Scores, P-Values:')
print(features_selected_tuple)

# print best parameters and corresponding score from gridSearchCV
print(' ')
print('Best parameters:')
print(gridSearch.best_params_)
print('')
print('Best score:')
print(gridSearch.best_score_)


# In[215]:

# make predictions based on test data split
pred = gridSearch.predict(features_test)

# plot predictions (regression) vs target (truth)
x = np.arange(len(pred))
plt.figure()
plt.plot(x, target_test, '.')
plt.plot(x, pred, '-')
plt.show()


# In[216]:

# print results
print('mean_absolute_error:', mean_absolute_error(target_test, pred))
print('mean_squared_error:', mean_squared_error(target_test, pred))
print('median_absolute_error:', median_absolute_error(target_test, pred))
print('r2_score:', r2_score(target_test, pred))


# I have tested 4 different machine learning regressors so far: linear, lasso, decision tree, and random forest regressors. The linear, lasso, and random forest regressors resulted in a r^2 score of less than 0.1 which is really poor. The latest test from the decision tree regressor yielded a result of -0.297 which is roughly 3 times better than the other regressors. The plot above shows the raw sales data in blue and the model's predictions for the same data. As can be seen, the result really isn't very good. During my search for a better fit, I have found that I am definitely limited in my optimial model search by a lack of computing power.
# 
# I've been able to eliminate PCA from the machine learning pipeline since it has been returned as None every time I have run the pipeline regardless of which algorithm (linear, lasso, DTree, RF) I'm using.

# ### Regression/Machine Learning Issues and Brainstorming
# I have noticed the quality of the fits of the models have all be lacking. One huge reason for this is a lack of computing power so I can't exhaustively try to find the best model. Also, I do not have much experience with time series data for such a varied population (different articles of clothing types) and I think it is causing problems. I have a few ideas to try that might help with getting a better handle on this dataset. Below are some possible next steps:
# 1. Return to do some more data wrangling to decompose and remove seasonality from the data. Then move forward with machine learning or a different type of modeling like ARIMA which is supposed to be good for time series data and forecasting.
# 2. Introduce a better way to split the data - i.e. use SKLearn's TimeSeriesSplit function within model_selection to make better use of the fact that the data is time series and therefore time dependent.
# 3. Return to data wrangling and just look at 1 article of clothing to see how I can do with a smaller dataset. If this works, then move on to other pieces of clothing. Considering how the attribute data has 476 entries, I would need to iterate through 476 different articles of clothing - this method would be very time consuming to apply on a production scale.

# ### Decompose and Remove Seasonality from Dataset
# Use python's statsmodels package for seasonal decomposition and for ARIMA model.
# 
# I need to make the index for the dataframe a datetime type index instead of an integer based index. I'll make a copy of the dataframe for this work forward.

# In[243]:

data_joined_dt = data_joined.copy()


# In[337]:

data_joined_dt.reset_index(inplace=True)
# data_joined_dt['retailweek'] = pd.to_datetime(data_joined_dt['retailweek'])
data_joined_dt = data_joined_dt.set_index('retailweek')

# set frequency for decomposition: since the data is weekly, but I am interested in seasonality,
# I'll multiply by 4 to get to months and then by 3 to get to seasons (4 seasons in a year).
decompFreq = 1*4*3

decomp = seasonal_decompose(data_joined_dt.sales, model = 'additive', freq = decompFreq)
decomp.plot()
plt.show()


# Okay, so this is now working how I thought it would. Initially, I had set frequency to 1 so I was looking for seasonality from a weekly perspective which really wouldn't give me that much. I saw next to zero seasonality data and all trend data which was not useful. When I changed the frequency parameter in the decomposition function to be seasonally I can see a nicely split out seasonal dataset. Now I can attempt to model again but this time I'll use the residuals from the seasonal decomposition since those have trend and seasonality removed.
# 
# Re-create residuals plot below compared with nominal sales data and then perform machine learning on results.

# In[354]:

plt.figure()
plt.plot(decomp.resid, '.')
plt.title('seasonal decomposition residuals of all sales data')
data_joined.plot(x = 'retailweek', y = 'sales')
plt.title('all sales data (raw)')
plt.show()


# ### Re-try Machine Learning Algorithms with Seasonal Decomposition Residuals

# Re-make training and testing data with new target data from seasonal decomposition. Feed in the same features data since that hasn't changed.

# In[452]:

target_SC = decomp.resid
features_train_SC, features_test_SC, target_train_SC, target_test_SC = train_test_split(df_features, target_SC,                                                                                         test_size=0.3, random_state=42)


# In[453]:

# make sure there are no NaN's in the created features and targets
# if there are NaN's then fill (interpolate) with median value of data
print(target_test_SC.isnull().any())
target_test_SC.fillna(target_test_SC.median(), inplace = True)
print(target_test_SC.isnull().any())


# In[454]:

print(np.isnan(features_test_SC).any())


# In[455]:

print(target_train_SC.isnull().any())
target_train_SC.fillna(target_train_SC.median(), inplace = True)
print(target_train_SC.isnull().any())


# In[456]:

print(np.isnan(features_train_SC).any())


# In[475]:

# create pipeline
pipe_SC = Pipeline([
    ('scaler', stdScaler),
    ('PCA', PCAreducer),
    ('KBest', KBestSelector),
    ('reg', RF) # LinReg, Las, DTree, RF
])

# create parameters for GridSearchCV using pipeline
params_SC = {
#     'PCA__n_components': [None, 13, 11, 9],
    'KBest__k': [8, 5, 3],
    'reg__n_estimators': [5, 8, 10, 15, 20],
    'reg__min_samples_split': [2, 5, 9, 12],
    'reg__min_samples_leaf': [1, 2, 5, 8]
#     'reg__alpha': [0.1, 0.3, 0.5, 0.7, 0.9, 1]
}
# scaler doesn't have any parameters to be used in this case
# linear regression algorithm doesn't have any params that need tuning

# create StratifiedShuffleSplit for cross-validation
# folds = 1000
# cvSetup = StratifiedShuffleSplit(target_np, folds, random_state=42)

# set up GridSearchCV
# gridSearch = GridSearchCV(pipe, param_grid = params, scoring = 'r2', \
#                          cv = cvSetup, verbose = 10)
gridSearch_SC = GridSearchCV(pipe_SC, param_grid = params_SC, scoring = 'r2', verbose = 10) # scoring = 'r2',

# fit pipeline and grid search to training features and target dataset
gridSearch_SC.fit(features_train_SC, target_train_SC)
# gridSearch.fit(df_features_np, target_np)

# store the best estimator (best classifier with parameters)
regr_SC = gridSearch_SC.best_estimator_


# In[476]:

# create handle
KBest_handle_SC = gridSearch_SC.best_estimator_.named_steps['KBest']

# Get SelectKBest scores rounded to 2 decimal places
feature_scores_SC = ['%.2f' % elem for elem in KBest_handle_SC.scores_]
# Get SelectKBest pvalues, rounded to 2 decimal places
feature_scores_pvalues_SC = ['%.2f' % elem for elem in  KBest_handle_SC.pvalues_]

# Create a tuple of SelectKBest feature names, scores and pvalues
features_selected_tuple_SC=[(features_list[i], feature_scores_SC[i],                           feature_scores_pvalues_SC[i]) for i in KBest_handle_SC.get_support(indices=True)]
# Sort by reverse score order
features_selected_tuple_SC = sorted(features_selected_tuple_SC, key=lambda feature: float(feature[1]) , reverse=True)

# Print selected machine learning algorithm along with feature names, scores, and p-values
print(' ')
print('Best machine learning algorithm:')
print('Random Forest')
print(' ')
print('Selected Features, Scores, P-Values:')
print(features_selected_tuple_SC)

# print best parameters and corresponding score from gridSearchCV
print(' ')
print('Best parameters:')
print(gridSearch_SC.best_params_)
print('')
print('Best score:')
print(gridSearch_SC.best_score_)


# In[477]:

# make predictions based on test data split
pred_SC = gridSearch_SC.predict(features_test_SC)

# plot predictions (regression) vs target (truth)
x = np.arange(len(pred_SC))
plt.figure()
plt.plot(x, target_test_SC, '.')
plt.plot(x, pred_SC, '-')
plt.show()


# In[478]:

# print results
print('mean_absolute_error:', mean_absolute_error(target_test_SC, pred_SC))
print('mean_squared_error:', mean_squared_error(target_test_SC, pred_SC))
print('median_absolute_error:', median_absolute_error(target_test_SC, pred_SC))
print('r2_score:', r2_score(target_test_SC, pred_SC))


# Plot residuals of the target test values and the predicted target values from the model.

# In[470]:

plt.figure()
plt.plot(x, (target_test_SC - pred_SC))
plt.show()


# I ended up using a random forest regressor that used the following 5 features:
# - year
# - week
# - regular_price
# - season
# - cost
# 
# My forecasting with the random forest regressor on 30% of the original data that I held out for testing and validating the regressor resulted in poor performance: a tested r^2 of 0.02 and a mean square error of 5577.92. The mean absolute error is 44.01 and the median absolute error is 25.99.
# 
# Even with the seasonal decomposition, the fit of the data is very poor. An r^2 value of 0.02 is really not good - it is actually pretty comparable to the random forest model when using the raw data (without seasonal decomposition) which gave a slightly better r^2 value but work mean and mediand absolute error and mean square error values. I'm questioning the quality of the seasonal decomposition - I wouldn't be surprised if I have made a mistake in implementing it. I honestly don't have any experience with analyzing sales data nor with forecasting demand / sales so I don't really know what to expect in terms of quality of fit and the resulting residuals.
# 
# I'm running into some issues with creating a useful model that can forecast sales numbers and the promotions seem to get lost in the other modeling terms so I haven't been able to pull out which promotion results in more sales. I'll approach the search for which promotion is better by grouping the sales numbers for when a promotion occurs (either promo1 or promo2) and see which has higher sales numbers. This is a gross and likely inaccurate way to find the best promotion, but it will work for a best effort given the time restrictions of the case study.

# ### Promo1 vs Promo2: Which had the greater impact on sales?

# Find the number of promo1 instances, mean value, median value, max/min values, etc. with the pandas describe function and find the sum of total sales. Repeat for promo2.

# In[481]:

data_joined.groupby(by=['promo1'])['sales'].describe()


# In[489]:

data_joined.groupby(by=['promo1'])['sales'].sum()


# In[483]:

data_joined.groupby(by=['promo2'])['sales'].describe()


# In[494]:

data_joined.groupby(by=['promo2'])['sales'].sum()


# One thing to point out before I make some declarations as to which promotion is better is that the sample size difference of the number of occurances of promotion 1 (5288) is significant when compared to the much smaller sample size for the number of occurances for promotion 2 (412). This in itself makes the comparison challenging and makes my inspection into the total sum of sales when each promotion is active useless. An important thing to note is that I didn't fit in a check for if the two promotions overlapped in the grouping above. I'll re-analyze the promotion data by conditionally grouping the data such that when one of the promotions is occurring (set to 1) the other promotion is not occurring.

# In[495]:

data_joined[(data_joined.promo1 == 1) & (data_joined.promo2 == 1)].shape


# There are 103 instances where both promo1 and promo2 are happening at the same time. Perform conditional groupby filtering so that the promotions are looked at on their own.

# In[502]:

g_p1 = data_joined.groupby(by=['promo1'])
g_p1.apply(lambda x: x[x['promo2'] != 1]['sales'].describe())


# In[499]:

g_p1.apply(lambda x: x[x['promo2'] != 1]['sales'].sum())


# Repeat for promo2.

# In[503]:

g_p2 = data_joined.groupby(by=['promo2'])
g_p2.apply(lambda x: x[x['promo1'] != 1]['sales'].describe())


# In[ ]:

g_p2.apply(lambda x: x[x['promo1'] != 1]['sales'].sum())


# Based on the descriptive statistics and with no overlap for when the promotions are going, promotion 1 is better at driving sales than promotion 2 when looking in a vaccuum of just comparing the two promotions with sales numbers.
# 
# ***Both the mean and median increase more when using promotion 1 (vs. not using promotion 1) when compared to when using promotion 2 (vs. not using promotion 2). Re-doing the analysis to ensure that both promotions couldn't be happening at the same time helped solidify my argument for promotion 1 being the more effective of the two promotions.***

# ## 5. Conclusions and What I Would Have Improved On If I Had More Time

# ### Improvements
# 1. Improve robustness of seasonality decomposition. Unsure if I implemented it correctly.
# 2. Improve robustness of machine learning GridSearchCV parameter exploration and of the initial algorithm selection.
# 3. Try more modeling options like the ARIMA and SARIMA models used for time series regressions. These are often used in sales data and demand forecasting.
# 4. Create a better analysis / study of the promotion comparisons. I didn't have enough time to really get a good grasp of each promotion and how I could best estimate their impact on the sales.
# 5. Improve validation methodology: As of now, I used 3 folds in the GridSearchCV parameter optimization and I used 30% of the data for testing and predictions but I could have done better like implementing some k-fold or StratifiedShuffleSplit cross-validation techniques.
# 6. Improve analysis and validation of promotion comparison on sales numbers.

# ### Conclusions
# 
# I found the sales demand forecasting to be challenging as I wasn't able to get a satisfactory model through machine learning (even after using time series seasonal decomposition). I ended up using a random forest regressor that used the following 5 features:
# - year
# - week
# - regular_price
# - season
# - cost
# 
# My forecasting with the random forest regressor on 30% of the original data that I held out for testing and validating the regressor resulted in poor performance: a tested r^2 of 0.02 and a mean square error of 5577.92. I did not create future data for predictions since I used 30% of the original data and treated that as future (or unknown to the trained regressor). Given more time, I would have created sample future data for one month in the future and used my regression model to predict the sales numbers and then evaluate the performance of the prediction.
# 
# Lastly, I found that promotion 1 (promo1) was significantly more effective than promo2 at increasing sales. I made sure to only include data for when the promotions were on exclusively, meaning that if one of the promotions was on then the other one was not on.
# 
# All in all, I really enjoyed the challenge and fun of digging into this data and trying to find a way to understand it and create a decent model. I love using data to improve processes and business decisions. There is so much more work I could have done (as I mentioned in the above Improvements section) and I would love the chance to be able to work for adidas on more projects like these.

# ## 6. Appendix: Extra Exploration

# ##### A side exploration is below for a single clothing article cateogry (Golf) that I performed while I was stuck on the full dataset seasonal decomposition. This analysis is not necessary for the rest of the analysis.
# 
# Take a step back and look at one article of clothing to see if I can get some seasonal decomposition or machine learning performed on it.

# ### Analyze Golf Sales: Clothing with Category = 'Golf'

# In[504]:

data_joined[data_joined.category == 'GOLF'].shape


# In[505]:

data_golfSales = data_joined[data_joined.category == 'GOLF']
data_golfSales.head()


# In[314]:

data_golfSales.describe()


# In[322]:

fig, axes = plt.subplots(nrows=2, ncols=2, figsize = (14,14))
data_golfSales.plot(ax = axes[0,0], x = 'retailweek', y = 'sales')
axes[0,0].set_title('Golf sales')
data_golfSales.plot(ax = axes[0,1], y = 'sales', kind = 'hist')
axes[0,1].set_title('Golf sales histogram')
data_golfSales.plot(ax = axes[1,0], y = 'sales', kind = 'box')
axes[1,0].set_title('Golf sales boxplot')
axes[1,1].set_title('INTENTIONALLY LEFT BLANK')
plt.show()


# In[323]:

fig, axes = plt.subplots(nrows=2, ncols=2, figsize = (14,14))
data_golfSales.plot(ax = axes[0,0], x = 'retailweek', y = ['regular_price', 'current_price'])
axes[0,0].set_title('Golf pricing')
data_golfSales.plot(ax = axes[0,1], y = ['regular_price', 'current_price'], kind = 'hist')
axes[0,1].set_title('Golf pricing histogram')
data_golfSales.plot(ax = axes[1,0], y = ['regular_price', 'current_price'], kind = 'box')
axes[1,0].set_title('Golf pricing boxplot')
data_golfSales.plot(ax = axes[1,1], x = 'retailweek', y = ['regular_price', 'current_price'])
data_golfSales.plot(ax = axes[1,1], secondary_y = True, x = 'retailweek', y = 'sales')
axes[1,1].set_title('Golf sales and pricing')
plt.show()


# In[282]:

plt.figure()
data_XS4279.plot(x = 'retailweek', y = 'sales')
plt.show()


# I can see a lot of correlation between the sales and when the current price drops with this subset of data for golf shoes sales (article ID XS4279). Next, I'll look into how the two promotions look when compared with sales data.

# In[324]:

fig, axes = plt.subplots(nrows=2, ncols=2, figsize = (14,14))
data_golfSales.plot(ax = axes[0,0], x = 'retailweek', y = ['promo1', 'promo2'])
axes[0,0].set_title('Golf promotions')
data_golfSales.plot(ax = axes[0,1], y = ['promo1', 'promo2'], kind = 'hist')
axes[0,1].set_title('Golf promotions histogram')
data_golfSales.plot(ax = axes[1,0], y = ['promo1', 'promo2'], kind = 'box')
axes[1,0].set_title('Golf promotions boxplot')
data_golfSales.plot(ax = axes[1,1], x = 'retailweek', y = ['promo1', 'promo2'])
data_golfSales.plot(ax = axes[1,1], secondary_y = True, x = 'retailweek', y = 'sales')
axes[1,1].set_title('Golf sales and promotions')
plt.show()


# There aren't any promo2 occurrances with this subset of data, but there are several promo1 instances.

# In[326]:

corr_golfSales = data_golfSales.corr()
sns.heatmap(corr_golfSales)
plt.show()


# I see similar correlations as to when I looked at the entire dataset. I'll see if I can do some seasonal decomposition on the data.

# In[339]:

data_golfSales_dt = data_golfSales.copy()

data_golfSales_dt.reset_index(inplace=True)
# data_golfSales_dt['retailweek'] = pd.to_datetime(data_golfSales_dt['retailweek'])
data_golfSales_dt = data_golfSales_dt.set_index('retailweek')
decomp_golfSales = seasonal_decompose(data_golfSales_dt.sales, model = 'additive', freq = decompFreq) 
decomp_golfSales.plot()
plt.show()


# Note: I decided to stop going down this route since I have the seasonality function working (I think!) for the full dataset.
# ##### End of side exploration of a single clothing article category
