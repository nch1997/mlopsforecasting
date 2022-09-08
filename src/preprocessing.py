"""
Preprocess and feature engineering data
https://stackoverflow.com/questions/64409231/is-there-any-wayby-setting-or-extension-to-view-and-use-variables-in-vscode-ot
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
import lightgbm
from sklearn.model_selection import train_test_split
import datetime
import joblib
from sklearn.model_selection import TimeSeriesSplit
import os
from sklearn.feature_selection import RFECV
import sklearn
from sklearn.metrics import fbeta_score, make_scorer
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

def my_custom_loss_func(y_true, y_pred):
    """
    Calculate pinball loss for alpha 0.5 used for sklearn scorer
    """
    
    return [max(0.5*(d-f), (1-0.5)*(f-d)) for d,f in zip(y_true,y_pred)][0]

def pinball_loss(actual, forecast, alpha):
    """
    Calculate pinball loss
    d - actual value
    f - quantile forecast
    alpha - quantile
    """
    
    return [max(alpha*(d-f), (1-alpha)*(f-d)) for d,f in zip(actual,forecast)]

# import training dataset
data_file = "data/data.csv"

# preprocess and freature engineering
df_15min = pd.read_csv(data_file)

# split date and time into own columns
df_15min['date'] = pd.to_datetime(df_15min['timestamp']).dt.date
df_15min['time'] = pd.to_datetime(df_15min['timestamp']).dt.time
df_15min['dayofweek'] = pd.to_datetime(df_15min['timestamp']).dt.weekday
df_15min['weekend'] = pd.to_datetime(df_15min['timestamp']).dt.weekday > 4

df_15min['n'] = df_15min['15min_traffic_count']
# use .shif(-1) to shift row by x to create lagging indicator
df_15min['n+1'] = df_15min['n'].shift(-1)

df_15min['xhour'] = df_15min['time'].map(lambda time: np.sin(2*np.pi*time.hour/24))
df_15min['yhour'] = df_15min['time'].map(lambda time: np.cos(2*np.pi*time.hour/24))
df_15min['xday'] = df_15min['dayofweek'].map(lambda day: np.sin(day * (2 * np.pi / 7)))
df_15min['yday'] = df_15min['dayofweek'].map(lambda day: np.cos(day * (2 * np.pi / 7)))
df_15min = df_15min.loc[df_15min['time'].between(datetime.time(6, 0),datetime.time(23, 59))]
df_15min = df_15min.drop(columns=['Unnamed: 0'])
df_15min = df_15min.dropna()
df_15min = df_15min.reset_index(drop=True)

# split into train and test
training_data, testing_data = train_test_split(df_15min, test_size=0.2, shuffle=False)
df_15min.to_csv("data/preprocess.csv")
training_data.to_csv("data/train.csv")
testing_data.to_csv("data/test.csv")

# import training dataset and process it
data_file = "data/train.csv"
data = pd.read_csv(data_file)
data['time'] = data['time'].map(lambda time: datetime.datetime.strptime(time, '%H:%M:%S').time())
data['date'] = data['date'].map(lambda date: datetime.datetime.strptime(date, '%Y-%m-%d').date())
data['timestamp'] = pd.to_datetime(data['timestamp'])
data["weekend"] = data["weekend"].astype(int)
data["n+1"] = data["n+1"].astype(int)
data = data.loc[data['time'].between(datetime.time(6, 0),datetime.time(23, 59))]
data = data.reset_index(drop=True)

# split to input output
x = data[['weekend', 'n', 'xhour', 'yhour', 'xday', 'yday']]
y = data[['n+1']]

# get feature importance
lgb = LGBMRegressor(objective='quantile',alpha=0.5)
lgb.fit(x, y)
lgb.booster_.feature_importance()
# importance of each attribute
fea_imp_ = pd.DataFrame({'cols':x.columns, 'fea_imp':lgb.feature_importances_})
fea_imp_.loc[fea_imp_.fea_imp > 0].sort_values(by=['fea_imp'], ascending = False)

# save feature importance graph
ax = lightgbm.plot_importance(lgb)
ax.figure.savefig("data/feature_importance.png")

score = make_scorer(my_custom_loss_func, greater_is_better=False)
tcsv = TimeSeriesSplit(gap=0, max_train_size=None, n_splits=5, test_size=None)
min_features_to_select = 1
selector = RFECV(lgb, step = 1, cv = tcsv, min_features_to_select = min_features_to_select, scoring=score)
selector = selector.fit(x, y)
#print(selector.support_)
fea_rank_ = pd.DataFrame({'cols':x.columns, 'fea_rank':selector.ranking_})
fea_rank_.loc[fea_rank_.fea_rank > 0].sort_values(by=['fea_rank'], ascending = True)

#print("Optimal number of features : %d" % selector.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure(figsize=(10, 6))
plt.xlabel("Number of features selected", fontsize=15)
plt.ylabel("Cross validation score (pinball loss)", fontsize=15)
plt.title("Recursive Feature Elimination ", fontsize=20)
plt.plot(range(min_features_to_select, len(selector.grid_scores_) + min_features_to_select),np.abs(selector.grid_scores_)
         , marker='x')
plt.legend(['cv1', 'cv2', 'cv3', 'cv4', 'cv5'], fontsize = 10)
plt.text(5,22,f'1 Feature = {abs(round(np.mean(selector.grid_scores_[0,:]),2))}')
plt.text(5,20,f'2 Features = {abs(round(np.mean(selector.grid_scores_[1,:]),2))}')
plt.text(5,18,f'3 Features = {abs(round(np.mean(selector.grid_scores_[2,:]),2))}')
plt.text(5,16,f'4 Features = {abs(round(np.mean(selector.grid_scores_[3,:]),2))}')
plt.text(5,14,f'5 Features = {abs(round(np.mean(selector.grid_scores_[4,:]),2))}')
plt.text(5,12,f'6 Features = {abs(round(np.mean(selector.grid_scores_[5,:]),2))}')
plt.tick_params(axis='both', which='major', labelsize=12, length = 7)
plt.savefig("data/featureselection.png")

#plt.show()


