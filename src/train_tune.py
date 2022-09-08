import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
import datetime
import optuna  # pip install optuna
from sklearn.model_selection import TimeSeriesSplit
from optuna.integration import LightGBMPruningCallback
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice
from optuna.samplers import TPESampler
from sklearn.preprocessing import MinMaxScaler
import yaml
import random

# open params file
params = yaml.safe_load(open("params.yaml"))["train_tune"]
random.seed(params["seed"])

# open training file
training_file = "data/train.csv"
df = pd.read_csv(training_file)
df = df.drop(("Unnamed: 0"), axis=1)
df['timestamp'] = pd.to_datetime(df['timestamp'])
time_str = df.time.unique()
df['time'] = pd.to_datetime(df['time']).dt.time
df['date'] = pd.to_datetime(df['date']).dt.date
df["weekend"] = df["weekend"].astype(int)
df["n+1"] = df["n+1"].astype(int)

x_data = df[['n', 'xhour', 'yhour', 'xday', 'yday']]
y_data = df[['n+1']]

def pinball_loss(actual, forecast, alpha):
    """
    Calculate pinball loss
    d - actual value
    f - quantile forecast
    alpha - quantile
    """
    
    return [max(alpha*(d-f), (1-alpha)*(f-d)) for d,f in zip(actual,forecast)]

def run_eval_model(models, x, y):
    """
    Runs and evaluates results
    models - dictionary containing models
    x - input in dataframe format
    y - output in dataframe format
    """
    loss_list = []
    full_result = {}
    for quantile_alpha, lgb in models.items():
        result = lgb.predict(x)
        loss = pinball_loss(y["n+1"].to_numpy(), result, quantile_alpha)
        loss_list.append(np.mean(loss))   
        full_result[quantile_alpha] = result
        
    sharpness = np.mean(full_result[0.95] - full_result[0.05])
    
    coverage = [] 
    for i in range(len(y["n+1"].to_numpy())):
        if full_result[0.05][i] <= y["n+1"].to_numpy()[i] <= full_result[0.95][i]:
            coverage.append(1)
        else:
            coverage.append(0)
        
    coverage = sum(coverage)/len(coverage)
    return np.mean(loss_list), sharpness, coverage

def objective(trial, X, Y):
    """
    X - x data
    Y - y data
    """
    param_grid = {
        # "device_type": trial.suggest_categorical("device_type", ['gpu']),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "num_leaves": trial.suggest_int("num_leaves", 2, 512),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_data_in_leaf": trial.suggest_categorical("min_data_in_leaf", [15,20,45,85]),
        "lambda_l1": trial.suggest_loguniform("lambda_l1", 1e-9, 100),
        "lambda_l2": trial.suggest_loguniform("lambda_l2", 1e-9, 100),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 10),
        #"bagging_fraction": trial.suggest_uniform("bagging_fraction", 0.1, 1),
        #"bagging_freq": trial.suggest_int("bagging_freq", 0, 15),
        #"feature_fraction": trial.suggest_float("feature_fraction", 0.2, 0.95, step=0.1),
    }
    
    tcsv = TimeSeriesSplit(gap=0, max_train_size=None, n_splits=5, test_size=None)

    # to access the randomize parameters from trial.suggest_int etc
    # https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html
    # trial.params['parameter name']
    loss_scores = []
    sharpness_scores = []
    coverage_scores = []
    alphas = [0.05, 0.25, 0.5, 0.75, 0.95]
    models = {}
    
    for train_index, test_index in tcsv.split(X):
        x_train, x_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]  
        
        for alpha in alphas:
            lgb = LGBMRegressor(objective='quantile', **param_grid, alpha=alpha)
            # make sure early_stopping_rounds is 10% of n_estimators
            lgb.fit(x_train,
                    y_train,
                    eval_metric="quantile",
                    eval_set = [(x_test, y_test)],
                    early_stopping_rounds = round(trial.params['n_estimators']*0.1))
            #lgb.fit(x_train, y_train)
            # store models in dictionary
            models[alpha] = lgb

        loss, sharpness, coverage = run_eval_model(models, x_test, y_test)
        loss_scores.append(loss)
        sharpness_scores.append(sharpness)
        coverage_scores.append(coverage)
        
        
    return np.mean(loss_scores), np.mean(sharpness_scores), np.mean(coverage_scores)


# start hyperparameter tuning
# https://optuna.readthedocs.io/en/stable/faq.html#how-can-i-obtain-reproducible-optimization-results
sampler = TPESampler(seed=params["seed"])  # Make the sampler behave in a deterministic way.

study = optuna.create_study(directions=["minimize", "minimize", "maximize"], study_name="first test", sampler=sampler)
func = lambda trial: objective(trial, x_data, y_data)
study.optimize(func, n_trials=params["n_trials"])


# save trials in dataframe into folder
heuristicdf = study.trials_dataframe()
heuristicdf.rename(columns = {'values_0':"PinballLoss", 'values_1':'Sharpness', 'values_2':"Coverage"}, inplace=True)
heuristicdf.to_csv('data/trials.csv')


# use weighted custom metric to chose best model from hyperparameter tuning
scaler = MinMaxScaler()

data = heuristicdf[['PinballLoss','Sharpness']]
# scale the data
scaler.fit_transform(data)

# inverse the scaling for pinball, sharpness so lowest value = 1 and biggest = 0
scaled = pd.DataFrame(scaler.fit_transform(data),columns=['pinball', 'sharpness'])
scaled['pinball'] = 1 - scaled['pinball'] 
scaled['sharpness'] = 1 - scaled['sharpness'] 
scaled['coverage'] = heuristicdf[['Coverage']]
scaled['metrics'] = 0.15*scaled['sharpness'] + 0.15*scaled['pinball'] + 0.7*scaled['coverage']

max_value = scaled['metrics'].max()
max_value_idx = scaled['metrics'].idxmax()
best_model = heuristicdf[['number','PinballLoss','Sharpness','Coverage']].loc[max_value_idx]





# train a model using the parameters from hyperparameter tuning
tcsv = TimeSeriesSplit(gap=0, max_train_size=None, n_splits=5, test_size=None)
for idx, (train_index, test_index) in enumerate(tcsv.split(x_data)):
    if idx == 4:
        x_train, x_test = x_data.iloc[train_index], x_data.iloc[test_index]
        y_train, y_test = y_data.iloc[train_index], y_data.iloc[test_index]  

best_params = {}
best_params.update(study.trials[int(best_model['number'])].params)

print(best_params)

alphas = [0.05, 0.25, 0.5, 0.75, 0.95]
models = {}

# create multiple models and store in dictionary
for alpha in alphas:
    lgb = LGBMRegressor(objective='quantile', **best_params, alpha=alpha)

#     lgb.fit(x_train, y_train, eval_metric="quantile",
#           eval_set = [(x_test, y_test)], early_stopping_rounds = round(best_params['n_estimators']*0.1))
    lgb.fit(x_train, y_train)

    
    # save model
    # https://stackoverflow.com/questions/55208734/save-lgbmregressor-model-from-python-lightgbm-package-to-disc
    file = "model/tunedmodel" + str(alpha) + ".txt"
    lgb.booster_.save_model(file)
    
    # store models in dictionary
    models[alpha] = lgb

optunaplot = optuna.visualization.plot_pareto_front(study, target_names=["loss", "sharpness", "coverage"])
optunaplot.write_image("data/tuning.png")