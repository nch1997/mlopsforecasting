import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
import lightgbm
from sklearn.model_selection import train_test_split
import datetime
import joblib
import os
import json
import random
import yaml

params = yaml.safe_load(open("params.yaml"))["evaluation"]
random.seed(params["seed"])

# open testfile
test_file = "data/test.csv"
df = pd.read_csv(test_file)
df = df.drop(("Unnamed: 0"), axis=1)
df['timestamp'] = pd.to_datetime(df['timestamp'])
time_str = df.time.unique()
df['time'] = pd.to_datetime(df['time']).dt.time
df['date'] = pd.to_datetime(df['date']).dt.date
df["weekend"] = df["weekend"].astype(int)
df["n+1"] = df["n+1"].astype(int)

x_test = df[['n', 'xhour', 'yhour', 'xday', 'yday']]
y_test = df[['n+1']]


model_dir = "model"
alphas = [0.05, 0.25, 0.5, 0.75, 0.95]
model_path = ["tunedmodel0.05.txt", "tunedmodel0.25.txt", "tunedmodel0.5.txt",
              "tunedmodel0.75.txt", "tunedmodel0.95.txt"]
models = {}
for file, key in zip(model_path,alphas):
    models[key] = lightgbm.Booster(model_file=os.path.join(model_dir, file))


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
    return full_result, np.mean(loss_list), sharpness, coverage

# evaluate test set
result, loss, sharpness, coverage = run_eval_model(models, x_test, y_test)
# save 
with open("data/metrics.json", 'w') as outfile:
        json.dump({ "pinball loss": loss, "sharpness": sharpness, "coverage":coverage}, outfile)


# plot some figures

#get unique dates and randomly pick one to plot
unique_dates = pd.to_datetime(df['date']).dt.date.unique()
chosen_date = random.choice(unique_dates)
next_date = chosen_date + datetime.timedelta(days=1)

df_plot = df.loc[(df['date'].between(chosen_date,next_date)) & 
                (df['time'].between(datetime.time(6, 0),datetime.time(23, 59)))].reset_index()

x_plot = df_plot[['n', 'xhour', 'yhour', 'xday', 'yday']]
y_plot = df_plot[['n+1']]
result2, loss2, sharpness2, coverage2 = run_eval_model(models, x_plot, y_plot)

fig = plt.figure(figsize=(20,10))
ax1 = fig.add_subplot()
ax1.plot(range(len(df_plot['n+1'])), df_plot['n+1'], label = 'original', color='k') 
ax1.plot(range(len(df_plot['n+1'])), result2[0.25], color='lightblue')
ax1.plot(range(len(df_plot['n+1'])), result2[0.50], color='lightblue')
ax1.plot(range(len(df_plot['n+1'])), result2[0.75], color='lightblue')
ax1.plot(range(len(df_plot['n+1'])), result2[0.05], color='red')
ax1.plot(range(len(df_plot['n+1'])), result2[0.95], color='green')
ax1.fill_between(range(len(df_plot['n+1'])), result2[0.05], result2[0.95],color='lightblue')
#ax1.fill_between(range(len(df_sample['n+1'])), percentile25.tolist(), percentile75.tolist(),color='lightblue')
ax1.tick_params(axis='both', which='major', labelsize=14, length = 3)
ax1.legend(fontsize=15)
ax1.set_title('Vehicle Daily Flow', fontsize=20)
ax1.set_xlabel('Time', fontsize=20)
ax1.set_ylabel('Vehicle Flow', fontsize=20)
#ax1.set_xticks(range(len(df_sample['n+1'])))
#ax1.set_xticklabels(df_sample['time'], rotation = 45)
ax1.xaxis.set_major_locator(plt.MaxNLocator(36))
ax1.figure.savefig("data/sampledailyprediciton.png")