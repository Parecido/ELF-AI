from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

import itertools
import lightgbm as lgbm
import numpy as np
import pandas as pd

def train_random_forest(nl, lr, md, X, y, bonds):
    metrics = []
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    for i, (train_index, test_index) in enumerate(skf.split(X, bonds)):
        X_train = X[train_index]
        y_train = y[train_index]

        X_test = X[test_index]
        y_test = y[test_index]

        model = lgbm.LGBMRegressor(
            device="gpu",
            n_estimators=1000, 
            num_leaves=nl, 
            learning_rate=lr, 
            max_depth=md, 
            verbose=-1
        )
    
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[
                lgbm.early_stopping(stopping_rounds=20),
            ]
        )
    
        y_pred = model.predict(X_test)
    
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        metrics.append([mse, mae, r2])

    metrics = np.array(metrics)
    mean = np.mean(metrics, axis=0)
    std = np.std(metrics, axis=0)
    return mean, std, metrics

# Load data
df = pd.read_csv("data/datasets.csv")
df = df[[
    "bond", "r", "N", 
    "a_env1", "b_env1",
    "a_env2", "b_env2",
    "a_env3", "b_env3",
    "a_env4", "b_env4",
    "a_env5", "b_env5",
]]

# Prepare data
df = df[df["bond"].isin(["CO", "CC", "CN", "NO", "NN"])]
df = df.reset_index(drop=True)

y = df["N"]
X = df[["bond", "r", 
        "a_env1", "b_env1", 
        "a_env2", "b_env2", 
        "a_env3", "b_env3", 
        "a_env4", "b_env4", 
        "a_env5", "b_env5"]]

transformer = ColumnTransformer(
    [("onehot", OneHotEncoder(), [
        "bond", 
        "a_env1", "b_env1", 
        "a_env2", "b_env2", 
        "a_env3", "b_env3", 
        "a_env4", "b_env4", 
        "a_env5", "b_env5"])],
    remainder="passthrough"
)

X = transformer.fit_transform(X)

# Hyperparameters
num_leaves = [11, 31, 51]
learning_rate = [0.1, 0.01, 0.001]
max_depth = [-1, 5, 10]

# Validation output
best_score_mean = float("inf")
best_score_std = float("inf")
best_params = None
best_validation = None

# Find best params
combinations = list(itertools.product(num_leaves, learning_rate, max_depth))
for combination in combinations:
    print(f"[RF] Starting: {combination}")
    nl = combination[0]
    lr = combination[1]
    md = combination[2]
    metrics = train_random_forest(nl, lr, md, X, y, df["bond"])

    if metrics[0][1] < best_score_mean:
        best_score_mean = metrics[0][1]
        best_score_std = metrics[1][1]
        best_validation = metrics[2][:,1]
        best_params = combination

best_score_mean = round(best_score_mean, 3)
best_score_std = round(best_score_std, 3)

print("[RF] Best:", best_score_mean, best_score_std, best_params)
print("[RF] Validation:", best_validation)
