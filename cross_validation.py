import os
os.environ["KERAS_BACKEND"] = "torch"

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, Dropout
from keras.losses import BinaryCrossentropy
from keras.models import Sequential
from keras.optimizers import Adam

import itertools
import lightgbm as lgbm
import numpy as np
import pandas as pd

def train_L2(a, fi, p, X, y, bonds):
    metrics = []
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    for i, (train_index, test_index) in enumerate(skf.split(X, bonds)):
        X_train = X[train_index]
        y_train = y[train_index]

        X_test = X[test_index]
        y_test = y[test_index]

        model = Ridge(alpha=a, fit_intercept=fi, positive=p)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        metrics.append([mse, mae, r2])

    metrics = np.array(metrics)
    mean = np.mean(metrics, axis=0)
    std = np.std(metrics, axis=0)
    return mean, std, metrics

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

def train_neural_network(n, lr, dr, X, y, bonds):
    def build_NN(shape):
        model = Sequential()
        model.add(Input(shape=(shape,)))
        model.add(Dense(n, activation='relu'))
        model.add(Dropout(dr))
        model.add(Dense(n, activation='relu'))
        model.add(Dense(1, activation='linear'))
        return model

    metrics = []
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    for i, (train_index, test_index) in enumerate(skf.split(X, bonds)):
        X_train = X[train_index]
        y_train = y[train_index]

        X_test = X[test_index]
        y_test = y[test_index]

        model = build_NN(shape=X_train.shape[1])

        model.compile(
            loss="mse",
            optimizer=Adam(learning_rate=lr),
        )
    
        model.fit(
            X_train, y_train, 
            batch_size=5120, epochs=1000,
            validation_data=(X_test, y_test), 
            callbacks=[EarlyStopping(
                monitor="val_loss", 
                patience=20, 
                restore_best_weights=True
            )],
            verbose=0
        )
    
        y_pred = model.predict(X_test, verbose=0)
    
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

# Prepare data
df = df[df["bond"].isin(["CO", "CC", "CN", "NO", "NN"])]
df = df[["bond", "r", "N"]]
df = df.reset_index(drop=True)

y = df["N"]
X = df[["bond", "r"]]

transformer = ColumnTransformer(
    [("onehot", OneHotEncoder(), ["bond"])],
    remainder="passthrough"
)

X = transformer.fit_transform(X)

# Train L2 model
alpha = [0.1, 1, 10]
fit_intercept = [True, False]
positive = [True, False]

best_score_mean = float("inf")
best_score_std = float("inf")
best_params = None
best_validation = None

combinations = list(itertools.product(alpha, fit_intercept, positive))
for combination in combinations:
    print(f"[LR] Starting: {combination}")
    a = combination[0]
    fi = combination[1]
    p = combination[2]
    metrics = train_L2(a, fi, p, X, y, df["bond"])

    if metrics[0][1] < best_score_mean:
        best_score_mean = metrics[0][1]
        best_score_std = metrics[1][1]
        best_validation = metrics[2][:,1]
        best_params = combination

best_score_mean = round(best_score_mean, 3)
best_score_std = round(best_score_std, 3)

print("[LR] Best:", best_score_mean, best_score_std, best_params)
print("[LR] Validation:", best_validation)

# Train GBRF
num_leaves = [11, 31, 51]
learning_rate = [0.1, 0.01, 0.001]
max_depth = [-1, 5, 10]

best_score_mean = float("inf")
best_score_std = float("inf")
best_params = None
best_validation = None

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

# Train FF-NN
neurons = [512, 2048, 5120]
dropouts = [0.25, 0.5, 0.75]
learning_rates = [0.001, 0.0001, 0.00001]

best_score_mean = float("inf")
best_score_std = float("inf")
best_params = None
best_validation = None

combinations = list(itertools.product(neurons, dropouts, learning_rates))
for combination in combinations:
    print(f"[NN] Starting: {combination}")
    n = combination[0]
    lr = combination[1]
    dr = combination[2]
    metrics = train_neural_network(n, lr, dr, X, y, df["bond"])

    if metrics[0][1] < best_score_mean:
        best_score_mean = metrics[0][1]
        best_score_std = metrics[1][1]
        best_validation = metrics[2][:,1]
        best_params = combination

best_score_mean = round(best_score_mean, 3)
best_score_std = round(best_score_std, 3)

print("[NN] Best:", best_score_mean, best_score_std, best_params)
print("[NN] Validation:", best_validation)
