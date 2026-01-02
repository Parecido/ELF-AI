from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import lightgbm as lgbm
import numpy as np
import pandas as pd
import pickle

def train_random_forest(nl, lr, md, X, y, bonds):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.10, 
        random_state=42, 
        stratify=bonds
    )

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

    return model

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

# Optimized hyperparameters
num_leaves = 51
learning_rate = 0.1
max_depth = -1

# Train model
model = train_random_forest(
    num_leaves, learning_rate, max_depth, X, y, df["bond"]
)

# Save transformer and model
with open('model/transformer.pkl', 'wb') as f:
    pickle.dump(transformer, f)

with open('model/model.pkl', 'wb') as file:
    pickle.dump(model, file)
