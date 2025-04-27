import pandas as pd
import numpy as np

#from numba import jit

import sklearn.metrics as ms
import sklearn.model_selection as mds
import sklearn.neural_network as nn
import sklearn.ensemble as en

import catboost as cb

DATA_DIR = "../data/kaggle"

NUM_FOLDS  = 2
ITERS      = 2500
PERIOD      = int(ITERS / 25)

infile_str = "poly3.20250426.213026"

df_train = pd.read_csv(f"{DATA_DIR}/train.{infile_str}.csv")
#df_train = pd.read_csv(f"{DATA_DIR}/podcast_dataset.csv")
df_test = pd.read_csv(f"{DATA_DIR}/test.{infile_str}.csv")

#df_ = pd.concat([df_train, df_test])

print("### Describe Training Data ###")
print(df_train.info())
print()

#print(numba.__version__)

all_colnames = df_train.columns.tolist()
y_colname = "Listening_Time_minutes"

exclude_columns = [ "id", "Num_Ads", "Episode_Completion_percentage", "Number_of_Ads",
                    "Episode_Length_minutes", "Length_Simple_Impute",
                    "Episode_Title", "Length_Clean", y_colname ]
x_colnames = [ c for c in all_colnames if c not in exclude_columns ]

X = df_train[ x_colnames ]
y = df_train[ y_colname ].values.ravel()

X_train, X_test, y_train, y_test = mds.train_test_split( X, y,
                                                         #random_state=42,
                                                         test_size=0.2
                                                       )
categorical_features_indices = [X.columns.get_loc(col) for col in X.columns if X[col].dtype == 'object']

print(f"Training data shape: {X_train.shape}")

GRID = False

if GRID:

    reg = cb.CatBoostRegressor( verbose=1,
                                metric_period=PERIOD
                              )

    param_grid = { 'iterations':  [ ITERS ],
                   'learning_rate': [ 0.2 ],
                   'loss_function': [ 'RMSE' ],
                   'depth': [ 7 ],
                   'l2_leaf_reg': [ 2 ],
                   'boosting_type': [ 'Plain' ]
                 }

    print("### Grid Search and Fit ###")
    reg_grid = mds.GridSearchCV( estimator=reg,
                                 param_grid=param_grid,
                                 scoring='neg_root_mean_squared_error',
                                 cv=NUM_FOLDS,
                                 verbose=4
                               )
    reg_grid.fit( X_train, y_train, cat_features=categorical_features_indices )
    print()

    best_params = reg_grid.best_params_
    model       = reg_grid.best_estimator_
    score       = reg_grid.best_score_

else:

    reg = cb.CatBoostRegressor( verbose=1,
                                metric_period=PERIOD,
                                iterations=ITERS,
                                od_type='Iter',
                                od_wait=10
                              )
    print("### Train CatBoost Regressor ###")
    reg.fit( X_train, y_train, cat_features=categorical_features_indices )
    print()

    model=reg


print("### Feature Importance ###")

importances = model.feature_importances_
feature_imp_df = pd.DataFrame(
    {'Feature': x_colnames, 'Gini Importance': importances} ).sort_values(
         'Gini Importance', ascending=False)
print(feature_imp_df[:20])
print()

if GRID:
    print("### Fit Results ###")

    print(f"Best parameters: {best_params}")
    print(f"Best score:      {score}")
    print()

# Make predictions on the test data
y_train_pred = model.predict(X_train)
y_pred = model.predict(X_test)

print("### Compare Train / Test Sets ###")
print(f"{ITERS} iterations")

# Evaluate the model on training data
rmse = np.sqrt(ms.mean_squared_error(y_train, y_train_pred))
print(f"RMSE for Train Data: {rmse}")

# Evaluate the model on training data
rmse = np.sqrt(ms.mean_squared_error(y_test, y_pred))
print(f"RMSE for Test Data: {rmse}")

#print(df_test.info())
# Write predictions to submissions file
df_test["Listening_Time_minutes"] = model.predict(df_test[x_colnames])
df_test[["id", "Listening_Time_minutes"]].to_csv("../data/kaggle/catboost_submission.csv", index=False)
