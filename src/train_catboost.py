import pandas as pd
import numpy as np

#from numba import jit

import sklearn.metrics as ms
import sklearn.model_selection as mds
import sklearn.neural_network as nn
import sklearn.ensemble as en

import catboost as cb

DATA_DIR = "../data/kaggle"

NUM_FOLDS  = 3
ITERS      = 10000
PERIOD      = int(ITERS / 10)

infile_str = "rollback.20250429.134917"

df_train = pd.read_csv(f"{DATA_DIR}/train.{infile_str}.csv")
#df_train = pd.read_csv(f"{DATA_DIR}/podcast_dataset.csv")
df_test = pd.read_csv(f"{DATA_DIR}/test.{infile_str}.csv")

#df_ = pd.concat([df_train, df_test])

#print(numba.__version__)

all_colnames = df_train.columns.tolist()
y_colname = "Listening_Time_minutes"

substr_columns = [ s for s in all_colnames if ("Poly" in s) and ("T1" in s) and ("T2" in s) ]
#substr_columns.remove("Poly T1_Ads_Mins T2_All_Cats")

#substr_columns.remove("Poly Length_Impute^2 T2_All_Cats")
#substr_columns = []
exclude_columns = [ "id", "Num_Ads", "Episode_Completion_percentage", "Number_of_Ads",
                    "Length_Impute", "Length_Simple_Impute", "Episode_Title", "Length_Clean",
                    "Cat_Secs", "Cat_Mins", y_colname ]
x_colnames = [ c for c in all_colnames if c not in exclude_columns ]
#x_colnames = x_colnames + substr_columns

x_colnames = ["T1_Ads_Mins", "T1_Cat_Secs", "Episode_Length_minutes", "Number_of_Ads",
              "Host_Popularity_percentage", "T1_All_Cats", "Guest_Popularity_percentage",
              "Poly Length_Impute T2_All_Cats", "Poly T1_Cat_Secs T2_Name_And_Episode",
              "All_Cats", "Ads_Mins", "T1_Name_And_Episode" ]

X = df_train[ x_colnames ]
y = df_train[ y_colname ].values.ravel()

print("### Describe Training Data ###")
print(df_train[ x_colnames ].info())
print()

X_train, X_test, y_train, y_test = mds.train_test_split( X, y,
                                                         #random_state=42,
                                                         test_size=0.2
                                                       )
categorical_features_indices = [X.columns.get_loc(col) for col in X.columns if X[col].dtype == 'object']

print(f"Training data shape: {X_train.shape}")

GRID = True

if GRID:

    reg = cb.CatBoostRegressor( verbose=1,
                                metric_period=PERIOD
                              )

    param_grid = { 'iterations':  [ ITERS ],
                   #'learning_rate': [ 0.02 ],
                   'loss_function': [ 'RMSE' ],
                   'boosting_type': [ 'Plain' ]
                 }

    kfold = mds.KFold(n_splits=NUM_FOLDS, shuffle=True)

    print("### Grid Search and Fit ###")
    reg_grid = mds.GridSearchCV( estimator=reg,
                                 param_grid=param_grid,
                                 scoring='neg_root_mean_squared_error',
                                 cv=kfold,
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
                                learning_rate=0.02,
                                od_type='Iter',
                                od_wait=5
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
df_test[["id", "Listening_Time_minutes"]].to_csv("../data/kaggle/catboost_cv_submission.csv", index=False)
