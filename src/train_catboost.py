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
ITERS      = 150
PERIOD      = int(ITERS / 15)

df_train = pd.read_csv(f"{DATA_DIR}/train.clean.20250412.092425.csv")
#df_train = pd.read_csv(f"{DATA_DIR}/podcast_dataset.csv")
df_test = pd.read_csv(f"{DATA_DIR}/test.clean.20250412.092431.csv")

#df_ = pd.concat([df_train, df_test])

print("### Describe Training Data ###")
print(df_train.info())
print()

#print(numba.__version__)

all_colnames = df_train.columns.tolist()
y_colname = "Listening_Time_minutes"

exclude_columns = [ y_colname, "id", "Episode_Completion" ]
x_colnames = [ c for c in all_colnames if c not in exclude_columns ]

X = df_train[ x_colnames ]
y = df_train[ y_colname ].values.ravel()

X_train, X_test, y_train, y_test = mds.train_test_split( X, y,
                                                         #random_state=42,
                                                         test_size=0.2
                                                       )
categorical_features_indices = [X.columns.get_loc(col) for col in X.columns if X[col].dtype == 'object']

print(f"Training data shape: {X_train.shape}")

cv_dataset = cb.Pool( data=X_train,
                      label=y_train,
                      cat_features=categorical_features_indices )



reg = cb.CatBoostRegressor( verbose=1,
                            metric_period=PERIOD )

param_grid = { 'iterations':  [ ITERS ],
               'learning_rate': [ 0.05 ],
               'loss_function': [ 'RMSE' ],
               'depth': [ 9 ],
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

print("### Fit Results ###")

print(f"Best parameters: {best_params}")
print(f"Best score:      {score}")
print()

print("### Feature Importance ###")

importances = model.feature_importances_
feature_imp_df = pd.DataFrame(
    {'Feature': x_colnames, 'Gini Importance': importances} ).sort_values(
         'Gini Importance', ascending=False)
print(feature_imp_df)
print()

# Make predictions on the test data
y_pred = model.predict(X_test)

print("### Run Model against Test Set ###")

# Evaluate the model
rmse = np.sqrt(ms.mean_squared_error(y_test, y_pred))
print(f"RMSE for Test Data: {rmse}")

#print(df_test.info())
# Write predictions to submissions file
df_test["Listening_Time_minutes"] = model.predict(df_test[x_colnames])
df_test[["id", "Listening_Time_minutes"]].to_csv("../data/kaggle/catboost_submission.csv", index=False)
