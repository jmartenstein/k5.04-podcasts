import pandas as pd
import numpy as np

#from numba import jit

import sklearn.metrics as ms
import sklearn.model_selection as mds
import sklearn.neural_network as nn
import sklearn.ensemble as en

import catboost as cb

DATA_DIR = "../data/kaggle"

NUM_TRIALS = 2
NUM_FOLDS  = 3

df_train = pd.read_csv(f"{DATA_DIR}/train.csv")
#df_train = pd.read_csv(f"{DATA_DIR}/podcast_dataset.csv")
#df_test = pd.read_csv(f"{DATA_DIR}/test.csv")

#df_ = pd.concat([df_train, df_test])

print("### Describe Training Data ###")
print(df_train.info())
print()

#print(numba.__version__)

x_colnames = [ "Podcast_Name", "Episode_Title", "Genre",
               "Host_Popularity_percentage", "Publication_Day",
               "Publication_Time", "Guest_Popularity_percentage",
               "Number_of_Ads", "Episode_Sentiment",
               "Episode_Length_minutes"
             ]
y_colname = "Listening_Time_minutes"

X = df_train[ x_colnames ]
y = df_train[ y_colname ].values.ravel()

X_train, X_test, y_train, y_test = mds.train_test_split( X, y,
                                                         #random_state=42,
                                                         test_size=0.2
                                                       )
categorical_features_indices = [X.columns.get_loc(col) for col in X.columns if X[col].dtype == 'object']

cv_dataset = cb.Pool( data=X_train,
                      label=y_train,
                      cat_features=categorical_features_indices )

reg = cb.CatBoostRegressor( loss_function='RMSE',
                            verbose=0 )

params = { 'iterations':  50,
           'learning_rate': 0.1,
           'loss_function': 'RMSE',
           'depth': 6
         }

inner_cv = mds.KFold( n_splits=NUM_FOLDS, shuffle=True )
outer_cv = mds.KFold( n_splits=NUM_FOLDS, shuffle=True )

#scores = cb.cv( cv_dataset,
#                params,
#                fold_count=NUM_FOLDS )

param_grid = { 'iterations':  [ 75 ],
               'learning_rate': [ 0.1, 0.15, 0.2 ],
               'loss_function': [ 'RMSE' ],
               'depth': [ 12 ]
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

#nested_scores = mds.cross_val_score( reg_grid,
#                                     cat_features=categorical_features_indices,
#                                     X=X_train, y=y_train,
#                                     cv=outer_cv
#                                   )

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

