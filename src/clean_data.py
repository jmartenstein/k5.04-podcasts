import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.experimental import enable_iterative_imputer

import sklearn.preprocessing as pre
import sklearn.impute as imp

import argparse

from datetime import datetime

### CONSTANTS ###

DATA_DIR = "../data/kaggle"
ORIG_DIR = "../data/original"

TARGET1 = "Listening_Time_minutes"
TARGET2 = "Episode_Completion_percentage"

### FUNCTIONS ###

def get_datetime_string( str_note ):

    now = datetime.now()

    s_date = now.strftime("%Y%m%d")
    s_time = now.strftime("%H%M%S")

    return f"{str_note}.{s_date}.{s_time}"

def bin_feature( s_feature, l_bins, l_labels ):
    pass

### MAIN ###

parser = argparse.ArgumentParser( description='Pre-process data from Podcasts dataset' )
parser.add_argument('-w', '--write', action='store_true', help='Write processed output to files')

args = vars( parser.parse_args() )

# load train data
df_train = pd.read_csv(f"{DATA_DIR}/train.csv")
df_test  = pd.read_csv(f"{DATA_DIR}/test.csv")
df_orig  = pd.read_csv(f"{ORIG_DIR}/podcast_dataset.csv")

# add the target column for the test data
df_test[TARGET1] = np.nan

df_ = pd.concat([df_train, df_test])
#df_.info()

df_["Length_Clean" ] = df_[ "Episode_Length_minutes" ].apply( lambda v: 120.5 if v > 120.5 else v )

df_["IsGuest"] = df_["Guest_Popularity_percentage"].apply( lambda v: 0 if np.isnan(v) else 1 )
df_["Num_Ads"] = df_["Number_of_Ads"].fillna(0).copy()
df_["Name_And_Episode"] = df_["Podcast_Name"] + " " + df_["Episode_Title"]
df_["Pub_Day_And_Time"] = df_["Publication_Day"] + " " + df_["Publication_Time"]

iter_impute = imp.IterativeImputer(missing_values=np.nan, add_indicator=True, max_iter=20)
impute_cols = ["Length_Clean", "Host_Popularity_percentage","Num_Ads",
               "Guest_Popularity_percentage" ]

iter_impute.fit(df_[impute_cols])
#df_train_clean["Length_Iter_Impute" ] = iter_impute.transform(df_train_clean[impute_cols])[:,0]

#iter_impute.fit(df_test_clean[impute_cols])
length_features = iter_impute.transform(df_[impute_cols])

#median_impute = imp.SimpleImputer(missing_values=np.nan, strategy='median', add_indicator=True)
#median_impute.fit(df_[['Length_Clean']])
#length_features = median_impute.transform(df_[['Length_Clean']])

df_['Length_Impute'] = length_features[:,0]
df_['Length_Missing'] = length_features[:,1]

df_[ TARGET2 ] = df_[TARGET1] / df_["Length_Impute"]
df_[ TARGET2 ] = df_[TARGET2].apply( lambda v: 1 if v > 1 else v )

feature = "Num_Ads"
hist, bin_edges = np.histogram( df_[feature],
                                bins=[ df_[feature].min(),
                                       0.5, 1.5, 2.5,
                                       df_[feature].max()
                                     ]
                               )
df_[f"{feature}_Bin"] = pd.cut( df_[feature], bins=bin_edges,
                             labels=[0, 1, 2, 3], include_lowest=True
                           )

df_["Ad_Rate_Per_Hour"] = df_["Num_Ads"] * (df_["Length_Impute"] / 60)
df_["Ads_And_Host_Pop"] = (df_["Num_Ads"] * 100) + df_["Host_Popularity_percentage"]
df_["Ads_Sentiment"] = str(df_["Num_Ads_Bin"]) + " " + df_["Episode_Sentiment"]


# prep the cleaned training data to write
df_train_clean = df_[ df_[ TARGET1 ].notna() ].copy()

# prep the cleaned test data to write
df_test_clean = df_[ df_[ TARGET1 ].isna() ].copy()
df_test_clean.drop( TARGET1, axis=1, inplace=True )

features = [ "Genre", "Num_Ads", "Num_Ads_Bin", "Name_And_Episode", "Publication_Day",
             "Publication_Time", "Podcast_Name", "Episode_Sentiment",
             "Pub_Day_And_Time", "Ads_Sentiment" ]
targets = [ TARGET1, TARGET2 ]

enc = pre.TargetEncoder( cv=3 )

for f in features:
    i = 1
    for t in targets:
        encoded_feature = f"T{i}_{f}_{t}"
        #print(encoded_feature)
        enc.fit( df_train_clean[[f]], df_train_clean[t] )
        df_train_clean[encoded_feature] = enc.transform(df_train_clean[[f]])
        df_test_clean[encoded_feature] = enc.transform(df_test_clean[[f]])
        i+=1

poly_features = [ "Length_Impute", "Num_Ads",
                  "T2_Name_And_Episode_Episode_Completion_percentage",
                  "T1_Name_And_Episode_Listening_Time_minutes",
                  "T2_Num_Ads_Episode_Completion_percentage",
                  "Host_Popularity_percentage",
                  "T2_Pub_Day_And_Time_Episode_Completion_percentage"
                ]

poly = pre.PolynomialFeatures(degree=2)
poly.fit( df_train_clean[ poly_features ] )
X_poly_train = poly.transform( df_train_clean[ poly_features ] )
X_poly_test = poly.transform( df_test_clean[ poly_features ] )

df_poly_train = pd.DataFrame(X_poly_train)
df_poly_test = pd.DataFrame(X_poly_test)

df_train_clean = df_train_clean.join(df_poly_train.add_prefix('Poly_'))
df_test_clean = df_test_clean.join(df_poly_test.add_prefix('Poly_'))

if args["write"]:

    outfile_str = get_datetime_string( "poly" )

    train_outfile = f"train.{outfile_str}.csv"
    print(f"write train shape: {df_train_clean.shape} to file: {train_outfile}")
    df_train_clean.to_csv(f"{DATA_DIR}/{train_outfile}", index=False)

    test_outfile = f"test.{outfile_str}.csv"
    print(f"write test shape:  {df_test_clean.shape} to file: {test_outfile}")
    df_test_clean.to_csv(f"{DATA_DIR}/{test_outfile}", index=False)

else:

    print(df_train_clean.info())

    print("No output written to files")
    print(f"  train shape: {df_train_clean.shape}")
    print(f"  test shape:  {df_test_clean.shape}")

