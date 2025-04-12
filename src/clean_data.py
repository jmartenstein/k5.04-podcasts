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

TARGET_COLUMN = "Listening_Time_minutes"


### FUNCTIONS ###

def get_outfile_name( in_file ):

    f_split = in_file.split(".")
    now = datetime.now()

    s_date = now.strftime("%Y%m%d")
    s_time = now.strftime("%H%M%S")

    return f"{f_split[0]}.clean.{s_date}.{s_time}.{f_split[1]}"


### MAIN ###

parser = argparse.ArgumentParser( description='Pre-process data from Podcasts dataset' )
parser.add_argument('-w', '--write', action='store_true', help='Write processed output to files')

args = vars( parser.parse_args() )

# load train data
df_train = pd.read_csv(f"{DATA_DIR}/train.csv")
df_test  = pd.read_csv(f"{DATA_DIR}/test.csv")
df_orig  = pd.read_csv(f"{ORIG_DIR}/podcast_dataset.csv")

# add the target column for the test data
df_test["Listening_Time_minutes"] = np.nan

df_ = pd.concat([df_train, df_test])
#df_.info()

df_["Length_Clean" ] = df_[ "Episode_Length_minutes" ].apply( lambda v: 120.5 if v > 120.5 else v )
df_["IsGuest"] = df_["Guest_Popularity_percentage"].apply( lambda v: 0 if np.isnan(v) else 1 )
df_["Num_Ads"] = df_["Number_of_Ads"].fillna(0)
df_["Name_And_Episode"] = df_["Podcast_Name"] + " " + df_["Episode_Title"]

df_["Episode_Completion"] = df_["Listening_Time_minutes"] / df_["Length_Clean"]
df_['Episode_Completion'] = df_['Episode_Completion'].apply( lambda v: 1 if v > 1 else v )

median_impute = imp.SimpleImputer(missing_values=np.nan, strategy='median', add_indicator=True)
median_impute.fit(df_[['Length_Clean']])
length_features = median_impute.transform(df_[['Length_Clean']])

df_['Length_Simple_Impute'] = length_features[:,0]
df_['Length_Missing'] = length_features[:,1]

hist, bin_edges = np.histogram( df_["Num_Ads"],
                                bins=[ df_["Num_Ads"].min(),
                                       0.5, 1.5, 2.5,
                                       df_["Num_Ads"].max()
                                     ]
                               )
df_["Num_Ads_Bin"] = pd.cut( df_[ "Num_Ads" ], bins=bin_edges,
                             labels=[0, 1, 2, 3], include_lowest=True
                           )

genre_enc = pre.TargetEncoder( cv=3 )
adsbin_enc = pre.TargetEncoder( cv=3 )
name_enc  = pre.TargetEncoder( cv=3 )

# prep the cleaned training data to write
df_train_clean = df_[ df_[ TARGET_COLUMN ].notna() ].copy()

genre_enc.fit( df_train_clean[["Genre"]], df_train_clean[TARGET_COLUMN] )
df_train_clean["Genre_Encoded"] = genre_enc.transform(df_train_clean[["Genre"]])

adsbin_enc.fit( df_train_clean[["Num_Ads_Bin"]], df_train_clean[TARGET_COLUMN] )
df_train_clean["Ads_Bin_Encoded"] = adsbin_enc.transform(df_train_clean[["Num_Ads_Bin"]])

name_enc.fit( df_train_clean[["Name_And_Episode"]], df_train_clean[TARGET_COLUMN] )
df_train_clean["Name_And_Episode_Encoded"] = name_enc.transform(df_train_clean[["Name_And_Episode"]])

#iter_impute = imp.IterativeImputer(missing_values=np.nan, max_iter=20)
#impute_cols = ["Length_Clean", "Completion_Simple_Impute", "Host_Popularity_percentage",
#               "Num_Ads_Bin", "Ads_Bin_Encoded", "Genre_Encoded", "Name_And_Episode_Encoded"]
#iter_impute.fit(df_train_clean[impute_cols])
#df_train_clean["Length_Iter_Impute" ] = iter_impute.transform(df_train_clean[impute_cols])[:,0]

# prep the cleaned test data to write
df_test_clean = df_[ df_[ TARGET_COLUMN ].isna() ].copy()
df_test_clean.drop( TARGET_COLUMN, axis=1, inplace=True )

df_test_clean["Genre_Encoded"] = genre_enc.transform(df_test_clean[["Genre"]])
df_test_clean["Ads_Bin_Encoded"] = adsbin_enc.transform(df_test_clean[["Num_Ads_Bin"]])
df_test_clean["Name_And_Episode_Encoded"] = name_enc.transform(df_test_clean[["Name_And_Episode"]])

#iter_impute.fit(df_test_clean[impute_cols])
#df_test_clean["Length_Iter_Impute" ] = iter_impute.transform(df_test_clean[impute_cols])[:,0]

if args["write"]:

    train_outfile = get_outfile_name("train.csv")
    print(f"write train shape: {df_train_clean.shape} to file: {train_outfile}")
    df_train_clean.to_csv(f"{DATA_DIR}/{train_outfile}", index=False)

    test_outfile = get_outfile_name("test.csv")
    print(f"write test shape:  {df_test_clean.shape} to file: {test_outfile}")
    df_test_clean.to_csv(f"{DATA_DIR}/{test_outfile}", index=False)

else:

    print(df_train_clean.info())

    print("No output written to files")
    print(f"  train shape: {df_train_clean.shape}")
    print(f"  test shape:  {df_test_clean.shape}")

